"""Fontes de quadros: janela ao vivo (mss), imagem fixa e mercado sintético (demonstração)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from ..common.geometry import Rect
from ..config import CaptureConfig
from .frame import Frame, WindowInfo
from .window import WindowLocator, is_windows

log = logging.getLogger(__name__)


class LiveWindowSource:
    """Captura só a área da janela da IQ Option, em qualquer monitor.

    `occluders` devolve os retângulos (pixels físicos) das janelas do próprio app que NÃO puderam
    ser escondidas da captura. Se cobrirem o gráfico, essa área é marcada como inválida.
    """

    def __init__(self, cfg: CaptureConfig, occluders: Callable[[], list[Rect]] | None = None) -> None:
        self.cfg = cfg
        self._locator = WindowLocator(cfg)
        self._occluders = occluders or (lambda: [])
        self._window: WindowInfo | None = None
        self._last_check = -1e9
        self._local = threading.local()
        self._lock = threading.Lock()

    @property
    def window(self) -> WindowInfo | None:
        return self._window

    def refresh_window(self) -> WindowInfo | None:
        with self._lock:
            self._window = self._locator.find()
            self._last_check = time.monotonic()
            return self._window

    def adjust_window(self, panel_w: int) -> Rect | None:
        w = self.refresh_window()
        if w is None:
            return None
        panel = self._locator.adjust(w, panel_w)
        self.refresh_window()
        return panel

    def _sct(self):  # type: ignore[no-untyped-def]
        # mss precisa ser criado na mesma thread que captura
        if getattr(self._local, "sct", None) is None:
            import mss

            self._local.sct = mss.mss()
        return self._local.sct

    def grab(self) -> Frame:
        now = time.time()
        if not is_windows():
            return Frame(None, now, status_hint="unsupported")
        if time.monotonic() - self._last_check >= self.cfg.window_check_interval_s:
            self.refresh_window()
        w = self._window
        if w is None:
            return Frame(None, now, status_hint="no_window")
        if w.minimized:
            return Frame(None, now, window=w, status_hint="minimized")
        try:
            sct = self._sct()
            virt = sct.monitors[0]
            screen = Rect(virt["left"], virt["top"], virt["width"], virt["height"])
            r = w.rect.intersect(screen)
            if r is None or r.w < 100 or r.h < 100:
                return Frame(None, now, window=w, status_hint="no_window")
            shot = sct.grab({"left": r.x, "top": r.y, "width": r.w, "height": r.h})
            img = np.asarray(shot)[:, :, :3].copy()  # BGRA -> BGR
        except Exception:  # noqa: BLE001
            log.exception("falha na captura")
            return Frame(None, now, window=w, status_hint="capture_error")
        mask = None
        try:
            occ = self._occluders()
        except Exception:  # noqa: BLE001
            occ = []
        for o in occ:
            ov = o.intersect(r)
            if ov is not None:
                if mask is None:
                    mask = np.ones(img.shape[:2], dtype=bool)
                mask[ov.y - r.y : ov.bottom - r.y, ov.x - r.x : ov.right - r.x] = False
        return Frame(img, now, window=w, valid_mask=mask, source="live")

    def close(self) -> None:
        sct = getattr(self._local, "sct", None)
        if sct is not None:
            try:
                sct.close()
            except Exception:  # noqa: BLE001
                pass


class ImageSource:
    """Repete uma imagem salva (print da tela) como se fosse ao vivo."""

    def __init__(self, path: str | Path) -> None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"não foi possível abrir a imagem {path}")
        self._img = img
        self._win = WindowInfo(0, Path(path).name, "virtual", Rect(0, 0, img.shape[1], img.shape[0]))

    def grab(self) -> Frame:
        return Frame(self._img.copy(), time.time(), window=self._win, source="image")

    def close(self) -> None:
        pass


class SyntheticSource:
    """Mercado sintético ao vivo (modo demonstração, funciona sem a IQ Option)."""

    def __init__(self, theme: str = "dark", timeframe_min: int = 1) -> None:
        from ..replay.synthetic import LiveSynthetic, SynthSpec

        self._mkt = LiveSynthetic(SynthSpec(theme=theme, timeframe_min=timeframe_min, seed=int(time.time()) % 1000))
        self._win = WindowInfo(0, "Demonstração (sintético)", "virtual", Rect(0, 0, 1920, 1040))

    def grab(self) -> Frame:
        img, _ = self._mkt.frame()
        return Frame(img, time.time(), window=self._win, source="synthetic")

    def close(self) -> None:
        pass
