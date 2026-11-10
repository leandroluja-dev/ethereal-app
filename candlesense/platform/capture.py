"""Captura só a janela da IQ Option (ScreenCaptureKit), em resolução nativa (Retina).

Fallback: CGWindowListCreateImage (macOS antigo). Para testes e para o modo
debug offline existe FileSource, que lê capturas salvas.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np

from .windows import WindowInfo, window_by_id

log = logging.getLogger(__name__)

BGRA_FOURCC = 1111970369  # kCVPixelFormatType_32BGRA


class FrameSource(Protocol):
    name: str

    def grab(self) -> Optional[np.ndarray]:
        """Quadro BGR (numpy) ou None se não deu para capturar agora."""
        ...

    def close(self) -> None: ...


# ------------------------------------------------------------ CGImage -> numpy


def cgimage_to_bgr(cg) -> Optional[np.ndarray]:
    import Quartz

    if cg is None:
        return None
    w = Quartz.CGImageGetWidth(cg)
    h = Quartz.CGImageGetHeight(cg)
    bpr = Quartz.CGImageGetBytesPerRow(cg)
    bpp = Quartz.CGImageGetBitsPerPixel(cg)
    if w == 0 or h == 0 or bpp != 32:
        return None
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(cg))
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    px = buf.reshape(h, bpr)[:, : w * 4].reshape(h, w, 4)
    info = Quartz.CGImageGetBitmapInfo(cg)
    alpha = Quartz.CGImageGetAlphaInfo(cg)
    little = (info & Quartz.kCGBitmapByteOrderMask) == Quartz.kCGBitmapByteOrder32Little
    alpha_first = alpha in (
        Quartz.kCGImageAlphaPremultipliedFirst, Quartz.kCGImageAlphaFirst, Quartz.kCGImageAlphaNoneSkipFirst,
    )
    if little and alpha_first:  # memória: B G R A (o caso normal de captura de tela)
        return np.ascontiguousarray(px[:, :, :3])
    if little:  # A B G R
        return np.ascontiguousarray(px[:, :, 1:4])
    if alpha_first:  # A R G B
        return np.ascontiguousarray(px[:, :, 1:4][:, :, ::-1])
    return np.ascontiguousarray(px[:, :, :3][:, :, ::-1])  # R G B A


def backing_scale_for(win: WindowInfo) -> float:
    """Escala Retina do monitor onde está o centro da janela."""
    try:
        from AppKit import NSScreen

        screens = NSScreen.screens()
        main_h = screens[0].frame().size.height
        cx = win.x + win.w / 2
        cy = main_h - (win.y + win.h / 2)  # CG (topo) -> Cocoa (base)
        for s in screens:
            f = s.frame()
            if f.origin.x <= cx < f.origin.x + f.size.width and f.origin.y <= cy < f.origin.y + f.size.height:
                return float(s.backingScaleFactor())
        return float(screens[0].backingScaleFactor())
    except Exception:
        return 2.0


def _wait_handler(timeout: float = 3.0):
    """Cria um completion handler para APIs assíncronas e uma função que espera o resultado."""
    ev = threading.Event()
    box: dict = {}

    def handler(*args):
        box["args"] = args
        ev.set()

    def wait():
        if not ev.wait(timeout):
            raise TimeoutError("o macOS não respondeu a tempo")
        return box["args"]

    return handler, wait


# ------------------------------------------------------- ScreenCaptureKit


class ScreenCaptureKitSource:
    """Captura de uma janela só (mesmo se estiver atrás de outras), macOS 14+."""

    name = "ScreenCaptureKit"

    def __init__(self, win: WindowInfo) -> None:
        import ScreenCaptureKit as SCK

        self._SCK = SCK
        self.win = win
        if not hasattr(SCK, "SCScreenshotManager"):
            raise RuntimeError("SCScreenshotManager exige macOS 14 ou mais novo")
        self._setup()

    def _setup(self) -> None:
        SCK = self._SCK
        handler, wait = _wait_handler()
        SCK.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
            True, True, handler
        )
        content, err = wait()
        if content is None:
            raise RuntimeError(f"ScreenCaptureKit sem acesso ao conteúdo: {err}")
        scwin = None
        for w in content.windows():
            if int(w.windowID()) == self.win.window_id:
                scwin = w
                break
        if scwin is None:
            raise RuntimeError("janela da IQ Option não está disponível para captura")
        self._filter = SCK.SCContentFilter.alloc().initWithDesktopIndependentWindow_(scwin)
        try:
            scale = float(self._filter.pointPixelScale())
        except Exception:
            scale = backing_scale_for(self.win)
        frame = scwin.frame()
        self.scale = scale
        cfg = SCK.SCStreamConfiguration.alloc().init()
        cfg.setWidth_(int(round(frame.size.width * scale)))
        cfg.setHeight_(int(round(frame.size.height * scale)))
        cfg.setShowsCursor_(False)
        cfg.setPixelFormat_(BGRA_FOURCC)
        self._cfg = cfg
        self._size = (frame.size.width, frame.size.height)

    def _window_changed(self) -> bool:
        cur = window_by_id(self.win.window_id)
        if cur is None:
            return True
        if (round(cur.w), round(cur.h)) != (round(self._size[0]), round(self._size[1])):
            self.win = cur
            return True
        return False

    def grab(self) -> Optional[np.ndarray]:
        SCK = self._SCK
        handler, wait = _wait_handler(2.0)
        SCK.SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
            self._filter, self._cfg, handler
        )
        try:
            cg, err = wait()
        except TimeoutError:
            return None
        if cg is None:
            log.debug("captura falhou: %s", err)
            if self._window_changed():
                try:
                    self._setup()
                except Exception as e:
                    log.warning("não deu para reconfigurar a captura: %s", e)
            return None
        img = cgimage_to_bgr(cg)
        return img

    def refresh_if_resized(self) -> None:
        if self._window_changed():
            self._setup()

    def close(self) -> None:
        pass


# --------------------------------------------------------- CGWindowList


class CGWindowSource:
    """Fallback para macOS 12/13 (API antiga, ainda funciona com aviso)."""

    name = "CGWindowList"

    def __init__(self, win: WindowInfo) -> None:
        import Quartz

        self._Q = Quartz
        self.win = win

    def grab(self) -> Optional[np.ndarray]:
        Q = self._Q
        cg = Q.CGWindowListCreateImage(
            Q.CGRectNull,
            Q.kCGWindowListOptionIncludingWindow,
            self.win.window_id,
            Q.kCGWindowImageBoundsIgnoreFraming | Q.kCGWindowImageBestResolution,
        )
        return cgimage_to_bgr(cg)

    def refresh_if_resized(self) -> None:
        pass

    def close(self) -> None:
        pass


def open_window_source(win: WindowInfo) -> FrameSource:
    try:
        return ScreenCaptureKitSource(win)
    except Exception as e:
        log.warning("ScreenCaptureKit indisponível (%s); usando CGWindowList", e)
        return CGWindowSource(win)


# ------------------------------------------------------------- arquivos


class FileSource:
    """Lê capturas salvas (PNG/JPG). Repete a lista em loop."""

    name = "arquivos"

    def __init__(self, paths: Sequence[str | Path], loop: bool = True) -> None:
        self.paths = [Path(p) for p in paths]
        self.loop = loop
        self._i = 0

    def grab(self) -> Optional[np.ndarray]:
        if not self.paths:
            return None
        if self._i >= len(self.paths):
            if not self.loop:
                return None
            self._i = 0
        p = self.paths[self._i]
        self._i += 1
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        return img

    def refresh_if_resized(self) -> None:
        pass

    def close(self) -> None:
        pass


def timed_grab(src: FrameSource) -> tuple[Optional[np.ndarray], float]:
    t = time.perf_counter()
    img = src.grab()
    return img, (time.perf_counter() - t) * 1000


class PrefetchSource:
    """Captura o próximo quadro em segundo plano enquanto o atual é lido.

    A captura do macOS leva ~150 ms; lida em paralelo, o tempo de cada ciclo
    passa a ser o maior dos dois (captura ou leitura), não a soma.
    """

    def __init__(self, src: FrameSource) -> None:
        import threading

        self.src = src
        self.name = getattr(src, "name", "captura") + " (em paralelo)"
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, name="captura", daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                img = self.src.grab()
            except Exception as e:  # nunca derruba o app
                log.debug("captura em paralelo falhou: %s", e)
                img = None
            if img is None:
                time.sleep(0.2)
                continue
            with self._lock:
                self._frame = img
            self._new.set()

    def grab(self) -> Optional[np.ndarray]:
        if not self._new.wait(2.5):
            return None
        with self._lock:
            img, self._frame = self._frame, None
            self._new.clear()
        return img

    def refresh_if_resized(self) -> None:
        fn = getattr(self.src, "refresh_if_resized", None)
        if fn:
            fn()

    def close(self) -> None:
        self._stop.set()
        self._t.join(timeout=3)
        self.src.close()
