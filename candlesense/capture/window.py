"""Localiza a janela da IQ Option (app desktop ou navegador) pela API do Windows.

Somente leitura de posição/título. A única ação sobre a janela é o botão "Ajustar janela",
que usa SetWindowPos (nunca mouse ou teclado).
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import sys
from dataclasses import dataclass

from ..common.geometry import Rect
from ..config import CaptureConfig
from .frame import WindowInfo

log = logging.getLogger(__name__)

BROWSER_HINTS = ("chrome", "edge", "firefox", "opera", "brave", "vivaldi", "yandex", "navegador", "browser")


def is_windows() -> bool:
    return sys.platform == "win32"


def set_dpi_awareness() -> str:
    """Torna o processo DPI-aware (coordenadas em pixels físicos em qualquer escala/monitor)."""
    if not is_windows():
        return "n/a"
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return "per-monitor-v2"
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per-monitor"
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return "system"
    except (AttributeError, OSError):
        return "falhou"


def exclude_from_capture(hwnd: int) -> bool:
    """Esconde uma janela nossa das capturas de tela (WDA_EXCLUDEFROMCAPTURE, Windows 10 2004+).

    Assim o painel pode ficar por cima da IQ Option sem atrapalhar a leitura.
    """
    if not is_windows():
        return False
    try:
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(ctypes.c_void_p(hwnd), 0x11))
    except (AttributeError, OSError):
        return False


def native_window_rect(hwnd: int) -> Rect | None:
    """Retângulo externo de uma janela em pixels físicos."""
    if not is_windows():
        return None
    try:
        import win32gui

        l, t, r, b = win32gui.GetWindowRect(hwnd)
        return Rect.from_ltrb(l, t, r, b)
    except Exception:  # noqa: BLE001
        return None


def move_native_window(hwnd: int, rect: Rect) -> bool:
    """Move uma janela NOSSA (o painel) para um retângulo em pixels físicos."""
    if not is_windows():
        return False
    try:
        import win32con
        import win32gui

        win32gui.SetWindowPos(hwnd, 0, rect.x, rect.y, rect.w, rect.h, win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE)
        return True
    except Exception:  # noqa: BLE001
        return False


def title_matches(title: str, patterns: list[str], exclude: list[str]) -> bool:
    if not title:
        return False
    if any(re.search(p, title, re.IGNORECASE) for p in exclude):
        return False
    return any(re.search(p, title, re.IGNORECASE) for p in patterns)


def classify_title(title: str) -> str:
    t = title.lower()
    return "browser" if any(b in t for b in BROWSER_HINTS) else "desktop"


def choose_window(cands: list[WindowInfo]) -> WindowInfo | None:
    """Prefere janelas visíveis (não minimizadas) e maiores."""
    if not cands:
        return None
    return max(cands, key=lambda w: (not w.minimized, w.rect.area))


def standard_layout(work: Rect, panel_w: int) -> tuple[Rect, Rect]:
    """Layout padrão: IQ Option ocupando a área de trabalho, menos uma coluna à direita para o painel."""
    panel_w = max(240, min(panel_w, work.w // 3))
    iq = Rect(work.x, work.y, work.w - panel_w, work.h)
    panel = Rect(work.x + work.w - panel_w, work.y, panel_w, work.h)
    return iq, panel


@dataclass
class WindowLocator:
    cfg: CaptureConfig

    def find(self) -> WindowInfo | None:
        if not is_windows():
            return None
        try:
            return self._find_win32()
        except Exception:  # noqa: BLE001
            log.exception("falha ao procurar a janela")
            return None

    def _find_win32(self) -> WindowInfo | None:
        import win32api
        import win32con
        import win32gui
        import win32process

        own_pid = os.getpid()
        found: list[WindowInfo] = []

        def cb(hwnd: int, _: object) -> bool:
            try:
                if not win32gui.IsWindowVisible(hwnd):
                    return True
                title = win32gui.GetWindowText(hwnd)
                if not title_matches(title, self.cfg.title_patterns, self.cfg.exclude_title_patterns):
                    return True
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid == own_pid:
                    return True
                minimized = bool(win32gui.IsIconic(hwnd))
                l, t, r, b = win32gui.GetClientRect(hwnd)
                x, y = win32gui.ClientToScreen(hwnd, (l, t))
                rect = Rect(x, y, r - l, b - t)
                work = None
                try:
                    mon = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTONEAREST)
                    wl, wt, wr, wb = win32api.GetMonitorInfo(mon)["Work"]
                    work = Rect.from_ltrb(wl, wt, wr, wb)
                except Exception:  # noqa: BLE001
                    pass
                dpi = 96
                try:
                    dpi = int(ctypes.windll.user32.GetDpiForWindow(ctypes.c_void_p(hwnd)))
                except (AttributeError, OSError):
                    pass
                if not minimized and rect.area < 200 * 150:
                    return True  # janelas minúsculas/ocultas
                found.append(WindowInfo(int(hwnd), title, classify_title(title), rect, minimized, work, dpi))
            except Exception:  # noqa: BLE001
                pass
            return True

        win32gui.EnumWindows(cb, None)
        return choose_window(found)

    def adjust(self, info: WindowInfo, panel_w: int) -> Rect | None:
        """Reposiciona a IQ Option no layout padrão (SetWindowPos). Devolve onde o painel deve ficar."""
        if not is_windows() or info.monitor_work is None:
            return None
        try:
            import win32con
            import win32gui

            iq, panel = standard_layout(info.monitor_work, panel_w)
            if win32gui.IsIconic(info.hwnd) or _is_maximized(info.hwnd):
                win32gui.ShowWindow(info.hwnd, win32con.SW_RESTORE)
            # SetWindowPos usa o retângulo externo; compensa a borda invisível do Windows 10/11.
            ext = _extended_frame(info.hwnd)
            wl, wt, wr, wb = win32gui.GetWindowRect(info.hwnd)
            dl, dt, dr, db = (ext[0] - wl, ext[1] - wt, wr - ext[2], wb - ext[3]) if ext else (0, 0, 0, 0)
            win32gui.SetWindowPos(
                info.hwnd, 0, iq.x - dl, iq.y - dt, iq.w + dl + dr, iq.h + dt + db,
                win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE,
            )
            log.info("janela ajustada para %s; painel em %s", iq.to_list(), panel.to_list())
            return panel
        except Exception:  # noqa: BLE001
            log.exception("falha ao ajustar a janela")
            return None


def _is_maximized(hwnd: int) -> bool:
    try:
        import win32con
        import win32gui

        return win32gui.GetWindowPlacement(hwnd)[1] == win32con.SW_SHOWMAXIMIZED
    except Exception:  # noqa: BLE001
        return False


def _extended_frame(hwnd: int) -> tuple[int, int, int, int] | None:
    try:
        import ctypes.wintypes

        rect = ctypes.wintypes.RECT()
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        res = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            ctypes.c_void_p(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
        )
        if res == 0:
            return rect.left, rect.top, rect.right, rect.bottom
    except Exception:  # noqa: BLE001
        pass
    return None
