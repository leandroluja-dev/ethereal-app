"""Acha a janela da IQ Option: app desktop ou aba de navegador.

Usa CGWindowListCopyWindowInfo (Quartz). O título das janelas só aparece com a
permissão de Gravação de Tela liberada.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

BROWSERS = {
    "google chrome", "google chrome beta", "google chrome canary", "chromium", "safari",
    "safari technology preview", "firefox", "firefox developer edition", "microsoft edge",
    "brave browser", "arc", "opera", "opera gx", "vivaldi", "orion", "zen browser", "yandex",
}
TITLE_RE = re.compile(r"iq\s*option|iqoption", re.IGNORECASE)
APP_RE = re.compile(r"^iq\s*option", re.IGNORECASE)
MIN_W, MIN_H = 500, 350


@dataclass(frozen=True)
class WindowInfo:
    window_id: int
    pid: int
    owner: str
    title: str
    x: float
    y: float
    w: float
    h: float
    layer: int = 0
    on_screen: bool = True

    @property
    def kind(self) -> str:
        return "app" if APP_RE.match(self.owner) else "navegador"

    @property
    def area(self) -> float:
        return self.w * self.h


def from_cg_dict(d: dict) -> Optional[WindowInfo]:
    try:
        b = d.get("kCGWindowBounds") or {}
        return WindowInfo(
            window_id=int(d["kCGWindowNumber"]),
            pid=int(d.get("kCGWindowOwnerPID", 0)),
            owner=str(d.get("kCGWindowOwnerName", "") or ""),
            title=str(d.get("kCGWindowName", "") or ""),
            x=float(b.get("X", 0)), y=float(b.get("Y", 0)),
            w=float(b.get("Width", 0)), h=float(b.get("Height", 0)),
            layer=int(d.get("kCGWindowLayer", 0)),
            on_screen=bool(d.get("kCGWindowIsOnscreen", True)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def is_iq_window(w: WindowInfo) -> bool:
    if w.layer != 0 or w.w < MIN_W or w.h < MIN_H:
        return False
    if APP_RE.match(w.owner):
        return True
    return w.owner.lower() in BROWSERS and bool(TITLE_RE.search(w.title))


def pick_iq_window(windows: Iterable[WindowInfo]) -> Optional[WindowInfo]:
    """Escolhe a janela da IQ Option: visível, maior área; app nativo vence empate."""
    cands = [w for w in windows if is_iq_window(w)]
    if not cands:
        return None
    return max(cands, key=lambda w: (w.on_screen, w.area, w.kind == "app"))


def list_windows() -> list[WindowInfo]:
    import Quartz

    opts = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    raw = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID) or []
    out = []
    for d in raw:
        w = from_cg_dict(dict(d))
        if w is not None:
            out.append(w)
    return out


def find_iq_window() -> Optional[WindowInfo]:
    return pick_iq_window(list_windows())


def window_by_id(window_id: int) -> Optional[WindowInfo]:
    for w in list_windows():
        if w.window_id == window_id:
            return w
    return None
