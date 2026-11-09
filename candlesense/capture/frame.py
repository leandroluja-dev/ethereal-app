"""Quadro capturado e a interface comum das fontes de quadros (ao vivo, imagem, gravação, demonstração)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..common.geometry import Rect


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    kind: str  # "desktop" | "browser" | "virtual"
    rect: Rect  # área cliente em pixels físicos da tela
    minimized: bool = False
    monitor_work: Rect | None = None
    dpi: int = 96


@dataclass
class Frame:
    image: np.ndarray | None  # BGR
    ts: float
    window: WindowInfo | None = None
    valid_mask: np.ndarray | None = None  # False onde a imagem não é confiável (painel por cima)
    source: str = "live"
    status_hint: str | None = None  # "no_window" | "minimized" | "capture_error" | "unsupported"


class FrameSource(Protocol):
    def grab(self) -> Frame: ...

    def close(self) -> None: ...
