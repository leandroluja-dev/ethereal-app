"""Retângulos em coordenadas de imagem (pixels físicos, origem no canto superior esquerdo)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @staticmethod
    def from_ltrb(left: float, top: float, right: float, bottom: float) -> Rect:
        l, t, r, b = int(round(left)), int(round(top)), int(round(right)), int(round(bottom))
        return Rect(l, t, max(0, r - l), max(0, b - t))

    def is_empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def crop(self, img: np.ndarray) -> np.ndarray:
        r = self.clip(img.shape[1], img.shape[0])
        return img[r.y : r.bottom, r.x : r.right]

    def clip(self, width: int, height: int) -> Rect:
        l = min(max(self.x, 0), width)
        t = min(max(self.y, 0), height)
        r = min(max(self.right, 0), width)
        b = min(max(self.bottom, 0), height)
        return Rect(l, t, max(0, r - l), max(0, b - t))

    def intersect(self, other: Rect) -> Rect | None:
        l = max(self.x, other.x)
        t = max(self.y, other.y)
        r = min(self.right, other.right)
        b = min(self.bottom, other.bottom)
        if r <= l or b <= t:
            return None
        return Rect(l, t, r - l, b - t)

    def union(self, other: Rect) -> Rect:
        return Rect.from_ltrb(
            min(self.x, other.x), min(self.y, other.y), max(self.right, other.right), max(self.bottom, other.bottom)
        )

    def expand(self, dx: int, dy: int | None = None) -> Rect:
        dy = dx if dy is None else dy
        return Rect(self.x - dx, self.y - dy, self.w + 2 * dx, self.h + 2 * dy)

    def offset(self, dx: int, dy: int) -> Rect:
        return Rect(self.x + dx, self.y + dy, self.w, self.h)

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def to_list(self) -> list[int]:
        return [self.x, self.y, self.w, self.h]

    @staticmethod
    def from_list(v: list[int] | tuple[int, int, int, int]) -> Rect:
        return Rect(int(v[0]), int(v[1]), int(v[2]), int(v[3]))
