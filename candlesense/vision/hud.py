"""Leitura periódica do payout e do nome do ativo nas posições achadas pelo layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common.geometry import Rect
from .ocr import OcrEngine
from .text import normalize_asset, parse_percent


@dataclass
class HudState:
    payout: int | None = None
    payout_ts: float = 0.0
    payout_text: str = ""
    asset: str | None = None
    asset_ts: float = 0.0
    asset_text: str = ""


class HudReader:
    def __init__(self, ocr: OcrEngine, interval_s: float = 2.0, stale_s: float = 10.0) -> None:
        self._ocr = ocr
        self._interval = interval_s
        self._stale = stale_s
        self._last = -1e9
        self.state = HudState()

    def reset(self, payout: int | None = None, asset: str | None = None, now: float = 0.0) -> None:
        self.state = HudState(payout, now if payout is not None else 0.0, "", asset, now if asset else 0.0, "")
        self._last = now

    def update(self, img: np.ndarray, payout_box: Rect | None, asset_box: Rect | None, now: float) -> HudState:
        if now - self._last < self._interval:
            return self.state
        self._last = now
        jobs: list[tuple[str, np.ndarray]] = []
        for kind, box in (("payout", payout_box), ("asset", asset_box)):
            if box is not None:
                crop = box.clip(img.shape[1], img.shape[0]).crop(img)
                if crop.size:
                    jobs.append((kind, crop))
        if not jobs:
            return self.state
        res = self._ocr.read_lines([c for _, c in jobs])
        for (kind, _), (text, _conf) in zip(jobs, res):
            if kind == "payout":
                self.state.payout_text = text
                v = parse_percent(text)
                if v is not None and v <= 100:
                    self.state.payout, self.state.payout_ts = v, now
            else:
                self.state.asset_text = text
                a = normalize_asset(text)
                if a is not None:
                    self.state.asset, self.state.asset_ts = a, now
        return self.state

    def payout_fresh(self, now: float) -> bool:
        return self.state.payout is not None and now - self.state.payout_ts <= self._stale

    def asset_fresh(self, now: float) -> bool:
        return self.state.asset is not None and now - self.state.asset_ts <= self._stale
