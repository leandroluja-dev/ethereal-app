"""Reproduz uma sessão gravada como fonte de quadros (para testar sem mercado aberto)."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import cv2

from ..capture.frame import Frame, WindowInfo
from ..common.geometry import Rect


def load_meta(session_dir: str | Path) -> list[dict]:
    d = Path(session_dir)
    meta_path = d / "meta.jsonl"
    if meta_path.is_file():
        rows = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows
    # pasta só com imagens: usa a ordem dos nomes e 0,5 s entre quadros
    files = sorted(p.name for p in (d / "frames" if (d / "frames").is_dir() else d).glob("*.png"))
    return [{"n": i + 1, "file": f, "ts": 1_700_000_000 + i * 0.5} for i, f in enumerate(files)]


class ReplaySource:
    def __init__(self, session_dir: str | Path, realtime: bool = True, loop: bool = False) -> None:
        self.dir = Path(session_dir)
        self._frames_dir = self.dir / "frames" if (self.dir / "frames").is_dir() else self.dir
        self.meta = load_meta(self.dir)
        if not self.meta:
            raise FileNotFoundError(f"nenhum quadro em {self.dir}")
        self._realtime = realtime
        self._loop = loop
        self._i = 0
        self._t0_wall: float | None = None
        self.finished = False

    def __len__(self) -> int:
        return len(self.meta)

    def _load(self, row: dict) -> Frame:
        img = cv2.imread(str(self._frames_dir / row["file"]), cv2.IMREAD_COLOR)
        w = row.get("window") or {}
        rect = Rect.from_list(w["rect"]) if w.get("rect") else Rect(0, 0, img.shape[1] if img is not None else 0,
                                                                     img.shape[0] if img is not None else 0)
        win = WindowInfo(0, w.get("title", "replay"), w.get("kind", "virtual"), rect)
        return Frame(img, float(row["ts"]), window=win, source="replay")

    def grab(self) -> Frame:
        if self._i >= len(self.meta):
            if not self._loop:
                self.finished = True
                row = self.meta[-1]
                return self._load(row)
            self._i = 0
            self._t0_wall = None
        row = self.meta[self._i]
        if self._realtime:
            t0 = float(self.meta[0]["ts"])
            if self._t0_wall is None:
                self._t0_wall = time.monotonic() - (float(row["ts"]) - t0)
            wait = (float(row["ts"]) - t0) - (time.monotonic() - self._t0_wall)
            if wait > 0:
                time.sleep(min(wait, 2.0))
        self._i += 1
        return self._load(row)

    def __iter__(self) -> Iterator[Frame]:
        for row in self.meta:
            yield self._load(row)

    def close(self) -> None:
        pass
