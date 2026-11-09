from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from candlesense.common.geometry import Rect  # noqa: E402
from candlesense.config import Config, OcrConfig  # noqa: E402
from candlesense.vision.ocr import RapidOcrEngine  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@dataclass
class Capture:
    """Captura salva + gabarito."""

    name: str
    image: np.ndarray
    truth: dict

    @property
    def plot(self) -> Rect:
        return Rect.from_list(self.truth["plot"])

    @property
    def axis(self) -> Rect:
        return Rect.from_list(self.truth["axis"])

    def y_of(self, price: float) -> float:
        return (price - self.truth["price_a"]) / self.truth["price_b"]

    @property
    def px_per_price(self) -> float:
        return 1.0 / abs(self.truth["price_b"])


def load_capture(name: str) -> Capture:
    img = cv2.imread(str(FIXTURES / f"{name}.png"), cv2.IMREAD_COLOR)
    assert img is not None, f"fixture ausente: {name} (rode python tests/make_fixtures.py)"
    truth = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return Capture(name, img, truth)


@pytest.fixture(scope="session")
def ocr():
    eng = RapidOcrEngine(OcrConfig())
    if not eng.available:
        pytest.skip("RapidOCR não instalado")
    return eng


@pytest.fixture
def cfg() -> Config:
    return Config()


class FakeOcr:
    """OCR de mentira: devolve textos pré-definidos (testes determinísticos)."""

    name = "fake"

    def __init__(self, lines: list[str] | None = None, tokens=None) -> None:  # type: ignore[no-untyped-def]
        self.lines = list(lines or [])
        self.tokens = tokens or []

    @property
    def available(self) -> bool:
        return True

    def read(self, img):  # type: ignore[no-untyped-def]
        return list(self.tokens)

    def read_lines(self, crops):  # type: ignore[no-untyped-def]
        out = []
        for _ in crops:
            out.append((self.lines.pop(0), 0.99) if self.lines else ("", 0.0))
        return out
