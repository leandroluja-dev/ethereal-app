"""Leitura completa (velas + OCR) nas capturas salvas, comparada ao gabarito."""

import time

import numpy as np
import pytest

from candlesense.vision.reader import ChartReader
from tests.conftest import load_fixture

ALL = ["app_1080", "navegador_1080", "retina_2x", "cripto_2casas", "doji_atual", "linha_compra_colada", "velas_finas", "sem_grafico"]


@pytest.fixture(scope="module")
def readings(ocr):
    out = {}
    for name in ALL:
        img, meta = load_fixture(name)
        reader = ChartReader(ocr)
        out[name] = (reader.read(img), meta, reader, img)
    return out


@pytest.mark.parametrize("name", ALL)
def test_status_matches_expectation(readings, name):
    r, meta, _, _ = readings[name]
    assert r.ok == meta["expect_ok"], r.summary()
    if not r.ok:
        assert r.reasons, "leitura incerta precisa dizer o motivo"


@pytest.mark.parametrize("name", [n for n in ALL if n not in ("velas_finas", "sem_grafico")])
def test_ohlc_matches_truth(readings, name):
    r, meta, _, _ = readings[name]
    ppx = meta["price_per_px"]
    unit = 10 ** -meta["decimals"]
    tol = 1.5 * ppx + unit / 2 + 1e-12  # 1,5 px de leitura + arredondamento do preço exibido
    truth = meta["candles"]
    assert len(r.candles) == len(truth)
    worst = 0.0
    for t, c in zip(truth, r.candles):
        for k in ("open", "high", "low", "close"):
            worst = max(worst, abs(t[k] - getattr(c, k)) / tol)
        assert c.high >= max(c.open, c.close) - 1e-12
        assert c.low <= min(c.open, c.close) + 1e-12
    assert worst <= 1.0, f"erro máximo = {worst:.2f} x tolerância"
    assert r.candles[-1].forming and not r.candles[-2].forming


@pytest.mark.parametrize("name", [n for n in ALL if n not in ("velas_finas", "sem_grafico")])
def test_timer_and_payout(readings, name):
    r, meta, _, _ = readings[name]
    assert r.timer_seconds == meta["timer_seconds"]
    assert r.payout_pct == meta["payout"]


def test_scale_quality(readings):
    r, meta, _, _ = readings["app_1080"]
    assert r.scale.inliers >= 5
    assert r.scale.residual_px <= 1.5
    assert abs(r.scale.price_per_px - meta["price_per_px"]) / meta["price_per_px"] < 0.01


def test_cached_frame_is_fast(readings):
    """Depois de achar o layout, cada quadro tem que caber folgado em 300 ms."""
    _, _, reader, img = readings["app_1080"]
    reader.read(img)
    t = time.perf_counter()
    r = reader.read(img)
    dt = (time.perf_counter() - t) * 1000
    assert r.ok
    assert dt < 300, f"{dt:.0f} ms"


def test_never_raises_on_garbage(ocr):
    reader = ChartReader(ocr)
    for img in (np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10), np.uint8),
                np.random.default_rng(0).integers(0, 255, (600, 800, 3), dtype=np.uint8),
                np.zeros((1080, 1920, 4), np.uint8)):
        r = reader.read(img)
        assert not r.ok
        assert r.reasons


class _BrokenAxisOcr:
    """OCR que lê a tela inteira certo, mas erra todos os números do eixo depois."""

    name = "quebrado"

    def __init__(self, real):
        self.real = real

    def read(self, bgr, roi=None):
        toks = self.real.read(bgr, roi)
        if roi is None:
            return toks
        from candlesense.models import OcrToken

        return [OcrToken("1.11111", t.box, t.score) if any(ch.isdigit() for ch in t.text) and "." in t.text else t
                for t in toks]


def test_bad_axis_reading_is_uncertain(ocr):
    img, _ = load_fixture("app_1080")
    r = ChartReader(_BrokenAxisOcr(ocr)).read(img)
    assert not r.ok
    assert any("eixo" in x for x in r.reasons)
