"""Detecção das velas em pixels, sem OCR: posição, cor e quantidade."""

import pytest

from candlesense.vision.candles import detect_candles
from tests.conftest import load_fixture

OK_FIXTURES = ["app_1080", "navegador_1080", "retina_2x", "cripto_2casas", "doji_atual", "linha_compra_colada"]


@pytest.mark.parametrize("name", OK_FIXTURES)
def test_candle_positions_and_colors(name):
    img, meta = load_fixture(name)
    det = detect_candles(img)
    truth = meta["candles"]
    assert not det.reasons, det.reasons
    assert len(det.candles) == len(truth)
    for t, c in zip(truth, det.candles):
        assert abs(c.x_center - t["x"]) <= 1.0
        assert c.color == t["color"]
    # a vela atual (mais à direita) é a última do gabarito
    assert abs(det.candles[-1].x_center - truth[-1]["x"]) <= 1.0


def test_buttons_and_text_are_not_candles():
    img, meta = load_fixture("app_1080")
    det = detect_candles(img)
    x1, y1, x2, y2 = meta["plot"]
    for c in det.candles:
        assert x1 <= c.x_center <= x2
        assert y1 <= c.high_y and c.low_y <= y2


def test_no_chart_means_no_candles():
    img, _ = load_fixture("sem_grafico")
    det = detect_candles(img)
    assert not det.candles
    assert det.reasons


def test_thin_candles_flagged():
    img, _ = load_fixture("velas_finas")
    det = detect_candles(img)
    assert any("finas" in r for r in det.reasons)
