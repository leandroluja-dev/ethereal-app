"""Leitura completa (layout + velas + OCR) a partir das capturas salvas."""

import json
from pathlib import Path

import numpy as np
import pytest

from candlesense.capture.frame import Frame
from candlesense.capture.layout import detect_layout
from candlesense.replay.synthetic import SynthSpec, render
from candlesense.vision.analyzer import FrameAnalyzer, Status, frame_from_image
from candlesense.vision.debug_draw import draw_overlay
from candlesense.vision.ocr import NullOcr

from .conftest import FIXTURES, load_capture

READABLE = ["synth_dark_m1", "synth_light_m1", "synth_dark_m5_wide", "synth_dark_small_window"]


@pytest.mark.ocr
@pytest.mark.parametrize("name", READABLE)
def test_layout_detection(name, ocr, cfg):
    cap = load_capture(name)
    lay = detect_layout(cap.image, ocr, cfg.vision)
    assert lay is not None
    p, t = lay.plot, cap.plot
    assert abs(p.x - t.x) <= 6 and abs(p.y - t.y) <= 6
    assert abs(p.right - t.right) <= 8 and abs(p.bottom - t.bottom) <= 8
    assert lay.price_axis.x <= cap.axis.x + 8 and lay.price_axis.right >= cap.axis.x + 50
    assert lay.time_axis is not None
    assert lay.payout_value == cap.truth["payout"]
    assert lay.asset_text == cap.truth["asset"]


@pytest.mark.ocr
@pytest.mark.parametrize("name", READABLE)
def test_full_reading(name, ocr, cfg):
    cap = load_capture(name)
    an = FrameAnalyzer(cfg, ocr)
    r = an.analyze(frame_from_image(cap.image, ts=1000.0))
    r = an.analyze(frame_from_image(cap.image, ts=1000.2))
    assert r.status == Status.OK, (r.message, r.conf_parts, r.notes)
    assert r.asset == cap.truth["asset"]
    assert r.payout == cap.truth["payout"]
    assert r.timeframe_min == cap.truth["timeframe_min"]
    assert r.timer_s == cap.truth["timer_s"] and r.timer_source == "ocr"
    tick = 10 ** -cap.truth["spec"]["decimals"]
    assert r.current_price == pytest.approx(cap.truth["current_price"], abs=tick)
    # velas fechadas não cobertas batem com o gabarito (até 2 px)
    truth = cap.truth["candles"]
    n = len(truth)
    tol = 2.0 / cap.px_per_price
    checked = 0
    for c in r.closed:
        if c.occluded:
            continue
        o, h, lo, cl = truth[n - 1 + c.slot]
        assert c.open == pytest.approx(o, abs=tol)
        assert c.close == pytest.approx(cl, abs=tol)
        assert c.high == pytest.approx(h, abs=tol)
        assert c.low == pytest.approx(lo, abs=tol)
        checked += 1
    assert checked >= 20
    assert r.timings_ms["total"] < 300
    json.dumps(r.summary())  # serializável
    out = draw_overlay(cap.image, r)
    assert out.shape == cap.image.shape


@pytest.mark.ocr
def test_zoomed_out_reports_zoom(ocr, cfg):
    cap = load_capture("synth_dark_zoom_out")
    r = FrameAnalyzer(cfg, ocr).analyze(frame_from_image(cap.image))
    assert r.status == Status.ZOOM
    assert "zoom" in r.message.lower()


def test_no_window(cfg):
    r = FrameAnalyzer(cfg, NullOcr()).analyze(Frame(None, 0.0, status_hint="no_window"))
    assert r.status == Status.NO_WINDOW
    assert r.message == "Abra a IQ Option"


def test_blank_screen_is_not_a_chart(cfg):
    img = np.full((800, 1200, 3), 30, np.uint8)
    r = FrameAnalyzer(cfg, NullOcr()).analyze(frame_from_image(img))
    assert r.status == Status.NO_CHART
    assert not r.candles


def test_without_ocr_never_invents_prices(cfg):
    """Sem OCR o eixo de preço é ilegível: a leitura sai incerta e sem preços."""
    cap = load_capture("synth_dark_m1")
    r = FrameAnalyzer(cfg, NullOcr()).analyze(frame_from_image(cap.image))
    assert r.status in (Status.UNCERTAIN, Status.NO_CHART)
    assert r.current_price is None
    assert not r.candles


@pytest.mark.ocr
def test_asset_change_detected(ocr, cfg):
    a, _ = render(SynthSpec(seed=7, asset="EUR/USD (OTC)"))
    b, _ = render(SynthSpec(seed=7, asset="AUD/CAD (OTC)", payout=79))
    an = FrameAnalyzer(cfg, ocr)
    ts = 1000.0
    for _ in range(2):
        r = an.analyze(frame_from_image(a, ts=ts))
        ts += 2.5
    assert r.asset == "EUR/USD (OTC)"
    events = []
    for _ in range(3):
        r = an.analyze(frame_from_image(b, ts=ts))
        events += r.events
        ts += 2.5
    assert r.asset == "AUD/CAD (OTC)"
    assert r.payout == 79
    assert any("troca de ativo" in e for e in events)
    assert r.seconds_since_reset is not None and r.seconds_since_reset < 6


@pytest.mark.ocr
def test_clock_timer_when_no_ocr_timer(ocr, cfg):
    img, _ = render(SynthSpec(seed=7, overlays=False))  # sem etiqueta/cronômetro no eixo
    cfg.vision.assume_timeframe_min = 1
    r = FrameAnalyzer(cfg, ocr).analyze(frame_from_image(img, ts=1_699_999_980.0 + 15))  # 15 s após o início do minuto
    assert r.timer_source == "relógio"
    assert r.timer_s == pytest.approx(45.0)


REAL = sorted((FIXTURES / "real").glob("*.png"))


@pytest.mark.ocr
@pytest.mark.parametrize("path", REAL, ids=[p.stem for p in REAL])
def test_real_captures(path: Path, ocr, cfg):
    """Prints reais da IQ Option em tests/fixtures/real/ com um .json ao lado contendo os valores esperados."""
    import cv2

    expected_path = path.with_suffix(".json")
    if not expected_path.is_file():
        pytest.skip("sem gabarito .json")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    r = FrameAnalyzer(cfg, ocr).analyze(frame_from_image(img))
    for key in ("status", "asset", "payout", "timeframe_min"):
        if key in expected:
            got = r.status.value if key == "status" else getattr(r, key)
            assert got == expected[key], key
    if "current_price" in expected:
        assert r.current_price == pytest.approx(expected["current_price"], abs=expected.get("price_tol", 1e-5))
