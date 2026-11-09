"""Eixo de preço/tempo: regressão robusta, segmentação de rótulos e leitura com OCR real."""

import numpy as np
import pytest

from candlesense.common.geometry import Rect
from candlesense.vision.price_axis import (
    PriceAxisReader,
    TimeAxisReader,
    fit_mapping,
    segment_lines,
    snap_timeframe,
)

from .conftest import FakeOcr, load_capture


def test_fit_mapping_ignores_outliers():
    a, b = 1.1, -0.00001
    pts = [(y, a + b * y) for y in (100, 180, 260, 340, 420)]
    pts.append((300, 9.99))  # leitura absurda
    pts.append((500, a + b * 500 + 0.0005))  # dígito errado
    m, flags = fit_mapping(pts)
    assert m is not None
    assert m.inliers == 5
    assert flags[5] is False and flags[6] is False
    assert m.price(200) == pytest.approx(a + b * 200, abs=1e-9)


def test_fit_mapping_needs_three_consistent_labels():
    m, _ = fit_mapping([(100, 1.2), (200, 1.1)])
    assert m is None


def test_fit_mapping_rejects_inverted_axis():
    m, _ = fit_mapping([(100, 1.0), (200, 1.1), (300, 1.2)])
    assert m is None


def test_segment_lines_finds_stacked_labels():
    mask = np.zeros((300, 80), np.uint8)
    for y in (20, 100, 180, 260):
        mask[y : y + 10, 10:60] = 1
    mask[150, :] = 1  # linha de grade atravessando
    lines = segment_lines(mask, axis=0)
    assert [r.y for r in lines] == [20, 100, 180, 260]


@pytest.mark.parametrize(("mpc", "tf"), [(1.0, 1), (0.97, 1), (5.2, 5), (15, 15), (3.3, None), (None, None)])
def test_snap_timeframe(mpc, tf):
    assert snap_timeframe(mpc) == tf


def test_reader_with_fake_ocr_uses_cache():
    cap = load_capture("synth_dark_m1")
    fake = FakeOcr(lines=[])
    reader = PriceAxisReader(fake)
    first = reader.read(cap.image, cap.axis)
    assert first.reocr
    second = reader.read(cap.image, cap.axis)
    assert not second.reocr  # eixo igual: não relê os rótulos


@pytest.mark.ocr
@pytest.mark.parametrize("name", ["synth_dark_m1", "synth_light_m1", "synth_dark_m5_wide", "synth_dark_small_window"])
def test_price_axis_real_ocr(name, ocr):
    cap = load_capture(name)
    r = PriceAxisReader(ocr).read(cap.image, cap.axis)
    assert r.mapping is not None
    assert r.mapping.inliers >= 3
    for y in range(cap.plot.y, cap.plot.bottom, 25):
        true_price = cap.truth["price_a"] + cap.truth["price_b"] * y
        assert abs(r.mapping.price(y) - true_price) * cap.px_per_price <= 2.0
    assert r.tag_price == pytest.approx(cap.truth["current_price"], abs=10 ** -cap.truth["spec"]["decimals"])


@pytest.mark.ocr
def test_timer_read_from_axis(ocr):
    cap = load_capture("synth_dark_m1")
    r = PriceAxisReader(ocr).read(cap.image, cap.axis)
    assert r.timer_s == cap.truth["timer_s"]


@pytest.mark.ocr
@pytest.mark.parametrize("name", ["synth_dark_m1", "synth_dark_m5_wide"])
def test_time_axis_gives_timeframe(name, ocr):
    cap = load_capture(name)
    band = Rect(cap.plot.x, cap.plot.bottom, cap.plot.w, 30)
    ta = TimeAxisReader(ocr).read(cap.image, band)
    assert snap_timeframe(ta.minutes_per_px * cap.truth["pitch"]) == cap.truth["timeframe_min"]
