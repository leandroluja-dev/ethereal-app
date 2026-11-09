"""Visão: cores e velas, comparadas com o gabarito das capturas salvas."""

import numpy as np
import pytest

from candlesense.config import VisionConfig
from candlesense.replay.synthetic import SynthSpec, render
from candlesense.vision.candles import detect_candles, remove_thin_horizontal_lines
from candlesense.vision.colors import discover_colors, hue_dist

from .conftest import load_capture

READABLE = ["synth_dark_m1", "synth_light_m1", "synth_dark_m5_wide", "synth_dark_small_window"]


@pytest.mark.parametrize("name", READABLE + ["synth_dark_zoom_out"])
def test_colors_bull_green_bear_red(name):
    cap = load_capture(name)
    c = discover_colors(cap.plot.crop(cap.image), VisionConfig())
    assert c is not None
    assert hue_dist(c.bull_hue, 60) <= 25, c
    assert hue_dist(c.bear_hue, 0) <= 15, c
    assert not c.ambiguous
    assert c.theme == ("light" if "light" in name else "dark")


@pytest.mark.parametrize("name", READABLE)
def test_candles_match_truth(name):
    cap = load_capture(name)
    cfg = VisionConfig()
    plot = cap.plot
    colors = discover_colors(plot.crop(cap.image), cfg)
    det = detect_candles(plot.crop(cap.image), colors, cfg, offset=(plot.x, plot.y))
    truth = cap.truth["candles"]
    n = len(truth)
    assert det.pitch == pytest.approx(cap.truth["pitch"], abs=0.2)
    assert det.body_width == pytest.approx(cap.truth["body_width"], abs=1)
    assert not det.too_small
    assert det.forming is not None and det.forming.slot == 0
    assert det.forming.cx == pytest.approx(cap.truth["centers_x"][-1], abs=1)

    visible = [i for i, x in enumerate(cap.truth["centers_x"]) if x - cap.truth["body_width"] / 2 > plot.x]
    found = {n - 1 + c.slot for c in det.candles}
    # no máximo 3 velas perdidas (escondidas sob o marcador de operação)
    assert len(set(visible) - found) <= 3

    clean = 0
    for c in det.candles:
        o, h, lo, cl = truth[n - 1 + c.slot]
        errs = [abs(c.wick_top - cap.y_of(h)), abs(c.wick_bottom - cap.y_of(lo)),
                abs(c.open_y - cap.y_of(o)), abs(c.close_y - cap.y_of(cl))]
        if max(errs) > 1.6:
            # toda vela com erro precisa estar marcada como coberta (nunca um valor errado "confiável")
            assert c.occluded, (name, c.slot, errs)
        else:
            clean += 1
            if abs(cap.y_of(o) - cap.y_of(cl)) > 2:
                assert c.bull == (cl > o)
    assert clean >= 0.6 * len(det.candles)
    assert det.confidence > 0.7


def test_zoomed_out_chart_is_flagged():
    cap = load_capture("synth_dark_zoom_out")
    cfg = VisionConfig()
    colors = discover_colors(cap.plot.crop(cap.image), cfg)
    det = detect_candles(cap.plot.crop(cap.image), colors, cfg)
    assert det.too_small
    assert any("zoom" in n for n in det.notes)


def test_thin_horizontal_lines_removed_but_bodies_kept():
    m = np.zeros((60, 200), np.uint8)
    m[30, :] = 1  # linha de preço
    m[10:50, 50:59] = 1  # corpo de vela
    out = remove_thin_horizontal_lines(m)
    assert out[30, 0:40].sum() == 0
    assert out[10:50, 50:59].all()


def test_only_bull_candles_still_detected():
    candles = []
    from candlesense.replay.synthetic import Ohlc

    p = 1.1
    for _ in range(40):
        candles.append(Ohlc(p, p + 0.0003, p - 0.0001, p + 0.0002))
        p += 0.0002
    img, t = render(SynthSpec(overlays=False), candles)
    cfg = VisionConfig()
    colors = discover_colors(t.plot.crop(img), cfg)
    assert colors is not None and hue_dist(colors.bull_hue, 60) <= 25
    det = detect_candles(t.plot.crop(img), colors, cfg)
    assert len(det.candles) >= 38
    assert all(c.bull for c in det.candles)


def test_empty_plot_gives_no_colors():
    img = np.full((400, 600, 3), (48, 35, 31), np.uint8)
    assert discover_colors(img, VisionConfig()) is None
