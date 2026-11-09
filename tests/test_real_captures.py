"""Capturas reais salvas em tests/fixtures/real (cada .png com seu gabarito .json)."""

import json

import cv2
import pytest

from candlesense.vision.reader import ChartReader
from tests.conftest import REAL

CASES = sorted(p for p in REAL.glob("*.png") if p.with_suffix(".json").exists())


@pytest.mark.skipif(not CASES, reason="nenhuma captura real com gabarito ainda")
@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_real_capture(ocr, path):
    exp = json.loads(path.with_suffix(".json").read_text())
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    r = ChartReader(ocr).read(img)
    if "expect_ok" in exp:
        assert r.ok == exp["expect_ok"], r.summary()
    if "payout" in exp:
        assert r.payout_pct == exp["payout"]
    if "timer_seconds" in exp:
        assert r.timer_seconds is not None and abs(r.timer_seconds - exp["timer_seconds"]) <= 1
    want = exp.get("ultimas_velas") or []
    if want:
        tol = exp.get("tolerancia", 0.0)
        got = r.candles[-len(want):]
        assert len(got) == len(want)
        for w, c in zip(want, got):
            for k in ("open", "high", "low", "close"):
                if k in w:
                    assert abs(getattr(c, k) - w[k]) <= tol, (k, w[k], getattr(c, k))
