"""Gera as capturas sintéticas salvas usadas pelos testes (com gabarito em JSON).

    python tests/make_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from candlesense.replay.synthetic import SynthSpec, render  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"

CASES = {
    "synth_dark_m1": SynthSpec(theme="dark", seed=7),
    "synth_light_m1": SynthSpec(theme="light", seed=3, asset="GBP/USD (OTC)", payout=91),
    "synth_dark_m5_wide": SynthSpec(theme="dark", seed=5, body_width=21, gap=7, timeframe_min=5,
                                    asset="BTC/USD", payout=80, decimals=2, start_price=64250.0, volatility=35.0),
    "synth_dark_small_window": SynthSpec(theme="dark", seed=8, width=1280, height=720, body_width=5, gap=2),
    "synth_dark_zoom_out": SynthSpec(theme="dark", seed=2, body_width=2, gap=1, n_candles=200),
}


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, spec in CASES.items():
        img, t = render(spec)
        cv2.imwrite(str(FIXTURES / f"{name}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        truth = {
            "spec": spec.__dict__,
            "candles": [[c.open, c.high, c.low, c.close] for c in t.candles],
            "centers_x": t.centers_x,
            "price_a": t.price_a,
            "price_b": t.price_b,
            "plot": t.plot.to_list(),
            "axis": t.axis.to_list(),
            "current_price": t.current_price,
            "timer_s": t.timer_s,
            "asset": t.asset,
            "payout": t.payout,
            "timeframe_min": t.timeframe_min,
            "body_width": t.body_width,
            "pitch": t.pitch,
        }
        (FIXTURES / f"{name}.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
        print("ok", name)


if __name__ == "__main__":
    main()
