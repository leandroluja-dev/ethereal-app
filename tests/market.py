"""Séries de velas sintéticas para testar indicadores, módulos e motor."""

from __future__ import annotations

import numpy as np

from candlesense.models import Candle, ChartReading, PriceScale


def candles_from_path(closes, wick=0.3, start=None, seed=0) -> list[Candle]:
    rng = np.random.default_rng(seed)
    closes = [float(x) for x in closes]
    out = []
    prev = closes[0] if start is None else start
    for c in closes:
        o = prev
        hi = max(o, c) + wick * abs(rng.normal(0, 1)) * 0.0002 + 0.00002
        lo = min(o, c) - wick * abs(rng.normal(0, 1)) * 0.0002 - 0.00002
        out.append(Candle(open=round(o, 6), high=round(hi, 6), low=round(lo, 6), close=round(c, 6),
                          color="verde" if c >= o else "vermelha"))
        prev = c
    return out


def mirror(candles: list[Candle], center: float = 1.1) -> list[Candle]:
    out = []
    for k in candles:
        o, h, l, c = (2 * center - x for x in (k.open, k.low, k.high, k.close))
        out.append(Candle(open=o, high=h, low=l, close=c, color="verde" if c >= o else "vermelha",
                          forming=k.forming))
    return out


def reading(candles: list[Candle], payout=86, purchase=40, timer=250, timeframe=300, ok=True) -> ChartReading:
    cs = [Candle(**{**k.__dict__}) for k in candles]
    cs[-1].forming = True
    r = ChartReading(ok=ok, candles=cs, current_price=cs[-1].close, payout_pct=payout,
                     purchase_seconds=purchase, timer_seconds=timer, timeframe_s=timeframe,
                     scale=PriceScale(slope=-1e-5, intercept=1.2, inliers=5, residual_px=0.5, decimals=6))
    if not ok:
        r.reasons.append("teste: leitura incerta")
    return r


def trend_path(n=40, step=0.0002, noise=0.00005, seed=1, start=1.1):
    rng = np.random.default_rng(seed)
    return start + np.cumsum(step + rng.normal(0, noise, n))


def zigzag_up(n=40, start=1.1, seed=2):
    """Alta com recuos: topos e fundos ascendentes."""
    rng = np.random.default_rng(seed)
    out, p = [], start
    for i in range(n):
        p += (0.0004 if (i // 3) % 2 == 0 else -0.00015) + rng.normal(0, 0.00003)
        out.append(p)
    return np.array(out)
