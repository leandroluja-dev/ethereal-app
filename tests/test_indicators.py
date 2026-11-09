import numpy as np

from candlesense.signals import indicators as ind
from tests.market import candles_from_path


def test_ema_sma_constant_series():
    x = np.full(30, 2.0)
    assert np.allclose(ind.ema(x, 9)[8:], 2.0)
    assert np.allclose(ind.sma(x, 5)[4:], 2.0)
    assert np.isnan(ind.ema(x, 9)[:8]).all()


def test_ema_follows_trend():
    x = np.arange(50, dtype=float)
    e = ind.ema(x, 9)
    assert e[-1] < x[-1] and e[-1] > x[-10]


def test_rsi_extremes():
    assert ind.last_valid(ind.rsi(np.arange(30, dtype=float))) == 100.0
    assert ind.last_valid(ind.rsi(-np.arange(30, dtype=float))) == 0.0
    assert ind.last_valid(ind.rsi(np.full(30, 1.0))) == 50.0


def test_atr_positive_and_scaled():
    s = ind.Series.from_candles(candles_from_path(1.1 + np.cumsum(np.full(30, 0.0002))))
    a = ind.last_valid(ind.atr(s))
    assert 0.0001 < a < 0.001


def test_swings_and_levels():
    # Três topos no mesmo preço viram uma resistência com 3 toques.
    path = [1.0, 1.1, 1.2, 1.1, 1.0, 1.1, 1.2, 1.1, 1.0, 1.1, 1.2, 1.1, 1.0, 1.05]
    s = ind.Series.from_candles(candles_from_path(path, wick=0))
    hi, lo = ind.swings(s, 2)
    assert len(hi) >= 2 and len(lo) >= 1
    lv = ind.levels(s, tol=0.02)
    top = max(lv, key=lambda x: x.price)
    assert top.touches >= 2 and top.kind == "resistencia"
