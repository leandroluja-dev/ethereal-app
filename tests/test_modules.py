import numpy as np

from candlesense.models import Candle
from candlesense.signals import modules as M
from tests.market import candles_from_path, mirror, reading, trend_path, zigzag_up


def ctx_of(candles, **kw):
    return M.MarketContext.from_reading(reading(candles, **kw))


def test_scores_are_bounded_and_explained():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        path = 1.1 + np.cumsum(rng.normal(0, 0.0003, 45))
        for m in M.run_all(ctx_of(candles_from_path(path, seed=seed))):
            assert -1.0 <= m.score <= 1.0
            assert 0.0 <= m.confidence <= 1.0
            assert m.explanation


def test_trend_up_and_down():
    up = M.trend(ctx_of(candles_from_path(zigzag_up())))
    down = M.trend(ctx_of(mirror(candles_from_path(zigzag_up()))))
    assert up.score > 0.5 and "alta" in up.explanation
    assert down.score < -0.5 and "baixa" in down.explanation


def test_trend_flat_is_neutral():
    rng = np.random.default_rng(3)
    path = 1.1 + rng.normal(0, 0.00005, 40)
    assert abs(M.trend(ctx_of(candles_from_path(path))).score) < 0.5


def test_modules_are_symmetric():
    cs = candles_from_path(zigzag_up(seed=5), seed=5)
    a = M.run_all(ctx_of(cs))
    b = M.run_all(ctx_of(mirror(cs)))
    for x, y in zip(a, b):
        assert abs(x.score + y.score) < 0.05, (x.name, x.score, y.score)


def test_resistance_nearby_pushes_down():
    # Bate três vezes em 1.1020 e volta para perto dela.
    path = []
    for _ in range(4):
        path += list(np.linspace(1.1000, 1.1020, 5)) + list(np.linspace(1.1018, 1.1000, 5))
    path += list(np.linspace(1.1002, 1.1016, 5))
    m = M.support_resistance(ctx_of(candles_from_path(path, wick=0)))
    assert m.score < 0 and "resistência" in m.explanation


def test_bullish_engulfing():
    cs = candles_from_path(1.1 - np.cumsum(np.full(30, 0.0001)))
    last = cs[-1].close
    cs.append(Candle(open=last, high=last + 0.00002, low=last - 0.0004, close=last - 0.00035, color="vermelha"))
    o = last - 0.00038
    cs.append(Candle(open=o, high=last + 0.0003, low=o - 0.00002, close=last + 0.00025, color="verde"))
    cs.append(Candle(open=last + 0.00025, high=last + 0.0003, low=last + 0.0002, close=last + 0.00026, color="verde"))
    m = M.price_action(ctx_of(cs))
    assert m.score > 0.5 and "engolfo de alta" in m.explanation


def test_rsi_overbought_dampens_momentum():
    straight = M.momentum(ctx_of(candles_from_path(trend_path(step=0.0004, noise=0.0))))
    assert straight.details["rsi"] > 90
    assert "sobrecomprado" in straight.explanation
    assert straight.score < 0.5


def test_microstructure_needs_time():
    cs = candles_from_path(trend_path())
    f = cs[-1]
    f.open, f.close, f.high, f.low = 1.2, 1.2006, 1.2006, 1.19995
    early = M.microstructure(ctx_of(cs, timer=290))
    late = M.microstructure(ctx_of(cs, timer=30))
    assert early.score > 0 and late.score > 0
    assert early.confidence < 0.3 < late.confidence


def test_context_requires_candles():
    assert M.MarketContext.from_reading(reading(candles_from_path([1.1, 1.2]))) is None
