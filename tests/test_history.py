import random

from candlesense.engine.rules import RuleEngine
from candlesense.history.scoreboard import learn, score
from candlesense.history.store import Prediction, Store, candle_end_ts
from tests.market import candles_from_path, reading, zigzag_up


def test_candle_end_alignment():
    assert candle_end_ts(1000.0, 200, 300) == 1200
    assert candle_end_ts(1199.4, 1, 300) == 1200


def test_record_and_resolve():
    st = Store(":memory:")
    eng = RuleEngine()
    cs = candles_from_path(zigzag_up())
    t0 = 300 * 1000  # início de uma vela de 5 min
    r = reading(cs, timer=290)
    st.record(r, eng.evaluate(r), asset="EURUSD", now=t0 + 10)
    preds = st.predictions()
    assert len(preds) == 1 and preds[0].phase == "início" and preds[0].candle_end == t0 + 300
    # Mesma fase da mesma vela: não grava de novo.
    st.record(r, eng.evaluate(r), asset="EURUSD", now=t0 + 20)
    assert len(st.predictions()) == 1
    # 1 vela depois: a vela prevista já fechou (é a última fechada).
    cs2 = cs + candles_from_path([cs[-1].close + 0.001], start=cs[-1].close)
    r2 = reading(cs2, timer=280)
    st.record(r2, eng.evaluate(r2), asset="EURUSD", now=t0 + 320)
    p = [x for x in st.predictions() if x.candle_end == t0 + 300][0]
    expected = 1 if cs[-1].close > p.price else 0 if cs[-1].close < p.price else -1
    assert p.out[1] == expected


def _preds(n, p_raw, true_rate, seed=0):
    rng = random.Random(seed)
    return [Prediction(i, float(i), 300, i * 300, "início", 1.0, {1: p_raw, 2: p_raw, 3: p_raw, 5: p_raw},
                       {h: int(rng.random() < true_rate) for h in (1, 2, 3, 5)}, "NÃO ENTRAR", 1, 0.56, 86)
            for i in range(n)]


def test_score_counts_hits_and_signals():
    ps = _preds(100, 0.56, 1.0)
    ps[0].action = "COMPRA"
    s = score(ps)[1]
    assert s.n == 100 and s.hits == 100 and s.rate == 1.0
    assert s.signals == 1 and s.signal_hits == 1 and abs(s.signal_profit - 0.86) < 1e-9


def test_calibration_learns_overconfidence():
    # O motor diz 58% mas só acerta 50%: a correção puxa para perto de 50%.
    cal = learn(_preds(3000, 0.58, 0.5, seed=1))
    assert cal.active
    assert abs(cal.apply(1, 0.58) - 0.5) < 0.04


def test_calibration_needs_data():
    cal = learn(_preds(50, 0.58, 0.5))
    assert not cal.active and "faltam" in cal.note[1]
    assert cal.apply(1, 0.58) == 0.58
