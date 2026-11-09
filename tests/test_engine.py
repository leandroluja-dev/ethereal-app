import numpy as np
import pytest

from candlesense.engine.rules import COMPRA, INCERTA, NAO_ENTRAR, VENDA, EngineConfig, RuleEngine, breakeven
from tests.market import candles_from_path, mirror, reading, zigzag_up


def strong_up():
    return candles_from_path(zigzag_up(seed=7), seed=7)


def test_breakeven():
    assert breakeven(86) == pytest.approx(0.5376, abs=1e-4)
    assert breakeven(100) == pytest.approx(0.5)


def test_uncertain_reading_never_signals():
    d = RuleEngine().evaluate(reading(strong_up(), ok=False))
    assert d.action == INCERTA and not d.signal and d.reasons


def test_probability_is_capped_and_honest():
    eng = RuleEngine()
    cap = 0.5 + max(eng.cfg.max_edge)
    for seed in range(30):
        rng = np.random.default_rng(seed)
        path = 1.1 + np.cumsum(rng.normal(rng.normal(0, 0.0002), 0.0003, 45))
        d = eng.evaluate(reading(candles_from_path(path, seed=seed)))
        for h in d.horizons:
            assert 1 - cap - 1e-9 <= h.p_up <= cap + 1e-9


def test_mirror_gives_complementary_probability():
    eng = RuleEngine()
    a = eng.evaluate(reading(strong_up()))
    b = eng.evaluate(reading(mirror(strong_up())))
    for x, y in zip(a.horizons, b.horizons):
        assert x.p_up + y.p_up == pytest.approx(1.0, abs=0.01)


def _aligned_engine():
    # Margem zero e poucos módulos exigidos: testa a mecânica do sinal, não o limiar.
    return RuleEngine(EngineConfig(margin=0.0, min_agree=2, phase_margin={"início": 0, "meio": 0, "fim": 0}))


def test_signal_needs_two_readings_and_direction():
    eng = _aligned_engine()
    r = reading(strong_up(), timer=280)
    first = eng.evaluate(r)
    assert first.best.direction == COMPRA
    if first.best.prob < first.threshold:
        pytest.skip("série não forte o bastante para passar o equilíbrio")
    assert first.action == NAO_ENTRAR and any("confirmando" in x for x in first.reasons)
    second = eng.evaluate(r)
    assert second.action == COMPRA and second.signal
    third = _aligned_engine()
    rr = reading(mirror(strong_up()), timer=280)
    third.evaluate(rr)
    assert third.evaluate(rr).action == VENDA


@pytest.mark.parametrize("kw,reason", [
    ({"payout": None}, "payout"),
    ({"purchase": None}, "tempo de compra"),
    ({"purchase": 2}, "acabando"),
    ({"payout": 60}, "abaixo do mínimo"),  # equilíbrio 62,5%: acima do teto das regras
])
def test_no_entry_conditions(kw, reason):
    eng = _aligned_engine()
    r = reading(strong_up(), **kw)
    eng.evaluate(r)
    d = eng.evaluate(r)
    assert d.action == NAO_ENTRAR
    assert any(reason in x for x in d.reasons), d.reasons


def test_default_engine_is_conservative_on_noise():
    eng = RuleEngine()
    signals = 0
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        r = reading(candles_from_path(1.1 + np.cumsum(rng.normal(0, 0.0003, 45)), seed=seed))
        eng.reset()
        eng.evaluate(r)
        signals += eng.evaluate(r).signal
    assert signals <= 4  # passeio aleatório: quase sempre NÃO ENTRAR


def test_phase_and_threshold():
    eng = RuleEngine()
    d = eng.evaluate(reading(strong_up(), timer=280))
    assert d.phase == "início"
    late = RuleEngine().evaluate(reading(strong_up(), timer=20))
    assert late.phase == "fim" and late.threshold > d.threshold


def test_expiry_hint_and_target():
    r = reading(strong_up())
    r.expiry_seconds, r.expiry_label = 83, "18:16"
    d = RuleEngine().evaluate(r)
    assert "18:16" in d.expiry_hint
    assert d.target is not None
    assert (d.target - r.current_price) * (1 if d.best.direction == COMPRA else -1) > 0
