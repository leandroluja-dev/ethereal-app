"""Modo medição do modelo de formatos: fita de velas de 1 min, cálculo uma vez por vela e conferência."""

import math
import sqlite3

from candlesense.engine.shapemodel import MIN_HISTORY, Bar, ShapeModel, features
from candlesense.engine.strong import StrongMode
from candlesense.models import Candle, ChartReading

T0 = 1_762_740_000  # múltiplo de 60 (UTC)


def _price(i: int) -> float:
    return 1.1 + 0.0004 * math.sin(i / 7.0) + 0.0002 * math.sin(i / 2.3)


def _bar(i: int) -> Candle:
    o, c = _price(i), _price(i + 1)
    return Candle(open=o, high=max(o, c) + 0.00005, low=min(o, c) - 0.00004, close=c, color="verde" if c >= o else "vermelha")


def _reading(i: int, n_visible: int = 60, timer: int = 55, tf: int = 60, ok: bool = True) -> ChartReading:
    cs = [_bar(k) for k in range(i - n_visible, i)]
    forming = _bar(i)
    forming.forming = True
    return ChartReading(ok=ok, candles=cs + [forming], timer_seconds=timer, timeframe_s=tf, payout_pct=86)


def _bars(n: int) -> list[Bar]:
    out = []
    for i in range(n):
        c = _bar(i)
        out.append(Bar(T0 + 60 * i, c.open, c.high, c.low, c.close))
    return out


def test_features_need_continuous_history():
    assert features(_bars(MIN_HISTORY - 1)) is None
    bars = _bars(MIN_HISTORY + 10)
    assert features(bars) is not None
    del bars[-30]  # buraco no meio
    assert features(bars) is None


def test_feature_values():
    bars = _bars(MIN_HISTORY + 5)
    f = features(bars)
    atr = sum(b.high - b.low for b in bars[-60:]) / 60
    assert abs(f["c1_1"] - (bars[-2].close - bars[-1].close) / atr) < 1e-12
    assert abs(f["h1_1"] - (bars[-1].high - bars[-1].close) / atr) < 1e-12
    assert f["hour"] == ((bars[-1].start // 3600) % 24)
    assert len([k for k in f if k.startswith(("c15_", "h15_", "l15_"))]) == 24


def test_model_gives_probability():
    m = ShapeModel()
    p = m.prob_up(features(_bars(MIN_HISTORY + 20)))
    assert 0.0 < p < 1.0


def test_strong_mode_measures_once_per_candle_and_resolves():
    db = sqlite3.connect(":memory:")
    sm = StrongMode(db, log=lambda *a, **k: None)
    got = []
    # 3 horas de leituras: a cada minuto, uma leitura logo depois de a vela fechar e outra no meio.
    for i in range(200, 200 + 180):
        start = T0 + 60 * i  # início da vela em formação
        for timer, after in ((55, 5), (30, 30)):
            p = sm.update(_reading(i, n_visible=60 if i < 300 else 40, timer=timer), "EUR/USD", now=start + after)
            if p is not None:
                got.append(p)
    # Primeiro minuto já grava as 60 velas visíveis; depois de ~2h15 de fita, mede uma vez por vela.
    n_fita = db.execute("SELECT COUNT(*) FROM fita").fetchone()[0]
    assert n_fita >= 180
    n_forte = db.execute("SELECT COUNT(*) FROM forte").fetchone()[0]
    assert n_forte == len(got) > 100
    assert db.execute("SELECT COUNT(*) FROM forte WHERE out5 IS NOT NULL").fetchone()[0] >= n_forte - 6
    rep = sm.report()
    assert rep[0.55][1] <= n_forte


def test_strong_mode_ignores_uncertain_otc_and_other_timeframes():
    db = sqlite3.connect(":memory:")
    sm = StrongMode(db, log=lambda *a, **k: None)
    for i in range(200, 400):
        now = T0 + 60 * i + 5
        assert sm.update(_reading(i, ok=False), "EUR/USD", now=now) is None
        assert sm.update(_reading(i, tf=300), "EUR/USD", now=now) is None
        assert sm.update(_reading(i), "EUR/USD (OTC)", now=now) is None
    assert db.execute("SELECT COUNT(*) FROM forte").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM fita WHERE asset='EUR/USD'").fetchone()[0] == 0


def test_simulator_counts_virtual_entries():
    db = sqlite3.connect(":memory:")
    sm = StrongMode(db, log=lambda *a, **k: None)
    rows = [("EUR/USD", 0, 1.0, 0.70, 1.1, 86, 1),   # compra certa: +8,60
            ("EUR/USD", 60, 1.0, 0.30, 1.1, 86, 1),  # venda errada: -10
            ("EUR/USD", 120, 1.0, 0.55, 1.1, 86, 0),  # abaixo do corte: fora
            ("GBP/USD", 180, 1.0, 0.25, 1.1, 90, 0),  # venda certa: +9
            ("GBP/USD", 240, 1.0, 0.80, 1.1, 70, 0),  # payout baixo: fora
            ("GBP/USD", 300, 1.0, 0.80, 1.1, None, -1)]  # empate
    db.executemany("INSERT INTO forte VALUES (?,?,?,?,?,?,?)", rows)
    s = sm.simulate(threshold=0.62, stake=10)
    assert (s["entradas"], s["ganhos"], s["perdas"], s["empates"]) == (4, 2, 1, 1)
    assert abs(s["lucro"] - (8.6 - 10 + 9)) < 1e-9
    assert "R$ +7.60" in sm.day_text(0)


def test_fita_without_timer_uses_clock():
    """Modo Blitz: sem cronômetro na tela, a vela em formação é a do minuto do relógio."""
    db = sqlite3.connect(":memory:")
    sm = StrongMode(db, log=lambda *a, **k: None)
    r = _reading(300)
    r.timer_seconds = None
    sm.update(r, "EUR/USD", now=T0 + 60 * 300 + 2)  # perto da virada: não grava
    assert db.execute("SELECT COUNT(*) FROM fita").fetchone()[0] == 0
    sm.update(r, "EUR/USD", now=T0 + 60 * 300 + 20)
    last = db.execute("SELECT MAX(start) FROM fita").fetchone()[0]
    assert last == T0 + 60 * 299
