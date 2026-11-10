import math
import sqlite3
import time

from candlesense import lite
from candlesense.engine.strong import StrongMode


def test_hours_and_window(tmp_path):
    assert lite.hours(tmp_path) == (9, 21)
    (tmp_path / "modo_leve").write_text("10-18\n")
    assert lite.hours(tmp_path) == (10, 18)
    (tmp_path / "modo_leve").write_text("lixo")
    assert lite.hours(tmp_path) == (9, 21)
    assert lite.payout(tmp_path) == 89
    (tmp_path / "payout.txt").write_text("85%")
    assert lite.payout(tmp_path) == 85
    wed_noon = time.mktime((2026, 11, 11, 12, 0, 0, 0, 0, -1))
    sat_noon = time.mktime((2026, 11, 14, 12, 0, 0, 0, 0, -1))
    assert lite.in_window(wed_noon, (9, 21)) and not lite.in_window(sat_noon, (9, 21))
    assert not lite.in_window(wed_noon, (13, 21))


def test_fetch_then_measure(monkeypatch, tmp_path):
    db = sqlite3.connect(":memory:")
    sm = StrongMode(db, log=lambda *a, **k: None)
    now = 1_794_340_812.0  # 12 s depois da virada do minuto
    minute = int(now // 60) * 60
    bars = [(minute - 60 * i, 1.3 + 0.001 * math.sin(i / 7), 1.3 + 0.001 * math.sin(i / 7) + 0.0003,
             1.3 + 0.001 * math.sin(i / 7) - 0.0003, 1.3 + 0.001 * math.sin((i - 1) / 7)) for i in range(0, 170)]
    import candlesense.feeds.twelvedata as td

    monkeypatch.setattr(td, "series", lambda *a, **k: bars)
    n = lite.fetch(db, "GBP/USD", "GBP/USD", now, True, tmp_path)
    assert n == 169  # a vela em formação fica de fora
    p = sm.measure_closed("GBP/USD", now, 89)
    row = db.execute("SELECT payout, start FROM forte").fetchone()
    if p is not None:
        assert row == (89, minute - 60)
    assert sm.measure_closed("GBP/USD", now, 89) is None  # a mesma vela não mede duas vezes
