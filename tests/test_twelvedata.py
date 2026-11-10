"""Twelve Data: completa a fita de 1 min sem passar dos limites do plano grátis."""

import sqlite3

from candlesense.engine.strong import StrongMode
from candlesense.feeds import twelvedata as td

T0 = 1_762_740_000


def test_symbol_of():
    assert td.symbol_of("GBP/USD") == "GBP/USD"
    assert td.symbol_of("GBP/USD Blitz") == "GBP/USD"
    assert td.symbol_of("EUR/USD (OTC)") is None
    assert td.symbol_of("?") is None


def test_filler_fills_missing_and_respects_screen(monkeypatch, tmp_path):
    db = sqlite3.connect(":memory:")
    StrongMode(db, log=lambda *a, **k: None)
    now = T0 + 60 * 300 + 20
    # a tela já leu as 10 últimas velas: essas não são trocadas
    db.executemany("INSERT INTO fita VALUES (?,?,?,?,?,?,?)",
                   [("GBP/USD", T0 + 60 * i, 9, 9, 9, 9, now) for i in range(290, 300)])
    calls = []

    def fake_series(sym, interval, outputsize, dados):
        calls.append(outputsize)
        return [(T0 + 60 * i, 1.0, 1.1, 0.9, 1.05) for i in range(300 - outputsize + 1, 301)]

    monkeypatch.setattr(td, "series", fake_series)
    f = td.FitaFiller(":memory:", tmp_path)
    f.asset = "GBP/USD"
    n = f.tick(db, now=now)
    assert calls == [155] and n > 140
    assert f.missing(db, "GBP/USD", now) <= 0
    assert db.execute("SELECT close FROM fita WHERE start=?", (T0 + 60 * 299,)).fetchone()[0] == 9  # tela manda
    assert db.execute("SELECT COUNT(*) FROM fita WHERE start=?", (T0 + 60 * 300,)).fetchone()[0] == 0  # em formação
    # completo: só confere de 3 em 3 min, com pedido pequeno
    assert f.tick(db, now=now + 30) == 0 and len(calls) == 1
    db.executemany("INSERT INTO fita VALUES (?,?,?,?,?,?,?)",  # a tela segue gravando
                   [("GBP/USD", T0 + 60 * i, 9, 9, 9, 9, now) for i in range(300, 303)])
    f.tick(db, now=now + 200)
    assert calls[-1] == 10
    f.asset = "EUR/USD (OTC)"
    assert f.tick(db, now=now + 400) == 0


def test_rate_limit(monkeypatch):
    td._recent.clear()
    td._day.update(date="", n=0)
    got = [td._allow() for _ in range(10)]
    assert got.count(True) == td.PER_MIN


def test_no_key_no_request(tmp_path):
    assert td.load_key(tmp_path) is None
    assert td.series("GBP/USD", dados=tmp_path) is None
    (tmp_path / "twelvedata.key").write_text(td.SKIP)
    assert td.load_key(tmp_path) is None and td.asked_before(tmp_path)
