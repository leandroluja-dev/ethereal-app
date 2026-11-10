"""Telas 2 e 3 ao lado do painel: entradas do simulador, resultados do dia e sons."""

import sqlite3
import wave

from candlesense import alerts
from candlesense.engine.strong import StrongMode
from candlesense.ui.sim_view import SideScreens, SimState, day_summary, groups, live_result, side_screens

T0 = 1_762_740_000


def _sm():
    db = sqlite3.connect(":memory:")
    sm = StrongMode(db, log=lambda *a, **k: None)
    rows = [("EUR/USD", T0, T0 + 5, 0.70, 1.1, 86, 1),        # compra certa
            ("EUR/USD", T0 + 60, T0 + 65, 0.30, 1.1, 86, 1),   # venda errada
            ("EUR/USD", T0 + 120, T0 + 125, 0.55, 1.1, 86, 0),  # abaixo do corte
            ("GBP/USD", T0 + 180, T0 + 185, 0.25, 1.2, 90, -1),  # empate
            ("GBP/USD", T0 + 240, T0 + 245, 0.64, 1.2, 90, None)]  # aberta
    db.executemany("INSERT INTO forte VALUES (?,?,?,?,?,?,?)", rows)
    return sm


def test_entries_follow_simulator_rules():
    sm = _sm()
    es = sm.entries(now=T0 + 300)
    assert [e["status"] for e in es] == ["ganhou", "perdeu", "empate", "aberta"]
    assert abs(es[0]["gain"] - 8.6) < 1e-9 and es[0]["profit"] == es[0]["gain"]
    assert es[1]["side"] == "VENDA" and es[1]["profit"] == -10.0
    assert abs(es[3]["chance"] - 0.64) < 1e-9 and es[3]["expiry_ts"] == T0 + 240 + 360
    # o lucro das telas bate com o resumo do simulador
    s = day_summary(es)
    assert abs(s["profit"] - sm.simulate()["lucro"]) < 1e-9
    assert (s["wins"], s["losses"], s["ties"], s["open"]) == (1, 1, 1, 1)
    # aberta há muito tempo sem conferência não conta como resultado
    assert sm.entries(now=T0 + 240 + 360 + 500)[-1]["status"] == "sem leitura"


def test_groups_by_pair_and_chance():
    es = _sm().entries(now=T0 + 300)
    by_pair, by_band = groups(es)
    assert dict(by_pair)["EUR/USD"]["n"] == 2
    assert [n for n, _ in by_band] == ["64–66%", "70% ou mais"]


def test_live_result_needs_same_pair():
    e = _sm().entries(now=T0 + 300)[-1]  # compra GBP/USD a 1.2
    assert live_result(e, "GBP/USD", 1.2001) == (9.0, "ganhando")
    assert live_result(e, "GBP/USD", 1.1999) == (-10.0, "perdendo")
    assert live_result(e, "EUR/USD", 1.3) == (None, None)


def test_history_minutes_counts_continuous_bars():
    sm = _sm()
    rows = [("EUR/USD", T0 + 60 * i, 1, 1, 1, 1, 0) for i in range(50)]
    del rows[10]
    sm.db.executemany("INSERT INTO fita VALUES (?,?,?,?,?,?,?)", rows)
    assert sm.history_minutes("EUR/USD", now=T0 + 60 * 50) == 39
    assert sm.history_minutes("EUR/USD", now=T0 + 60 * 60) == 0  # parou de ler
    assert sm.history_minutes("GBP/USD", now=T0) == 0


def test_side_screens_have_panel_height():
    es = _sm().entries(now=T0 + 300)
    es[0]["seen_at"] = T0 + 290
    for st in (SimState(entries=es, now=T0 + 300, asset="GBP/USD", price=1.2001, hist_min=135),
               SimState(hist_min=40), SimState(available=False), SimState(note="OTC")):
        img = side_screens(st, 700)
        assert img.shape[0] == 700 and img.shape[1] > 700 and img.shape[2] == 3
        assert img.std() > 5
    big = side_screens(SimState(entries=es * 10, now=T0 + 300), 1400, scale=1.3)
    assert big.shape[0] == 1400


def test_side_screens_cache_and_toggle():
    s = SideScreens()
    assert s.image(700) is None
    s.state = SimState(hist_min=10)
    a = s.image(700)
    assert s.image(700) is a  # nada mudou: não redesenha
    s.visible = False
    assert s.image(700) is None


def test_sounds_are_short_wavs(tmp_path):
    for gen in (alerts.money_samples, alerts.error_samples):
        p = alerts.write_wav(tmp_path / "x.wav", gen())
        with wave.open(str(p)) as w:
            secs = w.getnframes() / w.getframerate()
        assert 0.2 < secs < 1.2


def test_last_measure_and_waiting_card():
    sm = _sm()
    assert sm.last_measure("GBP/USD") == (T0 + 245, 0.64)
    assert sm.last_measure("AUD/USD") is None
    img = side_screens(SimState(now=T0 + 300, asset="EUR/USD", hist_min=135, last_p=0.43, last_ts=T0 + 250), 700)
    assert img.shape[0] == 700
