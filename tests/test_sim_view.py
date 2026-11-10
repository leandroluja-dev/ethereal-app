"""Telas ao lado do painel: três carteiras do simulador, resultados do dia e sons."""

import sqlite3
import wave

import numpy as np

from candlesense import alerts
from candlesense.engine.strong import StrongMode
from candlesense.ui.sim_view import (WALLETS, SideScreens, SimState, by_band, day_summary, live_result, merged,
                                     side_screens)

T0 = 1_762_740_000  # múltiplo de 300


def _sm():
    db = sqlite3.connect(":memory:")
    sm = StrongMode(db, log=lambda *a, **k: None)
    rows = [("EUR/USD", T0, T0 + 65, 0.70, 1.1, 86, 1),          # compra certa
            ("EUR/USD", T0 + 60, T0 + 125, 0.30, 1.1, 86, 1),    # venda errada
            ("EUR/USD", T0 + 120, T0 + 185, 0.59, 1.1, 86, 0),   # só a de 58%
            ("GBP/USD", T0 + 180, T0 + 245, 0.25, 1.2, 90, -1),  # empate
            ("GBP/USD", T0 + 240, T0 + 305, 0.64, 1.2, 90, None)]  # aberta
    db.executemany("INSERT INTO forte VALUES (?,?,?,?,?,?,?)", rows)
    return sm


def test_entries_follow_simulator_rules():
    sm = _sm()
    es = sm.entries(now=T0 + 300)
    assert [e["status"] for e in es] == ["ganhou", "perdeu", "empate", "aberta"]
    assert abs(es[0]["gain"] - 8.6) < 1e-9 and es[0]["profit"] == es[0]["gain"]
    assert es[1]["side"] == "VENDA" and es[1]["profit"] == -10.0
    assert es[3]["expiry_ts"] == T0 + 240 + 360
    s = day_summary(es)
    assert abs(s["profit"] - sm.simulate()["lucro"]) < 1e-9
    assert (s["wins"], s["losses"], s["ties"], s["open"]) == (1, 1, 1, 1)
    assert sm.entries(now=T0 + 240 + 360 + 500)[-1]["status"] == "sem leitura"
    assert len(sm.entries(threshold=0.58, now=T0 + 300)) == 5


def test_candle5_expires_at_end_of_5min_candle():
    sm = _sm()
    # medições que entram em T0+60, +120, +180 (min 1 a 3 da vela) vencem em T0+300; a que entraria em
    # T0+240 (sobra 1 min) fica de fora; a que entra em T0+300 abre a vela seguinte e vence em T0+600
    rows = [("EUR/USD", T0 + 240, 1, 1, 1, 1.2, T0 + 320), ("GBP/USD", T0 + 240, 1, 1, 1, 1.25, T0 + 305)]
    sm.db.executemany("INSERT INTO fita VALUES (?,?,?,?,?,?,?)", rows)
    es = sm.entries(threshold=0.58, now=T0 + 330, candle5=True)
    assert [e["start"] for e in es] == [T0, T0 + 60, T0 + 120, T0 + 240]
    assert [e["expiry_ts"] for e in es] == [T0 + 300] * 3 + [T0 + 600]
    st = {e["start"]: e["status"] for e in es}
    assert st[T0] == "ganhou"            # compra a 1.1, fechou 1.2
    assert st[T0 + 60] == "perdeu"       # venda a 1.1, fechou 1.2
    assert st[T0 + 120] == "ganhou"      # compra a 1.1 (59%), fechou 1.2
    assert st[T0 + 240] == "aberta"


def test_by_band_and_merged():
    sm = _sm()
    st = SimState(entries={w.key: sm.entries(threshold=w.cut, now=T0 + 300, mode=w.mode) for w in WALLETS},
                  now=T0 + 300)
    names = [n for n, _ in by_band(st.get("B"))]
    assert names == ["58–60%", "64–66%", "70% ou mais"]
    rows = merged(st)
    first = [r for r in rows if r["start"] == T0 and not r.get("expiry_ts") == T0 + 300]
    assert {w.key for w in first[0]["wallets"]} == {"A", "B"}


def test_live_result_needs_same_pair():
    e = _sm().entries(now=T0 + 300)[-1]  # compra GBP/USD a 1.2
    assert live_result(e, "GBP/USD", 1.2001) == (9.0, "ganhando")
    assert live_result(e, "GBP/USD", 1.1999) == (-10.0, "perdendo")
    assert live_result(e, "EUR/USD", 1.3) == (None, None)


def test_history_minutes_and_last_measure():
    sm = _sm()
    rows = [("EUR/USD", T0 + 60 * i, 1, 1, 1, 1, 0) for i in range(50)]
    del rows[10]
    sm.db.executemany("INSERT INTO fita VALUES (?,?,?,?,?,?,?)", rows)
    assert sm.history_minutes("EUR/USD", now=T0 + 60 * 50) == 39
    assert sm.history_minutes("EUR/USD", now=T0 + 60 * 60) == 0
    assert sm.last_measure("GBP/USD") == (T0 + 305, 0.64)
    assert sm.last_measure("AUD/USD") is None


def test_side_screens_have_requested_height():
    sm = _sm()
    ent = {w.key: sm.entries(threshold=w.cut, now=T0 + 300, mode=w.mode) for w in WALLETS}
    ent["A"][0]["seen_at"] = T0 + 290
    for st in (SimState(entries=ent, now=T0 + 300, asset="GBP/USD", price=1.2001, hist_min=135),
               SimState(hist_min=40), SimState(available=False), SimState(note="OTC"),
               SimState(now=T0 + 300, hist_min=135, last_p=0.43, last_ts=T0 + 250)):
        img = side_screens(st, 1000)
        assert img.shape[0] == 1000 and img.shape[1] > 700 and img.std() > 5
    big = {k: v * 10 for k, v in ent.items()}
    assert side_screens(SimState(entries=big, now=T0 + 300), 1300, scale=1.3).shape[0] == 1300


def test_side_screens_cache_toggle_and_height():
    s = SideScreens()
    assert s.image(940) is None
    s.state = SimState(hist_min=10)
    a = s.image(940)
    assert s.image(940) is a
    assert s.height(700) >= 940 and s.height(1500) == 1500
    s.visible = False
    assert s.image(940) is None


def test_sounds_are_short_wavs(tmp_path):
    for gen in (alerts.money_samples, alerts.error_samples):
        p = alerts.write_wav(tmp_path / "x.wav", gen())
        with wave.open(str(p)) as w:
            secs = w.getnframes() / w.getframerate()
        assert 0.2 < secs < 1.2


def test_one_minute_and_thirty_second_risk_wallets():
    sm = _sm()
    # 1 min: a conta da vela T0 (compra 70% a 1.1) vence no fechamento da vela T0+60
    sm.db.execute("INSERT INTO fita VALUES ('EUR/USD', ?, 1, 1, 1, 1.05, ?)", (T0 + 60, T0 + 135))
    es = {e["start"]: e for e in sm.entries(threshold=0.55, now=T0 + 140, mode="1min")}
    assert es[T0]["expiry_ts"] == T0 + 120 and es[T0]["status"] == "perdeu"
    assert es[T0 + 60]["status"] == "aberta"
    # 30 s: preço lido na tela no instante da conta e 30 s depois
    sm.db.executemany("INSERT INTO precos VALUES (?,?,?)",
                      [("EUR/USD", T0 + 65, 1.1000), ("EUR/USD", T0 + 96, 1.1003),
                       ("EUR/USD", T0 + 125, 1.1000), ("EUR/USD", T0 + 154, 1.0990)])
    es = {e["start"]: e for e in sm.entries(threshold=0.55, now=T0 + 160, mode="30s")}
    assert es[T0]["expiry_ts"] == T0 + 95 and es[T0]["status"] == "ganhou"   # compra, subiu
    assert es[T0 + 60]["status"] == "ganhou"                                 # venda, caiu
    assert es[T0 + 120]["status"] == "aberta"                               # vence em T0+215


def test_screen_price_stored_once_per_second():
    from types import SimpleNamespace

    sm = _sm()
    r = SimpleNamespace(ok=True, current_price=1.2345)
    for now in (T0 + 0.1, T0 + 0.6, T0 + 1.2):
        sm._store_price(r, "EUR/USD", now)
    assert sm.db.execute("SELECT COUNT(*) FROM precos").fetchone()[0] == 2
    assert sm.price_at("EUR/USD", T0 + 2) == 1.2345
    assert sm.price_at("EUR/USD", T0 + 9) is None


def test_dragging_window_corner_changes_scale(monkeypatch):
    import types

    from candlesense.ui import panel as P

    rect = [0, 0, 800, 1000]
    fake = types.SimpleNamespace(getWindowImageRect=lambda n: tuple(rect), resizeWindow=lambda *a: None)
    w = P.PanelWindow.__new__(P.PanelWindow)
    w.cv2, w.scale = fake, 1.0
    clock = [100.0]
    monkeypatch.setattr(P.time, "monotonic", lambda: clock[0])
    w._fit_window(np.zeros((1000, 800, 3), np.uint8))
    clock[0] += 1; w._watch_user_resize()            # referência: 800x1000
    rect[2:] = [1200, 1500]
    clock[0] += 0.1; w._watch_user_resize()          # puxando
    assert w.pop_scale_change() is None
    clock[0] += 0.5; w._watch_user_resize()          # soltou
    assert w.pop_scale_change() == 1.5 and w.scale == 1.5
    assert w.set_scale(9) and w.scale == 2.5
