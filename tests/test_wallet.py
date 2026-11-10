import numpy as np
import cv2

from candlesense.history.store import Store
from candlesense.history.wallet import WalletTracker, summarize
from candlesense.models import Box
from candlesense.vision.wallet import account_kind, parse_money

T0 = 1_800_000_000.0


def _feed(w, t, bal, price=1.0, acc="treino", **kw):
    """Duas leituras iguais seguidas (o saldo precisa se confirmar)."""
    ev = w.update(acc, bal, price=price, now=t, **kw)
    ev += w.update(acc, bal, price=price, now=t + 1, **kw)
    return ev


def test_parse_money():
    assert parse_money("$923.22") == 923.22
    assert parse_money("R$ 1.234,56") == 1234.56
    assert parse_money("$10,923.22") == 10923.22
    assert parse_money("$1") == 1.0
    assert parse_money("abc") is None


def test_account_colour():
    img = np.zeros((40, 120, 3), np.uint8)
    cv2.putText(img, "$923", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 140, 240), 3)  # laranja (BGR)
    assert account_kind(img, Box(0, 0, 120, 40)) == "treino"
    img[:] = 0
    cv2.putText(img, "$923", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 200, 40), 3)  # verde
    assert account_kind(img, Box(0, 0, 120, 40)) == "real"


def test_win_after_drop_and_rise():
    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("treino", 1000.0, price=1.0, now=T0)
    w.note_signal("COMPRA", 0.58, now=T0 + 5)
    ev = _feed(w, T0 + 10, 990.0, price=1.0, payout=85, expiry_seconds=60, tf=60)
    assert [e.kind for e in ev] == ["entrada"]
    assert ev[0].amount == 10.0 and ev[0].direction == "COMPRA"
    for k in range(12, 70):
        w.update("treino", 990.0, price=1.0 + (k - 10) * 1e-4, now=T0 + k)
    ev = _feed(w, T0 + 72, 1008.5, price=1.006)
    assert [e.kind for e in ev] == ["ganho"]
    assert ev[0].profit == 8.5 and ev[0].direction == "COMPRA"
    day = summarize(s.trades())["treino"]
    assert (day.n, day.wins, day.losses) == (1, 1, 0)
    assert round(day.profit, 2) == 8.5
    assert day.followed == [1, 1]


def test_loss_when_no_rise_and_direction_from_price():
    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("real", 500.0, price=2.0, now=T0)
    _feed(w, T0 + 1, 480.0, price=2.0, acc="real", payout=80, expiry_seconds=30)
    for k in range(3, 40):
        w.update("real", 480.0, price=2.0 + k * 1e-3, now=T0 + k)  # preço subiu
    ev = w.update("real", 480.0, price=2.05, now=T0 + 60)
    assert [e.kind for e in ev] == ["perda"]
    assert ev[0].profit == -20.0
    assert ev[0].direction == "VENDA"  # perdeu com o preço subindo: apostou na baixa
    t = s.trades()[0]
    assert t["account"] == "real" and t["result"] == "perda" and t["app_action"] is None


def test_single_glitch_and_account_switch_are_not_trades():
    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("treino", 1000.0, now=T0)
    assert w.update("treino", 100.0, now=T0 + 1) == []      # erro de OCR de uma leitura só
    w.update("treino", 1000.0, now=T0 + 2)
    _feed(w, T0 + 3, 250.0, acc="real")                       # trocou para a conta real
    _feed(w, T0 + 6, 1000.0, acc="treino")                    # voltou
    assert s.trades() == []


def test_tie_and_early_sale():
    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("treino", 100.0, now=T0)
    _feed(w, T0 + 1, 95.0, payout=80, expiry_seconds=60)
    ev = _feed(w, T0 + 20, 98.0)                              # vendeu antes, recuperou 3
    assert ev[0].kind == "vendida" and ev[0].profit == -2.0
    _feed(w, T0 + 30, 93.0, payout=80, expiry_seconds=10)
    ev = _feed(w, T0 + 42, 98.0)                              # devolveu o valor: empate
    assert ev[0].kind == "empate" and ev[0].profit == 0.0


def test_exit_alert_against_followed_signal():
    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("treino", 100.0, now=T0)
    w.note_signal("COMPRA", 0.6, now=T0 + 1)
    _feed(w, T0 + 2, 90.0, payout=80, expiry_seconds=120)
    assert w.exit_alert(0.3, now=T0 + 10) is None
    assert w.exit_alert(-0.2, now=T0 + 10) is None
    assert "contra" in w.exit_alert(-0.6, now=T0 + 10)


def test_wallet_view_renders():
    from candlesense.history.scoreboard import score
    from candlesense.ui.wallet_view import render_wallet

    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("treino", 100.0, now=T0)
    _feed(w, T0 + 1, 95.0, payout=80, expiry_seconds=60, tf=60)
    _feed(w, T0 + 62, 104.0)
    img = render_wallet(summarize(s.trades()), score([]), [])
    assert img.ndim == 3 and img.shape[0] > 200
    assert render_wallet({}, None, None).shape[0] > 100


def test_two_wins_closing_together_are_split():
    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("treino", 100.0, now=T0)
    _feed(w, T0 + 1, 90.0, payout=80, expiry_seconds=60)     # entrada de 10
    _feed(w, T0 + 5, 85.0, payout=80, expiry_seconds=56)     # entrada de 5, mesma expiração
    ev = _feed(w, T0 + 62, 112.0)                             # 18 + 9 = 27 de volta
    assert sorted((e.kind, e.amount, e.profit) for e in ev) == [("ganho", 5.0, 4.0), ("ganho", 10.0, 8.0)]
    assert w.open == []
    assert w.update("treino", 112.0, now=T0 + 200) == []     # nenhuma "perda" falsa depois


def test_win_and_tie_together():
    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("treino", 100.0, now=T0)
    _feed(w, T0 + 1, 90.0, payout=80, expiry_seconds=60)
    _feed(w, T0 + 5, 80.0, payout=80, expiry_seconds=56)
    ev = _feed(w, T0 + 62, 108.0)                             # 18 (ganho) + 10 (empate)
    assert sorted(e.kind for e in ev) == ["empate", "ganho"]


def test_implausible_drop_is_ignored_then_becomes_baseline():
    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("treino", 923.22, now=T0)
    assert _feed(w, T0 + 1, 92.32) == []                      # OCR perdeu um dígito, estável
    assert s.trades() == [] and len(w.ignored) == 1
    assert w.balance["treino"] == 923.22
    _feed(w, T0 + 60, 92.32)                                  # ficou na tela: novo saldo de partida
    assert w.balance["treino"] == 92.32 and s.trades() == []


def test_big_drop_matching_invest_is_an_entry():
    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("real", 100.0, now=T0)
    ev = _feed(w, T0 + 1, 30.0, acc="real", invest=70.0, payout=80, expiry_seconds=60)
    assert [e.kind for e in ev] == ["entrada"] and ev[0].amount == 70.0
    assert _feed(w, T0 + 5, 29.5, acc="real") == []            # abaixo do mínimo de $1


def test_one_wrong_colour_read_does_not_mix_accounts():
    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("treino", 1000.0, now=T0)
    w.update("real", 1000.0, now=T0 + 1)                       # cor lida errada uma vez
    w.update("treino", 1000.0, now=T0 + 2)
    assert "real" not in w.balance
    _feed(w, T0 + 3, 990.0, payout=80, expiry_seconds=60)
    assert [t["account"] for t in s.trades()] == ["treino"]


def test_report_today_history_and_expiry():
    from candlesense.history.report import wallet_report

    s = Store(":memory:")
    w = WalletTracker(s)
    w.update("treino", 100.0, now=T0 - 86400)
    _feed(w, T0 - 86400 + 1, 90.0, payout=80, expiry_seconds=40, tf=60)
    _feed(w, T0 - 86400 + 45, 108.0)                           # ontem: 1 ganho
    _feed(w, T0 + 1, 98.0, payout=80, expiry_seconds=290, tf=300)
    w.update("treino", 98.0, now=T0 + 400)                     # hoje: 1 perda (5 min)
    _feed(w, T0 + 500, 93.0, payout=80, expiry_seconds=100, tf=60)
    rep = wallet_report(s, w, now=T0 + 510)
    assert rep.account == "treino" and rep.balance == 93.0
    assert len(rep.open) == 1 and rep.open[0].amount == 5.0 and 80 <= rep.open[0].seconds_left <= 100
    assert (rep.today.n, rep.today.wins, rep.today.losses) == (2, 0, 1)
    assert rep.today.by_expiry[5][0] == 1 and rep.today.by_expiry[2][0] == 1
    assert rep.all_time.n == 3 and round(rep.all_time.profit, 2) == -2.0
    assert [d.n for d in rep.days] == [2, 1]
    assert rep.days[1].profit == 8.0 and rep.days[1].balance_close == 108.0
    assert rep.days[0].balance_open == 108.0


def test_report_without_balance():
    from candlesense.history.report import wallet_report

    rep = wallet_report(Store(":memory:"), None)
    assert rep.account is None and rep.today is None
