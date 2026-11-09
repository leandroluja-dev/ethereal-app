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
