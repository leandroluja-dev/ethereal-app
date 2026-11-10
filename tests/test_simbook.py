"""Saldo das carteiras do simulador e histórico por dia."""

import sqlite3

from candlesense.history import simbook as sb
from candlesense.history.simbook import SimBook, balances


def _e(start, status, profit, ts=None):
    return {"asset": "GBP/USD", "start": start, "mode": "5min", "ts": ts or start + 66, "side": "VENDA",
            "chance": 0.6, "payout": 89, "stake": 10.0, "status": status, "profit": profit}


def test_book_keeps_finished_entries_and_sums_balance():
    book = SimBook(sqlite3.connect(":memory:"))
    t = sb.START_TS + 3600
    assert book.record("A", [_e(t, "ganhou", 8.9), _e(t + 60, "perdeu", -10.0), _e(t + 120, "aberta", None)]) == 2
    book.record("A", [_e(t, "ganhou", 8.9)])  # repetida: não conta duas vezes
    past = book.past_days(sb.START_TS + 86400, now=t + 86400)
    w = past[0]["w"]["A"]
    assert len(past) == 1 and (w["n"], w["wins"], w["losses"]) == (2, 1, 1) and abs(w["profit"] + 1.1) < 1e-9
    b = balances(past, {"A": 8.9}, ["A", "B"])
    assert abs(b["A"] - (10_000 - 1.1 + 8.9)) < 1e-9 and b["B"] == 10_000
    assert book.record("A", [_e(sb.START_TS - 600, "ganhou", 8.9)]) == 0  # antes do começo: fora


def test_wallet_b_never_keeps_entries_below_its_cut():
    import sqlite3

    from candlesense.history.simbook import START_TS, SimBook

    db = sqlite3.connect(":memory:")
    book = SimBook(db)
    e = {"asset": "X", "start": int(START_TS) + 600, "mode": "5min", "ts": START_TS + 602, "side": "VENDA",
         "chance": 0.553, "payout": 89, "stake": 10.0, "status": "perdeu", "profit": -10.0}
    book.record("B", [e])
    book.record("C", [e])
    SimBook(db)  # abrir de novo limpa
    assert [r[0] for r in db.execute("SELECT wallet FROM sim_hist")] == ["C"]
