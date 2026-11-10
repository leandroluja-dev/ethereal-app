"""Livro das carteiras do simulador: saldo que acumula dia a dia e o histórico.

Cada carteira começa com R$ 10.000,00 (dinheiro de mentira) no dia 10/11/2026 e
soma os ganhos e perdas de todas as entradas desde então. As entradas que já
saíram ficam guardadas na tabela ``sim_hist``: o resultado não muda depois (as de
30 s dependem do preço de cada segundo, que só fica 3 dias no banco).
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

START_BALANCE = 10_000.0
START_TS = time.mktime((2026, 11, 10, 0, 0, 0, 0, 0, -1))  # 10/11/2026, 00:00 no horário do Mac
DONE = ("ganhou", "perdeu", "empate", "sem leitura")
WEEK = ("seg", "ter", "qua", "qui", "sex", "sáb", "dom")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sim_hist (
    wallet TEXT NOT NULL, asset TEXT NOT NULL, start INTEGER NOT NULL, mode TEXT,
    ts REAL NOT NULL, side TEXT, chance REAL, payout INTEGER, stake REAL,
    status TEXT NOT NULL, profit REAL,
    PRIMARY KEY (wallet, asset, start)
);
"""


def day_label(day: str) -> str:
    """'2026-11-10' -> 'ter 10/11'."""
    t = time.strptime(day, "%Y-%m-%d")
    return f"{WEEK[t.tm_wday]} {t.tm_mday:02d}/{t.tm_mon:02d}"


class SimBook:
    def __init__(self, db) -> None:
        self.db = db
        self.db.executescript(SCHEMA)
        self.db.commit()
        self._past = None
        self._past_at = 0.0
        self._past_day = None

    def record(self, wallet: str, entries: Iterable[dict]) -> int:
        """Guarda as entradas que já têm resultado (as abertas ficam de fora)."""
        rows = [(wallet, e["asset"], int(e["start"]), e.get("mode"), e["ts"], e["side"], e["chance"],
                 e["payout"], e["stake"], e["status"], e["profit"] or 0.0)
                for e in entries if e["status"] in DONE and e["ts"] >= START_TS]
        if rows:
            self.db.executemany("INSERT OR IGNORE INTO sim_hist VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
            self.db.commit()
        return len(rows)

    def past_days(self, today_start: float, now: Optional[float] = None) -> list[dict]:
        """Dias anteriores a hoje: [{day, label, w: {letra: {n, wins, losses, profit}}}], do mais antigo."""
        now = time.time() if now is None else now
        if self._past is not None and self._past_day == today_start and now - self._past_at < 60:
            return self._past
        q = ("SELECT date(ts, 'unixepoch', 'localtime') AS d, wallet, COUNT(*), "
             "SUM(status='ganhou'), SUM(status='perdeu'), SUM(profit) "
             "FROM sim_hist WHERE ts>=? AND ts<? GROUP BY d, wallet ORDER BY d")
        days: dict = {}
        for d, w, n, wins, losses, profit in self.db.execute(q, (START_TS, today_start)):
            days.setdefault(d, {})[w] = {"n": n, "wins": wins or 0, "losses": losses or 0, "profit": profit or 0.0}
        self._past = [{"day": d, "label": day_label(d), "w": ws} for d, ws in days.items()]
        self._past_at, self._past_day = now, today_start
        return self._past


def day_entries(db, day: str) -> list[dict]:
    """Entradas já resolvidas de um dia ('2026-11-10'), no formato das entradas ao vivo."""
    q = ("SELECT wallet, asset, start, mode, ts, side, chance, payout, stake, status, profit FROM sim_hist "
         "WHERE date(ts, 'unixepoch', 'localtime') = ? ORDER BY ts")
    cols = ("wallet", "asset", "start", "mode", "ts", "side", "chance", "payout", "stake", "status", "profit")
    return [dict(zip(cols, r)) for r in db.execute(q, (day,))]


def balances(past: list[dict], today: dict, keys: Iterable[str]) -> dict:
    """Saldo de cada carteira: R$ 10.000 + dias anteriores + hoje (``today`` = {letra: lucro})."""
    out = {}
    for k in keys:
        out[k] = START_BALANCE + sum(d["w"].get(k, {}).get("profit", 0.0) for d in past) + today.get(k, 0.0)
    return out
