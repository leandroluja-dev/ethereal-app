"""Histórico em SQLite: velas lidas, previsões do motor e entradas da carteira.

Tudo fica no Mac da pessoa, em ``dados/candlesense.db`` na pasta do projeto.
Nada é enviado para fora.

Horários são em segundos desde 1970 (relógio do Mac). O fim de cada vela é
alinhado ao múltiplo do tempo da vela, como a IQ Option faz (velas de 5 min
fecham em :00, :05, :10...).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..engine.rules import Decision
from ..models import ChartReading

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    asset TEXT NOT NULL, tf INTEGER NOT NULL, end_ts INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL, updated REAL,
    PRIMARY KEY (asset, tf, end_ts)
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, asset TEXT NOT NULL, tf INTEGER NOT NULL, candle_end INTEGER NOT NULL,
    phase TEXT NOT NULL, price REAL NOT NULL,
    p1 REAL, p2 REAL, p3 REAL, p5 REAL,
    action TEXT, best_h INTEGER, threshold REAL, payout INTEGER,
    out1 INTEGER, out2 INTEGER, out3 INTEGER, out5 INTEGER,
    UNIQUE (asset, tf, candle_end, phase)
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opened REAL NOT NULL, account TEXT NOT NULL, amount REAL NOT NULL,
    asset TEXT, tf INTEGER, entry_price REAL, expiry_ts REAL, payout INTEGER,
    app_action TEXT, app_prob REAL,
    closed REAL, result TEXT, profit REAL, direction TEXT, exit_price REAL
);
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, asset TEXT NOT NULL, tf INTEGER NOT NULL, candle_end INTEGER NOT NULL,
    action TEXT NOT NULL, prob REAL, best_h INTEGER, price REAL, payout INTEGER,
    outcome INTEGER
);
CREATE TABLE IF NOT EXISTS balances (
    ts REAL NOT NULL, account TEXT NOT NULL, balance REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bal ON balances (account, ts);
CREATE INDEX IF NOT EXISTS idx_pred_open ON predictions (asset, tf, candle_end);
CREATE INDEX IF NOT EXISTS idx_trades_day ON trades (opened);
"""

HCOLS = {1: ("p1", "out1"), 2: ("p2", "out2"), 3: ("p3", "out3"), 5: ("p5", "out5")}


def candle_end_ts(now: float, timer_seconds: int, tf: int) -> int:
    """Fim da vela atual, alinhado ao tempo da vela."""
    return int(round((now + timer_seconds) / tf) * tf)


@dataclass
class Prediction:
    id: int
    ts: float
    tf: int
    candle_end: int
    phase: str
    price: float
    p: dict[int, Optional[float]]
    out: dict[int, Optional[int]]
    action: str
    best_h: Optional[int]
    threshold: Optional[float]
    payout: Optional[int]


class Store:
    def __init__(self, path: str | Path = "dados/candlesense.db") -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=5, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ---------------------------------------------------------------- escrita

    def record(self, r: ChartReading, d: Decision, asset: str = "?", now: Optional[float] = None) -> None:
        """Guarda as velas fechadas e a previsão desta leitura (uma por fase da vela)."""
        if not r.ok or not r.timeframe_s or r.timer_seconds is None or not r.candles:
            return
        now = time.time() if now is None else now
        tf = int(r.timeframe_s)
        end = candle_end_ts(now, r.timer_seconds, tf)
        closed = [c for c in r.candles if not c.forming]
        # As 3 últimas fechadas: a leitura delas já está estável.
        rows = []
        for k, c in enumerate(reversed(closed[-3:])):
            rows.append((asset, tf, end - (k + 1) * tf, c.open, c.high, c.low, c.close, now))
        self.db.executemany(
            "INSERT INTO candles VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(asset,tf,end_ts) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, updated=excluded.updated",
            rows)
        if d.horizons and d.phase and r.current_price is not None:
            p = {h.candles: (h.p_raw if h.p_raw is not None else h.p_up) for h in d.horizons}
            self.db.execute(
                "INSERT OR IGNORE INTO predictions (ts, asset, tf, candle_end, phase, price, p1, p2, p3, p5, "
                "action, best_h, threshold, payout) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (now, asset, tf, end, d.phase, r.current_price, p.get(1), p.get(2), p.get(3), p.get(5),
                 d.action, d.best.candles if d.best else None, d.threshold, r.payout_pct))
        self.db.commit()
        self.resolve()

    def record_signal(self, r: ChartReading, d: Decision, asset: str = "?", now: Optional[float] = None) -> None:
        """Um sinal (COMPRA/VENDA) que acabou de acender."""
        if not d.signal or not r.timeframe_s or r.timer_seconds is None or r.current_price is None:
            return
        now = time.time() if now is None else now
        tf = int(r.timeframe_s)
        self.db.execute(
            "INSERT INTO signals (ts, asset, tf, candle_end, action, prob, best_h, price, payout) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (now, asset, tf, candle_end_ts(now, r.timer_seconds, tf), d.action,
             d.best.prob if d.best else None, d.best.candles if d.best else 1, r.current_price, r.payout_pct))
        self.db.commit()

    def signals(self, since: float = 0.0) -> list[dict]:
        cur = self.db.execute("SELECT id, ts, asset, tf, action, prob, best_h, price, payout, outcome "
                              "FROM signals WHERE ts >= ? ORDER BY ts", (since,))
        keys = ("id", "ts", "asset", "tf", "action", "prob", "best_h", "price", "payout", "outcome")
        return [dict(zip(keys, row)) for row in cur]

    # ------------------------------------------------------------ carteira

    def open_trade(self, **kw) -> int:
        cols = ", ".join(kw)
        q = ", ".join("?" for _ in kw)
        cur = self.db.execute(f"INSERT INTO trades ({cols}) VALUES ({q})", tuple(kw.values()))
        self.db.commit()
        return int(cur.lastrowid)

    def close_trade(self, trade_id: int, **kw) -> None:
        sets = ", ".join(f"{k}=?" for k in kw)
        self.db.execute(f"UPDATE trades SET {sets} WHERE id=?", (*kw.values(), trade_id))
        self.db.commit()

    def log_balance(self, ts: float, account: str, balance: float) -> None:
        """Saldo confirmado (para a evolução da carteira por dia)."""
        self.db.execute("INSERT INTO balances VALUES (?,?,?)", (ts, account, balance))
        self.db.commit()

    def balances(self, since: float = 0.0, account: Optional[str] = None) -> list[tuple[float, str, float]]:
        q = "SELECT ts, account, balance FROM balances WHERE ts >= ?"
        args: list = [since]
        if account:
            q += " AND account = ?"
            args.append(account)
        return [tuple(r) for r in self.db.execute(q + " ORDER BY ts", args)]

    def trades(self, since: float = 0.0, account: Optional[str] = None) -> list[dict]:
        q = "SELECT * FROM trades WHERE opened >= ?"
        args: list = [since]
        if account:
            q += " AND account = ?"
            args.append(account)
        cur = self.db.execute(q + " ORDER BY opened", args)
        names = [c[0] for c in cur.description]
        return [dict(zip(names, row)) for row in cur]

    def resolve(self) -> int:
        """Confere as previsões cujo prazo já fechou: 1 = subiu, 0 = caiu, -1 = empate."""
        n = 0
        for h, (_, oc) in HCOLS.items():
            q = (f"SELECT p.id, p.price, c.close FROM predictions p JOIN candles c "
                 f"ON c.asset=p.asset AND c.tf=p.tf AND c.end_ts = p.candle_end + ({h} - 1) * p.tf "
                 f"WHERE p.{oc} IS NULL AND c.updated >= c.end_ts + 10")
            res = self.db.execute(q).fetchall()
            for pid, price, close in res:
                out = 1 if close > price else 0 if close < price else -1
                self.db.execute(f"UPDATE predictions SET {oc}=? WHERE id=?", (out, pid))
                n += 1
        # Sinais: confere na vela do prazo sugerido (best_h).
        q = ("SELECT s.id, s.price, s.action, c.close FROM signals s JOIN candles c "
             "ON c.asset=s.asset AND c.tf=s.tf AND c.end_ts = s.candle_end + (s.best_h - 1) * s.tf "
             "WHERE s.outcome IS NULL AND c.updated >= c.end_ts + 10")
        for sid, price, action, close in self.db.execute(q).fetchall():
            if close == price:
                out = -1
            else:
                out = int((close > price) == (action == "COMPRA"))
            self.db.execute("UPDATE signals SET outcome=? WHERE id=?", (out, sid))
            n += 1
        if n:
            self.db.commit()
        return n

    # ---------------------------------------------------------------- leitura

    def predictions(self, since: float = 0.0, resolved_h: Optional[int] = None) -> list[Prediction]:
        q = "SELECT id, ts, tf, candle_end, phase, price, p1, p2, p3, p5, out1, out2, out3, out5, " \
            "action, best_h, threshold, payout FROM predictions WHERE ts >= ?"
        if resolved_h is not None:
            q += f" AND {HCOLS[resolved_h][1]} IS NOT NULL"
        out = []
        for row in self.db.execute(q + " ORDER BY ts", (since,)):
            out.append(Prediction(
                id=row[0], ts=row[1], tf=row[2], candle_end=row[3], phase=row[4], price=row[5],
                p={1: row[6], 2: row[7], 3: row[8], 5: row[9]},
                out={1: row[10], 2: row[11], 3: row[12], 5: row[13]},
                action=row[14], best_h=row[15], threshold=row[16], payout=row[17]))
        return out
