"""Modo medição do modelo de formatos ("sinal forte").

Só registra. Não acende sinal no painel, não toca som e nunca opera.

* Guarda cada vela de 1 min lida na tela (tabela ``fita``), com o horário
  dela em UTC, para montar o histórico contínuo que o modelo precisa
  (cerca de 2h15 de gráfico de 1 min).
* Uma vez por vela, logo depois que ela fecha, calcula a chance de subir em
  5 min e guarda em ``forte``. Cinco minutos depois confere pelo fechamento
  lido na tela: 1 = subiu, 0 = caiu, -1 = empate.
* Se a leitura estiver incerta, o gráfico não for de 1 min ou faltar vela no
  histórico, não calcula nada.
"""

from __future__ import annotations

import time
from typing import Optional

from ..models import ChartReading
from .shapemodel import MIN_HISTORY, Bar, ShapeModel, features

SCHEMA = """
CREATE TABLE IF NOT EXISTS fita (
    asset TEXT NOT NULL, start INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL, updated REAL,
    PRIMARY KEY (asset, start)
);
CREATE TABLE IF NOT EXISTS forte (
    asset TEXT NOT NULL, start INTEGER NOT NULL, ts REAL NOT NULL,
    p_up REAL NOT NULL, price REAL NOT NULL, payout INTEGER, out5 INTEGER,
    PRIMARY KEY (asset, start)
);
"""

STRONG = 0.66  # "sinal forte": chance de 66% ou mais para um dos lados
EVAL_AFTER_S = 3  # espera a vela recém-fechada assentar na tela
EVAL_UNTIL_S = 20  # passou disso, a entrada já ficou tarde: não mede


class StrongMode:
    def __init__(self, db, model: Optional[ShapeModel] = None, log=print) -> None:
        self.db = db
        self.db.executescript(SCHEMA)
        self.db.commit()
        self.model = model or ShapeModel()
        self.log = log
        self._done: set[tuple[str, int]] = set()

    # ------------------------------------------------------------ fita

    def _store_bars(self, r: ChartReading, asset: str, now: float) -> Optional[int]:
        """Grava as velas fechadas visíveis. Devolve o início da vela em formação (UTC)."""
        if not r.ok or r.timeframe_s != 60 or r.timer_seconds is None or not r.candles:
            return None
        if not r.candles[-1].forming:
            return None
        forming_start = int(round((now + r.timer_seconds) / 60)) * 60 - 60
        closed = [c for c in r.candles if not c.forming]
        rows = []
        for k, c in enumerate(reversed(closed)):
            start = forming_start - (k + 1) * 60
            rows.append((asset, start, c.open, c.high, c.low, c.close, now))
        # A vela que acabou de fechar ainda pode mudar 1 ou 2 px: as 3 últimas são
        # sempre regravadas; as mais antigas só entram se ainda não existirem.
        self.db.executemany(
            "INSERT INTO fita VALUES (?,?,?,?,?,?,?) ON CONFLICT(asset,start) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, "
            "updated=excluded.updated", rows[:3])
        self.db.executemany("INSERT OR IGNORE INTO fita VALUES (?,?,?,?,?,?,?)", rows[3:])
        return forming_start

    def bars(self, asset: str, last_start: int, n: int = MIN_HISTORY) -> list[Bar]:
        cur = self.db.execute(
            "SELECT start, open, high, low, close FROM fita WHERE asset=? AND start<=? AND start>? ORDER BY start",
            (asset, last_start, last_start - n * 60))
        return [Bar(*row) for row in cur]

    # ------------------------------------------------------------ medição

    def update(self, r: ChartReading, asset: str, now: Optional[float] = None) -> Optional[float]:
        """Chamado a cada leitura. Devolve a chance de subir quando calcula uma nova."""
        now = time.time() if now is None else now
        forming = self._store_bars(r, asset, now)
        self.resolve()
        self.db.commit()
        if forming is None or asset == "?" or "OTC" in asset.upper():
            return None
        since_close = now - forming
        last = forming - 60  # vela que acabou de fechar
        if not (EVAL_AFTER_S <= since_close <= EVAL_UNTIL_S) or (asset, last) in self._done:
            return None
        self._done.add((asset, last))
        f = features(self.bars(asset, last))
        if f is None:
            return None
        p = self.model.prob_up(f)
        if p is None:
            return None
        price = self.db.execute("SELECT close FROM fita WHERE asset=? AND start=?", (asset, last)).fetchone()[0]
        self.db.execute("INSERT OR IGNORE INTO forte VALUES (?,?,?,?,?,?,NULL)",
                        (asset, last, now, p, price, r.payout_pct))
        self.db.commit()
        if p >= STRONG or p <= 1 - STRONG:
            lado = "COMPRA" if p >= 0.5 else "VENDA"
            self.log(f"sinal forte (só medição): {lado} {asset} {max(p, 1 - p):.0%} expiração 5 min", flush=True)
        return p

    def resolve(self) -> int:
        """Confere as medições cujo prazo de 5 min já fechou na tela."""
        q = ("SELECT f.asset, f.start, f.price, b.close FROM forte f JOIN fita b "
             "ON b.asset=f.asset AND b.start=f.start + 300 "
             "WHERE f.out5 IS NULL AND b.updated >= b.start + 70")
        rows = self.db.execute(q).fetchall()
        for asset, start, price, close in rows:
            out = 1 if close > price else 0 if close < price else -1
            self.db.execute("UPDATE forte SET out5=? WHERE asset=? AND start=?", (out, asset, start))
        return len(rows)

    def report(self, since: float = 0.0) -> dict[float, tuple[int, int]]:
        """Acertos e total por nível de corte (só medições conferidas, sem empates)."""
        rows = self.db.execute("SELECT p_up, out5 FROM forte WHERE ts>=? AND out5 IN (0,1)", (since,)).fetchall()
        res = {}
        for th in (0.55, 0.58, 0.60, 0.62, 0.64, 0.66):
            hits = n = 0
            for p, out in rows:
                if p >= th or p <= 1 - th:
                    n += 1
                    hits += int((p >= 0.5) == (out == 1))
            res[th] = (hits, n)
        return res

    def simulate(self, since: float = 0.0, threshold: float = 0.62, stake: float = 10.0,
                 min_payout: int = 80) -> dict:
        """Conta de mentira: uma entrada de ``stake`` em cada medição acima do corte.

        Ganho = stake x payout lido na tela naquele minuto (86% se não foi lido);
        perda = -stake; empate devolve o valor. Entradas com payout abaixo de
        ``min_payout`` ficam de fora, como uma pessoa faria.
        """
        rows = self.db.execute(
            "SELECT asset, p_up, out5, payout FROM forte WHERE ts>=? AND out5 IS NOT NULL ORDER BY ts",
            (since,)).fetchall()
        res = {"entradas": 0, "ganhos": 0, "perdas": 0, "empates": 0, "lucro": 0.0, "por_par": {}}
        for asset, p, out, payout in rows:
            if not (p >= threshold or p <= 1 - threshold):
                continue
            pay = payout if payout else 86
            if pay < min_payout:
                continue
            par = res["por_par"].setdefault(asset, [0, 0])
            res["entradas"] += 1
            if out == -1:
                res["empates"] += 1
                continue
            par[1] += 1
            if (p >= 0.5) == (out == 1):
                res["ganhos"] += 1
                par[0] += 1
                res["lucro"] += stake * pay / 100.0
            else:
                res["perdas"] += 1
                res["lucro"] -= stake
        return res

    def day_text(self, since: float, threshold: float = 0.62, stake: float = 10.0) -> str:
        """Resumo do simulador para o log e para dados/simulador.txt."""
        s = self.simulate(since, threshold, stake)
        n = s["ganhos"] + s["perdas"]
        acc = f"{s['ganhos'] / n:.0%}" if n else "-"
        pares = ", ".join(f"{a} {w}/{t}" for a, (w, t) in s["por_par"].items()) or "nenhum"
        return (f"Simulador (só medição, R$ {stake:.0f} por entrada, corte {threshold:.0%}): "
                f"{s['entradas']} entradas, {s['ganhos']} ganhos, {s['perdas']} perdas, {s['empates']} empates, "
                f"acerto {acc}, resultado R$ {s['lucro']:+.2f}. Por par: {pares}.")

    # ------------------------------------------------------------ telas do simulador

    def entries(self, since: float = 0.0, threshold: float = 0.62, stake: float = 10.0,
                min_payout: int = 80, now: Optional[float] = None) -> list[dict]:
        """Entradas virtuais do simulador, uma por medição acima do corte (mais antiga primeiro).

        Mesmas regras de ``simulate``. Cada entrada traz: asset, start, ts, side
        (COMPRA/VENDA), chance, price (preço de entrada), payout, stake, gain (quanto
        ganha se acertar), expiry_ts, status (aberta, ganhou, perdeu, empate ou
        sem leitura) e profit (None enquanto aberta).
        """
        now = time.time() if now is None else now
        rows = self.db.execute(
            "SELECT asset, start, ts, p_up, price, payout, out5 FROM forte WHERE ts>=? ORDER BY ts",
            (since,)).fetchall()
        out = []
        for asset, start, ts, p, price, payout, res in rows:
            if not (p >= threshold or p <= 1 - threshold):
                continue
            pay = payout if payout else 86
            if pay < min_payout:
                continue
            side = "COMPRA" if p >= 0.5 else "VENDA"
            gain = stake * pay / 100.0
            expiry = start + 360  # fecha a vela start+300: 5 min depois da entrada
            if res is None:
                status = "aberta" if now < expiry + 120 else "sem leitura"
                profit = None
            elif res == -1:
                status, profit = "empate", 0.0
            elif (side == "COMPRA") == (res == 1):
                status, profit = "ganhou", gain
            else:
                status, profit = "perdeu", -stake
            out.append({"asset": asset, "start": start, "ts": ts, "side": side, "chance": max(p, 1 - p),
                        "price": price, "payout": pay, "stake": stake, "gain": gain, "expiry_ts": expiry,
                        "status": status, "profit": profit})
        return out

    def history_minutes(self, asset: str, now: Optional[float] = None) -> int:
        """Quantos minutos seguidos de gráfico de 1 min o app já tem deste par (precisa de MIN_HISTORY)."""
        now = time.time() if now is None else now
        rows = self.db.execute("SELECT start FROM fita WHERE asset=? AND start<=? ORDER BY start DESC LIMIT ?",
                               (asset, int(now), MIN_HISTORY + 5)).fetchall()
        if not rows or now - rows[0][0] > 180:
            return 0
        n, prev = 0, None
        for (s,) in rows:
            if prev is not None and prev - s != 60:
                break
            n, prev = n + 1, s
        return n
