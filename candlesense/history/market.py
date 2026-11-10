"""Histórico longo do mercado: anos, meses, semanas e dias, além do que a tela mostra.

A tela da IQ Option mostra só uma ou duas horas de gráfico. Para saber se o par
está em alta ou em baixa hoje, ontem, em 7 dias, em 30 dias e no ano, o app pede
velas diárias e de 1 hora à mesma fonte que completa a fita de 1 min
(Twelve Data, com a chave da pessoa: ``feeds/twelvedata.py``) e guarda tudo na
tabela ``mercado`` do ``dados/candlesense.db``, sem apagar.

* Velas diárias: até 5000 (cerca de 19 anos), a cada 6 horas.
* Velas de 1 hora: últimos ~45 dias, a cada hora.
* São 2 pedidos por hora por par, dentro do limite diário que o app divide.
* Não é a IQ Option nem uma API dela. OTC não existe fora da IQ: sem histórico.
* Roda numa linha separada: sem chave ou sem internet, o app segue normal.
"""

from __future__ import annotations

import calendar
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS mercado (
    asset TEXT NOT NULL, tf TEXT NOT NULL, start INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    PRIMARY KEY (asset, tf, start)
);
"""

DAY = 86400
DAILY_EVERY_S = 6 * 3600
HOURLY_EVERY_S = 3600
HOURS = 45 * 24


def symbol(asset: str) -> Optional[str]:
    """'GBP/USD' -> 'GBP/USD' (nome da Twelve Data). OTC e nomes estranhos -> None."""
    if not asset or "OTC" in asset.upper():
        return None
    a = asset.strip().upper().split()[0]
    if len(a) == 7 and a[3] == "/" and a.replace("/", "").isalpha():
        return a
    return None


def _utc(y: int, mo: int, d: int = 1, h: int = 0) -> int:
    return calendar.timegm((y, mo, d, h, 0, 0))


def aggregate(bars: Iterable[tuple], size: int) -> list[tuple[int, float, float, float, float]]:
    """Junta velas menores em velas de ``size`` segundos (dia = 86400), alinhadas em UTC."""
    out: dict[int, list[float]] = {}
    for s, o, h, l, c in sorted(bars):
        k = s - s % size
        b = out.get(k)
        if b is None:
            out[k] = [o, h, l, c]
        else:
            b[1] = max(b[1], h)
            b[2] = min(b[2], l)
            b[3] = c
    return [(k, *v) for k, v in sorted(out.items())]


# ------------------------------------------------------------------ contexto


@dataclass
class Period:
    name: str          # "hoje", "ontem", "7 dias", "30 dias", "ano"
    change: float      # variação em % do começo do período até agora (ou até o fim, para "ontem")
    pos: float         # onde o preço está entre a mínima (0) e a máxima (1) do período
    high: float
    low: float

    @property
    def trend(self) -> str:
        return "alta" if self.change > 0 else "baixa" if self.change < 0 else "parado"


def _local_midnight(now: float) -> int:
    """Meia-noite de hoje no relógio do Mac (Brasília para o usuário), em segundos UTC."""
    lt = time.localtime(now)
    return int(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1)))


def context(db: sqlite3.Connection, asset: str, price: Optional[float] = None,
            now: Optional[float] = None) -> list[Period]:
    """Alta ou baixa hoje, ontem, nos últimos 7 dias, nos últimos 30 dias e no ano (desde 1º de janeiro).

    Hoje e ontem seguem a meia-noite do relógio do Mac; os demais usam velas diárias (UTC).
    """
    now = time.time() if now is None else now
    midnight = _local_midnight(now)
    utc_today = int(now) - int(now) % DAY

    def rows(tf: str, since: int, until: int) -> list[tuple]:
        return db.execute("SELECT start, open, high, low, close FROM mercado WHERE asset=? AND tf=? "
                          "AND start>=? AND start<? ORDER BY start", (asset, tf, since, until)).fetchall()

    hours = rows("h1", midnight - 40 * DAY, int(now) + 3600)
    days = rows("d1", utc_today - 400 * DAY, utc_today)
    have = {r[0] for r in days}
    days = sorted(days + [r for r in aggregate(hours, DAY) if r[0] not in have and r[0] < utc_today])
    if not days and not hours:
        return []
    today_bars = [r for r in hours if r[0] >= midnight]
    last = price if price is not None else (hours[-1][4] if hours else days[-1][4])
    recent = [r for r in hours if r[0] >= utc_today]  # parte de hoje que a vela diária ainda não tem

    def make(name: str, sel: list[tuple], ref: Optional[float] = None) -> Optional[Period]:
        if not sel:
            return None
        ref = sel[-1][4] if ref is None else ref
        hi = max([r[2] for r in sel] + [ref])
        lo = min([r[3] for r in sel] + [ref])
        return Period(name, (ref / sel[0][1] - 1) * 100, (ref - lo) / (hi - lo) if hi > lo else 0.5, hi, lo)

    found = [
        make("hoje", today_bars, last),
        make("ontem", [r for r in hours if midnight - DAY <= r[0] < midnight]),
        make("7 dias", [r for r in days if r[0] >= utc_today - 7 * DAY] + recent, last),
        make("30 dias", [r for r in days if r[0] >= utc_today - 30 * DAY] + recent, last),
        make("ano", [r for r in days if r[0] >= _utc(time.gmtime(now).tm_year, 1)] + recent, last),
    ]
    return [p for p in found if p is not None]


def context_text(periods: list[Period]) -> str:
    """Uma linha curta: 'Hoje ▲0,12% · Ontem ▼0,30% · 7 dias ▲0,80% · ...'."""
    parts = []
    for p in periods:
        arrow = "▲" if p.change > 0 else "▼" if p.change < 0 else "="
        parts.append(f"{p.name.capitalize()} {arrow}{abs(p.change):.2f}%".replace(".", ","))
    return " · ".join(parts)


# ------------------------------------------------------------------ download em segundo plano


class MarketHistory:
    """Mantém o histórico diário e de 1 hora do par que está na tela, numa linha separada."""

    def __init__(self, db_path: Path, dados="dados", log: Callable[..., None] = print, fetch=None) -> None:
        self.db_path = Path(db_path)
        self.dados = dados
        self.log = log
        self.fetch = fetch  # (símbolo, intervalo, quantidade) -> velas ou None; padrão: Twelve Data
        self._want: Optional[str] = None
        self._last: dict[tuple[str, str], float] = {}
        self._wake = threading.Event()
        self._stop = False
        db = self._connect()
        db.executescript(SCHEMA)
        db.commit()
        db.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=30)

    def start(self) -> "MarketHistory":
        threading.Thread(target=self._run, name="mercado", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop = True
        self._wake.set()

    def want(self, asset: Optional[str]) -> None:
        """Pede o histórico do par da tela (barato: só acorda a linha quando algo venceu)."""
        if not asset or symbol(asset) is None:
            return
        if asset != self._want or self._due(asset):
            self._want = asset
            self._wake.set()

    def _due(self, asset: str, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        return (now - self._last.get((asset, "d1"), 0) > DAILY_EVERY_S
                or now - self._last.get((asset, "h1"), 0) > HOURLY_EVERY_S)

    def _run(self) -> None:
        while not self._stop:
            self._wake.wait(60)
            self._wake.clear()
            asset = self._want
            if not asset or not self._due(asset):
                continue
            try:
                n = self.update(asset)
                if n:
                    self.log(f"histórico do mercado: {asset} {n} velas atualizadas", flush=True)
            except Exception as e:  # nunca derruba o app
                self.log(f"histórico do mercado indisponível agora: {e!r}", flush=True)

    def _get(self, sym: str, interval: str, size: int):
        if self.fetch is not None:
            return self.fetch(sym, interval, size)
        from ..feeds.twelvedata import series

        return series(sym, interval, size, dados=self.dados)

    def update(self, asset: str, now: Optional[float] = None) -> int:
        """Busca o que venceu (diário a cada 6 h, 1 hora a cada hora). Devolve quantas velas gravou."""
        sym = symbol(asset)
        if sym is None:
            return 0
        now = time.time() if now is None else now
        n = 0
        db = self._connect()
        try:
            for tf, interval, size, every in (("d1", "1day", 5000, DAILY_EVERY_S),
                                              ("h1", "1h", HOURS, HOURLY_EVERY_S)):
                if now - self._last.get((asset, tf), 0) <= every:
                    continue
                bars = self._get(sym, interval, size)
                if bars is None:  # sem chave, limite ou rede: tenta de novo em 10 min
                    self._last[(asset, tf)] = now - every + 600
                    continue
                self._last[(asset, tf)] = now
                db.executemany("INSERT OR REPLACE INTO mercado VALUES (?,?,?,?,?,?,?)",
                               [(asset, tf, *b) for b in bars])
                db.commit()
                n += len(bars)
        finally:
            db.close()
        return n
