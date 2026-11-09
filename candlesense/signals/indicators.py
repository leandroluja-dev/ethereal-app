"""Indicadores clássicos calculados sobre as velas lidas da tela.

Tudo em numpy puro, sem dependências extras. As funções recebem arrays de
preço (mais antigo primeiro) e devolvem arrays do mesmo tamanho, com NaN
onde ainda não há dados suficientes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import Candle


@dataclass
class Series:
    """Velas como arrays. ``o, h, l, c`` têm o mesmo tamanho."""

    o: np.ndarray
    h: np.ndarray
    l: np.ndarray  # noqa: E741
    c: np.ndarray

    @classmethod
    def from_candles(cls, candles: list[Candle]) -> "Series":
        a = np.array([[k.open, k.high, k.low, k.close] for k in candles], dtype=float).reshape(-1, 4)
        return cls(a[:, 0], a[:, 1], a[:, 2], a[:, 3])

    def __len__(self) -> int:
        return len(self.c)

    @property
    def body(self) -> np.ndarray:
        return self.c - self.o

    @property
    def range(self) -> np.ndarray:
        return self.h - self.l


def ema(x: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < period or period <= 0:
        return out
    alpha = 2.0 / (period + 1)
    out[period - 1] = float(np.mean(x[:period]))
    for i in range(period, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def sma(x: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < period or period <= 0:
        return out
    cs = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out[period - 1 :] = (cs[period:] - cs[:-period]) / period
    return out


def rsi(c: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI de Wilder."""
    out = np.full(len(c), np.nan)
    if len(c) <= period:
        return out
    d = np.diff(c)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = float(np.mean(up[:period]))
    ad = float(np.mean(dn[:period]))

    def val(u: float, w: float) -> float:
        if u == 0 and w == 0:
            return 50.0
        return 100.0 if w == 0 else 100.0 - 100.0 / (1.0 + u / w)

    out[period] = val(au, ad)
    for i in range(period + 1, len(c)):
        au = (au * (period - 1) + up[i - 1]) / period
        ad = (ad * (period - 1) + dn[i - 1]) / period
        out[i] = val(au, ad)
    return out


def true_range(s: Series) -> np.ndarray:
    prev = np.concatenate([[s.c[0]], s.c[:-1]]) if len(s) else s.c
    return np.maximum(s.h - s.l, np.maximum(np.abs(s.h - prev), np.abs(s.l - prev)))


def atr(s: Series, period: int = 14) -> np.ndarray:
    """ATR de Wilder; com poucas velas usa a média do que houver."""
    tr = true_range(s)
    out = np.full(len(tr), np.nan)
    if len(tr) == 0:
        return out
    p = min(period, len(tr))
    out[p - 1] = float(np.mean(tr[:p]))
    for i in range(p, len(tr)):
        out[i] = (out[i - 1] * (p - 1) + tr[i]) / p
    return out


def bollinger(c: np.ndarray, period: int = 20, k: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(c, period)
    sd = np.full(len(c), np.nan)
    for i in range(period - 1, len(c)):
        sd[i] = float(np.std(c[i - period + 1 : i + 1]))
    return mid - k * sd, mid, mid + k * sd


def swings(s: Series, k: int = 2) -> tuple[list[int], list[int]]:
    """Topos e fundos (fractais): máxima/mínima maior/menor que as k velas de cada lado."""
    highs, lows = [], []
    for i in range(k, len(s) - k):
        win_h = s.h[i - k : i + k + 1]
        win_l = s.l[i - k : i + k + 1]
        # Empate (o fechamento de uma vela é a abertura da seguinte): vale o primeiro.
        if int(np.argmax(win_h)) == k:
            highs.append(i)
        if int(np.argmin(win_l)) == k:
            lows.append(i)
    return highs, lows


@dataclass
class Level:
    price: float
    touches: int
    last_index: int
    kind: str  # "resistencia" | "suporte" | "misto"


def levels(s: Series, tol: float, k: int = 2) -> list[Level]:
    """Agrupa topos e fundos próximos (dentro de ``tol``) em níveis de preço."""
    highs, lows = swings(s, k)
    pts = [(float(s.h[i]), i, "resistencia") for i in highs] + [(float(s.l[i]), i, "suporte") for i in lows]
    pts.sort()
    groups: list[list[tuple[float, int, str]]] = []
    for p in pts:
        if groups and p[0] - groups[-1][-1][0] <= tol:
            groups[-1].append(p)
        else:
            groups.append([p])
    out = []
    for g in groups:
        kinds = {x[2] for x in g}
        out.append(Level(
            price=float(np.mean([x[0] for x in g])),
            touches=len(g),
            last_index=max(x[1] for x in g),
            kind=kinds.pop() if len(kinds) == 1 else "misto",
        ))
    return out


def last_valid(x: np.ndarray, default: float = float("nan")) -> float:
    v = x[~np.isnan(x)]
    return float(v[-1]) if v.size else default
