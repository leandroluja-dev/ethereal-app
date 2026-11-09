"""Módulos de sinal. Cada um olha uma parte do gráfico e devolve uma nota.

Nota de -1 (pressão de venda) a +1 (pressão de compra), uma confiança de 0 a 1
(quanto o módulo tem de dados para opinar) e uma frase curta em português.
Nenhum módulo decide nada sozinho: o motor (engine/rules.py) junta as notas.

Todas as distâncias de preço são medidas em ATR (o tamanho médio das velas),
para valer igual em qualquer ativo e em qualquer escala do gráfico.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..models import Candle, ChartReading
from . import indicators as ind


@dataclass
class ModuleResult:
    key: str
    name: str
    score: float  # -1 venda ... +1 compra
    confidence: float  # 0 ... 1
    explanation: str
    details: dict = field(default_factory=dict)

    @property
    def weighted(self) -> float:
        return self.score * self.confidence


@dataclass
class MarketContext:
    """O que os módulos precisam, tirado de uma leitura OK."""

    closed: ind.Series  # velas fechadas (sem a atual)
    forming: Optional[Candle]  # vela atual, ainda formando
    price: float  # preço atual (etiqueta do eixo)
    atr: float
    elapsed_frac: Optional[float]  # quanto da vela atual já passou (0 = abriu agora)
    decimals: int = 5

    @property
    def closes_now(self) -> np.ndarray:
        """Fechamentos + preço atual (a vela atual entra com o preço de agora)."""
        return np.append(self.closed.c, self.price)

    @classmethod
    def from_reading(cls, r: ChartReading) -> Optional["MarketContext"]:
        if not r.candles or len(r.candles) < 3:
            return None
        forming = r.candles[-1] if r.candles[-1].forming else None
        closed = r.candles[:-1] if forming else r.candles
        s = ind.Series.from_candles(closed)
        price = r.current_price if r.current_price is not None else closed[-1].close
        a = ind.last_valid(ind.atr(s, 14), 0.0)
        if not a or a <= 0:
            a = float(np.median(s.range)) or 1e-9
        frac = None
        if r.timer_seconds is not None and r.timeframe_s:
            frac = float(np.clip(1.0 - r.timer_seconds / r.timeframe_s, 0.0, 1.0))
        dec = r.scale.decimals if r.scale else 5
        return cls(closed=s, forming=forming, price=float(price), atr=a, elapsed_frac=frac, decimals=dec)

    def fmt(self, p: float) -> str:
        return f"{p:.{self.decimals}f}"


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, x)))


def _data_conf(n: int, need: int) -> float:
    return _clip(n / need, 0.0, 1.0)


def _br(x: float, nd: int = 1) -> str:
    """Número com vírgula decimal."""
    return f"{x:.{nd}f}".replace(".", ",")


# ------------------------------------------------------------------ tendência

def trend(ctx: MarketContext) -> ModuleResult:
    c = ctx.closes_now
    n = len(c)
    fast, slow = ind.ema(c, 9), ind.ema(c, 21)
    if n < 12 or np.isnan(fast[-4]):
        return ModuleResult("tendencia", "Tendência", 0.0, 0.0, "poucas velas para medir a tendência")
    slope = (fast[-1] - fast[-4]) / (3 * ctx.atr)
    s_slow = ind.last_valid(slow)
    gap = (fast[-1] - s_slow) / ctx.atr if not math.isnan(s_slow) else 0.0

    # Estrutura: últimos dois topos e dois fundos das velas fechadas.
    hi, lo = ind.swings(ctx.closed, 2)
    struct = 0.0
    if len(hi) >= 2 and len(lo) >= 2:
        hh = ctx.closed.h[hi[-1]] > ctx.closed.h[hi[-2]]
        hl = ctx.closed.l[lo[-1]] > ctx.closed.l[lo[-2]]
        if hh and hl:
            struct = 1.0
        elif not hh and not hl:
            struct = -1.0
    score = _clip(0.4 * math.tanh(1.5 * slope) + 0.3 * math.tanh(gap) + 0.3 * struct)

    parts = []
    parts.append("média rápida subindo" if slope > 0.05 else "média rápida caindo" if slope < -0.05 else "média rápida de lado")
    if not math.isnan(s_slow):
        parts.append("acima da lenta" if gap > 0 else "abaixo da lenta")
    if struct > 0:
        parts.append("topos e fundos subindo")
    elif struct < 0:
        parts.append("topos e fundos descendo")
    head = "alta" if score > 0.25 else "baixa" if score < -0.25 else "sem tendência clara"
    return ModuleResult("tendencia", "Tendência", score, _data_conf(n, 30),
                        f"{head}: " + ", ".join(parts),
                        {"inclinacao_atr": slope, "distancia_medias_atr": gap, "estrutura": struct})


# -------------------------------------------------------- suporte/resistência

def support_resistance(ctx: MarketContext) -> ModuleResult:
    s = ctx.closed
    if len(s) < 10:
        return ModuleResult("sr", "Suporte e resistência", 0.0, 0.0, "poucas velas para achar níveis")
    lv = [x for x in ind.levels(s, tol=0.3 * ctx.atr, k=2) if x.touches >= 2]
    p = ctx.price
    score = 0.0
    notes = []

    # Rompimento: a última vela fechada passou um nível com corpo.
    if len(s) >= 2:
        for x in lv:
            if s.c[-2] <= x.price < s.c[-1] and s.c[-1] - x.price > 0.15 * ctx.atr:
                score += 0.5
                notes.append(f"rompeu a resistência {ctx.fmt(x.price)}")
                break
            if s.c[-2] >= x.price > s.c[-1] and x.price - s.c[-1] > 0.15 * ctx.atr:
                score -= 0.5
                notes.append(f"perdeu o suporte {ctx.fmt(x.price)}")
                break

    above = [x for x in lv if x.price > p]
    below = [x for x in lv if x.price < p]
    near_r = min(above, key=lambda x: x.price - p) if above else None
    near_s = max(below, key=lambda x: x.price) if below else None
    if near_r is not None:
        d = (near_r.price - p) / ctx.atr
        if d < 1.0 and not notes:
            score -= (1.0 - d) * min(1.0, near_r.touches / 3)
            notes.append(f"resistência perto ({ctx.fmt(near_r.price)}, {near_r.touches} toques, {_br(d)} ATR acima)")
    if near_s is not None:
        d = (p - near_s.price) / ctx.atr
        if d < 1.0 and not notes:
            score += (1.0 - d) * min(1.0, near_s.touches / 3)
            notes.append(f"suporte perto ({ctx.fmt(near_s.price)}, {near_s.touches} toques, {_br(d)} ATR abaixo)")
    opinion = bool(notes)
    if not notes:
        notes.append("preço longe de suporte e resistência")
    # Sem nível por perto o módulo não tem opinião: pesa pouco na média.
    conf = _data_conf(len(s), 30) * (1.0 if lv else 0.5) * (1.0 if opinion else 0.3)
    return ModuleResult("sr", "Suporte e resistência", _clip(score), conf, "; ".join(notes),
                        {"resistencia": near_r.price if near_r else None,
                         "suporte": near_s.price if near_s else None})


# ------------------------------------------------------------- price action

def price_action(ctx: MarketContext) -> ModuleResult:
    s = ctx.closed
    if len(s) < 4:
        return ModuleResult("price_action", "Price action", 0.0, 0.0, "poucas velas fechadas")
    a = ctx.atr
    o, h, l, c = s.o[-1], s.h[-1], s.l[-1], s.c[-1]
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    up_w = h - max(o, c)
    lo_w = min(o, c) - l
    found: list[tuple[float, str]] = []

    po, pc = s.o[-2], s.c[-2]
    if c > o and pc < po and c >= po and o <= pc and body > 0.3 * a:
        found.append((0.7, "engolfo de alta"))
    if c < o and pc > po and c <= po and o >= pc and body > 0.3 * a:
        found.append((-0.7, "engolfo de baixa"))

    at_low = l <= s.l[-5:].min()
    at_high = h >= s.h[-5:].max()
    if rng > 0.6 * a and lo_w >= 2 * body and up_w <= max(body, 0.15 * rng):
        found.append((0.5 * (1.3 if at_low else 1.0), "martelo (rejeitou a mínima)" + (" no fundo" if at_low else "")))
    if rng > 0.6 * a and up_w >= 2 * body and lo_w <= max(body, 0.15 * rng):
        found.append((-0.5 * (1.3 if at_high else 1.0), "estrela cadente (rejeitou a máxima)" + (" no topo" if at_high else "")))

    if body >= 0.8 * rng and rng >= 1.0 * a:
        found.append((0.4 if c > o else -0.4, "vela de força " + ("verde" if c > o else "vermelha")))

    last3 = s.body[-3:]
    if (last3 > 0).all() and s.c[-1] > s.c[-2] > s.c[-3]:
        found.append((0.35, "três velas verdes seguidas"))
    if (last3 < 0).all() and s.c[-1] < s.c[-2] < s.c[-3]:
        found.append((-0.35, "três velas vermelhas seguidas"))

    if not found:
        if body <= 0.1 * rng:
            return ModuleResult("price_action", "Price action", 0.0, 0.4, "última vela foi um doji (indecisão)")
        return ModuleResult("price_action", "Price action", 0.0, 0.3, "nenhum padrão claro nas últimas velas")
    best = max(found, key=lambda x: abs(x[0]))
    agree = [f for f in found if f[0] * best[0] > 0]
    score = _clip(best[0] + 0.15 * (len(agree) - 1) * (1 if best[0] > 0 else -1))
    conflict = any(f[0] * best[0] < 0 for f in found)
    conf = 0.5 if conflict else 0.9
    return ModuleResult("price_action", "Price action", score, conf, ", ".join(f[1] for f in found),
                        {"padroes": [f[1] for f in found]})


# ----------------------------------------------------------------- momentum

def momentum(ctx: MarketContext) -> ModuleResult:
    c = ctx.closes_now
    if len(c) < 16:
        return ModuleResult("momentum", "Momento", 0.0, 0.0, "poucas velas para o RSI")
    r = ind.last_valid(ind.rsi(c, 14), 50.0)
    roc = (c[-1] - c[-4]) / (ctx.atr * math.sqrt(3))
    score = 0.6 * math.tanh(roc) + 0.4 * _clip((r - 50) / 25)
    note = ""
    if r > 75:
        score -= 0.8 * (r - 75) / 25 + 0.3
        note = " (sobrecomprado: risco de virar)"
    elif r < 25:
        score += 0.8 * (25 - r) / 25 + 0.3
        note = " (sobrevendido: risco de virar)"
    move = "subindo" if roc > 0.2 else "caindo" if roc < -0.2 else "parado"
    return ModuleResult("momentum", "Momento", _clip(score), _data_conf(len(c), 30),
                        f"RSI {r:.0f}, preço {move} nas últimas 3 velas ({_br(roc * math.sqrt(3))} ATR){note}",
                        {"rsi": r, "roc_atr": roc})


# -------------------------------------------------------------- volatilidade

def volatility(ctx: MarketContext) -> ModuleResult:
    """Pouco diz sobre direção; diz o quanto confiar nos outros módulos.

    ``details['fator']`` multiplica a vantagem final: mercado parado demais ou
    agitado demais deixa os padrões menos confiáveis.
    """
    s = ctx.closed
    if len(s) < 15:
        return ModuleResult("volatilidade", "Volatilidade", 0.0, 0.0, "poucas velas", {"fator": 0.7})
    tr = ind.true_range(s)
    ratio = float(np.mean(tr[-5:]) / max(np.median(tr), 1e-12))
    lo, mid, up = ind.bollinger(ctx.closes_now, 20, 2.0)
    score = 0.0
    band = ""
    if not np.isnan(up[-1]) and up[-1] > lo[-1]:
        pb = (ctx.price - lo[-1]) / (up[-1] - lo[-1])
        if pb > 1.0:
            score = -0.4 * min(1.0, (pb - 1.0) * 4 + 0.5)
            band = ", preço acima da banda de Bollinger"
        elif pb < 0.0:
            score = 0.4 * min(1.0, -pb * 4 + 0.5)
            band = ", preço abaixo da banda de Bollinger"
    if ratio < 0.5:
        fator, txt = 0.6, "mercado parado"
    elif ratio > 2.5:
        fator, txt = 0.5, "mercado muito agitado"
    elif ratio > 1.8:
        fator, txt = 0.75, "mercado agitado"
    else:
        fator, txt = 1.0, "volatilidade normal"
    conf = _data_conf(len(s), 30) * (1.0 if score else 0.3)
    return ModuleResult("volatilidade", "Volatilidade", _clip(score), conf,
                        f"{txt} (velas recentes {_br(ratio)}x o normal){band}",
                        {"fator": fator, "razao": ratio})


# --------------------------------------------------- microestrutura da vela

def microstructure(ctx: MarketContext) -> ModuleResult:
    f = ctx.forming
    if f is None:
        return ModuleResult("micro", "Vela atual", 0.0, 0.0, "vela atual não identificada")
    hi, lo = max(f.high, ctx.price), min(f.low, ctx.price)
    rng = hi - lo
    body = ctx.price - f.open
    pos = (ctx.price - lo) / rng if rng > 0 else 0.5
    up_w = hi - max(f.open, ctx.price)
    lo_w = min(f.open, ctx.price) - lo
    score = 0.6 * math.tanh(1.5 * body / ctx.atr) + 0.4 * (2 * pos - 1)
    notes = [("verde" if body > 0 else "vermelha" if body < 0 else "sem corpo")
             + f" ({_br(body / ctx.atr)} ATR)"]
    if rng > 0.3 * ctx.atr:
        if up_w > 2 * abs(body) and up_w > 0.3 * rng:
            score -= 0.3
            notes.append("pavio de cima grande (vendedores rejeitaram a alta)")
        elif lo_w > 2 * abs(body) and lo_w > 0.3 * rng:
            score += 0.3
            notes.append("pavio de baixo grande (compradores rejeitaram a queda)")
        else:
            notes.append("preço perto da máxima" if pos > 0.75 else "preço perto da mínima" if pos < 0.25 else "preço no meio da vela")
    frac = ctx.elapsed_frac
    # No começo da vela quase não há informação; ela cresce com o tempo passado.
    conf = 0.3 if frac is None else _clip(0.15 + 0.85 * frac, 0.0, 1.0)
    when = "" if frac is None else f", {int(round(frac * 100))}% da vela já passou"
    return ModuleResult("micro", "Vela atual", _clip(score), conf, "vela " + ", ".join(notes) + when,
                        {"posicao": pos, "corpo_atr": body / ctx.atr})


MODULES: list[Callable[[MarketContext], ModuleResult]] = [
    trend, support_resistance, price_action, momentum, volatility, microstructure,
]


def run_all(ctx: MarketContext) -> list[ModuleResult]:
    return [m(ctx) for m in MODULES]
