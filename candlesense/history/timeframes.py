"""Qual tempo de vela (1, 2, 5 min...) tem dado mais certo, pelo histórico.

Compara, para cada tempo de vela:

* os sinais do app já conferidos: acerto e resultado médio por $1 apostado
  (ganho = payout, perda = -1);
* o acerto do motor em todas as previsões (mais dados, mesmo sem sinal);
* as suas entradas registradas na carteira.

Só recomenda um tempo quando há sinais conferidos suficientes e, mesmo no
cenário pessimista (limite inferior de Wilson a 90%), o acerto passa do ponto de
equilíbrio do payout, 1 / (1 + payout). Se nenhum passa, diz isso: não inventa
recomendação.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..models import fmt_duration
from .store import Prediction

MIN_SIGNALS = 30
Z90 = 1.645


def wilson_low(hits: int, n: int, z: float = Z90) -> float:
    if n == 0:
        return 0.0
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d


@dataclass
class TimeframeStat:
    tf: int
    signals: int = 0          # sinais conferidos (ganho ou perda)
    hits: int = 0
    payout_sum: float = 0.0
    preds: int = 0            # previsões conferidas (1 vela)
    pred_hits: int = 0
    trades: int = 0           # suas entradas nesse tempo
    trade_wins: int = 0
    trade_profit: float = 0.0

    @property
    def label(self) -> str:
        return fmt_duration(self.tf)

    @property
    def rate(self) -> Optional[float]:
        return self.hits / self.signals if self.signals else None

    @property
    def payout(self) -> Optional[float]:
        return self.payout_sum / self.signals if self.signals else None

    @property
    def breakeven(self) -> Optional[float]:
        return 1 / (1 + self.payout) if self.payout else None

    @property
    def ev(self) -> Optional[float]:
        """Resultado médio por $1 nos sinais conferidos."""
        if not self.signals or self.payout is None:
            return None
        return self.rate * self.payout - (1 - self.rate)

    @property
    def low(self) -> float:
        return wilson_low(self.hits, self.signals)

    @property
    def pred_rate(self) -> Optional[float]:
        return self.pred_hits / self.preds if self.preds else None

    @property
    def proven(self) -> bool:
        be = self.breakeven
        return self.signals >= MIN_SIGNALS and be is not None and self.low > be


@dataclass
class TimeframeAdvice:
    stats: list[TimeframeStat]
    best: Optional[TimeframeStat]
    text: str


def compare(signals: list[dict], preds: list[Prediction], trades: Optional[list[dict]] = None,
            asset: Optional[str] = None) -> TimeframeAdvice:
    by: dict[int, TimeframeStat] = {}

    def get(tf) -> Optional[TimeframeStat]:
        return by.setdefault(int(tf), TimeframeStat(int(tf))) if tf else None

    for s in signals:
        if asset and s.get("asset") != asset:
            continue
        if s.get("outcome") not in (0, 1):
            continue
        st = get(s.get("tf"))
        if st is None:
            continue
        st.signals += 1
        st.hits += int(s["outcome"] == 1)
        st.payout_sum += (s.get("payout") or 0) / 100.0
    for p in preds:
        st = get(p.tf)
        if st is None or p.p.get(1) is None or p.out.get(1) not in (0, 1):
            continue
        st.preds += 1
        st.pred_hits += int((p.out[1] == 1) == (p.p[1] >= 0.5))
    for t in trades or []:
        if asset and t.get("asset") != asset:
            continue
        st = get(t.get("tf"))
        if st is None or t.get("result") is None:
            continue
        st.trades += 1
        st.trade_wins += int(t["result"] == "ganho")
        st.trade_profit += t.get("profit") or 0.0

    stats = sorted(by.values(), key=lambda s: s.tf)
    proven = [s for s in stats if s.proven]
    if proven:
        best = max(proven, key=lambda s: s.low - s.breakeven)
        text = (f"Melhor tempo de vela até agora: {best.label} "
                f"({best.hits}/{best.signals} sinais certos, {best.rate:.0%}; "
                f"o payout pede {best.breakeven:.0%})")
        return TimeframeAdvice(stats, best, text)
    if not any(s.signals for s in stats):
        text = "Ainda sem sinais conferidos para comparar os tempos de vela"
    elif all(s.signals < MIN_SIGNALS for s in stats):
        top = max(stats, key=lambda s: s.signals)
        text = (f"Ainda poucos sinais para escolher o tempo de vela "
                f"(o que tem mais: {top.label}, {top.signals} de {MIN_SIGNALS})")
    else:
        text = "Nenhum tempo de vela passou do ponto de equilíbrio com segurança: melhor esperar mais dados"
    return TimeframeAdvice(stats, None, text)
