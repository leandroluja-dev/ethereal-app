"""Placar: quanto o motor acerta de verdade, e a correção que ele aprende com isso.

* Acerto por expiração, por fase da vela (início/meio/fim) e por faixa de
  probabilidade (o motor disse 55%: acertou perto de 55%?).
* Sinais (COMPRA/VENDA que acenderam): acertos e resultado simulado com o payout.
* Calibração: com previsões suficientes, aprende uma correção da probabilidade
  (regressão logística sobre a probabilidade bruta). Só passa a valer se for
  melhor que a bruta nas previsões mais recentes, que ela não viu ao aprender.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .store import Prediction

HORIZONS = (1, 2, 3, 5)
BINS = (0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 1.01)


@dataclass
class HorizonScore:
    candles: int
    n: int = 0
    hits: int = 0
    by_phase: dict[str, tuple[int, int]] = field(default_factory=dict)  # fase -> (n, acertos)
    by_bin: list[tuple[float, float, int, int, float]] = field(default_factory=list)  # (de, até, n, acertos, prob média)
    signals: int = 0
    signal_hits: int = 0
    signal_profit: float = 0.0  # em unidades de entrada (ganho = payout, perda = -1)

    @property
    def rate(self) -> Optional[float]:
        return self.hits / self.n if self.n else None

    @property
    def signal_rate(self) -> Optional[float]:
        return self.signal_hits / self.signals if self.signals else None


def _usable(p: Prediction, h: int) -> bool:
    return p.p.get(h) is not None and p.out.get(h) in (0, 1)


def score(preds: list[Prediction]) -> dict[int, HorizonScore]:
    out = {}
    for h in HORIZONS:
        s = HorizonScore(h)
        bins = [[lo, hi, 0, 0, 0.0] for lo, hi in zip(BINS, BINS[1:])]
        for p in preds:
            if not _usable(p, h):
                continue
            pu = p.p[h]
            up = pu >= 0.5
            hit = int((p.out[h] == 1) == up)
            s.n += 1
            s.hits += hit
            n0, h0 = s.by_phase.get(p.phase, (0, 0))
            s.by_phase[p.phase] = (n0 + 1, h0 + hit)
            conf = max(pu, 1 - pu)
            for b in bins:
                if b[0] <= conf < b[1]:
                    b[2] += 1
                    b[3] += hit
                    b[4] += conf
                    break
            if p.action in ("COMPRA", "VENDA") and p.best_h == h:
                s.signals += 1
                win = (p.out[h] == 1) == (p.action == "COMPRA")
                s.signal_hits += int(win)
                s.signal_profit += ((p.payout or 0) / 100.0) if win else -1.0
        s.by_bin = [(b[0], b[1], b[2], b[3], (b[4] / b[2]) if b[2] else 0.0) for b in bins]
        out[h] = s
    return out


# ------------------------------------------------------------- calibração

def _logit(p: float) -> float:
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logloss(pairs: list[tuple[float, int]], a: float, b: float) -> float:
    tot = 0.0
    for x, y in pairs:
        q = min(1 - 1e-9, max(1e-9, _sig(a * x + b)))
        tot -= y * math.log(q) + (1 - y) * math.log(1 - q)
    return tot / max(1, len(pairs))


def _fit(pairs: list[tuple[float, int]], prior: float = 20.0) -> tuple[float, float]:
    """Regressão logística y ~ sig(a*logit(p) + b), puxada para a=1, b=0 (sem correção)."""
    a, b = 1.0, 0.0
    for _ in range(25):
        ga = gb = 0.0
        haa = hab = hbb = 0.0
        for x, y in pairs:
            q = _sig(a * x + b)
            e = q - y
            w = q * (1 - q)
            ga += e * x
            gb += e
            haa += w * x * x
            hab += w * x
            hbb += w
        # penalidade: prior pontos imaginários dizendo "a bruta está certa"
        ga += prior * (a - 1.0) * 0.01
        gb += prior * b
        haa += prior * 0.01
        hbb += prior
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) < 1e-7 and abs(db) < 1e-7:
            break
    return a, b


@dataclass
class Calibration:
    """Correção por expiração: p_corrigida = sig(a * logit(p_bruta) + b)."""

    params: dict[int, tuple[float, float]] = field(default_factory=dict)
    n: dict[int, int] = field(default_factory=dict)
    note: dict[int, str] = field(default_factory=dict)
    cap: float = 0.70  # mesmo com dados, não afirma mais que isso

    def apply(self, h: int, p: float) -> float:
        if h not in self.params:
            return p
        a, b = self.params[h]
        q = _sig(a * _logit(p) + b)
        return min(self.cap, max(1 - self.cap, q))

    @property
    def active(self) -> bool:
        return bool(self.params)


def learn(preds: list[Prediction], min_n: int = 200, holdout: float = 0.3) -> Calibration:
    """Aprende a correção com as previsões antigas e testa nas mais recentes."""
    cal = Calibration()
    for h in HORIZONS:
        pairs = [(_logit(p.p[h]), int(p.out[h] == 1)) for p in preds if _usable(p, h)]
        cal.n[h] = len(pairs)
        if len(pairs) < min_n:
            cal.note[h] = f"faltam {min_n - len(pairs)} previsões conferidas"
            continue
        cut = int(len(pairs) * (1 - holdout))
        train, test = pairs[:cut], pairs[cut:]
        a, b = _fit(train)
        if _logloss(test, a, b) < _logloss(test, 1.0, 0.0):
            cal.params[h] = _fit(pairs)  # validou: aprende de novo com tudo
            cal.note[h] = "correção ativa (melhor que a bruta nas previsões recentes)"
        else:
            cal.note[h] = "correção não melhorou as previsões recentes: usando a bruta"
    return cal
