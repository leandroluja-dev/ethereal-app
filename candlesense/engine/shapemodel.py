"""Modelo de formatos: o formato das últimas velas de 1, 3, 5 e 15 min -> chance de subir em 5 min.

Treinado em EUR/USD 2015-2020 (velas de 1 min da Dukascopy) e testado em anos e
pares que ele nunca viu. O arquivo ``modelo_formatos.json.gz`` guarda as árvores
do modelo; aqui elas são avaliadas em Python puro (sem lightgbm no Mac).

A conta das características repete a do treino (pesquisa/shapes.py) para a
última vela fechada de uma sequência contínua de velas de 1 min.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

MODEL_PATH = Path(__file__).with_name("modelo_formatos.json.gz")

# Velas de 1 min contínuas necessárias: 8 velas de 15 min (até 134 min para trás)
# e a média de amplitude de 120 min.
MIN_HISTORY = 135


@dataclass(frozen=True)
class Bar:
    """Vela de 1 min. ``start`` = início da vela em segundos UTC (múltiplo de 60)."""

    start: int
    open: float
    high: float
    low: float
    close: float


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def features(bars: Sequence[Bar]) -> Optional[dict[str, float]]:
    """Características da última vela de ``bars`` (contínuas, 1 min cada).

    Devolve None se faltar histórico ou se as velas não forem contínuas.
    """
    if len(bars) < MIN_HISTORY:
        return None
    bars = list(bars[-MIN_HISTORY:])
    for a, b in zip(bars, bars[1:]):
        if b.start - a.start != 60:
            return None
    rng = [b.high - b.low for b in bars]
    atr = _mean(rng[-60:])
    if atr <= 0:
        return None
    base = _mean(rng[-120:])
    t = bars[-1]
    c = t.close
    f: dict[str, float] = {}
    for j in range(1, 21):
        f[f"c1_{j}"] = (bars[-1 - j].close - c) / atr
        if j <= 10:
            f[f"h1_{j}"] = (bars[-j].high - c) / atr
            f[f"l1_{j}"] = (bars[-j].low - c) / atr
    end = t.start + 60  # fim da vela atual
    for tf, nb in ((3, 10), (5, 10), (15, 8)):
        size = tf * 60
        last_end = end - (end % size)  # fim da última vela de tf min já fechada
        for j in range(nb):
            b_end = last_end - j * size
            grp = [b for b in bars if b_end - size <= b.start < b_end]
            if len(grp) < tf:  # histórico contínuo: a vela grande tem que estar inteira
                return None
            f[f"c{tf}_{j}"] = (grp[-1].close - c) / atr
            f[f"h{tf}_{j}"] = (max(b.high for b in grp) - c) / atr
            f[f"l{tf}_{j}"] = (min(b.low for b in grp) - c) / atr
    f["min_hour"] = float((t.start // 60) % 60)
    f["hour"] = float((t.start // 3600) % 24)
    f["atr_rel"] = atr / base if base > 0 else float("nan")
    return f


class ShapeModel:
    """Soma das árvores (gradient boosting) e sigmoide, como o LightGBM faz."""

    def __init__(self, path: Path = MODEL_PATH) -> None:
        with gzip.open(path, "rt") as fh:
            data = json.load(fh)
        self.names: list[str] = data["features"]
        self.trees: list[dict] = data["trees"]

    @staticmethod
    def _leaf(tree: dict, x: list[float]) -> float:
        feat, thr, left, right = tree["f"], tree["t"], tree["l"], tree["r"]
        dleft, miss = tree["d"], tree["m"]
        node = 0
        while True:
            v = x[feat[node]]
            if math.isnan(v) or (miss[node] == 1 and v == 0.0):
                go_left = dleft[node] == 1
            else:
                go_left = v <= thr[node]
            nxt = left[node] if go_left else right[node]
            if nxt < 0:
                return tree["v"][~nxt]
            node = nxt

    def prob_up(self, feats: dict[str, float]) -> Optional[float]:
        x = [feats.get(n, float("nan")) for n in self.names]
        if any(math.isnan(v) for v in x):
            return None
        raw = sum(self._leaf(t, x) for t in self.trees)
        return 1.0 / (1.0 + math.exp(-raw))
