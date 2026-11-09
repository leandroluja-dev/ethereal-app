"""Eixo de preço: rótulos lidos por OCR viram um mapeamento linear y -> preço."""

from __future__ import annotations

import math

from itertools import combinations
from typing import Optional

import numpy as np

from ..models import OcrToken, PriceScale
from .ocr import parse_number


def axis_labels(tokens: list[OcrToken]) -> list[tuple[float, float, int]]:
    """(y do centro, preço, casas decimais) dos tokens que são números."""
    out = []
    for t in tokens:
        p = parse_number(t.text)
        if p is None or p[1] == 0:
            continue  # preço sempre tem casas decimais; inteiros são de indicadores (RSI 70, 30...)
        out.append((t.box.cy, p[0], p[1]))
    return out


def fit_scale(
    labels: list[tuple[float, float, int]],
    tol_px: float = 3.0,
    min_inliers: int = 3,
) -> Optional[PriceScale]:
    """Ajuste robusto (todas as duplas, conta quem concorda) e refino por mínimos quadrados.

    Preço cresce para cima na tela, então a inclinação precisa ser negativa.
    Rótulos lidos errado pelo OCR viram outliers e são descartados.
    """
    if len(labels) < 2:
        return None
    ys = np.array([l[0] for l in labels], dtype=float)
    ps = np.array([l[1] for l in labels], dtype=float)
    best: tuple[int, float, np.ndarray] | None = None
    for i, j in combinations(range(len(labels)), 2):
        dy = ys[j] - ys[i]
        dp = ps[j] - ps[i]
        if abs(dy) < 4 or dp == 0:
            continue
        slope = dp / dy
        if slope >= 0:
            continue
        icpt = ps[i] - slope * ys[i]
        res_px = np.abs((ps - icpt) / slope - ys)
        inl = res_px <= tol_px
        n = int(inl.sum())
        err = float(res_px[inl].mean())
        if best is None or n > best[0] or (n == best[0] and err < best[1]):
            best = (n, err, inl)
    if best is None or best[0] < min_inliers:
        return None
    inl = best[2]
    slope, icpt = np.polyfit(ys[inl], ps[inl], 1)
    if slope >= 0:
        return None
    res = np.abs((ps[inl] - icpt) / slope - ys[inl])
    decimals = max(l[2] for l, ok in zip(labels, inl) if ok)
    return PriceScale(
        slope=float(slope),
        intercept=float(icpt),
        inliers=int(inl.sum()),
        residual_px=float(res.max()),
        decimals=int(decimals),
        labels=[(float(y), float(p)) for y, p, ok in zip(ys, ps, inl) if ok],
    )


def grid_profile(bgr: np.ndarray, x1: int, x2: int, y1: int, y2: int) -> np.ndarray:
    """Brilho mediano de cada linha da área do gráfico (as linhas da grade viram picos).

    Devolve um vetor do tamanho da altura da imagem (fora de y1..y2 fica zero).
    """
    h = bgr.shape[0]
    prof = np.zeros(h, dtype=float)
    x1, x2 = max(0, x1), min(bgr.shape[1], x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return prof
    band = bgr[y1:y2, x1:x2:3]
    gray = band.mean(axis=2)
    prof[y1:y2] = np.median(gray, axis=1)
    return prof


def snap_to_grid(labels: list[tuple[float, float, int]], prof: np.ndarray,
                 label_h: float, min_contrast: float = 3.0) -> list[tuple[float, float, int]]:
    """Move o y de cada rótulo para a linha de grade mais próxima, se houver uma clara.

    A caixa do OCR fica ~1 px fora do centro do texto; a linha da grade é a
    posição exata do preço.
    """
    if not prof.any():
        return labels
    win = max(4, int(round(0.7 * label_h)))
    out = []
    for y, p, d in labels:
        c = int(round(y))
        lo, hi = max(0, c - win), min(len(prof), c + win + 1)
        seg = prof[lo:hi]
        if seg.size < 3:
            out.append((y, p, d))
            continue
        base = float(np.median(seg))
        peak = seg - base >= min_contrast
        if not peak.any():
            out.append((y, p, d))
            continue
        rows = np.nonzero(peak)[0]
        # grupos contíguos; escolhe o mais perto do rótulo
        groups = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
        best = min(groups, key=lambda g: abs(lo + g.mean() + 0.5 - y))
        out.append((lo + float(best.mean()) + 0.5, p, d))
    return out


def grid_lines(bgr: np.ndarray, x1: int, x2: int, y1: int, y2: int, min_frac: float = 0.55) -> list[float]:
    """Centros das linhas horizontais da grade entre y1 e y2.

    Uma linha da grade é contínua de ponta a ponta; as fileiras de pontinhos do
    mapa-múndi ao fundo, não. Conta em quantas colunas a linha é mais clara que
    as vizinhas (3 px acima e abaixo).
    """
    h = bgr.shape[0]
    x1, x2 = max(0, x1), min(bgr.shape[1], x2)
    y1, y2 = max(3, y1), min(h - 3, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return []
    g = bgr[y1 - 3:y2 + 3, x1:x2].mean(axis=2)
    mid = g[3:-3]
    nb = np.maximum(g[:-6], g[6:])
    frac = ((mid - nb) > 1.5).mean(axis=1)
    rows = np.nonzero(frac >= min_frac)[0]
    if rows.size == 0:
        return []
    groups = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
    return [y1 + float(g_.mean()) + 0.5 for g_ in groups if len(g_) <= 6]


def nice_step(x: float) -> Optional[float]:
    """Arredonda para o passo 'redondo' mais próximo (1, 2, 2.5, 5 x 10^k) se estiver a até 8%."""
    if x <= 0:
        return None
    k = math.floor(math.log10(x))
    best = None
    for m in (1, 2, 2.5, 5, 10):
        v = m * 10 ** k
        err = abs(x - v) / v
        if best is None or err < best[0]:
            best = (err, v)
    return best[1] if best and best[0] <= 0.08 else None


def scale_from_grid(labels: list[tuple[float, float, int]], lines: list[float],
                    price_label: tuple[float, float, int]) -> Optional[PriceScale]:
    """Escala com só 1 rótulo legível (a etiqueta do preço pode cobrir o outro).

    A etiqueta do preço dá a inclinação aproximada. Para cada passo redondo de
    preço (0.001, 0.0025, 0.005...), prevê onde estaria a próxima linha da
    grade e procura uma linha ali. Achou: a inclinação exata sai do rótulo e
    dessa linha. Quem chama ainda confere a etiqueta do preço.
    """
    if not labels or not lines:
        return None
    y_l, p_l, dec_l = labels[0]
    y_p, p_p, _ = price_label
    if abs(y_p - y_l) < 20 or p_p == p_l:
        return None
    slope_est = (p_p - p_l) / (y_p - y_l)
    if slope_est >= 0:
        return None
    best = None
    k0 = math.floor(math.log10(abs(slope_est) * 20))
    for k in range(k0, k0 + 4):
        for m in (1, 2, 2.5, 5):
            step = m * 10 ** k
            for mult in (1, -1, 2, -2):
                y_exp = y_l + mult * step / abs(slope_est)
                near = [ln for ln in lines if abs(ln - y_exp) <= 0.08 * step / abs(slope_est) and abs(ln - y_l) > 8]
                for ln in near:
                    slope = -mult * step / (ln - y_l)
                    if slope >= 0:
                        continue
                    err = abs((p_p - p_l) / slope + y_l - y_p)
                    if best is None or err < best[0]:
                        best = (err, slope, step)
    if best is None:
        return None
    _, slope, step = best
    icpt = p_l - slope * y_l
    dec_step = len(f"{step:.10f}".rstrip("0").split(".")[1]) if step < 1 else 0
    return PriceScale(slope=slope, intercept=icpt, inliers=1, residual_px=0.0,
                      decimals=max(dec_l, dec_step), labels=[(y_l, p_l)])
