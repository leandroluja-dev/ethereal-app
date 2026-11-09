"""Eixo de preço: rótulos lidos por OCR viram um mapeamento linear y -> preço."""

from __future__ import annotations

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
        if p is None:
            continue
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
