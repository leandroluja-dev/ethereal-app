"""Descobre sozinho as cores das velas de alta e de baixa e o tema (claro/escuro)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import VisionConfig


@dataclass(frozen=True)
class CandleColors:
    bull_bgr: tuple[int, int, int]
    bear_bgr: tuple[int, int, int]
    bull_hue: int
    bear_hue: int
    chroma_min: int
    val_min: int
    theme: str
    bg_bgr: tuple[int, int, int]
    confidence: float
    ambiguous: bool = False  # cores fora do padrão verde/vermelho: confirmar pela continuidade das velas


def hue_dist(h1: float, h2: float) -> float:
    d = abs(h1 - h2) % 180
    return min(d, 180 - d)


def estimate_background(img: np.ndarray) -> tuple[int, int, int]:
    """Cor de fundo = cor mais frequente (quantizada)."""
    small = img[::4, ::4].reshape(-1, 3)
    q = (small // 8).astype(np.int32)
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    top = vals[np.argmax(counts)]
    sel = small[keys == top]
    m = sel.mean(axis=0)
    return int(m[0]), int(m[1]), int(m[2])


@dataclass(frozen=True)
class ColorPlanes:
    """Matiz (0-179, OpenCV), croma (max-min dos canais) e brilho de cada pixel."""

    hue: np.ndarray
    chroma: np.ndarray
    val: np.ndarray

    @staticmethod
    def of(img: np.ndarray) -> ColorPlanes:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(img)
        mx = cv2.max(cv2.max(b, g), r)
        mn = cv2.min(cv2.min(b, g), r)
        return ColorPlanes(hsv[..., 0], cv2.subtract(mx, mn), mx)


def hue_distance_map(hue_plane: np.ndarray, hue: int) -> np.ndarray:
    """Distância circular de matiz de cada pixel até `hue` (via tabela, rápido)."""
    h = np.arange(256, dtype=np.int16)
    d = np.abs(h - hue) % 180
    lut = np.minimum(d, 180 - d).clip(0, 255).astype(np.uint8)
    return cv2.LUT(hue_plane, lut)


def saturated_mask(planes: ColorPlanes, chroma_min: int, val_min: int) -> np.ndarray:
    # Croma (e não a saturação HSV): azul-marinho escuro do fundo/grade tem saturação alta mas croma baixo.
    return ((planes.chroma >= chroma_min) & (planes.val >= val_min)).astype(np.uint8)


def hue_mask(planes: ColorPlanes, hue: int, tol: int, chroma_min: int, val_min: int) -> np.ndarray:
    return ((hue_distance_map(planes.hue, hue) <= tol) & (planes.chroma >= chroma_min)
            & (planes.val >= val_min)).astype(np.uint8)


def _bar_score(mask: np.ndarray, max_w: int) -> tuple[int, np.ndarray]:
    """Conta componentes com formato de vela (barras verticais estreitas) e devolve seus pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 3 or w > max_w:
            continue
        if h <= 2 and w >= 12:  # linha horizontal
            continue
        if area < 0.25 * w * h and w > 6:  # linha diagonal / curva (indicador)
            continue
        keep[i] = True
    return int(keep.sum()), keep[labels]


def hue_peaks(hues: np.ndarray, max_peaks: int = 5, min_sep: int = 12) -> list[int]:
    hist = np.bincount(hues.astype(np.int64), minlength=180).astype(np.float64)
    k = np.array([1, 2, 3, 2, 1], dtype=np.float64)
    smooth = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), k / k.sum(), mode="valid")
    peaks: list[int] = []
    work = smooth.copy()
    top = float(work.max())
    for _ in range(max_peaks):
        p = int(np.argmax(work))
        if work[p] <= 0.01 * top:
            break
        peaks.append(p)
        for h in range(180):
            if hue_dist(h, p) <= min_sep:
                work[h] = 0
    return peaks


def discover_colors(plot_img: np.ndarray, cfg: VisionConfig) -> CandleColors | None:
    """Escolhe as duas cores que mais formam barras verticais (velas) na área do gráfico.

    Linhas de indicadores, marcadores e textos também são coloridos, mas não formam
    dezenas de barras verticais regulares, então perdem na pontuação.
    """
    if plot_img.size == 0:
        return None
    bg = estimate_background(plot_img)
    theme = "dark" if (0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2]) < 128 else "light"
    planes = ColorPlanes.of(plot_img)
    sat = saturated_mask(planes, cfg.chroma_min, cfg.val_min).astype(bool)
    hues = planes.hue[sat]
    if hues.size < 30:
        return None
    max_w = max(40, plot_img.shape[1] // 25)
    scored: list[tuple[int, int, np.ndarray]] = []
    for p in hue_peaks(hues):
        m = hue_mask(planes, p, 8, cfg.chroma_min, cfg.val_min)
        count, pixels = _bar_score(m, max_w)
        if count > 0:
            scored.append((count, p, pixels))
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])

    def median_bgr(pixels: np.ndarray) -> tuple[int, int, int]:
        sel = plot_img[pixels]
        if sel.size == 0:
            return (0, 0, 0)
        m = np.median(sel, axis=0)
        return int(m[0]), int(m[1]), int(m[2])

    c1, p1, px1 = scored[0]
    if len(scored) < 2 or scored[1][0] < max(2, 0.05 * c1):
        # Só uma cor de vela visível (ex.: todas de alta). Atribui pela cor.
        is_bull = hue_dist(p1, 60) < hue_dist(p1, 0)
        c = median_bgr(px1)
        other = (p1 + 90) % 180
        return CandleColors(
            bull_bgr=c if is_bull else (0, 0, 0), bear_bgr=(0, 0, 0) if is_bull else c,
            bull_hue=p1 if is_bull else other, bear_hue=other if is_bull else p1,
            chroma_min=cfg.chroma_min, val_min=cfg.val_min, theme=theme, bg_bgr=bg, confidence=0.4,
        )
    c2, p2, px2 = scored[1]
    # Verde = alta, vermelho = baixa (padrão da IQ Option). A mais "verde" vira alta.
    if hue_dist(p1, 60) <= hue_dist(p2, 60):
        (bull, bpx), (bear, rpx) = (p1, px1), (p2, px2)
    else:
        (bull, bpx), (bear, rpx) = (p2, px2), (p1, px1)
    canonical = hue_dist(bull, 60) <= 30 and hue_dist(bear, 0) <= 20
    balance = min(c1, c2) / max(c1, c2)
    conf = 0.9 if canonical else 0.5
    conf *= 0.7 + 0.3 * min(1.0, balance * 3)
    return CandleColors(
        bull_bgr=median_bgr(bpx), bear_bgr=median_bgr(rpx), bull_hue=bull, bear_hue=bear,
        chroma_min=cfg.chroma_min, val_min=cfg.val_min, theme=theme, bg_bgr=bg, confidence=float(conf),
        ambiguous=not canonical,
    )


def swap(colors: CandleColors) -> CandleColors:
    return CandleColors(colors.bear_bgr, colors.bull_bgr, colors.bear_hue, colors.bull_hue, colors.chroma_min,
                        colors.val_min, colors.theme, colors.bg_bgr, colors.confidence, colors.ambiguous)
