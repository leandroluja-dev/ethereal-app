"""Detecção de velas por segmentação de cor: corpo e pavio de cada vela, em pixels."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import VisionConfig
from .colors import CandleColors, ColorPlanes, hue_distance_map, hue_mask


@dataclass(frozen=True)
class PixelCandle:
    """Vela em coordenadas da janela. y cresce para baixo."""

    x_left: int
    x_right: int
    body_top: int
    body_bottom: int
    wick_top: int
    wick_bottom: int
    bull: bool
    slot: int = 0  # 0 = vela mais à direita; -1 a anterior, etc.
    partial: bool = False  # vela cortada pela borda da área do gráfico
    occluded: bool = False  # outra cor (indicador, marcador) cobre parte da vela: valores menos confiáveis

    @property
    def cx(self) -> float:
        return (self.x_left + self.x_right) / 2.0

    @property
    def width(self) -> int:
        return self.x_right - self.x_left + 1

    @property
    def area(self) -> int:
        return self.width * (self.body_bottom - self.body_top + 1)

    # Lados do corpo que correspondem à abertura e ao fechamento.
    @property
    def open_y(self) -> int:
        return self.body_bottom if self.bull else self.body_top

    @property
    def close_y(self) -> int:
        return self.body_top if self.bull else self.body_bottom


@dataclass
class CandleDetection:
    candles: list[PixelCandle] = field(default_factory=list)
    pitch: float = 0.0
    body_width: float = 0.0
    regularity: float = 0.0
    rejected: int = 0
    too_small: bool = False
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def forming(self) -> PixelCandle | None:
        return self.candles[-1] if self.candles else None

    @property
    def closed(self) -> list[PixelCandle]:
        return self.candles[:-1]


def remove_thin_horizontal_lines(mask: np.ndarray, min_len: int = 40) -> np.ndarray:
    """Apaga linhas horizontais finas (linha de preço, linha de operação) sem tocar em corpos/pavios."""
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1)))
    if not horiz.any():
        return mask
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    line = (horiz > 0) & (vertical == 0)
    out = mask.copy()
    out[line] = 0
    return out


@dataclass
class _Cand:
    x_left: int
    x_right: int
    body_top: int
    body_bottom: int
    wick_top: int
    wick_bottom: int
    bull: bool
    rect_score: float
    comp_x0: int = 0
    comp_x1: int = 0

    @property
    def width(self) -> int:
        return self.x_right - self.x_left + 1

    @property
    def cx(self) -> float:
        return (self.x_left + self.x_right) / 2.0


def _components(mask: np.ndarray, bull: bool) -> list[_Cand]:
    out: list[_Cand] = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 2:
            continue
        sub = labels[y : y + h, x : x + w] == i
        row_w = sub.sum(axis=1)
        maxw = int(row_w.max())
        thr = max(1, int(np.ceil(0.6 * maxw)))
        rows = np.flatnonzero(row_w >= thr)
        # trechos de linhas "largas" separados por falhas pequenas (linha sobreposta) formam o corpo
        splits = np.flatnonzero(np.diff(rows) > 4)
        runs = np.split(rows, splits + 1)
        best = max(runs, key=len)
        run = best[np.isin(best, rows)]
        lefts = np.array([np.argmax(sub[r]) for r in run])
        rights = np.array([w - 1 - np.argmax(sub[r][::-1]) for r in run])
        xl = int(np.median(lefts))
        xr = int(np.median(rights))
        full = float(np.mean(row_w[run] >= 0.9 * maxw))
        out.append(_Cand(x + xl, x + xr, y + int(run[0]), y + int(run[-1]), y, y + h - 1, bull, full, x, x + w - 1))
    return out


def _extend_wick(raw: np.ndarray, c: _Cand, max_gap: int = 3) -> tuple[int, int]:
    """Segue o pavio na máscara original, atravessando pequenas falhas (linhas sobrepostas)."""
    cx = int(round(c.cx))
    x0, x1 = max(0, cx - 1), min(raw.shape[1], cx + 2)
    col = raw[:, x0:x1].any(axis=1)
    top = min(c.wick_top, c.body_top)
    gap = 0
    y = top - 1
    while y >= 0:
        if col[y]:
            top = y
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
        y -= 1
    bottom = max(c.wick_bottom, c.body_bottom)
    gap = 0
    y = bottom + 1
    while y < raw.shape[0]:
        if col[y]:
            bottom = y
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
        y += 1
    return top, bottom


def detect_candles(
    plot_img: np.ndarray,
    colors: CandleColors,
    cfg: VisionConfig,
    offset: tuple[int, int] = (0, 0),
    valid_mask: np.ndarray | None = None,
) -> CandleDetection:
    det = CandleDetection()
    if plot_img.size == 0:
        det.notes.append("área do gráfico vazia")
        return det
    planes = ColorPlanes.of(plot_img)
    raw_masks = {
        True: hue_mask(planes, colors.bull_hue, cfg.hue_tolerance, colors.chroma_min, colors.val_min),
        False: hue_mask(planes, colors.bear_hue, cfg.hue_tolerance, colors.chroma_min, colors.val_min),
    }
    # Pixels coloridos que não são das velas (indicadores, marcadores) -> usados para marcar oclusão.
    lim = cfg.hue_tolerance + 6
    far = (hue_distance_map(planes.hue, colors.bull_hue) > lim) & (hue_distance_map(planes.hue, colors.bear_hue) > lim)
    foreign = (far & (planes.chroma >= colors.chroma_min) & (planes.val >= colors.val_min)).astype(np.uint8)
    if valid_mask is not None:
        for m in raw_masks.values():
            m[~valid_mask] = 0
        foreign[~valid_mask] = 1
    cands: list[_Cand] = []
    vclose = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    for bull, raw in raw_masks.items():
        # fecha falhas verticais curtas (linhas de indicador cruzando corpo/pavio)
        raw_masks[bull] = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, vclose)
        cleaned = remove_thin_horizontal_lines(raw_masks[bull])
        foreign[(raw_masks[bull] > 0) & (cleaned == 0)] = 1  # linhas removidas também cobrem velas
        cands += _components(cleaned, bull)
    if not cands:
        det.notes.append("nenhuma vela encontrada")
        return det

    widths = np.array([c.width for c in cands])
    med_w = float(np.median(widths))
    good: list[_Cand] = []
    rejected = 0
    for c in cands:
        if c.width > 1.7 * med_w or c.width < 0.5 * med_w:
            rejected += 1
            continue
        if c.width >= 4 and (c.body_bottom - c.body_top) >= 3 and c.rect_score < 0.7:
            rejected += 1  # forma arredondada: marcador de operação, não vela
            continue
        good.append(c)
    for c in cands:
        if c not in good:  # marcadores e outros formatos rejeitados também cobrem velas
            foreign[c.wick_top : c.wick_bottom + 1, c.comp_x0 : c.comp_x1 + 1] = 1
    foreign = cv2.dilate(foreign, np.ones((3, 3), np.uint8))
    if len(good) < 2:
        det.rejected = rejected
        det.notes.append("poucas velas reconhecidas")
        return det

    good.sort(key=lambda c: c.cx)
    med_w = float(np.median([c.width for c in good]))
    xs = np.array([c.cx for c in good])
    diffs = np.diff(xs)
    diffs = diffs[diffs > 0.5 * med_w]
    if diffs.size == 0:
        det.notes.append("velas sem espaçamento regular")
        return det
    pitch = float(np.median(diffs))
    ks = np.round(diffs / pitch)
    ok = (ks >= 1) & (np.abs(diffs / np.maximum(ks, 1) - pitch) < 0.2 * pitch)
    if ok.any():
        pitch = float(diffs[ok].sum() / ks[ok].sum())

    # Encaixa cada vela num "slot" da grade. Pedaços no mesmo slot são unidos.
    ref = xs[-1]
    slots: dict[int, list[_Cand]] = {}
    regular = 0
    for c in good:
        s = (c.cx - ref) / pitch
        k = int(round(s))
        if abs(s - k) <= 0.25:
            regular += 1
        slots.setdefault(k, []).append(c)

    combined: dict[int, np.ndarray] = {b: m for b, m in raw_masks.items()}
    ox, oy = offset
    out: list[PixelCandle] = []
    conflicts = 0
    for k in sorted(slots):
        group = slots[k]
        by_color: dict[bool, list[_Cand]] = {}
        for c in group:
            by_color.setdefault(c.bull, []).append(c)
        if len(by_color) > 1:
            conflicts += 1
        bull = max(by_color, key=lambda b: sum(c.width * (c.body_bottom - c.body_top + 1) for c in by_color[b]))
        parts = by_color[bull]
        main = max(parts, key=lambda c: c.body_bottom - c.body_top)
        merged = _Cand(
            x_left=int(np.median([c.x_left for c in parts])), x_right=int(np.median([c.x_right for c in parts])),
            body_top=min(c.body_top for c in parts), body_bottom=max(c.body_bottom for c in parts),
            wick_top=min(c.wick_top for c in parts), wick_bottom=max(c.wick_bottom for c in parts),
            bull=bull, rect_score=main.rect_score,
        )
        wt, wb = _extend_wick(combined[bull], merged)
        partial = merged.x_left <= 0 or merged.x_right >= plot_img.shape[1] - 1
        occ = bool(foreign[max(0, wt - 1) : wb + 2, merged.x_left : merged.x_right + 1].any())
        out.append(PixelCandle(merged.x_left + ox, merged.x_right + ox, merged.body_top + oy, merged.body_bottom + oy,
                               wt + oy, wb + oy, bull, k, partial, occ))

    det.candles = out
    det.pitch = pitch
    det.body_width = med_w
    det.rejected = rejected
    det.regularity = regular / len(good)
    det.too_small = med_w < cfg.min_body_width_px or pitch < cfg.min_pitch_px
    count_f = min(1.0, len(out) / max(1, cfg.min_candles))
    rej_ratio = rejected / (rejected + len(good))
    occ_ratio = sum(c.occluded for c in out) / len(out)
    conf = det.regularity * count_f * (1 - 0.5 * rej_ratio) * (1 - 0.5 * conflicts / len(out)) * (1 - 0.3 * occ_ratio)
    if occ_ratio > 0.3:
        det.notes.append("muitas velas cobertas por indicadores")
    if det.too_small:
        det.notes.append("velas pequenas demais: aumente o zoom do gráfico")
        conf *= 0.3
    det.confidence = float(max(0.0, min(1.0, conf)))
    return det

