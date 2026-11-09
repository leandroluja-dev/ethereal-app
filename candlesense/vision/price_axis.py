"""Eixo de preço: lê os rótulos por OCR e ajusta preço = a + b·y (regressão robusta).

Também lê a etiqueta colorida do preço atual e o cronômetro que fica junto dela.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..common.geometry import Rect
from .ocr import OcrEngine
from .text import parse_clock, parse_number, repair_decimals

log = logging.getLogger(__name__)


@dataclass
class AxisLabel:
    box: Rect
    text: str
    conf: float
    value: float | None
    inlier: bool = False


@dataclass(frozen=True)
class PriceMapping:
    a: float
    b: float
    decimals: int
    inliers: int
    total: int
    rms_px: float
    confidence: float

    def price(self, y: float) -> float:
        return self.a + self.b * y

    def y(self, price: float) -> float:
        return (price - self.a) / self.b

    def fmt(self, price: float) -> str:
        return f"{price:.{self.decimals}f}"


@dataclass
class AxisReading:
    mapping: PriceMapping | None = None
    labels: list[AxisLabel] = field(default_factory=list)
    tag_box: Rect | None = None
    tag_text: str = ""
    tag_price: float | None = None
    timer_box: Rect | None = None
    timer_text: str = ""
    timer_s: int | None = None
    reocr: bool = False
    tag_residual_px: float | None = None
    notes: list[str] = field(default_factory=list)


def text_mask(img: np.ndarray, thresh: int = 40) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    bg = float(np.median(g))
    return (np.abs(g.astype(np.int16) - bg) > thresh).astype(np.uint8)


def find_colored_boxes(img: np.ndarray, min_w_frac: float = 0.45, chroma_min: int = 50) -> list[Rect]:
    """Caixas coloridas cheias (etiqueta do preço atual, etiquetas de operações)."""
    mx = img.max(axis=2).astype(np.int16)
    mn = img.min(axis=2).astype(np.int16)
    m = ((mx - mn) >= chroma_min).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out: list[Rect] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if w >= min_w_frac * img.shape[1] and 8 <= h <= 60 and area >= 0.5 * w * h:
            out.append(Rect(x, y, w, h))
    return out


def segment_lines(mask: np.ndarray, axis: int, min_size: int = 5, max_size: int = 40, gap: int = 1,
                  merge_gap: int = 12) -> list[Rect]:
    """Separa blocos de texto. axis=0: linhas empilhadas (eixo de preço); axis=1: rótulos lado a lado."""
    if axis == 1:
        return [Rect(r.y, r.x, r.h, r.w) for r in segment_lines(mask.T.copy(), 0, min_size, max_size, gap, merge_gap)]
    h, w = mask.shape
    rows = mask.sum(axis=1)
    rows[rows > 0.7 * w] = 0  # linha de grade, não texto
    on = rows > 0
    out: list[Rect] = []
    y = 0
    while y < h:
        if not on[y]:
            y += 1
            continue
        y0 = y
        miss = 0
        while y < h and (on[y] or miss < gap):
            miss = 0 if on[y] else miss + 1
            y += 1
        y1 = y - miss
        if not (min_size <= y1 - y0 <= max_size):
            continue
        cols = np.flatnonzero(mask[y0:y1].any(axis=0))
        if cols.size == 0:
            continue
        # agrupa colunas próximas e fica com o grupo mais largo
        groups = np.split(cols, np.flatnonzero(np.diff(cols) > merge_gap) + 1)
        g = max(groups, key=lambda c: c[-1] - c[0])
        out.append(Rect(int(g[0]), y0, int(g[-1] - g[0] + 1), y1 - y0))
    return out


def fit_mapping(points: list[tuple[float, float]], tol_px: float = 2.5) -> tuple[PriceMapping | None, list[bool]]:
    """RANSAC sobre pares (y, preço). Preço sobe para cima: b < 0."""
    n = len(points)
    flags = [False] * n
    if n < 3:
        return None, flags
    ys = np.array([p[0] for p in points], dtype=np.float64)
    vs = np.array([p[1] for p in points], dtype=np.float64)
    best: tuple[int, float, np.ndarray] | None = None
    for i, j in itertools.combinations(range(n), 2):
        if abs(ys[i] - ys[j]) < 5:
            continue
        b = (vs[j] - vs[i]) / (ys[j] - ys[i])
        if b >= 0:
            continue
        a = vs[i] - b * ys[i]
        res_px = np.abs(ys - (vs - a) / b)
        inl = res_px <= tol_px
        cnt = int(inl.sum())
        err = float(res_px[inl].mean())
        if best is None or cnt > best[0] or (cnt == best[0] and err < best[1]):
            best = (cnt, err, inl)
    if best is None or best[0] < 3:
        return None, flags
    inl = best[2]
    b, a = np.polyfit(ys[inl], vs[inl], 1)
    if b >= 0:
        return None, flags
    res_px = np.abs(ys[inl] - (vs[inl] - a) / b)
    rms = float(np.sqrt(np.mean(res_px**2)))
    cnt = int(inl.sum())
    conf = (cnt / n) * min(1.0, cnt / 4) * float(np.exp(-max(0.0, rms - 0.7) / 1.5))
    flags = [bool(f) for f in inl]
    return PriceMapping(float(a), float(b), 0, cnt, n, rms, float(conf)), flags


class PriceAxisReader:
    """Lê o eixo de preço. Rótulos fixos só são relidos quando o eixo muda (reescala)."""

    def __init__(self, ocr: OcrEngine, change_threshold: float = 3.0) -> None:
        self._ocr = ocr
        self._thr = change_threshold
        self._sig: np.ndarray | None = None
        self._sig_rect: Rect | None = None
        self._cached: tuple[PriceMapping | None, list[AxisLabel], list[str]] | None = None

    def reset(self) -> None:
        self._sig = None
        self._cached = None

    def _signature(self, band: np.ndarray, exclude: list[Rect]) -> np.ndarray:
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for r in exclude:
            e = r.expand(2, 4).clip(g.shape[1], g.shape[0])
            g[e.y : e.bottom, :] = 0
        return cv2.resize(g, (max(1, g.shape[1] // 2), max(1, g.shape[0] // 2)), interpolation=cv2.INTER_AREA)

    def read(self, img: np.ndarray, axis: Rect) -> AxisReading:
        axis = axis.clip(img.shape[1], img.shape[0])
        out = AxisReading()
        if axis.is_empty():
            out.notes.append("eixo de preço não encontrado")
            return out
        band = axis.crop(img)
        tags = find_colored_boxes(band)
        tag = max(tags, key=lambda r: r.area) if tags else None
        mask = text_mask(band)
        for r in tags:
            e = r.expand(2).clip(mask.shape[1], mask.shape[0])
            mask[e.y : e.bottom, e.x : e.right] = 0
        lines = segment_lines(mask, axis=0)

        # Cronômetro: linha de texto logo abaixo/acima da etiqueta do preço atual.
        dynamic: list[Rect] = []
        if tag is not None:
            for ln in lines:
                if abs(ln.cy - tag.cy) <= 3.5 * tag.h and not ln.intersect(tag.expand(1)):
                    dynamic.append(ln)
        static = [ln for ln in lines if ln not in dynamic]

        sig = self._signature(band, ([tag] if tag else []) + dynamic)
        changed = (
            self._cached is None
            or self._sig is None
            or self._sig.shape != sig.shape
            or self._sig_rect != axis
            or float(np.mean(np.abs(self._sig - sig))) > self._thr
        )
        if changed:
            self._read_static(band, axis, static, out)
            self._sig, self._sig_rect = sig, axis
            self._cached = (out.mapping, list(out.labels), list(out.notes))
            out.reocr = True
        else:
            assert self._cached is not None
            out.mapping, labels, notes = self._cached
            out.labels, out.notes = list(labels), list(notes)

        # Etiqueta do preço atual e cronômetro: lidos a cada quadro (recortes pequenos, rápido).
        crops: list[np.ndarray] = []
        if tag is not None:
            crops.append(tag.crop(band))
        dyn_crops = [d.expand(2).crop(band) for d in dynamic]
        res = self._ocr.read_lines(crops + dyn_crops) if (crops or dyn_crops) else []
        if tag is not None and res:
            text, _ = res[0]
            out.tag_box = tag.offset(axis.x, axis.y)
            out.tag_text = text
            pn = parse_number(text)
            if pn is not None:
                out.tag_price = pn.value
                if out.mapping is not None and out.mapping.decimals and pn.decimals != out.mapping.decimals:
                    fixed = repair_decimals([pn] + [parse_number(lb.text) for lb in out.labels if lb.inlier])
                    out.tag_price = fixed[0]
            if out.mapping is not None and out.tag_price is not None:
                out.tag_residual_px = float(out.tag_box.cy - out.mapping.y(out.tag_price))
        extra: list[AxisLabel] = []
        for d, (text, conf) in zip(dynamic, res[len(crops) :]):
            clk = parse_clock(text)
            if clk is not None and len(clk) == 2:
                if out.timer_box is None:
                    out.timer_box = d.offset(axis.x, axis.y)
                    out.timer_text = text
                    out.timer_s = clk[0] * 60 + clk[1]
                continue
            pn = parse_number(text)
            if pn is not None and pn.had_separator:
                extra.append(AxisLabel(d.offset(axis.x, axis.y), text, conf, pn.value))
        if extra and out.labels:
            # rótulos perto da etiqueta também entram na regressão (reajuste é barato)
            labels = [AxisLabel(lb.box, lb.text, lb.conf, lb.value) for lb in out.labels] + extra
            self._fit(labels, [parse_number(lb.text) for lb in labels], out)
            if out.mapping is not None and out.tag_price is not None:
                out.tag_residual_px = float(out.tag_box.cy - out.mapping.y(out.tag_price)) if out.tag_box else None
        return out

    def _read_static(self, band: np.ndarray, axis: Rect, lines: list[Rect], out: AxisReading) -> None:
        if not lines:
            out.notes.append("nenhum rótulo no eixo de preço")
            return
        res = self._ocr.read_lines([ln.expand(2).crop(band) for ln in lines])
        parsed = [parse_number(t) for t, _ in res]
        values = repair_decimals(parsed)
        labels = [
            AxisLabel(ln.offset(axis.x, axis.y), t, c, v) for ln, (t, c), v in zip(lines, res, values)
        ]
        self._fit(labels, parsed, out)

    @staticmethod
    def _fit(labels: list[AxisLabel], parsed: list, out: AxisReading) -> None:  # type: ignore[type-arg]
        pts = [(lb.box.cy, lb.value) for lb in labels if lb.value is not None]
        idx = [i for i, lb in enumerate(labels) if lb.value is not None]
        mapping, flags = fit_mapping(pts)  # type: ignore[arg-type]
        for i, f in zip(idx, flags):
            labels[i].inlier = f
        if mapping is not None:
            decs = [p.decimals for p, lb in zip(parsed, labels) if p is not None and lb.inlier and p.had_separator]
            dec = max(set(decs), key=decs.count) if decs else 0
            mapping = PriceMapping(mapping.a, mapping.b, dec, mapping.inliers, mapping.total, mapping.rms_px,
                                   mapping.confidence)
        else:
            out.notes.append("rótulos do eixo de preço inconsistentes")
        out.mapping = mapping
        out.labels = labels


@dataclass
class TimeAxisReading:
    minutes_per_px: float | None = None
    labels: list[tuple[Rect, str, float | None]] = field(default_factory=list)
    reocr: bool = False


class TimeAxisReader:
    """Lê os horários do eixo de tempo para descobrir quantos minutos cada vela representa."""

    def __init__(self, ocr: OcrEngine, change_threshold: float = 3.0) -> None:
        self._ocr = ocr
        self._thr = change_threshold
        self._sig: np.ndarray | None = None
        self._cached: TimeAxisReading | None = None

    def reset(self) -> None:
        self._sig = None
        self._cached = None

    def read(self, img: np.ndarray, band_rect: Rect) -> TimeAxisReading:
        band_rect = band_rect.clip(img.shape[1], img.shape[0])
        if band_rect.is_empty():
            return TimeAxisReading()
        band = band_rect.crop(img)
        g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sig = cv2.resize(g, (max(1, g.shape[1] // 4), max(1, g.shape[0] // 2)), interpolation=cv2.INTER_AREA)
        if self._cached is not None and self._sig is not None and self._sig.shape == sig.shape and \
                float(np.mean(np.abs(self._sig - sig))) <= self._thr:
            return self._cached
        out = TimeAxisReading(reocr=True)
        mask = text_mask(band)
        boxes = segment_lines(mask, axis=1, min_size=8, max_size=120, gap=8, merge_gap=4)
        res = self._ocr.read_lines([b.expand(2).crop(band) for b in boxes]) if boxes else []
        pts: list[tuple[float, float]] = []
        for b, (t, _) in zip(boxes, res):
            clk = parse_clock(t)
            minute = float(clk[0] * 60 + clk[1]) if clk is not None and clk[0] < 24 else None
            box = b.offset(band_rect.x, band_rect.y)
            out.labels.append((box, t, minute))
            if minute is not None:
                pts.append((box.cx, minute))
        pts.sort()
        # atravessar a meia-noite
        for i in range(1, len(pts)):
            while pts[i][1] < pts[i - 1][1]:
                pts[i] = (pts[i][0], pts[i][1] + 1440)
        if len(pts) >= 2:
            xs = np.array([p[0] for p in pts])
            ms = np.array([p[1] for p in pts])
            slopes = np.diff(ms) / np.maximum(np.diff(xs), 1e-6)
            slopes = slopes[slopes > 0]
            if slopes.size:
                out.minutes_per_px = float(np.median(slopes))
        self._sig, self._cached = sig, out
        return out


TIMEFRAMES_MIN = (1, 2, 5, 10, 15, 30, 60, 120, 240, 1440)


def snap_timeframe(minutes_per_candle: float | None) -> int | None:
    if minutes_per_candle is None or minutes_per_candle <= 0:
        return None
    best = min(TIMEFRAMES_MIN, key=lambda t: abs(np.log(minutes_per_candle / t)))
    return best if abs(minutes_per_candle / best - 1) <= 0.2 else None
