"""Descobre, dentro da janela, onde estão as velas, o eixo de preço, o eixo de tempo, o payout e o ativo.

Estratégia (sem nenhuma configuração):
1. Acha as velas: muitas barras verticais coloridas, com espaçamento regular, lado a lado.
2. Cresce a partir delas até onde vai o fundo do gráfico.
3. Lê todos os textos da janela (OCR) uma vez:
   - uma coluna de números alinhados à direita das velas, com relação linear com y = eixo de preço;
   - horários HH:MM numa linha abaixo das velas = eixo de tempo;
   - "+87%" = payout (prioriza o painel de operação à direita);
   - "EUR/USD (OTC)" acima do gráfico = ativo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..common.geometry import Rect
from ..config import VisionConfig
from ..vision.colors import ColorPlanes, estimate_background, hue_mask, hue_peaks, saturated_mask
from ..vision.ocr import OcrEngine, OcrToken
from ..vision.price_axis import fit_mapping
from ..vision.text import asset_score, normalize_asset, parse_clock, parse_number, parse_percent

log = logging.getLogger(__name__)


@dataclass
class ChartLayout:
    frame_size: tuple[int, int]
    chart: Rect
    plot: Rect
    price_axis: Rect
    time_axis: Rect | None = None
    payout_box: Rect | None = None
    payout_value: int | None = None
    asset_box: Rect | None = None
    asset_text: str | None = None
    bg_bgr: tuple[int, int, int] = (0, 0, 0)
    confidence: float = 0.0
    manual: bool = False
    notes: list[str] = field(default_factory=list)
    tokens: list[OcrToken] = field(default_factory=list)


def _bar_components(img: np.ndarray, cfg: VisionConfig) -> np.ndarray:
    """Barras verticais coloridas (candidatas a vela), separadas por matiz. Array Nx4: x, y, w, h.

    Separar por matiz evita que uma linha de indicador de outra cor "cole" todas as velas num bloco só.
    """
    planes = ColorPlanes.of(img)
    sat = saturated_mask(planes, cfg.chroma_min, cfg.val_min).astype(bool)
    hues = planes.hue[sat]
    if hues.size < 30:
        return np.zeros((0, 4), dtype=np.int64)
    max_w = max(40, img.shape[1] // 30)
    keep = []
    for peak in hue_peaks(hues):
        m = hue_mask(planes, peak, 8, cfg.chroma_min, cfg.val_min)
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < 3 or w > max_w or h < 3:
                continue
            if area < 0.25 * w * h and w > 6:
                continue
            if h > 0.95 * img.shape[0]:
                continue
            keep.append((x, y, w, h))
    return np.array(keep, dtype=np.int64).reshape(-1, 4)


def _candle_cluster(bars: np.ndarray, max_gap: int = 80) -> np.ndarray | None:
    if len(bars) < 8:
        return None
    cx = bars[:, 0] + bars[:, 2] / 2.0
    order = np.argsort(cx)
    bars, cx = bars[order], cx[order]
    splits = np.flatnonzero(np.diff(cx) > max_gap) + 1
    groups = np.split(np.arange(len(bars)), splits)
    best = max(groups, key=len)
    if len(best) < 8:
        return None
    return bars[best]


def _expand(img: np.ndarray, box: Rect, bg: tuple[int, int, int], tol: float) -> Rect:
    """Cresce o retângulo enquanto as colunas/linhas forem majoritariamente fundo do gráfico."""
    H, W = img.shape[:2]
    small = img.astype(np.int16)
    bgv = np.array(bg, dtype=np.int16)
    close = (np.abs(small - bgv).sum(axis=2) <= tol * 1.5).astype(np.float32)
    col_frac = close[box.y : box.bottom].mean(axis=0)
    row_frac = close[:, box.x : box.right].mean(axis=1)

    def grow(frac: np.ndarray, start: int, step: int, limit: int) -> int:
        pos, bad = start, 0
        i = start + step
        while 0 <= i < limit:
            if frac[i] >= 0.5:
                pos, bad = i, 0
            else:
                bad += 1
                if bad >= 3:
                    break
            i += step
        return pos

    left = grow(col_frac, box.x, -1, W)
    right = grow(col_frac, box.right - 1, 1, W)
    top = grow(row_frac, box.y, -1, H)
    bottom = grow(row_frac, box.bottom - 1, 1, H)
    return Rect.from_ltrb(left, top, right + 1, bottom + 1)


def _find_price_axis(tokens: list[OcrToken], chart: Rect, candles_right: int) -> tuple[Rect | None, float]:
    nums = []
    for t in tokens:
        if t.box.cx <= candles_right or not chart.contains(t.box.cx, t.box.cy):
            continue
        pn = parse_number(t.text)
        if pn is None or len(pn.digits) < 2:
            continue
        nums.append((t, pn.value))
    if len(nums) < 3:
        return None, 0.0
    best: tuple[int, list[tuple[OcrToken, float]], float] | None = None
    for anchor, _ in nums:
        col = [(t, v) for t, v in nums if abs(t.box.x - anchor.box.x) <= 14 or abs(t.box.right - anchor.box.right) <= 14]
        if len(col) < 3:
            continue
        mapping, flags = fit_mapping([(t.box.cy, v) for t, v in col])
        if mapping is None:
            continue
        inl = [c for c, f in zip(col, flags) if f]
        if best is None or len(inl) > best[0]:
            best = (len(inl), inl, mapping.confidence)
    if best is None:
        return None, 0.0
    col_tokens = [t for t, _ in best[1]]
    x0 = min(t.box.x for t in col_tokens) - 8
    x1 = max(t.box.right for t in col_tokens) + 8
    return Rect.from_ltrb(x0, chart.y, max(x1, min(chart.right, x1 + 40)), chart.bottom), best[2]


def _find_time_axis(tokens: list[OcrToken], chart: Rect, axis_x: int, candles_bottom: int) -> Rect | None:
    clocks = [t for t in tokens if parse_clock(t.text) is not None and chart.x <= t.box.cx < axis_x
              and t.box.cy > candles_bottom]
    if len(clocks) < 2:
        return None
    ys = np.array([t.box.cy for t in clocks])
    med = float(np.median(ys))
    row = [t for t in clocks if abs(t.box.cy - med) <= 8]
    if len(row) < 2:
        return None
    y0 = min(t.box.y for t in row) - 4
    y1 = max(t.box.bottom for t in row) + 4
    return Rect.from_ltrb(chart.x, y0, axis_x, y1)


def _region_brightness(img: np.ndarray, r: Rect) -> float:
    c = r.crop(img)
    return float(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).mean()) if c.size else 0.0


def _find_payout(tokens: list[OcrToken], chart: Rect, axis: Rect | None) -> tuple[Rect | None, int | None]:
    best: tuple[float, OcrToken, int] | None = None
    for t in tokens:
        v = parse_percent(t.text)
        if v is None or v > 100:
            continue
        score = t.box.h / 20.0
        if axis is not None and t.box.x >= axis.right - 4:
            score += 2.0  # painel de operação
        if "+" in t.text:
            score += 1.0
        if chart.contains(t.box.cx, t.box.cy):
            score -= 3.0
        if best is None or score > best[0]:
            best = (score, t, v)
    if best is None:
        return None, None
    b = best[1].box
    return Rect(b.x - 10, b.y - 3, b.w + 30, b.h + 6), best[2]


def _find_asset(img: np.ndarray, tokens: list[OcrToken], chart: Rect) -> tuple[Rect | None, str | None]:
    top_band = Rect(0, 0, img.shape[1], max(1, chart.y))
    band_brightness = _region_brightness(img, top_band)
    best: tuple[float, OcrToken] | None = None
    for t in tokens:
        if t.box.cy >= chart.y:
            continue
        s = asset_score(t.text)
        if s <= 0:
            continue
        # aba ativa: fundo destacado em relação à barra superior
        around = t.box.expand(10, 6).clip(img.shape[1], img.shape[0])
        contrast = abs(_region_brightness(img, around) - band_brightness) / 40.0
        score = s + min(contrast, 0.6)
        if best is None or score > best[0]:
            best = (score, t)
    if best is None:
        return None, None
    t = best[1]
    box = Rect(t.box.x - 6, t.box.y - 3, t.box.w + 80, t.box.h + 6).clip(img.shape[1], img.shape[0])
    return box, normalize_asset(t.text)


def detect_layout(img: np.ndarray, ocr: OcrEngine, cfg: VisionConfig) -> ChartLayout | None:
    H, W = img.shape[:2]
    notes: list[str] = []
    manual = cfg.manual_plot_rect is not None
    if manual:
        assert cfg.manual_plot_rect is not None
        seed = Rect.from_list(cfg.manual_plot_rect).clip(W, H)
        if seed.is_empty():
            return None
        cands_right, cands_bottom = seed.right - 40, seed.bottom
        bg = estimate_background(seed.crop(img))
        chart = seed.union(Rect(seed.x, seed.y, min(W - seed.x, seed.w + 200), seed.h + 40)).clip(W, H)
        notes.append("área manual")
    else:
        bars = _bar_components(img, cfg)
        cluster = _candle_cluster(bars)
        if cluster is None:
            log.info("layout: velas não encontradas (%d barras)", len(bars))
            return None
        x0, y0 = int(cluster[:, 0].min()), int(cluster[:, 1].min())
        x1, y1 = int((cluster[:, 0] + cluster[:, 2]).max()), int((cluster[:, 1] + cluster[:, 3]).max())
        seed = Rect.from_ltrb(x0, y0, x1, y1)
        bg = estimate_background(seed.crop(img))
        chart = _expand(img, seed, bg, cfg.bg_tolerance / 2)
        cands_right, cands_bottom = x1, y1

    tokens = ocr.read(img)
    axis, axis_conf = _find_price_axis(tokens, chart.union(Rect(chart.x, chart.y, W - chart.x, chart.h)), cands_right)
    if axis is None:
        notes.append("eixo de preço não encontrado")
        axis = Rect(chart.right - 90, chart.y, 90, chart.h)
        axis_conf = 0.0
    chart = Rect.from_ltrb(chart.x, chart.y, max(chart.right, axis.right), chart.bottom)
    time_axis = _find_time_axis(tokens, chart, axis.x, cands_bottom)
    plot_bottom = time_axis.y - 1 if time_axis is not None else chart.bottom
    if manual:
        plot = seed
    else:
        plot = Rect.from_ltrb(chart.x, chart.y, axis.x - 1, plot_bottom)
    axis = Rect(axis.x, chart.y, axis.w, plot.bottom - chart.y)
    payout_box, payout = _find_payout(tokens, chart, axis)
    asset_box, asset = _find_asset(img, tokens, chart)
    if time_axis is None:
        notes.append("eixo de tempo não encontrado")
    if payout is None:
        notes.append("payout não encontrado")
    if asset is None:
        notes.append("ativo não encontrado")
    conf = 0.25 + 0.55 * axis_conf + (0.1 if time_axis else 0) + (0.1 if payout_box else 0)
    layout = ChartLayout(
        frame_size=(W, H), chart=chart, plot=plot, price_axis=axis, time_axis=time_axis,
        payout_box=payout_box, payout_value=payout, asset_box=asset_box, asset_text=asset, bg_bgr=bg,
        confidence=float(min(1.0, conf)), manual=manual, notes=notes, tokens=tokens,
    )
    log.info("layout: plot=%s eixo=%s tempo=%s payout=%s ativo=%s conf=%.2f %s", plot.to_list(), axis.to_list(),
             time_axis.to_list() if time_axis else None, payout, asset, layout.confidence, notes)
    return layout
