"""Leitura completa de um quadro: velas em pixels -> velas OHLC em preço.

Regra principal: se qualquer parte essencial da leitura for duvidosa, o
resultado sai com ``ok=False`` ("leitura incerta") e nada deve ser sinalizado.
O leitor nunca lança exceção para quem chama.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..models import Box, Candle, ChartReading, Layout, OcrToken, PriceScale
from .axis import axis_labels, fit_scale, grid_profile, snap_to_grid
from .candles import ColorConfig, CandleDetection, detect_candles
from .layout import build_layout
from .ocr import OcrEngine, parse_number, parse_payout, parse_timer

log = logging.getLogger(__name__)


@dataclass
class ReaderConfig:
    min_candles: int = 20
    axis_tol_px: float = 3.0
    max_residual_px: float = 2.5
    min_axis_labels: int = 3
    relayout_cooldown_s: float = 2.0
    label_tol_px: float = 3.0  # etiqueta do preço atual x fechamento da vela atual


class _OcrCache:
    """Evita repetir OCR quando o recorte não mudou."""

    def __init__(self) -> None:
        self.key: Optional[str] = None
        self.tokens: list[OcrToken] = []
        self.labels: list[tuple[float, float, int]] = []


def _crop_key(bgr: np.ndarray, box: Box, ignore_colored_rows: bool = False) -> str:
    crop = bgr[box.y : box.y2, box.x : box.x2]
    if crop.size == 0:
        return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if ignore_colored_rows:
        # A etiqueta colorida do preço atual anda pelo eixo: as linhas dela não contam.
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        colored = (hsv[..., 1] > 80).any(axis=1)
        if colored.any():
            rows = np.nonzero(colored)[0]
            lo = max(0, rows.min() - 4)
            hi = rows.max() + 5
            gray = gray.copy()
            gray[lo:hi] = 0
    small = gray[::2, ::2]
    return hashlib.blake2b(small.tobytes(), digest_size=12).hexdigest()


def _saturated_fraction(bgr: np.ndarray, box: Box) -> float:
    crop = bgr[box.y : box.y2, box.x : box.x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 1] > 80) & (hsv[..., 2] > 70)).mean())


def _colored_boxes(bgr: np.ndarray, axis: Box) -> list[Box]:
    """Faixas de fundo colorido dentro do eixo (candidatas à etiqueta do preço atual)."""
    crop = bgr[axis.y : axis.y2, axis.x : axis.x2]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] > 80) & (hsv[..., 2] > 70)
    rows = sat.mean(axis=1) > 0.35
    if not rows.any():
        return []
    idx = np.nonzero(rows)[0]
    groups = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    out = []
    for g in sorted(groups, key=len, reverse=True):
        if len(g) < 6:
            continue
        cols = np.nonzero(sat[g].mean(axis=0) > 0.5)[0]
        if cols.size == 0:
            continue
        out.append(Box(axis.x + int(cols.min()), axis.y + int(g.min()),
                       int(cols.max() - cols.min() + 1), int(len(g))))
    return out


class ChartReader:
    def __init__(self, ocr: OcrEngine, cfg: Optional[ReaderConfig] = None,
                 colors: Optional[ColorConfig] = None) -> None:
        self.ocr = ocr
        self.cfg = cfg or ReaderConfig()
        self.colors = colors or ColorConfig()
        self.layout: Optional[Layout] = None
        self._last_layout_t = 0.0
        self._axis_cache = _OcrCache()
        self._timer_cache = _OcrCache()
        self._payout_cache = _OcrCache()
        self._label_cache = _OcrCache()
        self.last_tokens: list[OcrToken] = []  # do último OCR de tela inteira (debug)

    def reset(self) -> None:
        self.layout = None
        self._axis_cache = _OcrCache()
        self._timer_cache = _OcrCache()
        self._payout_cache = _OcrCache()
        self._label_cache = _OcrCache()

    # ------------------------------------------------------------------ API

    def read(self, bgr: np.ndarray) -> ChartReading:
        try:
            return self._read(bgr)
        except Exception as e:  # nunca derruba o app por erro de leitura
            log.exception("erro na leitura")
            return ChartReading(ok=False, reasons=[f"erro interno na leitura: {e!r}"])

    # ------------------------------------------------------------ internals

    def _ocr_cached(self, cache: _OcrCache, bgr: np.ndarray, box: Box, ignore_colored: bool = False) -> list[OcrToken]:
        key = _crop_key(bgr, box, ignore_colored)
        if key and key == cache.key:
            return cache.tokens
        cache.tokens = self.ocr.read(bgr, box)
        cache.key = key
        return cache.tokens

    def _detect(self, bgr: np.ndarray) -> CandleDetection:
        roi = None
        lay = self.layout
        if lay is not None and lay.width == bgr.shape[1] and lay.height == bgr.shape[0]:
            x1 = max(0, int(lay.chart.x - 3 * lay.candle_pitch))
            x2 = lay.axis.x if lay.axis else lay.width
            roi = Box(x1, 0, max(1, x2 - x1), lay.height)
        return detect_candles(bgr, self.colors, roi=roi, min_candles=self.cfg.min_candles)

    def _needs_relayout(self, bgr: np.ndarray, det: CandleDetection) -> bool:
        lay = self.layout
        if lay is None:
            return True
        if (lay.width, lay.height) != (bgr.shape[1], bgr.shape[0]):
            return True
        if lay.axis is None:
            return True
        if not det.candles:
            return True
        if abs(det.pitch - lay.candle_pitch) > 0.15 * lay.candle_pitch:
            return True  # zoom mudou
        return False

    def _relayout(self, bgr: np.ndarray, det: CandleDetection) -> None:
        now = time.monotonic()
        if self.layout is not None and now - self._last_layout_t < self.cfg.relayout_cooldown_s:
            return
        self._last_layout_t = now
        if det.box is None:
            # Pode ter detectado com ROI antiga: tenta a tela inteira.
            self.layout = None
            det = detect_candles(bgr, self.colors, min_candles=self.cfg.min_candles)
            if det.box is None:
                return
        tokens = self.ocr.read(bgr, None)
        self.last_tokens = tokens
        self.layout = build_layout(bgr, det, tokens)
        self._axis_cache = _OcrCache()
        self._timer_cache = _OcrCache()
        self._payout_cache = _OcrCache()
        self._label_cache = _OcrCache()

    def _read(self, bgr: np.ndarray) -> ChartReading:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        if bgr.ndim == 3 and bgr.shape[2] == 4:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
        det = self._detect(bgr)
        timings["velas"] = (time.perf_counter() - t0) * 1000

        if self._needs_relayout(bgr, det):
            t = time.perf_counter()
            self._relayout(bgr, det)
            if self.layout is not None and not det.candles:
                det = self._detect(bgr)
            timings["layout"] = (time.perf_counter() - t) * 1000

        reading = ChartReading(ok=True, layout=self.layout, timings_ms=timings)
        reading.reasons.extend(det.reasons)
        reading.pixel_candles = det.candles
        lay = self.layout
        if lay is None or not det.candles:
            reading.ok = False
            if not reading.reasons:
                reading.reasons.append("gráfico não encontrado na captura")
            return reading

        # Eixo de preço
        t = time.perf_counter()
        scale: Optional[PriceScale] = None
        label = None
        if lay.axis is None:
            reading.reasons.append("eixo de preço não encontrado")
        else:
            scale = self._axis_scale(bgr, lay, reading)
            label = self._current_price_label(bgr, lay)
        timings["eixo"] = (time.perf_counter() - t) * 1000
        reading.scale = scale

        # Cronômetro e payout (não bloqueiam a leitura das velas, mas avisam)
        t = time.perf_counter()
        if lay.timer is not None:
            toks = self._ocr_cached(self._timer_cache, bgr, lay.timer)
            vals = [v for v in (parse_timer(x.text) for x in toks) if v is not None]
            reading.timer_seconds = vals[0] if vals else None
        if reading.timer_seconds is None:
            reading.warnings.append("cronômetro não lido")
        if lay.payout is not None:
            toks = self._ocr_cached(self._payout_cache, bgr, lay.payout)
            vals = [v for v in (parse_payout(x.text) for x in toks) if v is not None]
            reading.payout_pct = vals[0] if vals else None
        if reading.payout_pct is None:
            reading.warnings.append("payout não lido")
        timings["textos"] = (time.perf_counter() - t) * 1000

        if len(det.candles) < self.cfg.min_candles:
            reading.ok = False
        if det.reasons:
            reading.ok = False
        if scale is None or scale.residual_px > self.cfg.max_residual_px:
            reading.ok = False

        if scale is not None:
            dec = scale.decimals
            n = len(det.candles)
            for i, pc in enumerate(det.candles):
                def p(y: int) -> float:
                    return round(scale.price(y + 0.5), dec)
                reading.candles.append(Candle(
                    open=p(pc.open_y), high=p(pc.high_y), low=p(pc.low_y), close=p(pc.close_y),
                    color=pc.color, forming=(i == n - 1), x_center=pc.x_center,
                ))
            reading.current_price = reading.candles[-1].close if reading.candles else None
            if label is not None and reading.candles:
                self._apply_price_label(reading, scale, label)
            elif det.candles and not det.candles[-1].body_certain:
                reading.ok = False
                reading.reasons.append("corpo da vela atual encoberto e etiqueta do preço não lida")
            if any(not pc.body_certain for pc in det.candles[-4:-1]):
                reading.ok = False
                reading.reasons.append("corpo de uma vela recente não pôde ser medido")
        return reading

    def _axis_scale(self, bgr: np.ndarray, lay: Layout, reading: ChartReading) -> Optional[PriceScale]:
        assert lay.axis is not None
        key = _crop_key(bgr, lay.axis, ignore_colored_rows=True)
        cache = self._axis_cache
        if not (key and key == cache.key):
            tokens = self.ocr.read(bgr, lay.axis)
            # a etiqueta do preço atual é lida à parte, a cada quadro
            tokens = [tk for tk in tokens if _saturated_fraction(bgr, tk.box) < 0.25]
            labels = axis_labels(tokens)
            if labels:
                lh = float(np.median([tk.box.h for tk in tokens])) if tokens else 10.0
                prof = grid_profile(bgr, lay.chart.x, lay.axis.x, lay.axis.y, lay.axis.y2)
                labels = snap_to_grid(labels, prof, lh)
            cache.tokens = tokens
            cache.key = key
            cache.labels = labels
        labels = cache.labels
        scale = fit_scale(labels, tol_px=self.cfg.axis_tol_px, min_inliers=self.cfg.min_axis_labels)
        if scale is None:
            reading.reasons.append(f"eixo de preço ilegível ({len(labels)} rótulos lidos)")
            self._axis_cache = _OcrCache()  # tenta de novo no próximo quadro
        elif scale.residual_px > self.cfg.max_residual_px:
            reading.reasons.append(f"rótulos do eixo inconsistentes ({scale.residual_px:.1f}px)")
        return scale

    def _current_price_label(self, bgr: np.ndarray, lay: Layout) -> Optional[tuple[float, float]]:
        """Etiqueta colorida do preço atual no eixo: (y do centro, preço)."""
        assert lay.axis is not None
        for box in _colored_boxes(bgr, lay.axis)[:2]:
            toks = self._ocr_cached(self._label_cache, bgr, box)
            for tk in toks:
                p = parse_number(tk.text)
                if p is not None:
                    return (box.cy, p[0])
        return None

    def _apply_price_label(self, reading: ChartReading, scale: PriceScale,
                           label: tuple[float, float]) -> None:
        """Confere a vela atual com a etiqueta do preço e usa o valor exato dela."""
        last = reading.candles[-1]
        pc = reading.pixel_candles[-1]
        price = label[1]
        if not pc.body_certain:
            # Corpo encoberto pela linha do preço: o corpo está na etiqueta.
            y = scale.y_of(price)
            if pc.high_y - 2 <= y <= pc.low_y + 3:
                last.open = last.close = price
                last.high = max(last.high, price)
                last.low = min(last.low, price)
                reading.current_price = price
                return
            reading.ok = False
            reading.reasons.append("etiqueta do preço fora da vela atual")
            return
        dpx = abs(scale.y_of(price) - scale.y_of(last.close))
        if dpx > self.cfg.label_tol_px:
            reading.ok = False
            reading.reasons.append(
                f"preço atual da etiqueta ({price}) não confere com a vela ({dpx:.1f}px)")
            return
        last.close = price
        last.high = max(last.high, price)
        last.low = min(last.low, price)
        reading.current_price = price
