"""Leitura completa de um quadro: velas em pixels -> velas OHLC em preço.

Regra principal: se qualquer parte essencial da leitura for duvidosa, o
resultado sai com ``ok=False`` ("leitura incerta") e nada deve ser sinalizado.
O leitor nunca lança exceção para quem chama.
"""

from __future__ import annotations

import hashlib
import math
import logging
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..models import Box, Candle, ChartReading, Layout, OcrToken, PriceScale
from .axis import axis_labels, fit_scale, grid_profile, snap_to_grid
from .candles import ColorConfig, CandleDetection, detect_candles, measure_forming
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
    axis_max_age_s: float = 2.0  # reler os rótulos do eixo no máximo a cada N s
    payout_max_age_s: float = 5.0
    timer_ocr_every_s: float = 1.0  # entre leituras, o cronômetro é descontado pelo relógio
    max_candle_height_ratio: float = 4.0  # vela atual mais alta que N x a maior recente = suspeita


class _OcrCache:
    """Evita repetir OCR quando o recorte não mudou."""

    def __init__(self) -> None:
        self.key: Optional[str] = None
        self.t: float = -1e9  # quando o OCR foi feito (time.monotonic)
        self.value: Optional[int] = None  # último valor lido (cronômetro)
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

    def _ocr_cached(self, cache: _OcrCache, bgr: np.ndarray, box: Box, max_age: float = 0.0) -> list[OcrToken]:
        """OCR do recorte, reaproveitado se a imagem é a mesma ou se o resultado é recente."""
        now = time.monotonic()
        if cache.tokens and now - cache.t < max_age:
            return cache.tokens
        key = _crop_key(bgr, box)
        if key and key == cache.key:
            return cache.tokens
        cache.tokens = self.ocr.read(bgr, box)
        cache.key = key
        cache.t = now
        return cache.tokens

    def _detect(self, bgr: np.ndarray) -> CandleDetection:
        roi = None
        lay = self.layout
        if lay is not None and lay.width == bgr.shape[1] and lay.height == bgr.shape[0]:
            x1 = max(0, int(lay.chart.x - 3 * lay.candle_pitch))
            x2 = lay.axis.x if lay.axis else lay.width
            y1, y2 = 0, lay.height
            if lay.axis is not None:
                # Só o painel do gráfico: menos pixels para processar (Retina pesa).
                m = int(0.15 * lay.axis.h)
                y1, y2 = max(0, lay.axis.y - m), min(lay.height, lay.axis.y2 + m)
            roi = Box(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
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
        reading.pixel_candles = list(det.candles)
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
            reading.timer_seconds = self._read_timer(bgr, lay.timer)
        if reading.timer_seconds is None:
            reading.warnings.append("cronômetro não lido")
        if lay.payout is not None:
            toks = self._ocr_cached(self._payout_cache, bgr, lay.payout, max_age=self.cfg.payout_max_age_s)
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
            # A grade pode ter poucas casas (1.105) mas a resolução do gráfico é bem maior:
            # arredonda na casa que um pixel consegue distinguir, nunca menos que a etiqueta.
            px_dec = int(math.ceil(-math.log10(scale.price_per_px / 2))) if scale.price_per_px > 0 else 0
            dec = max(scale.decimals, px_dec, label[2] if label else 0)
            scale.decimals = dec
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
                self._apply_price_label(reading, scale, label, bgr, det)
            elif det.candles and not det.candles[-1].body_certain:
                reading.ok = False
                reading.reasons.append("corpo da vela atual encoberto e etiqueta do preço não lida")
            pcs = reading.pixel_candles
            if len(pcs) >= 6:
                prev_h = max(pc.low_y - pc.high_y + 1 for pc in pcs[-21:-1])
                last_h = pcs[-1].low_y - pcs[-1].high_y + 1
                if last_h > self.cfg.max_candle_height_ratio * prev_h:
                    reading.ok = False
                    reading.reasons.append("vela atual alta demais: algo sobreposto a ela (linha vertical?)")
            if any(not pc.body_certain for pc in det.candles[-4:-1]):
                reading.ok = False
                reading.reasons.append("corpo de uma vela recente não pôde ser medido")
        return reading

    def _read_timer(self, bgr: np.ndarray, box: Box) -> Optional[int]:
        """Lê mm:ss no máximo 1x por segundo; entre leituras desconta o tempo passado."""
        now = time.monotonic()
        c = self._timer_cache
        last = c.value
        if last is not None and now - c.t < self.cfg.timer_ocr_every_s:
            return max(0, int(round(last - (now - c.t))))
        toks = self.ocr.read(bgr, box)
        vals = [v for v in (parse_timer(x.text) for x in toks) if v is not None]
        c.value = vals[0] if vals else None
        c.t = now
        return c.value

    def _axis_scale(self, bgr: np.ndarray, lay: Layout, reading: ChartReading) -> Optional[PriceScale]:
        assert lay.axis is not None
        # Os rótulos do eixo só mudam quando a escala muda. Reaproveita por alguns segundos;
        # se a escala mudar nesse meio tempo, a etiqueta do preço atual não confere,
        # a leitura sai incerta e o eixo é lido de novo no quadro seguinte.
        cache = self._axis_cache
        if time.monotonic() - cache.t >= self.cfg.axis_max_age_s:
            tokens = self.ocr.read(bgr, lay.axis)
            # a etiqueta do preço atual é lida à parte, a cada quadro
            tokens = [tk for tk in tokens if _saturated_fraction(bgr, tk.box) < 0.25]
            labels = axis_labels(tokens)
            if labels:
                lh = float(np.median([tk.box.h for tk in tokens])) if tokens else 10.0
                prof = grid_profile(bgr, lay.chart.x, lay.axis.x, lay.axis.y, lay.axis.y2)
                labels = snap_to_grid(labels, prof, lh)
            cache.tokens = tokens
            cache.t = time.monotonic()
            cache.labels = labels
        labels = cache.labels
        scale = fit_scale(labels, tol_px=self.cfg.axis_tol_px, min_inliers=self.cfg.min_axis_labels)
        if scale is None:
            reading.reasons.append(f"eixo de preço ilegível ({len(labels)} rótulos lidos)")
            self._axis_cache = _OcrCache()  # tenta de novo no próximo quadro
        elif scale.residual_px > self.cfg.max_residual_px:
            reading.reasons.append(f"rótulos do eixo inconsistentes ({scale.residual_px:.1f}px)")
        return scale

    def _current_price_label(self, bgr: np.ndarray, lay: Layout) -> Optional[tuple[float, float, int]]:
        """Etiqueta colorida do preço atual no eixo: (y do centro, preço)."""
        assert lay.axis is not None
        for box in _colored_boxes(bgr, lay.axis)[:2]:
            toks = self._ocr_cached(self._label_cache, bgr, box)
            for tk in toks:
                p = parse_number(tk.text)
                if p is not None:
                    return (box.cy, p[0], p[1])
        return None

    def _apply_price_label(self, reading: ChartReading, scale: PriceScale,
                           label: tuple[float, float, int], bgr: np.ndarray,
                           det: CandleDetection) -> None:
        """Remede a vela atual a partir da etiqueta do preço (fechamento exato).

        A vela atual tem um brilho por cima e às vezes acabou de abrir (sem
        corpo ainda). Procura primeiro uma vela nova um passo à direita da
        última detectada; se não houver, remede a última. Se nenhuma encostar
        no preço da etiqueta, a leitura fica incerta.
        """
        price, dec = label[1], label[2]
        if dec > scale.decimals:
            scale.decimals = dec  # a etiqueta tem mais casas que a grade (ex.: 1.107005)
        lay = self.layout
        assert lay is not None and lay.axis is not None
        y = scale.y_of(price)
        pcs = reading.pixel_candles
        last = pcs[-1]
        y_lo, y_hi = lay.axis.y, lay.axis.y2
        max_h = self.cfg.max_candle_height_ratio * max(pc.low_y - pc.high_y + 1 for pc in pcs[-21:])
        new = measure_forming(bgr, last.x_center + det.pitch, det.body_width, y, y_lo, y_hi, max_h)
        replace = False
        if new is None:
            new = measure_forming(bgr, last.x_center, det.body_width, y, y_lo, y_hi, max_h)
            replace = True
        if new is None:
            reading.ok = False
            reading.reasons.append(f"nenhuma vela encosta no preço atual da etiqueta ({price})")
            self._axis_cache = _OcrCache()  # a escala pode ter mudado: relê o eixo
            return

        def p(row: int) -> float:
            return round(scale.price(row + 0.5), scale.decimals)

        op = p(new.open_y) if new.body_certain else price
        candle = Candle(open=op, high=max(p(new.high_y), price, op), low=min(p(new.low_y), price, op),
                        close=price, color=new.color, forming=True, x_center=new.x_center)
        if replace:
            pcs[-1] = new
            reading.candles[-1] = candle
        else:
            reading.candles[-1].forming = False
            pcs.append(new)
            reading.candles.append(candle)
        reading.current_price = price
