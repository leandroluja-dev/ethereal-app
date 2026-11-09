"""Orquestra a visão quadro a quadro: layout -> cores -> velas -> eixo -> payout/ativo -> leitura final.

Regra de ouro: se alguma parte essencial falhar, a leitura sai com status "incerta" e nada é inventado.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..capture.frame import Frame
from ..capture.layout import ChartLayout, detect_layout
from ..config import Config
from .candles import CandleDetection, PixelCandle, detect_candles
from .colors import CandleColors, discover_colors
from .hud import HudReader
from .ocr import OcrEngine
from .price_axis import AxisReading, PriceAxisReader, TimeAxisReader, TimeAxisReading, snap_timeframe
from .state import ChangeDetector

log = logging.getLogger(__name__)


class Status(str, Enum):
    NO_WINDOW = "NO_WINDOW"
    MINIMIZED = "MINIMIZED"
    DETECTING = "DETECTING"
    NO_CHART = "NO_CHART"
    ZOOM = "ZOOM"
    UNCERTAIN = "UNCERTAIN"
    OK = "OK"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


STATUS_TEXT = {
    Status.NO_WINDOW: "Abra a IQ Option",
    Status.MINIMIZED: "A janela da IQ Option está minimizada",
    Status.DETECTING: "Procurando o gráfico…",
    Status.NO_CHART: "Gráfico de velas não encontrado",
    Status.ZOOM: "Aumente o zoom do gráfico",
    Status.UNCERTAIN: "Leitura incerta",
    Status.OK: "Leitura OK",
    Status.PAUSED: "Pausado",
    Status.ERROR: "Erro na leitura",
}


@dataclass(frozen=True)
class Candle:
    open: float
    high: float
    low: float
    close: float
    bull: bool
    slot: int
    occluded: bool
    partial: bool
    px: PixelCandle


@dataclass
class Reading:
    ts: float
    status: Status
    message: str
    window_title: str | None = None
    layout: ChartLayout | None = None
    colors: CandleColors | None = None
    detection: CandleDetection | None = None
    axis: AxisReading | None = None
    time_axis: TimeAxisReading | None = None
    candles: list[Candle] = field(default_factory=list)  # fechadas + a em formação (última)
    current_price: float | None = None
    current_price_source: str | None = None
    asset: str | None = None
    payout: int | None = None
    timeframe_min: int | None = None
    timeframe_source: str | None = None
    timer_s: float | None = None
    timer_source: str | None = None
    timer_ocr_s: int | None = None
    confidence: float = 0.0
    conf_parts: dict[str, float] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    seconds_since_reset: float | None = None

    @property
    def forming(self) -> Candle | None:
        return self.candles[-1] if self.candles else None

    @property
    def closed(self) -> list[Candle]:
        return self.candles[:-1]

    def summary(self) -> dict:
        """Resumo serializável (JSON) para registro, testes e depuração."""
        f = self.forming
        return {
            "ts": self.ts,
            "status": self.status.value,
            "message": self.message,
            "asset": self.asset,
            "payout": self.payout,
            "timeframe_min": self.timeframe_min,
            "timeframe_source": self.timeframe_source,
            "timer_s": self.timer_s,
            "timer_source": self.timer_source,
            "current_price": self.current_price,
            "current_price_source": self.current_price_source,
            "n_candles": len(self.candles),
            "forming": None if f is None else {"open": f.open, "high": f.high, "low": f.low, "close": f.close},
            "last_closed": None if len(self.candles) < 2 else {
                k: getattr(self.candles[-2], k) for k in ("open", "high", "low", "close")},
            "confidence": round(self.confidence, 3),
            "conf_parts": {k: round(v, 3) for k, v in self.conf_parts.items()},
            "events": self.events,
            "notes": self.notes,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "plot_rect": self.layout.plot.to_list() if self.layout else None,
            "candles": [[c.open, c.high, c.low, c.close, int(c.bull), c.slot, int(c.occluded)] for c in self.candles],
        }


class FrameAnalyzer:
    def __init__(self, cfg: Config, ocr: OcrEngine) -> None:
        self.cfg = cfg
        self.ocr = ocr
        self.layout: ChartLayout | None = None
        self._layout_attempt = -1e9
        self._fail_count = 0
        self._colors: CandleColors | None = None
        self._colors_ts = -1e9
        self._axis = PriceAxisReader(ocr, cfg.vision.axis_change_threshold)
        self._time = TimeAxisReader(ocr)
        self._hud = HudReader(ocr, cfg.vision.hud_ocr_interval_s)
        self._changes = ChangeDetector()
        self._timeframe: int | None = None
        self._force_layout = False

    def invalidate_layout(self) -> None:
        self._force_layout = True

    def _reading(self, frame: Frame, status: Status, extra: str | None = None, **kw) -> Reading:  # type: ignore[no-untyped-def]
        msg = STATUS_TEXT[status] + (f": {extra}" if extra else "")
        return Reading(ts=frame.ts, status=status, message=msg,
                       window_title=frame.window.title if frame.window else None, **kw)

    def analyze(self, frame: Frame) -> Reading:
        try:
            return self._analyze(frame)
        except Exception as e:  # noqa: BLE001 - a análise nunca derruba o app
            log.exception("falha na análise do quadro")
            return self._reading(frame, Status.ERROR, type(e).__name__)

    def _analyze(self, frame: Frame) -> Reading:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        if frame.image is None:
            st = Status.MINIMIZED if frame.status_hint == "minimized" else Status.NO_WINDOW
            extra = "captura ao vivo só funciona no Windows" if frame.status_hint == "unsupported" else None
            return self._reading(frame, st, extra)
        img = frame.image
        now = frame.ts
        h, w = img.shape[:2]

        # 1. Layout (lento: só quando necessário).
        need = self.layout is None or self.layout.frame_size != (w, h) or self._force_layout
        if need:
            if not self._force_layout and self.layout is None and now - self._layout_attempt < self.cfg.vision.layout_retry_s:
                return self._reading(frame, Status.NO_CHART, "verifique se o gráfico de velas está visível")
            self._force_layout = False
            self._layout_attempt = now
            t = time.perf_counter()
            self.layout = detect_layout(img, self.ocr, self.cfg.vision)
            timings["layout"] = (time.perf_counter() - t) * 1000
            self._colors = None
            self._axis.reset()
            self._time.reset()
            self._timeframe = None
            self._fail_count = 0
            if self.layout is not None:
                self._hud.reset(self.layout.payout_value, self.layout.asset_text, now)
        layout = self.layout
        if layout is None:
            return self._reading(frame, Status.NO_CHART, "verifique se o gráfico de velas está visível",
                                 timings_ms=timings)

        # 2. Cores das velas.
        plot_img = layout.plot.crop(img)
        valid = layout.plot.crop(frame.valid_mask) if frame.valid_mask is not None else None
        if self._colors is None or now - self._colors_ts > self.cfg.vision.color_refresh_s:
            t = time.perf_counter()
            found = discover_colors(plot_img, self.cfg.vision)
            timings["colors"] = (time.perf_counter() - t) * 1000
            if found is not None:
                self._colors, self._colors_ts = found, now
        colors = self._colors
        if colors is None:
            self._note_failure(now)
            return self._reading(frame, Status.UNCERTAIN, "cores das velas não identificadas", layout=layout,
                                 timings_ms=timings)

        # 3. Velas.
        t = time.perf_counter()
        det = detect_candles(plot_img, colors, self.cfg.vision, offset=(layout.plot.x, layout.plot.y),
                             valid_mask=valid)
        timings["candles"] = (time.perf_counter() - t) * 1000
        if len(det.candles) < 3:
            self._note_failure(now)
            return self._reading(frame, Status.UNCERTAIN, "velas não reconhecidas", layout=layout, colors=colors,
                                 detection=det, timings_ms=timings)
        self._fail_count = 0

        # 4. Eixo de preço (rótulos relidos só quando o eixo muda).
        t = time.perf_counter()
        axis = self._axis.read(img, layout.price_axis)
        timings["axis"] = (time.perf_counter() - t) * 1000

        # 5. Eixo de tempo -> timeframe.
        ta: TimeAxisReading | None = None
        tf_source = None
        if layout.time_axis is not None:
            t = time.perf_counter()
            ta = self._time.read(img, layout.time_axis)
            timings["time_axis"] = (time.perf_counter() - t) * 1000
            if ta.minutes_per_px is not None and det.pitch > 0:
                tf = snap_timeframe(ta.minutes_per_px * det.pitch)
                if tf is not None:
                    self._timeframe, tf_source = tf, "eixo de tempo"
        if tf_source is None and self.cfg.vision.assume_timeframe_min:
            self._timeframe, tf_source = int(self.cfg.vision.assume_timeframe_min), "config"
        elif tf_source is None and self._timeframe is not None:
            tf_source = "última leitura"
        timeframe = self._timeframe

        # 6. Payout e ativo.
        t = time.perf_counter()
        hud = self._hud.update(img, layout.payout_box, layout.asset_box, now)
        timings["hud"] = (time.perf_counter() - t) * 1000
        payout = hud.payout if self._hud.payout_fresh(now) else None
        asset = hud.asset if self._hud.asset_fresh(now) else None

        # 7. Velas em preço.
        mapping = axis.mapping
        candles: list[Candle] = []
        if mapping is not None:
            for pc in det.candles:
                candles.append(Candle(mapping.price(pc.open_y), mapping.price(pc.wick_top),
                                      mapping.price(pc.wick_bottom), mapping.price(pc.close_y), pc.bull,
                                      pc.slot, pc.occluded, pc.partial, pc))

        # 8. Preço atual (etiqueta do eixo) conferido com o fechamento da vela em formação.
        current = None
        source = None
        price_check = 0.6
        notes = list(layout.notes) + det.notes + axis.notes
        if candles:
            forming_close = candles[-1].close
            current, source = forming_close, "vela"
            if axis.tag_price is not None and mapping is not None:
                diff_px = abs(mapping.y(axis.tag_price) - det.candles[-1].close_y)
                if diff_px <= 3.0:
                    current, source, price_check = axis.tag_price, "etiqueta", 1.0
                else:
                    price_check = max(0.0, 1.0 - (diff_px - 3.0) / 10.0)
                    notes.append(f"etiqueta do preço difere da vela ({diff_px:.0f}px)")

        # 9. Cronômetro da vela.
        timer, timer_src = self._timer(now, timeframe, axis.timer_s)

        # 10. Troca de ativo / timeframe.
        events = self._changes.update(now, asset, timeframe)
        for e in events:
            log.info(e)

        # 11. Confiança.
        parts = {
            "layout": layout.confidence,
            "cores": colors.confidence,
            "velas": det.confidence,
            "eixo": mapping.confidence if mapping else 0.0,
            "preço": price_check,
            "payout": 1.0 if payout is not None else 0.5,
            "ativo": 1.0 if asset is not None else 0.7,
        }
        core = min(parts["velas"], parts["eixo"], parts["cores"])
        overall = core * (0.8 + 0.2 * price_check) * (0.9 + 0.1 * parts["layout"])
        if det.too_small:
            status, extra = Status.ZOOM, "velas estreitas demais para leitura confiável"
        elif mapping is None:
            status, extra = Status.UNCERTAIN, "eixo de preço ilegível"
        elif overall < self.cfg.vision.min_confidence:
            status, extra = Status.UNCERTAIN, None
        else:
            status, extra = Status.OK, None
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return self._reading(
            frame, status, extra, layout=layout, colors=colors, detection=det, axis=axis, time_axis=ta,
            candles=candles, current_price=current, current_price_source=source, asset=asset, payout=payout,
            timeframe_min=timeframe, timeframe_source=tf_source, timer_s=timer, timer_source=timer_src,
            timer_ocr_s=axis.timer_s, confidence=overall, conf_parts=parts, events=events, notes=notes,
            timings_ms=timings, seconds_since_reset=self._changes.seconds_since_reset(now),
        )

    def _timer(self, now: float, timeframe: int | None, ocr_s: int | None) -> tuple[float | None, str | None]:
        mode = self.cfg.vision.timer_source
        clock: float | None = None
        if timeframe:
            period = timeframe * 60
            clock = period - ((now + self.cfg.vision.clock_offset_s) % period)
        plausible_ocr = ocr_s is not None and (timeframe is None or 0 <= ocr_s <= timeframe * 60)
        if mode == "ocr":
            return (float(ocr_s), "ocr") if plausible_ocr else (None, None)
        if mode == "clock":
            return (clock, "relógio") if clock is not None else (None, None)
        if plausible_ocr:
            return float(ocr_s), "ocr"  # type: ignore[arg-type]
        if clock is not None:
            return clock, "relógio"
        return None, None

    def _note_failure(self, now: float) -> None:
        self._fail_count += 1
        if self._fail_count >= 3 and now - self._layout_attempt >= self.cfg.vision.layout_retry_s:
            log.info("leituras falhando: redetectando o layout")
            self._force_layout = True
            self._fail_count = 0


def frame_from_image(img: np.ndarray, ts: float | None = None, source: str = "image") -> Frame:
    return Frame(image=img, ts=time.time() if ts is None else ts, source=source)
