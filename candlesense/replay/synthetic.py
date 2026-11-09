"""Gerador de telas sintéticas no estilo da IQ Option, com gabarito (valores verdadeiros).

Usado pelos testes automatizados (comparar o que a visão lê com o que foi desenhado)
e pelo modo demonstração (`python -m candlesense demo`), que roda o app sem a IQ Option.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..common.geometry import Rect

DARK = {
    "window": (26, 19, 17),
    "topbar": (24, 18, 16),
    "tab_active": (58, 44, 40),
    "panel": (30, 23, 21),
    "chart": (48, 35, 31),
    "grid": (60, 47, 43),
    "text": (205, 200, 195),
    "muted": (140, 132, 128),
    "bull": (108, 196, 46),
    "bear": (75, 75, 240),
    "price_line": (175, 175, 175),
    "ma": (0, 170, 255),
}
LIGHT = {
    "window": (226, 224, 222),
    "topbar": (214, 212, 210),
    "tab_active": (250, 250, 250),
    "panel": (222, 220, 218),
    "chart": (252, 251, 250),
    "grid": (232, 230, 228),
    "text": (40, 40, 45),
    "muted": (110, 110, 115),
    "bull": (90, 175, 35),
    "bear": (60, 60, 220),
    "price_line": (120, 120, 120),
    "ma": (0, 140, 230),
}


@dataclass
class Ohlc:
    open: float
    high: float
    low: float
    close: float


@dataclass
class SynthSpec:
    width: int = 1920
    height: int = 1040
    theme: str = "dark"
    n_candles: int = 60
    body_width: int = 9
    gap: int = 4
    decimals: int = 5
    start_price: float = 1.08450
    volatility: float = 0.00012
    timeframe_min: int = 1
    asset: str = "EUR/USD (OTC)"
    other_asset: str = "GBP/JPY"
    payout: int = 87
    other_payout: int = 82
    timer_s: int = 34
    end_time_min: int = 15 * 60 + 30  # minuto do dia da vela em formação
    overlays: bool = True
    seed: int = 7


@dataclass
class SynthTruth:
    candles: list[Ohlc]
    centers_x: list[float]
    price_a: float  # preço = a + b * y
    price_b: float
    plot: Rect
    axis: Rect
    chart: Rect
    current_price: float
    timer_s: int
    asset: str
    payout: int
    timeframe_min: int
    body_width: int
    pitch: int
    extra: dict = field(default_factory=dict)

    def y_of(self, price: float) -> float:
        return (price - self.price_a) / self.price_b


def random_walk(n: int, start: float, vol: float, seed: int) -> list[Ohlc]:
    rng = np.random.default_rng(seed)
    out: list[Ohlc] = []
    price = start
    for _ in range(n):
        o = price
        steps = rng.normal(0, vol / 3, 12).cumsum() + o
        c = float(steps[-1])
        h = max(o, c, float(steps.max())) + abs(rng.normal(0, vol / 4))
        lo = min(o, c, float(steps.min())) - abs(rng.normal(0, vol / 4))
        out.append(Ohlc(o, h, lo, c))
        price = c
    return out


def _nice_step(span: float, target: int = 8) -> float:
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _text(img: np.ndarray, s: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.45,
          thick: int = 1) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(spec: SynthSpec, candles: list[Ohlc] | None = None) -> tuple[np.ndarray, SynthTruth]:
    pal = DARK if spec.theme == "dark" else LIGHT
    W, H = spec.width, spec.height
    img = np.full((H, W, 3), pal["window"], np.uint8)
    if candles is None:
        candles = random_walk(spec.n_candles, spec.start_price, spec.volatility, spec.seed)

    side_w, top_h, right_w, axis_w, time_h = 64, 60, 200, 92, 30
    chart = Rect(side_w, top_h, W - side_w - right_w, H - top_h)
    img[chart.y : chart.bottom, chart.x : chart.right] = pal["chart"]
    img[0:top_h, :] = pal["topbar"]
    img[top_h:, chart.right :] = pal["panel"]
    axis = Rect(chart.right - axis_w, chart.y, axis_w, chart.h - time_h)
    plot = Rect(chart.x, chart.y, chart.w - axis_w, chart.h - time_h)

    # Barra superior: abas de ativos (a primeira é a ativa).
    cv2.rectangle(img, (side_w + 16, 8), (side_w + 236, top_h - 6), pal["tab_active"], -1)
    _text(img, spec.asset, (side_w + 30, 32), pal["text"], 0.55)
    _text(img, f"{spec.payout}%", (side_w + 30, 50), pal["muted"], 0.4)
    _text(img, spec.other_asset, (side_w + 270, 32), pal["muted"], 0.55)
    _text(img, f"{spec.other_payout}%", (side_w + 270, 50), pal["muted"], 0.4)
    _text(img, "IQ Option", (12, 36), pal["text"], 0.5)

    # Painel direito de operação.
    px = chart.right + 18
    _text(img, "Valor", (px, top_h + 40), pal["muted"], 0.45)
    _text(img, "$ 10", (px, top_h + 68), pal["text"], 0.6)
    _text(img, "Expiracao", (px, top_h + 110), pal["muted"], 0.45)
    end_min = spec.end_time_min + spec.timeframe_min
    _text(img, f"{end_min // 60 % 24:02d}:{end_min % 60:02d}", (px, top_h + 138), pal["text"], 0.6)
    _text(img, "Lucro", (px, top_h + 190), pal["muted"], 0.45)
    _text(img, f"+{spec.payout}%", (px, top_h + 228), pal["bull"], 0.9, 2)
    cv2.rectangle(img, (px, top_h + 300), (W - 18, top_h + 370), pal["bull"], -1)
    _text(img, "ALTA", (px + 50, top_h + 343), (255, 255, 255), 0.7, 2)
    cv2.rectangle(img, (px, top_h + 390), (W - 18, top_h + 460), pal["bear"], -1)
    _text(img, "BAIXA", (px + 45, top_h + 433), (255, 255, 255), 0.7, 2)

    # Mapeamento preço -> pixel.
    pmax = max(c.high for c in candles)
    pmin = min(c.low for c in candles)
    span = pmax - pmin
    pmax += span * 0.12
    pmin -= span * 0.12
    y_top, y_bot = plot.y + 10, plot.bottom - 10
    scale = (y_bot - y_top) / (pmax - pmin)
    b = -1.0 / scale
    a = pmax - b * y_top

    def y_of(p: float) -> float:
        return y_top + (pmax - p) * scale

    # Grade e rótulos do eixo de preço.
    step = _nice_step(pmax - pmin)
    first = math.ceil(pmin / step) * step
    v = first
    y_cur = y_of(candles[-1].close)
    while v <= pmax:
        y = int(round(y_of(v)))
        cv2.line(img, (plot.x, y), (plot.right, y), pal["grid"], 1)
        # como na plataforma, rótulos sob a etiqueta do preço atual/cronômetro ficam ocultos
        if not (spec.overlays and -16 <= y - y_cur <= 40):
            _text(img, f"{v:.{spec.decimals}f}", (axis.x + 8, y + 5), pal["muted"], 0.42)
        v += step

    pitch = spec.body_width + spec.gap
    last_x = plot.right - 140
    n = len(candles)
    centers = [last_x - (n - 1 - i) * pitch for i in range(n)]

    # Eixo de tempo.
    for i, cx in enumerate(centers):
        minute = spec.end_time_min - (n - 1 - i) * spec.timeframe_min
        if minute % (10 * spec.timeframe_min) == 0:
            cv2.line(img, (int(cx), plot.y), (int(cx), plot.bottom), pal["grid"], 1)
            _text(img, f"{minute // 60 % 24:02d}:{minute % 60:02d}", (int(cx) - 20, plot.bottom + 20), pal["muted"], 0.42)

    # Velas.
    half = spec.body_width // 2
    for c, cx in zip(candles, centers):
        col = pal["bull"] if c.close >= c.open else pal["bear"]
        x0 = int(cx) - half
        x1 = x0 + spec.body_width - 1
        yb0 = int(round(y_of(max(c.open, c.close))))
        yb1 = int(round(y_of(min(c.open, c.close))))
        cv2.line(img, (int(cx), int(round(y_of(c.high)))), (int(cx), int(round(y_of(c.low)))), col, 1)
        cv2.rectangle(img, (x0, yb0), (x1, max(yb0, yb1)), col, -1)

    current = candles[-1].close
    extra: dict = {}
    if spec.overlays:
        # Média móvel (linha laranja sobre as velas).
        closes = np.array([c.close for c in candles])
        ma = np.convolve(closes, np.ones(9) / 9, mode="valid")
        pts = np.array([[int(centers[i + 8]), int(round(y_of(m)))] for i, m in enumerate(ma)], np.int32)
        cv2.polylines(img, [pts], False, pal["ma"], 1, cv2.LINE_AA)
        # Marcador de operação: círculo verde com seta + linha horizontal até a expiração.
        k = n - 15
        my = int(round(y_of(candles[k].close)))
        cv2.line(img, (int(centers[k]), my), (int(centers[k]) + 12 * pitch, my), pal["bull"], 1)
        cv2.circle(img, (int(centers[k]), my), 9, pal["bull"], -1)
        cv2.arrowedLine(img, (int(centers[k]), my + 5), (int(centers[k]), my - 5), (255, 255, 255), 1, tipLength=0.5)
        extra["marker_x"] = centers[k]
        # Linha do preço atual (tracejada) e etiqueta no eixo.
        yc = int(round(y_of(current)))
        for x in range(plot.x, plot.right, 8):
            cv2.line(img, (x, yc), (x + 4, yc), pal["price_line"], 1)
        tag_col = pal["bull"] if candles[-1].close >= candles[-1].open else pal["bear"]
        cv2.rectangle(img, (axis.x + 2, yc - 10), (axis.right - 4, yc + 10), tag_col, -1)
        _text(img, f"{current:.{spec.decimals}f}", (axis.x + 8, yc + 5), (255, 255, 255), 0.42)
        _text(img, f"{spec.timer_s // 60:02d}:{spec.timer_s % 60:02d}", (axis.x + 22, yc + 30), pal["text"], 0.42)

    truth = SynthTruth(candles, [float(c) for c in centers], a, b, plot, axis, chart, current, spec.timer_s,
                       spec.asset, spec.payout, spec.timeframe_min, spec.body_width, pitch, extra)
    return img, truth


class LiveSynthetic:
    """Mercado sintético em tempo real (velas alinhadas ao relógio), para o modo demonstração."""

    def __init__(self, spec: SynthSpec | None = None, history: int = 70) -> None:
        self.spec = spec or SynthSpec()
        self._rng = np.random.default_rng(self.spec.seed)
        self._tf_s = self.spec.timeframe_min * 60
        self._closed = random_walk(history, self.spec.start_price, self.spec.volatility, self.spec.seed)
        self._bucket = int(time.time() // self._tf_s)
        self._forming = self._new_candle(self._closed[-1].close)

    def _new_candle(self, price: float) -> Ohlc:
        return Ohlc(price, price, price, price)

    def frame(self) -> tuple[np.ndarray, SynthTruth]:
        now = time.time()
        bucket = int(now // self._tf_s)
        while bucket > self._bucket:
            self._closed.append(self._forming)
            self._closed = self._closed[-120:]
            self._forming = self._new_candle(self._forming.close)
            self._bucket += 1
        f = self._forming
        f.close += float(self._rng.normal(0, self.spec.volatility / 6))
        f.high = max(f.high, f.close)
        f.low = min(f.low, f.close)
        remaining = int(self._tf_s - (now % self._tf_s))
        lt = time.localtime(bucket * self._tf_s)
        spec = SynthSpec(**{**self.spec.__dict__, "timer_s": remaining,
                            "end_time_min": lt.tm_hour * 60 + lt.tm_min})
        visible = self._closed[-(spec.n_candles - 1):] + [Ohlc(f.open, f.high, f.low, f.close)]
        return render(spec, visible)
