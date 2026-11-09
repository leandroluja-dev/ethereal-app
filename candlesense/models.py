"""Estruturas de dados compartilhadas pela leitura do gráfico."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Box:
    """Retângulo em pixels da captura (x, y no canto superior esquerdo)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    def expand(self, dx: int, dy: int, limit_w: int, limit_h: int) -> "Box":
        x = max(0, self.x - dx)
        y = max(0, self.y - dy)
        x2 = min(limit_w, self.x2 + dx)
        y2 = min(limit_h, self.y2 + dy)
        return Box(x, y, max(0, x2 - x), max(0, y2 - y))


@dataclass(frozen=True)
class OcrToken:
    """Texto reconhecido e sua caixa (coordenadas da captura inteira)."""

    text: str
    box: Box
    score: float = 1.0


@dataclass
class PixelCandle:
    """Vela medida em pixels (y cresce para baixo)."""

    x_center: float
    body_left: int
    body_right: int
    high_y: int
    low_y: int
    body_top_y: int
    body_bottom_y: int
    color: str  # "verde" | "vermelha"
    body_certain: bool = True  # False: corpo encoberto/indistinguível do pavio

    @property
    def body_width(self) -> int:
        return self.body_right - self.body_left + 1

    @property
    def open_y(self) -> int:
        return self.body_bottom_y if self.color == "verde" else self.body_top_y

    @property
    def close_y(self) -> int:
        return self.body_top_y if self.color == "verde" else self.body_bottom_y


@dataclass
class Candle:
    """Vela convertida em preço."""

    open: float
    high: float
    low: float
    close: float
    color: str
    forming: bool = False
    x_center: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.open, self.high, self.low, self.close)


@dataclass
class PriceScale:
    """Mapeamento linear y -> preço: preço = slope * y + intercept."""

    slope: float
    intercept: float
    inliers: int
    residual_px: float
    decimals: int
    labels: list[tuple[float, float]] = field(default_factory=list)  # (y, preço)

    def price(self, y: float) -> float:
        return self.slope * y + self.intercept

    def y_of(self, price: float) -> float:
        return (price - self.intercept) / self.slope

    @property
    def price_per_px(self) -> float:
        return abs(self.slope)


@dataclass
class Layout:
    """Regiões do gráfico detectadas numa captura."""

    width: int
    height: int
    chart: Box
    candle_pitch: float
    candle_width: int
    axis: Optional[Box] = None
    timer: Optional[Box] = None
    payout: Optional[Box] = None
    green_hsv: Optional[tuple[int, int, int]] = None
    red_hsv: Optional[tuple[int, int, int]] = None


@dataclass
class ChartReading:
    """Resultado de uma leitura. Se ``ok`` for falso, nada deve ser sinalizado."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    candles: list[Candle] = field(default_factory=list)
    pixel_candles: list[PixelCandle] = field(default_factory=list)
    scale: Optional[PriceScale] = None
    layout: Optional[Layout] = None
    current_price: Optional[float] = None
    timer_seconds: Optional[int] = None
    payout_pct: Optional[int] = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "ok" if self.ok else "leitura incerta"

    def summary(self) -> str:
        parts = [f"status={self.status}"]
        if self.candles:
            last = self.candles[-1]
            parts.append(f"velas={len(self.candles)}")
            parts.append(
                f"atual O={last.open:.{self._dec()}f} H={last.high:.{self._dec()}f} "
                f"L={last.low:.{self._dec()}f} C={last.close:.{self._dec()}f}"
            )
        if self.timer_seconds is not None:
            parts.append(f"cronômetro={self.timer_seconds}s")
        if self.payout_pct is not None:
            parts.append(f"payout={self.payout_pct}%")
        if self.timings_ms:
            parts.append(f"tempo={sum(self.timings_ms.values()):.0f}ms")
        if self.reasons:
            parts.append("motivos: " + "; ".join(self.reasons))
        if self.warnings:
            parts.append("avisos: " + "; ".join(self.warnings))
        return " | ".join(parts)

    def _dec(self) -> int:
        return self.scale.decimals if self.scale else 5
