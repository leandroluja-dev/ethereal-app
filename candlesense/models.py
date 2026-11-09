"""Estruturas de dados compartilhadas pela leitura do gráfico."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def fmt_duration(seconds: float) -> str:
    """300 -> '5 min', 30 -> '30 s', 3600 -> '1 h'."""
    s = int(round(seconds))
    if s % 3600 == 0 and s >= 3600:
        return f"{s // 3600} h"
    if s % 60 == 0 and s >= 60:
        return f"{s // 60} min"
    if s > 60:
        return f"{s // 60}:{s % 60:02d} min"
    return f"{s} s"


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
    purchase: Optional[Box] = None  # "PURCHASE TIME mm:ss" (prazo para comprar)
    expiry: Optional[Box] = None  # expiração escolhida na IQ (HH:MM)
    clock: Optional[Box] = None  # "CURRENT TIME ... HH:MM:SS"
    time_axis: Optional[Box] = None  # faixa dos rótulos de hora abaixo do gráfico
    timeframe_s: Optional[int] = None  # tempo de cada vela, em segundos
    asset: Optional[str] = None  # ativo do gráfico, ex.: "EUR/USD (OTC)"
    balance: Optional[Box] = None  # saldo no topo
    invest: Optional[Box] = None  # valor da entrada ("Invest")
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
    purchase_seconds: Optional[int] = None  # quanto falta para fechar a compra
    timeframe_s: Optional[int] = None  # tempo de cada vela (60 = 1 min)
    expiry_label: Optional[str] = None  # expiração escolhida na IQ, ex.: "18:16"
    expiry_seconds: Optional[int] = None  # quanto falta até essa expiração
    balance: Optional[float] = None  # saldo mostrado na IQ
    account: Optional[str] = None  # "real" (saldo verde) | "treino" (saldo laranja)
    invest_amount: Optional[float] = None  # valor da entrada escolhido na IQ
    asset: Optional[str] = None
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
        if self.timeframe_s is not None:
            parts.append(f"vela={fmt_duration(self.timeframe_s)}")
        if self.purchase_seconds is not None:
            parts.append(f"compra={self.purchase_seconds}s")
        if self.expiry_label is not None:
            parts.append(f"expiração={self.expiry_label}")
        if self.timings_ms:
            parts.append(f"tempo={sum(self.timings_ms.values()):.0f}ms")
        if self.reasons:
            parts.append("motivos: " + "; ".join(self.reasons))
        if self.warnings:
            parts.append("avisos: " + "; ".join(self.warnings))
        return " | ".join(parts)

    def _dec(self) -> int:
        return self.scale.decimals if self.scale else 5
