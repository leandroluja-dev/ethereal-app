"""Modo debug: desenha tudo o que foi detectado por cima da captura."""

from __future__ import annotations

import cv2
import numpy as np

from ..models import Box, ChartReading

CYAN = (255, 220, 0)
YELLOW = (0, 230, 255)
MAGENTA = (255, 0, 255)
ORANGE = (0, 150, 255)
WHITE = (255, 255, 255)
RED = (60, 60, 255)
GREEN = (80, 220, 80)


def _rect(img: np.ndarray, b: Box, color, label: str = "", th: int = 1) -> None:
    cv2.rectangle(img, (b.x, b.y), (b.x2 - 1, b.y2 - 1), color, th)
    if label:
        cv2.putText(img, label, (b.x, max(12, b.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw(bgr: np.ndarray, r: ChartReading) -> np.ndarray:
    out = bgr.copy()
    if out.ndim == 3 and out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
    lay = r.layout
    if lay is not None:
        _rect(out, lay.chart, CYAN, "grafico")
        if lay.axis:
            _rect(out, lay.axis, YELLOW, "eixo de preco")
        if lay.timer:
            _rect(out, lay.timer, MAGENTA, f"cronometro {r.timer_seconds}s" if r.timer_seconds is not None else "cronometro ?")
        if lay.payout:
            _rect(out, lay.payout, ORANGE, f"payout {r.payout_pct}%" if r.payout_pct is not None else "payout ?")

    # velas: retângulo do corpo e linha do pavio desenhados ao lado (não por cima)
    for i, pc in enumerate(r.pixel_candles):
        col = YELLOW if i == len(r.pixel_candles) - 1 else (WHITE if pc.body_certain else RED)
        cv2.rectangle(out, (pc.body_left - 1, pc.body_top_y - 1), (pc.body_right + 1, pc.body_bottom_y + 1), col, 1)
        x = int(round(pc.x_center))
        cv2.line(out, (x, pc.high_y), (x, pc.body_top_y - 1), col, 1)
        cv2.line(out, (x, pc.body_bottom_y + 1), (x, pc.low_y), col, 1)

    # rótulos do eixo usados no ajuste de escala
    if r.scale is not None and lay is not None:
        x1 = lay.chart.x
        x2 = lay.axis.x if lay.axis else lay.chart.x2
        for y, p in r.scale.labels:
            yy = int(round(y - 0.5))
            cv2.line(out, (x2 - 40, yy), (x2, yy), GREEN, 1)
        if r.candles:
            last = r.candles[-1]
            y = int(round(r.scale.y_of(last.close) - 0.5))
            cv2.line(out, (x1, y), (x2, y), YELLOW, 1)

    # resumo
    lines = [f"CandleSense debug - {r.status.upper()}"]
    if r.candles:
        d = r.scale.decimals if r.scale else 5
        c = r.candles[-1]
        lines.append(f"velas: {len(r.candles)}  atual O {c.open:.{d}f} H {c.high:.{d}f} L {c.low:.{d}f} C {c.close:.{d}f}")
    if r.scale is not None:
        lines.append(f"eixo: {r.scale.inliers} rotulos, erro max {r.scale.residual_px:.1f}px, {r.scale.price_per_px:.3g}/px")
    if r.timings_ms:
        lines.append("tempo: " + ", ".join(f"{k} {v:.0f}ms" for k, v in r.timings_ms.items()))
    for reason in r.reasons:
        lines.append("! " + reason)
    for w in r.warnings:
        lines.append("aviso: " + w)
    y = 22
    pad_w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for s in lines) + 16
    cv2.rectangle(out, (4, 4), (4 + pad_w, 10 + 20 * len(lines)), (0, 0, 0), -1)
    for s in lines:
        color = GREEN if r.ok and not s.startswith("!") else (RED if s.startswith("!") or not r.ok else WHITE)
        cv2.putText(out, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 20
    return out
