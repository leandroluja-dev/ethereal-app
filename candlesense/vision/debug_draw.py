"""Desenha por cima da captura tudo o que a visão leu, para conferir a olho.

Legenda:
- retângulo ciano: área das velas; amarelo: eixo de preço; magenta: eixo de tempo
- contorno verde/vermelho: corpo de cada vela detectada; linha: pavio
- contorno branco grosso: vela em formação
- X magenta: vela parcialmente coberta por indicador/marcador (valores menos confiáveis)
- rótulos do eixo: verde = usado na regressão, vermelho = descartado
- linhas ciano finas: onde o mapeamento coloca cada preço lido (devem passar no meio dos rótulos)
"""

from __future__ import annotations

import unicodedata

import cv2
import numpy as np

from ..common.geometry import Rect
from .analyzer import Reading, Status

CYAN = (255, 220, 0)
YELLOW = (0, 220, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
GREEN = (80, 220, 80)
RED = (60, 60, 255)
ORANGE = (0, 160, 255)


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.45) -> None:
    text = _ascii(text)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + 3), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _rect(img: np.ndarray, r: Rect | None, color: tuple[int, int, int], thick: int = 1) -> None:
    if r is not None:
        cv2.rectangle(img, (r.x, r.y), (r.right - 1, r.bottom - 1), color, thick)


def draw_overlay(image: np.ndarray, reading: Reading) -> np.ndarray:
    img = image.copy()
    L = reading.layout
    if L is not None:
        _rect(img, L.chart, (120, 120, 120))
        _rect(img, L.plot, CYAN, 2)
        _rect(img, L.price_axis, YELLOW, 2)
        _rect(img, L.time_axis, MAGENTA, 2)
        if L.payout_box is not None:
            _rect(img, L.payout_box, GREEN, 2)
            _label(img, f"payout {reading.payout}%", (L.payout_box.x, L.payout_box.bottom + 16), GREEN)
        if L.asset_box is not None:
            _rect(img, L.asset_box, ORANGE, 2)
            _label(img, f"ativo {reading.asset}", (L.asset_box.x, L.asset_box.bottom + 16), ORANGE)

    det = reading.detection
    if det is not None:
        for c in det.candles:
            col = GREEN if c.bull else RED
            cx = int(round(c.cx))
            cv2.line(img, (cx, c.wick_top), (cx, c.wick_bottom), col, 1)
            cv2.rectangle(img, (c.x_left - 1, c.body_top - 1), (c.x_right + 1, c.body_bottom + 1), col, 1)
            if c.occluded:
                cv2.drawMarker(img, (cx, c.wick_top - 6), MAGENTA, cv2.MARKER_TILTED_CROSS, 8, 1)
        f = det.forming
        if f is not None:
            cv2.rectangle(img, (f.x_left - 3, f.wick_top - 3), (f.x_right + 3, f.wick_bottom + 3), WHITE, 2)

    ax = reading.axis
    if ax is not None and L is not None:
        for lb in ax.labels:
            col = GREEN if lb.inlier else RED
            _rect(img, lb.box, col, 1)
            txt = "?" if lb.value is None else (ax.mapping.fmt(lb.value) if ax.mapping else f"{lb.value:g}")
            _label(img, txt, (max(0, lb.box.x - 80), lb.box.bottom - 2), col, 0.38)
            if ax.mapping is not None and lb.value is not None and lb.inlier:
                y = int(round(ax.mapping.y(lb.value)))
                cv2.line(img, (L.plot.x, y), (L.price_axis.x, y), CYAN, 1)
        if ax.tag_box is not None:
            _rect(img, ax.tag_box, WHITE, 2)
            _label(img, f"atual {ax.tag_text}", (ax.tag_box.x - 120, ax.tag_box.y + 14), WHITE, 0.4)
        if ax.timer_box is not None:
            _rect(img, ax.timer_box, MAGENTA, 1)
    if reading.time_axis is not None:
        for box, _text, minute in reading.time_axis.labels:
            _rect(img, box, GREEN if minute is not None else RED, 1)

    _draw_summary(img, reading)
    return img


def _draw_summary(img: np.ndarray, r: Reading) -> None:
    status_col = {Status.OK: GREEN, Status.UNCERTAIN: ORANGE, Status.ZOOM: ORANGE}.get(r.status, RED)
    lines: list[tuple[str, tuple[int, int, int]]] = [(r.message, status_col)]
    lines.append((f"confianca {r.confidence:.2f}  " + " ".join(f"{k}={v:.2f}" for k, v in r.conf_parts.items()),
                  WHITE))
    tf = f"M{r.timeframe_min}" if r.timeframe_min else "?"
    timer = "?" if r.timer_s is None else f"{int(r.timer_s) // 60:02d}:{int(r.timer_s) % 60:02d} ({r.timer_source})"
    lines.append((f"ativo {r.asset or '?'}  payout {r.payout or '?'}%  {tf}  cronometro {timer}", WHITE))
    if r.axis is not None and r.axis.mapping is not None:
        m = r.axis.mapping
        price = "?" if r.current_price is None else m.fmt(r.current_price)
        lines.append((f"preco atual {price} ({r.current_price_source})  velas {len(r.candles)}", WHITE))
        if len(r.candles) >= 2:
            c = r.candles[-2]
            lines.append((f"ultima fechada O {m.fmt(c.open)} H {m.fmt(c.high)} L {m.fmt(c.low)} C {m.fmt(c.close)}",
                          WHITE))
    lines.append(("tempo " + " ".join(f"{k}={v:.0f}ms" for k, v in r.timings_ms.items()), (180, 180, 180)))
    for n in r.notes[:3]:
        lines.append((n, ORANGE))
    # no canto inferior esquerdo da área das velas (não cobre ativo/payout)
    x0 = r.layout.plot.x + 8 if r.layout else 10
    bottom = r.layout.plot.bottom - 10 if r.layout else img.shape[0] - 10
    y = max(22, bottom - 22 * (len(lines) - 1))
    for text, col in lines:
        _label(img, text, (x0, y), col, 0.5)
        y += 22
