"""Acha as regiões do gráfico numa captura, sem posição fixa.

Roda uma vez (ou quando a janela muda): as velas definem a área do gráfico,
e um OCR da tela inteira localiza o eixo de preço, o cronômetro e o payout.
Depois disso, cada quadro só faz OCR em recortes pequenos.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models import Box, Layout, OcrToken
from .candles import CandleDetection
from .ocr import is_clock_text, parse_number, parse_payout, parse_timer


def _cluster(values: list[float], tol: float) -> list[list[int]]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    groups: list[list[int]] = []
    for i in order:
        if groups and abs(values[i] - values[groups[-1][-1]]) <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def find_axis(tokens: list[OcrToken], det: CandleDetection, img_w: int, img_h: int) -> Optional[Box]:
    """Coluna de números à direita da última vela."""
    cbox = det.box
    if cbox is None:
        return None
    last_x = det.candles[-1].body_right
    band_top = cbox.y - 0.5 * cbox.h
    band_bot = cbox.y2 + 0.5 * cbox.h
    nums = [
        t for t in tokens
        if parse_number(t.text) is not None
        and "." in t.text.replace(",", ".")
        and t.box.x > last_x
        and band_top <= t.box.cy <= band_bot
    ]
    if len(nums) < 3:
        return None
    groups = _cluster([t.box.x2 for t in nums], tol=max(12.0, 0.012 * img_w))
    groups += _cluster([t.box.x for t in nums], tol=max(12.0, 0.012 * img_w))
    best = max(groups, key=len)
    if len(best) < 3:
        return None
    sel = [nums[i] for i in best]
    x1 = min(t.box.x for t in sel)
    x2 = max(t.box.x2 for t in sel)
    pad = max(6, int(0.004 * img_w))
    # Vertical: do primeiro ao último rótulo, mais um espaçamento de cada lado.
    # A área do gráfico não muda quando a escala muda, só os números.
    cys = sorted(t.box.cy for t in sel)
    spacing = float(np.median(np.diff(cys))) if len(cys) >= 2 else 20.0
    # Meio espaçamento: não pode invadir painéis de indicador logo abaixo (RSI etc.).
    y1 = int(max(0, min(t.box.y for t in sel) - 0.5 * spacing))
    y2 = int(min(img_h, max(t.box.y2 for t in sel) + 0.5 * spacing))
    x1 = max(0, x1 - pad)
    x2 = min(img_w, x2 + 3 * pad)
    return Box(x1, y1, x2 - x1, y2 - y1)


def find_timer(tokens: list[OcrToken], det: CandleDetection, axis: Optional[Box],
               img_w: int, img_h: int) -> Optional[Box]:
    """Cronômetro mm:ss perto da vela atual. Rótulos de hora do eixo X ficam de fora."""
    clocks = [t for t in tokens if is_clock_text(t.text) and parse_timer(t.text) is not None]
    if not clocks:
        return None
    rows = _cluster([t.box.cy for t in clocks], tol=max(4.0, 0.004 * img_h))
    in_row = {i for g in rows if len(g) >= 3 for i in g}  # eixo do tempo: vários na mesma linha
    cand = [t for i, t in enumerate(clocks) if i not in in_row]
    if not cand:
        return None
    last = det.candles[-1]
    lx, ly = last.x_center, (last.high_y + last.low_y) / 2
    best = min(cand, key=lambda t: (t.box.cx - lx) ** 2 + (t.box.cy - ly) ** 2)
    return best.box.expand(best.box.w // 2 + 4, best.box.h // 2 + 2, img_w, img_h)


def find_payout(tokens: list[OcrToken], axis: Optional[Box], img_w: int, img_h: int) -> Optional[Box]:
    """Texto '+87%'. Prefere o painel à direita do eixo e a fonte maior."""
    pcts = [t for t in tokens if parse_payout(t.text) is not None]
    if not pcts:
        return None
    right_of = axis.x2 if axis else int(0.7 * img_w)
    right = [t for t in pcts if t.box.x >= right_of - 4]
    pool = right or pcts
    plus = [t for t in pool if "+" in t.text]
    pool = plus or pool
    best = max(pool, key=lambda t: t.box.h)
    return best.box.expand(best.box.w // 2 + 6, best.box.h // 2 + 2, img_w, img_h)


def build_layout(bgr: np.ndarray, det: CandleDetection, tokens: list[OcrToken]) -> Layout:
    h, w = bgr.shape[:2]
    cbox = det.box
    assert cbox is not None
    axis = find_axis(tokens, det, w, h)
    timer = find_timer(tokens, det, axis, w, h)
    payout = find_payout(tokens, axis, w, h)
    chart_x2 = axis.x if axis else min(w, cbox.x2 + int(12 * det.pitch))
    chart = Box(cbox.x, cbox.y, max(1, chart_x2 - cbox.x), cbox.h)
    return Layout(
        width=w, height=h, chart=chart,
        candle_pitch=det.pitch, candle_width=det.body_width,
        axis=axis, timer=timer, payout=payout,
        green_hsv=det.green_hsv, red_hsv=det.red_hsv,
    )
