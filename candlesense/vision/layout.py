"""Acha as regiões do gráfico numa captura, sem posição fixa.

Roda uma vez (ou quando a janela muda): as velas definem a área do gráfico,
e um OCR da tela inteira localiza o eixo de preço, o cronômetro e o payout.
Depois disso, cada quadro só faz OCR em recortes pequenos.
"""

from __future__ import annotations

import re

from typing import Optional

import numpy as np

from ..models import Box, Layout, OcrToken
from .candles import CandleDetection
from .wallet import find_balance, find_invest
from .ocr import (is_clock_text, parse_clock, parse_number, parse_payout, parse_timeframe,
                  parse_timer)


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
    if len(nums) < 2:
        return None
    groups = _cluster([t.box.x2 for t in nums], tol=max(12.0, 0.012 * img_w))
    groups += _cluster([t.box.x for t in nums], tol=max(12.0, 0.012 * img_w))
    best = max(groups, key=len)
    if len(best) < 2:
        return None
    sel = [nums[i] for i in best]
    x1 = min(t.box.x for t in sel)
    x2 = max(t.box.x2 for t in sel)
    pad = max(6, int(0.004 * img_w))
    # Vertical: do primeiro ao último rótulo, mais um espaçamento de cada lado.
    # A área do gráfico não muda quando a escala muda, só os números.
    cys = sorted(t.box.cy for t in sel)
    spacing = float(np.median(np.diff(cys))) if len(cys) >= 2 else 20.0
    if len(sel) < 3:
        # Gráfico baixo: poucos rótulos. A área vai das velas até meia altura além delas.
        x1 = max(0, min(t.box.x for t in sel) - max(6, int(0.004 * img_w)))
        x2 = min(img_w, max(t.box.x2 for t in sel) + 3 * max(6, int(0.004 * img_w)))
        y1 = int(max(0, min(cbox.y - 0.25 * cbox.h, min(t.box.y for t in sel) - 10)))
        y2 = int(min(img_h, max(cbox.y2 + 0.25 * cbox.h, max(t.box.y2 for t in sel) + 10)))
        return Box(x1, y1, x2 - x1, y2 - y1)
    # Meio espaçamento: não pode invadir painéis de indicador logo abaixo (RSI etc.).
    y1 = int(max(0, min(t.box.y for t in sel) - 0.5 * spacing))
    y2 = int(min(img_h, max(t.box.y2 for t in sel) + 0.5 * spacing))
    x1 = max(0, x1 - pad)
    x2 = min(img_w, x2 + 3 * pad)
    return Box(x1, y1, x2 - x1, y2 - y1)


_PURCHASE_WORDS = ("PURCHASE", "COMPRA", "COMPRAR")


def _purchase_token(tokens: list[OcrToken]) -> Optional[OcrToken]:
    """O mm:ss logo à direita do texto 'PURCHASE TIME' / 'TEMPO DE COMPRA'."""
    words = [t for t in tokens if any(k in t.text.upper() for k in _PURCHASE_WORDS)]
    best = None
    for wtk in words:
        for t in tokens:
            if not is_clock_text(t.text) or parse_timer(t.text) is None:
                continue
            dx = t.box.x - wtk.box.x2
            same_row = t.box.y - 2 * wtk.box.h <= wtk.box.cy <= t.box.y2 + 2 * wtk.box.h
            if -4 <= dx <= 3 * max(t.box.h, wtk.box.h) and same_row:
                if best is None or dx < best[0]:
                    best = (dx, t)
    return best[1] if best else None


_CLOCK_TAIL = re.compile(r"(\d{1,2}\s*[:.]\s*\d{2})\s*$")


def _purchase_inline(tokens: list[OcrToken]) -> Optional[OcrToken]:
    """O Vision às vezes junta tudo num texto só ('PURCHASE TIME 00:31' ou 'TIME 00:31')."""
    for t in tokens:
        up = t.text.upper()
        if not any(k in up for k in _PURCHASE_WORDS + ("TIME", "TEMPO")):
            continue
        m = _CLOCK_TAIL.search(t.text)
        if not m or m.start() == 0:
            continue
        frac = m.start() / max(1, len(t.text))
        x1 = t.box.x + int(frac * t.box.w)
        return OcrToken(m.group(1), Box(x1, t.box.y, t.box.x2 - x1, t.box.h), t.score)
    return None


def _purchase_beside_label(tokens: list[OcrToken], img_w: int, img_h: int) -> Optional[Box]:
    """Sem os dígitos no OCR da tela inteira: recorta a área à direita de 'PURCHASE / TIME'."""
    words = [t for t in tokens if any(k in t.text.upper() for k in _PURCHASE_WORDS)]
    if not words:
        return None
    w = words[0]
    below = [t for t in tokens if t.text.strip().upper() in ("TIME", "TEMPO")
             and 0 <= t.box.y - w.box.y2 <= 2 * w.box.h and abs(t.box.x2 - w.box.x2) <= 3 * w.box.h]
    y1 = w.box.y
    y2 = max([w.box.y2] + [t.box.y2 for t in below])
    x1 = max([w.box.x2] + [t.box.x2 for t in below]) + 2
    hh = y2 - y1
    pad = hh // 6 + 2
    x2 = min(img_w, x1 + int(2.6 * hh) + 4)
    return Box(x1, max(0, y1 - pad), x2 - x1, min(img_h, y2 + pad) - max(0, y1 - pad))


def find_purchase(tokens: list[OcrToken], img_w: int, img_h: int) -> Optional[Box]:
    t = _purchase_token(tokens) or _purchase_inline(tokens)
    if t is None:
        return _purchase_beside_label(tokens, img_w, img_h)
    # Só os dígitos: o texto "PURCHASE TIME" ao lado atrapalha o OCR do recorte.
    pad_y = t.box.h // 4 + 2
    x1, x2 = max(0, t.box.x - 3), min(img_w, t.box.x2 + t.box.w // 4)
    y1, y2 = max(0, t.box.y - pad_y), min(img_h, t.box.y2 + pad_y)
    return Box(x1, y1, x2 - x1, y2 - y1)


def find_expiry(tokens: list[OcrToken], axis: Optional[Box], img_w: int, img_h: int) -> Optional[Box]:
    """Hora de expiração escolhida no painel da direita (abaixo de 'Expiration'/'Expiração')."""
    labels = [t for t in tokens if t.text.strip().lower().startswith(("expira", "expiry"))]
    right_of = axis.x if axis else int(0.75 * img_w)
    cands = [t for t in tokens if parse_clock(t.text) is not None and t.box.x >= right_of
             and not any(k in t.text.upper() for k in ("CURRENT", "ATUAL"))]
    if labels:
        lab = labels[0]
        below = [t for t in cands if 0 <= t.box.y - lab.box.y2 <= 4 * lab.box.h
                 and abs(t.box.x - lab.box.x) <= 4 * lab.box.h]
        if below:
            t = min(below, key=lambda t: t.box.y)
            return t.box.expand(t.box.w // 3 + 6, t.box.h // 3 + 2, img_w, img_h)
    return None


def find_clock(tokens: list[OcrToken], img_w: int, img_h: int) -> Optional[Box]:
    """Relógio da IQ ('CURRENT TIME: 9 NOVEMBER, 18:14:37')."""
    for t in tokens:
        up = t.text.upper()
        if ("CURRENT" in up or "ATUAL" in up) and parse_clock(t.text, need_seconds=True) is not None:
            return t.box.expand(8, t.box.h // 3 + 2, img_w, img_h)
    return None


def _time_axis_tokens(tokens: list[OcrToken], det: CandleDetection) -> list[OcrToken]:
    """Rótulos de hora alinhados numa linha abaixo das velas."""
    cbox = det.box
    clocks = [t for t in tokens if is_clock_text(t.text, allow_seconds=True)
              and (cbox is None or t.box.cy > cbox.y2)]
    if len(clocks) < 2:
        return []
    rows = _cluster([t.box.cy for t in clocks], tol=6.0)
    best = max(rows, key=len)
    return [clocks[i] for i in best] if len(best) >= 2 else []


def find_time_axis(tokens: list[OcrToken], det: CandleDetection, img_w: int, img_h: int) -> Optional[Box]:
    row = _time_axis_tokens(tokens, det)
    if not row:
        return None
    y1 = min(t.box.y for t in row)
    y2 = max(t.box.y2 for t in row)
    pad = max(4, (y2 - y1) // 2)
    x1 = det.box.x if det.box is not None else 0
    return Box(max(0, x1 - 4), max(0, y1 - pad), img_w - max(0, x1 - 4), min(img_h, y2 + pad) - max(0, y1 - pad))


STANDARD_TIMEFRAMES = (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400)


def timeframe_from_axis(tokens: list[OcrToken], pitch: float) -> Optional[int]:
    """Tempo de cada vela pelos rótulos de hora: segundos por pixel x passo das velas."""
    row = sorted(tokens, key=lambda t: t.box.cx)
    rates = []
    for a, b in zip(row, row[1:]):
        ta, tb = parse_clock(a.text), parse_clock(b.text)
        if ta is None or tb is None:
            continue
        dt = (tb - ta) % 86400
        dx = b.box.cx - a.box.cx
        if dt <= 0 or dx <= 5:
            continue
        rates.append(dt / dx)
    if not rates or pitch <= 0:
        return None
    raw = float(np.median(rates)) * pitch
    best = min(STANDARD_TIMEFRAMES, key=lambda s: abs(np.log(raw / s)))
    return best if abs(np.log(raw / best)) <= 0.15 else None


def find_timeframe(tokens: list[OcrToken], det: CandleDetection) -> Optional[int]:
    """Pelo eixo de horas; se não der, pelo botão '5m'/'1m' da barra lateral."""
    tf = timeframe_from_axis(_time_axis_tokens(tokens, det), det.pitch)
    if tf is not None:
        return tf
    cbox = det.box
    vals = [parse_timeframe(t.text) for t in tokens
            if cbox is None or t.box.x2 < cbox.x + 0.25 * cbox.w]
    vals = [v for v in vals if v is not None and v <= 4 * 3600]
    return vals[0] if len(set(vals)) == 1 else None


def find_timer(tokens: list[OcrToken], det: CandleDetection, axis: Optional[Box],
               img_w: int, img_h: int) -> Optional[Box]:
    """Cronômetro mm:ss perto da vela atual. Rótulos de hora do eixo X ficam de fora."""
    purchase = _purchase_token(tokens)
    clocks = [t for t in tokens if is_clock_text(t.text) and parse_timer(t.text) is not None
              and t is not purchase]
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


_ASSET_RE = __import__("re").compile(r"\b([A-Z]{3,5})\s*/\s*([A-Z]{3,5})\b(\s*\(OTC\))?")


def find_asset(tokens: list[OcrToken], det: CandleDetection) -> Optional[str]:
    """Nome do ativo (ex.: 'EUR/USD (OTC)'): o maior texto do tipo AAA/BBB na tela."""
    cands = []
    for t in tokens:
        m = _ASSET_RE.search(t.text.upper())
        if m:
            name = f"{m.group(1)}/{m.group(2)}" + (" (OTC)" if m.group(3) or "OTC" in t.text.upper() else "")
            cands.append((t.box.h, name))
    if not cands:
        return None
    return max(cands)[1]


def build_layout(bgr: np.ndarray, det: CandleDetection, tokens: list[OcrToken]) -> Layout:
    h, w = bgr.shape[:2]
    cbox = det.box
    assert cbox is not None
    axis = find_axis(tokens, det, w, h)
    timer = find_timer(tokens, det, axis, w, h)
    payout = find_payout(tokens, axis, w, h)
    purchase = find_purchase(tokens, w, h)
    expiry = find_expiry(tokens, axis, w, h)
    clock = find_clock(tokens, w, h)
    time_axis = find_time_axis(tokens, det, w, h)
    timeframe = find_timeframe(tokens, det)
    chart_x2 = axis.x if axis else min(w, cbox.x2 + int(12 * det.pitch))
    chart = Box(cbox.x, cbox.y, max(1, chart_x2 - cbox.x), cbox.h)
    return Layout(
        width=w, height=h, chart=chart,
        candle_pitch=det.pitch, candle_width=det.body_width,
        axis=axis, timer=timer, payout=payout,
        purchase=purchase, expiry=expiry, clock=clock, time_axis=time_axis, timeframe_s=timeframe,
        asset=find_asset(tokens, det),
        balance=find_balance(tokens, chart, w, h),
        invest=find_invest(tokens, w, h),
        green_hsv=det.green_hsv, red_hsv=det.red_hsv,
    )
