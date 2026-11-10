"""Saldo, tipo de conta e valor da entrada, lidos da tela da IQ Option.

* Saldo: o valor em dinheiro no topo, à esquerda do botão de depósito.
* Conta: a cor do saldo. Verde = conta real; laranja = conta de treino.
* Entrada: o campo "Invest" do painel da direita.

Só leitura de pixels, como o resto do app.
"""

from __future__ import annotations

import re
from typing import Optional

import cv2
import numpy as np

from ..models import Box, OcrToken

_MONEY_RE = re.compile(r"^\s*(?:R\$|US\$|\$|€|£|₹|₽|R\s*\$)?\s*([0-9][0-9.,\s]*)\s*$")
_TRAIL_JUNK = re.compile(r"[\s~▾▼▿⌄vV^|_\-–—]+$")
_CURRENCY_HINT = re.compile(r"[$€£₹₽]")


def parse_money(text: str) -> Optional[float]:
    """'$923.22' -> 923.22; 'R$ 1.234,56' -> 1234.56; '$1' -> 1.0; '$10,923.22' -> 10923.22."""
    # A seta do menu ao lado do saldo vira "~", "v", "▾"...; o cursor do campo Invest vira "|".
    t = _TRAIL_JUNK.sub("", text.strip()).replace("O", "0").replace("o", "0")
    m = _MONEY_RE.match(t)
    if not m:
        return None
    num = m.group(1).replace(" ", "")
    if not num:
        return None
    if "," in num and "." in num:
        dec = "," if num.rfind(",") > num.rfind(".") else "."
        thou = "." if dec == "," else ","
        num = num.replace(thou, "").replace(dec, ".")
    elif "," in num:
        head, _, tail = num.rpartition(",")
        num = (head.replace(",", "") + "." + tail) if len(tail) == 2 else num.replace(",", "")
    elif num.count(".") > 1:
        head, _, tail = num.rpartition(".")
        num = head.replace(".", "") + ("." + tail if len(tail) == 2 else tail)
    try:
        return float(num)
    except ValueError:
        return None


def find_balance(tokens: list[OcrToken], chart: Optional[Box], img_w: int, img_h: int) -> Optional[Box]:
    """O valor com símbolo de moeda mais alto na tela, na metade direita, acima do gráfico."""
    top = chart.y if chart is not None else int(0.2 * img_h)
    cands = [t for t in tokens if _CURRENCY_HINT.search(t.text) and parse_money(t.text) is not None
             and t.box.cy < top and t.box.cx > 0.5 * img_w]
    if not cands:
        return None
    t = min(cands, key=lambda t: t.box.y)
    return t.box.expand(t.box.w // 3 + 8, t.box.h // 4 + 3, img_w, img_h)


def find_invest(tokens: list[OcrToken], img_w: int, img_h: int) -> Optional[Box]:
    """O valor logo abaixo de 'Invest'/'Investimento'/'Valor'."""
    labels = [t for t in tokens if t.text.strip().lower().startswith(("invest", "valor", "amount"))]
    for lab in labels:
        below = [t for t in tokens if parse_money(t.text) is not None and t is not lab
                 and 0 <= t.box.y - lab.box.y2 <= 3 * lab.box.h and abs(t.box.x - lab.box.x) <= 3 * lab.box.h]
        if below:
            t = min(below, key=lambda t: t.box.y)
            x2 = min(img_w, t.box.x + max(t.box.w * 3, 4 * t.box.h))  # o valor pode crescer ($1 -> $100)
            pad = t.box.h // 4 + 2
            return Box(max(0, t.box.x - 4), max(0, t.box.y - pad), x2 - max(0, t.box.x - 4), t.box.h + 2 * pad)
    return None


def account_kind(bgr: np.ndarray, box: Box) -> Optional[str]:
    """'real' (texto verde), 'treino' (texto laranja) ou None."""
    crop = bgr[box.y:box.y2, box.x:box.x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 1] > 80) & (hsv[..., 2] > 80)
    if m.mean() < 0.03:
        return None
    hue = float(np.median(hsv[..., 0][m]))
    if 4 <= hue <= 30:
        return "treino"
    if 35 <= hue <= 95:
        return "real"
    return None
