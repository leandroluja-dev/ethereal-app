"""Gera capturas sintéticas no estilo do tema escuro da IQ Option, com OHLC conhecido.

Usado para criar as capturas salvas em tests/fixtures (ver tools/make_fixtures.py).
Não é uma cópia da interface: é só uma tela com os mesmos elementos que o leitor
precisa encontrar (velas, eixo de preço, cronômetro, payout, botões coloridos,
rótulos de hora) para testar a leitura com gabarito exato.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def _font(size: int, bold: bool = False):
    for p in (BOLD_CANDIDATES if bold else []) + FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


GREEN = (46, 189, 89)
RED = (233, 71, 71)
BG = (30, 34, 49)
PANEL = (24, 27, 39)
GRID = (40, 45, 62)
TEXT = (150, 156, 175)
WHITE = (235, 238, 245)


@dataclass
class Scene:
    width: int = 1920
    height: int = 1080
    scale: int = 1  # 2 = Retina
    n_candles: int = 70
    pitch: int = 14
    body_w: int = 9
    start_price: float = 1.08350
    tick: float = 0.00010  # passo dos rótulos
    vol: float = 0.00012  # volatilidade por vela
    decimals: int = 5
    seed: int = 1
    browser_bar: int = 0  # altura da barra do navegador (0 = app)
    timer_text: str = "00:23"
    payout: int = 87
    clock_start: tuple[int, int] = (14, 0)
    candle_minutes: int = 1
    doji_every: int = 0  # força doji a cada N velas (0 = nunca)
    noise: int = 0  # ruído de pixel (+-), para testes de robustez
    deadline_at: float = 0.0  # linha vermelha contínua a N passos da vela atual (0 = tracejada longe)
    truth: dict = field(default_factory=dict)


def _walk(sc: Scene) -> list[tuple[float, float, float, float]]:
    rnd = random.Random(sc.seed)
    out = []
    price = sc.start_price
    for i in range(sc.n_candles):
        o = price
        c = o + rnd.gauss(0, sc.vol)
        if sc.doji_every and i % sc.doji_every == sc.doji_every - 1:
            c = o
        hi = max(o, c) + abs(rnd.gauss(0, sc.vol * 0.6))
        lo = min(o, c) - abs(rnd.gauss(0, sc.vol * 0.6))
        out.append((o, hi, lo, c))
        price = c
    return out


def render(sc: Scene) -> tuple[np.ndarray, dict]:
    s = sc.scale
    W, H = sc.width * s, sc.height * s
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f_small = _font(12 * s)
    f_mid = _font(14 * s)
    f_big = _font(22 * s, bold=True)

    top = sc.browser_bar * s
    if top:
        d.rectangle([0, 0, W, top], fill=(53, 54, 58))
        d.rounded_rectangle([90 * s, 8 * s, 420 * s, top - 6 * s], 6 * s, fill=(33, 34, 37))
        d.text((110 * s, 14 * s), "IQ Option - plataforma", font=f_mid, fill=(220, 220, 220))
    bar_h = 56 * s
    d.rectangle([0, top, W, top + bar_h], fill=PANEL)
    d.text((80 * s, top + 18 * s), "EUR/USD (OTC)", font=f_mid, fill=WHITE)
    d.text((210 * s, top + 18 * s), f"{sc.payout}%", font=f_mid, fill=GREEN)
    # botão largo verde (depósito) no topo: não pode virar "vela"
    d.rounded_rectangle([W - 330 * s, top + 12 * s, W - 220 * s, top + 44 * s], 4 * s, fill=GREEN)
    d.text((W - 312 * s, top + 20 * s), "Depositar", font=f_mid, fill=WHITE)

    left_w = 64 * s
    d.rectangle([0, top + bar_h, left_w, H], fill=PANEL)
    for k in range(6):
        y = top + bar_h + 30 * s + k * 70 * s
        d.rounded_rectangle([18 * s, y, 46 * s, y + 28 * s], 5 * s, outline=(90, 95, 110), width=2 * s)

    panel_w = 210 * s
    px = W - panel_w
    d.rectangle([px, top + bar_h, W, H], fill=PANEL)
    d.text((px + 20 * s, top + bar_h + 30 * s), "Valor", font=f_small, fill=TEXT)
    d.text((px + 20 * s, top + bar_h + 50 * s), "$ 10", font=f_mid, fill=WHITE)
    d.text((px + 20 * s, top + bar_h + 100 * s), "Expiração", font=f_small, fill=TEXT)
    d.text((px + 20 * s, top + bar_h + 120 * s), "14:36", font=f_mid, fill=WHITE)
    d.text((px + 20 * s, top + bar_h + 180 * s), "Lucro", font=f_small, fill=TEXT)
    d.text((px + 20 * s, top + bar_h + 200 * s), f"+{sc.payout}%", font=f_big, fill=GREEN)
    d.rounded_rectangle([px + 20 * s, top + bar_h + 280 * s, W - 20 * s, top + bar_h + 340 * s], 6 * s, fill=GREEN)
    d.text((px + 70 * s, top + bar_h + 300 * s), "ACIMA", font=f_mid, fill=WHITE)
    d.rounded_rectangle([px + 20 * s, top + bar_h + 360 * s, W - 20 * s, top + bar_h + 420 * s], 6 * s, fill=RED)
    d.text((px + 70 * s, top + bar_h + 380 * s), "ABAIXO", font=f_mid, fill=WHITE)

    axis_w = 84 * s
    plot_x1, plot_x2 = left_w, px - axis_w
    plot_y1, plot_y2 = top + bar_h + 40 * s, H - 50 * s

    ohlc = _walk(sc)
    pmax = max(c[1] for c in ohlc)
    pmin = min(c[2] for c in ohlc)
    span = pmax - pmin
    pmax += 0.15 * span
    pmin -= 0.15 * span

    def y_of(p: float) -> float:
        return plot_y1 + (pmax - p) / (pmax - pmin) * (plot_y2 - plot_y1)

    def p_of_row(r: int) -> float:  # preço no centro da linha r
        return pmax - (r + 0.5 - plot_y1) / (plot_y2 - plot_y1) * (pmax - pmin)

    ppx = (pmax - pmin) / (plot_y2 - plot_y1)

    # grade e rótulos do eixo
    first = math.ceil(pmin / sc.tick)
    last = math.floor(pmax / sc.tick)
    step = max(1, int(round((last - first) / 12)))
    labels = []
    for k in range(first, last + 1, step):
        p = k * sc.tick
        y = y_of(p)
        gy = int(math.floor(y))
        d.line([plot_x1, gy, plot_x2, gy], fill=GRID, width=1)
        txt = f"{p:.{sc.decimals}f}"
        tb = d.textbbox((0, 0), txt, font=f_small, anchor="lm")
        d.text((plot_x2 + 10 * s, y), txt, font=f_small, fill=TEXT, anchor="lm")
        labels.append((y, p))

    pitch, bw = sc.pitch * s, sc.body_w * s
    x_last = plot_x2 - 140 * s
    candles_px = []
    truth = []
    for i, (o, hi, lo, c) in enumerate(ohlc):
        cx = x_last - (len(ohlc) - 1 - i) * pitch
        if cx - bw < plot_x1 + 4 * s:
            continue
        color = GREEN if c >= o else RED
        r_hi, r_lo = int(math.floor(y_of(hi))), int(math.floor(y_of(lo)))
        r_top, r_bot = int(math.floor(y_of(max(o, c)))), int(math.floor(y_of(min(o, c))))
        x1 = cx - bw // 2
        x2 = x1 + bw - 1
        wick_w = max(1, s)
        wx = cx - wick_w // 2
        d.rectangle([wx, r_hi, wx + wick_w - 1, r_lo], fill=color)
        d.rectangle([x1, r_top, x2, r_bot], fill=color)
        # gabarito como o leitor deve ver: preço no centro de cada linha de pixel
        if c >= o:
            to, tc = p_of_row(r_bot), p_of_row(r_top)
        else:
            to, tc = p_of_row(r_top), p_of_row(r_bot)
        truth.append({
            "open": to, "high": p_of_row(r_hi), "low": p_of_row(r_lo), "close": tc,
            "color": "verde" if c >= o else "vermelha", "x": cx,
        })
        candles_px.append((cx, r_top, r_bot))

    # linha do preço atual (tracejada, clara) e etiqueta colorida no eixo
    lc = ohlc[-1][3]
    ly = y_of(lc)
    for x in range(plot_x1, plot_x2, 10 * s):
        d.line([x, ly, x + 5 * s, ly], fill=(200, 205, 215), width=max(1, s))
    lab_col = GREEN if ohlc[-1][3] >= ohlc[-1][0] else RED
    d.rectangle([plot_x2 + 2 * s, ly - 10 * s, plot_x2 + axis_w - 4 * s, ly + 10 * s], fill=lab_col)
    d.text((plot_x2 + 10 * s, ly), f"{lc:.{sc.decimals}f}", font=f_small, fill=WHITE, anchor="lm")

    # linha vertical tracejada vermelha (fim das compras): fina, não é vela
    if sc.deadline_at:
        vx = int(x_last + sc.deadline_at * pitch)
        d.rectangle([vx, plot_y1, vx + 2 * s - 1, plot_y2], fill=RED)
    else:
        vx = x_last + 60 * s
        for y in range(plot_y1, plot_y2, 12 * s):
            d.line([vx, y, vx, y + 6 * s], fill=RED, width=max(1, s))
    # cronômetro perto da vela atual
    d.text((x_last + 24 * s, ly - 34 * s), sc.timer_text, font=f_mid, fill=WHITE)

    # eixo do tempo
    hh, mm = sc.clock_start
    for k in range(0, len(ohlc), 10):
        cx = x_last - (len(ohlc) - 1 - k) * pitch
        if cx < plot_x1 + 20 * s:
            continue
        t = hh * 60 + mm + k * sc.candle_minutes
        d.text((cx, plot_y2 + 22 * s), f"{(t // 60) % 24:02d}:{t % 60:02d}", font=f_small, fill=TEXT, anchor="mm")

    arr = np.asarray(img)[:, :, ::-1].copy()  # BGR
    if sc.noise:
        rng = np.random.default_rng(sc.seed)
        noise = rng.integers(-sc.noise, sc.noise + 1, size=arr.shape[:2], dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise[..., None], 0, 255).astype(np.uint8)

    meta = {
        "price_per_px": ppx,
        "decimals": sc.decimals,
        "timer_seconds": int(sc.timer_text.split(":")[0]) * 60 + int(sc.timer_text.split(":")[1]),
        "payout": sc.payout,
        "candles": truth,
        "scale": s,
        "plot": [plot_x1, plot_y1, plot_x2, plot_y2],
    }
    return arr, meta
