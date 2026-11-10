"""Painel do CandleSense: uma janela só, sempre por cima, no estilo de app de corretora.

Fundo preto, cartões cinza-escuro arredondados, preço grande, gráfico em linha,
cartão do sinal com medidor e as duas pílulas Compra / Venda com a porcentagem.

Desenhado com Pillow (fonte Inter, que vem junto com o app) e mostrado pelo
OpenCV. Tudo é desenhado no dobro do tamanho e reduzido no fim, para as curvas
ficarem lisas. A mesma função gera a imagem para testes e para o modo debug.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..engine.rules import COMPRA, INCERTA, NAO_ENTRAR, VENDA, Decision
from ..models import ChartReading, fmt_duration

# ---------------------------------------------------------------------------
# Tokens de design (unidades do painel; a escala multiplica tudo)
# ---------------------------------------------------------------------------
W = 380          # largura do painel
PAD = 16         # margem lateral
GAP = 12         # espaço entre blocos
R_CARD = 18      # raio dos cartões
SS = 2           # superamostragem (desenha 2x e reduz)

BG = (0, 0, 0)
CARD = (28, 28, 30)
CARD_2 = (44, 44, 47)       # pílulas e trilhos dentro de cartões
CARD_3 = (66, 66, 70)       # pílula selecionada
FG = (245, 245, 247)
DIM = (152, 152, 158)
FAINT = (92, 92, 97)
LINE = (52, 52, 56)
GREEN = (60, 196, 142)      # verde menta da referência
RED = (240, 84, 84)
ORANGE = (245, 160, 60)
GREY = (120, 120, 126)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

HEAD_COLOR = {COMPRA: GREEN, VENDA: RED, NAO_ENTRAR: FG, INCERTA: ORANGE}

_FONT_DIR = Path(__file__).with_name("fonts")
_WEIGHTS = {"regular": "Inter-Regular.otf", "medium": "Inter-Medium.otf",
            "semibold": "Inter-SemiBold.otf", "bold": "Inter-Bold.otf"}
_FALLBACK = {
    False: ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"],
    True: ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"],
}


@lru_cache(maxsize=64)
def _font(size: int, bold=False):
    """Fonte Inter. ``bold`` aceita True/False ou o peso: regular, medium, semibold, bold."""
    weight = bold if isinstance(bold, str) else ("bold" if bold else "regular")
    path = _FONT_DIR / _WEIGHTS.get(weight, _WEIGHTS["regular"])
    try:
        return ImageFont.truetype(str(path), max(1, size))
    except OSError:
        pass
    for p in _FALLBACK[weight in ("semibold", "bold")]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, max(1, size))
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow antigo
        return ImageFont.load_default()


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%".replace(".", ",")


def _pct0(p: float) -> str:
    return f"{p * 100:.0f}%"


def _mmss(s: int) -> str:
    return f"{s // 60:02d}:{s % 60:02d}"


def _money(v: float) -> str:
    sign = "+" if v > 0 else "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class _Canvas:
    """Folha de desenho. Coordenadas em unidades do painel; ``y`` corre em pixels internos."""

    def __init__(self, scale: float, height: int = 1400) -> None:
        self.out_scale = scale
        self.s = scale * SS
        self.w = int(round(W * self.s))
        self.img = Image.new("RGB", (self.w, int(height * self.s)), BG)
        self.d = ImageDraw.Draw(self.img)
        self.y = 0

    # medidas
    def px(self, v: float) -> int:
        return int(round(v * self.s))

    @property
    def u(self) -> float:
        """y atual em unidades do painel."""
        return self.y / self.s

    def font(self, size: float, bold=False):
        return _font(self.px(size), bold)

    def tw(self, txt: str, size: float, bold=False) -> float:
        """Largura do texto em unidades do painel."""
        return self.d.textlength(txt, font=self.font(size, bold)) / self.s

    # texto
    def text(self, x: float, txt: str, size: float, color=FG, bold=False, dy: float = 0) -> None:
        self.d.text((self.px(x), self.y + self.px(dy)), txt, font=self.font(size, bold), fill=color)

    def at(self, x: float, y: float, txt: str, size: float, color=FG, bold=False, anchor: str = "la") -> None:
        self.d.text((self.px(x), self.px(y)), txt, font=self.font(size, bold), fill=color, anchor=anchor)

    def para(self, txt: str, size: float, color=FG, x: float = PAD, width: Optional[float] = None,
             bold=False, gap: float = 4) -> None:
        f = self.font(size, bold)
        width = width if width is not None else W - x - PAD
        for ln in _wrap(self.d, txt, f, self.px(width)):
            self.d.text((self.px(x), self.y), ln, font=f, fill=color)
            self.y += self.px(size + gap)

    def space(self, v: float) -> None:
        self.y += self.px(v)

    def rule(self) -> None:
        self.space(8)
        self.d.line([(self.px(PAD), self.y), (self.w - self.px(PAD), self.y)], fill=LINE, width=self.px(1))
        self.space(10)

    # formas
    def rrect(self, x1, y1, x2, y2, r, fill=None, outline=None, width: float = 1) -> None:
        self.d.rounded_rectangle([self.px(x1), self.px(y1), self.px(x2), self.px(y2)], radius=self.px(r),
                                 fill=fill, outline=outline, width=self.px(width) if outline else 0)

    def card(self, y1: float, y2: float, fill=CARD) -> None:
        self.rrect(PAD, y1, W - PAD, y2, R_CARD, fill=fill)

    def crop(self) -> np.ndarray:
        h = self.y + self.px(PAD)
        img = self.img.crop((0, 0, self.w, h))
        out_w = int(round(W * self.out_scale))
        out_h = max(1, int(round(h / SS)))
        img = img.resize((out_w, out_h), Image.BOX)
        return np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR (OpenCV)


@dataclass
class PanelExtras:
    """O que o painel mostra além da decisão: carteira, operação aberta e alerta de saída."""

    wallet_line: Optional[str] = None
    wallet_profit: Optional[float] = None
    wallet_account: Optional[str] = None   # "real" | "treino"
    wallet_n: Optional[int] = None         # entradas hoje
    wallet_wins: Optional[int] = None
    wallet_losses: Optional[int] = None
    open_line: Optional[str] = None
    # operação aberta: amount, direction (COMPRA/VENDA/None), entry_price, payout, seconds_left
    open_trade: Optional[dict] = None
    exit_alert: Optional[str] = None
    details: bool = False       # explicação dos módulos (tecla d)
    chart: bool = True          # gráfico ao vivo
    buttons: bool = True        # botão Carteira no topo


# Botões desenhados no próprio painel. (x1, y1, x2, y2) em unidades do painel, antes da escala.
BUTTONS = {
    "carteira": (W - PAD - 92, 16, W - PAD, 48),
}


def hit_button(x: float, y: float, scale: float = 1.0) -> Optional[str]:
    """Qual botão do painel está no ponto (x, y) da imagem, em pixels."""
    ux, uy = x / scale, y / scale
    for name, (x1, y1, x2, y2) in BUTTONS.items():
        if x1 - 4 <= ux <= x2 + 4 and y1 - 4 <= uy <= y2 + 4:
            return name
    return None


# ---------------------------------------------------------------------------
# Ícones simples (linhas finas, como os da referência)
# ---------------------------------------------------------------------------

def _icon(c: _Canvas, kind: str, x: float, y: float, color=DIM, size: float = 14) -> None:
    """Ícone de ``size`` unidades com canto em (x, y)."""
    px, s = c.px, size
    lw = max(1, px(1.6))
    if kind in ("up", "down"):
        if kind == "up":
            pts = [(x + s / 2, y + 2), (x + s - 1, y + s - 3), (x + 1, y + s - 3)]
        else:
            pts = [(x + 1, y + 3), (x + s - 1, y + 3), (x + s / 2, y + s - 2)]
        c.d.polygon([(px(a), px(b)) for a, b in pts], fill=color)
    elif kind == "clock":
        c.d.ellipse([px(x + 1), px(y + 1), px(x + s - 1), px(y + s - 1)], outline=color, width=lw)
        cx, cy = x + s / 2, y + s / 2
        c.d.line([(px(cx), px(cy)), (px(cx), px(y + 4))], fill=color, width=lw)
        c.d.line([(px(cx), px(cy)), (px(cx + 3), px(cy + 2))], fill=color, width=lw)
    elif kind == "target":
        c.d.ellipse([px(x + 1), px(y + 1), px(x + s - 1), px(y + s - 1)], outline=color, width=lw)
        c.d.ellipse([px(x + 5), px(y + 5), px(x + s - 5), px(y + s - 5)], fill=color)
    elif kind == "bars":
        for i, h in enumerate((5, 9, 13)):
            bx = x + 1 + i * 4.5
            c.d.rectangle([px(bx), px(y + s - h), px(bx + 2.6), px(y + s - 1)], fill=color)
    elif kind == "dot":
        c.d.ellipse([px(x + 3), px(y + 3), px(x + s - 3), px(y + s - 3)], fill=color)


def _icon_line(c: _Canvas, kind: str, txt: str, color=FG, icolor=DIM, size: float = 14, x: float = PAD + 18,
               width: Optional[float] = None) -> None:
    """Linha com ícone à esquerda e texto (quebra se precisar)."""
    _icon(c, kind, x, c.u + 2, icolor, 14)
    width = width if width is not None else W - PAD - 18 - (x + 24)
    c.para(txt, size, color, x=x + 24, width=width, gap=6)


# ---------------------------------------------------------------------------
# Blocos
# ---------------------------------------------------------------------------

# Estado do mouse sobre os botões (atualizado pela janela): None, "hover" ou "down"
BUTTON_STATE: dict[str, Optional[str]] = {"carteira": None}


def draw_button(c: _Canvas, label: str = "Carteira") -> None:
    """Pílula do canto de cima (Carteira no painel, Voltar na carteira).

    Normal: cinza. Mouse em cima: mais clara com contorno. Apertada: branca, como o Buy da referência.
    """
    x1, y1, x2, y2 = BUTTONS["carteira"]
    st = BUTTON_STATE.get("carteira")
    if st == "down":
        fill, fg, outline = FG, BLACK, None
    elif st == "hover":
        fill, fg, outline = CARD_3, WHITE, (120, 120, 126)
    else:
        fill, fg, outline = CARD_2, FG, None
    c.rrect(x1, y1, x2, y2, (y2 - y1) / 2, fill=fill, outline=outline, width=1.2)
    c.at((x1 + x2) / 2, (y1 + y2) / 2, label, 13, fg, "semibold" if st else "medium", anchor="mm")


def _header(c: _Canvas, reading: Optional[ChartReading], ex: PanelExtras) -> None:
    asset = None
    if reading is not None:
        asset = reading.asset or (reading.layout.asset if reading.layout is not None else None)
    title = asset or "CandleSense"
    room = (BUTTONS["carteira"][0] - PAD - 10) if ex.buttons else W - 2 * PAD
    size = 19
    while size > 13 and c.tw(title, size, "semibold") > room:
        size -= 1
    c.at(PAD, 16, title, size, FG, "semibold")
    sub = []
    if reading is not None and reading.timeframe_s:
        sub.append(f"vela de {fmt_duration(reading.timeframe_s)}")
    if reading is not None and reading.payout_pct:
        sub.append(f"payout {reading.payout_pct}%")
    c.at(PAD, 41, " · ".join(sub) if sub else "lendo a IQ Option", 13, DIM)
    if ex.buttons:
        draw_button(c)
    c.y = c.px(66)


def _price(c: _Canvas, reading: ChartReading, cs: list) -> None:
    """Preço grande, com os dois últimos dígitos menores (como na IQ e na referência)."""
    y0 = c.u
    p = reading.current_price
    dec = reading.scale.decimals if reading.scale else 5
    if p is None:
        c.at(PAD, y0, "—", 40, FG, "bold")
        c.y = c.px(y0 + 52)
        return
    s = f"{p:.{dec}f}"
    big, small = (s[:-2], s[-2:]) if dec >= 3 else (s, "")
    c.at(PAD, y0 + 40, big, 42, FG, "bold", anchor="ls")
    if small:
        c.at(PAD + c.tw(big, 42, "bold") + 1, y0 + 40, small, 28, FG, "bold", anchor="ls")
    # variação na janela mostrada
    if len(cs) >= 2:
        ref = cs[0].open
        diff = p - ref
        col = GREEN if diff > 0 else RED if diff < 0 else DIM
        rel = diff / ref * 100 if ref else 0.0
        a = f"{'+' if diff >= 0 else '-'}{abs(diff):.{dec}f}"
        b = f"{abs(rel):.2f}%".replace(".", ",")
        x = PAD
        y = y0 + 52
        c.at(x, y, a, 14, DIM, "medium")
        x += c.tw(a, 14, "medium") + 7
        _icon(c, "up" if diff >= 0 else "down", x, y + 1.5, col, 12)
        x += 15
        c.at(x, y, b, 14, col, "medium")
        x += c.tw(b, 14, "medium") + 6
        c.at(x, y, f"· últimas {len(cs)} velas", 14, DIM)
    c.y = c.px(y0 + 80)


def _smooth(pts: list[tuple[float, float]], steps: int = 6) -> list[tuple[float, float]]:
    """Curva Catmull-Rom passando pelos pontos (linha macia como na referência)."""
    if len(pts) < 3:
        return pts
    out = []
    p = [pts[0]] + pts + [pts[-1]]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for k in range(steps):
            t = k / steps
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in (0, 1)))
    out.append(pts[-1])
    return out


def _chart(c: _Canvas, reading: ChartReading, cs: list, h: float = 150) -> None:
    """Gráfico em linha com o fechamento das velas lidas; confira se bate com a IQ Option."""
    y0 = c.u
    x0, x1 = PAD + 2, W - PAD - 56
    closes = [k.close for k in cs]
    if reading.current_price is not None and closes:
        closes[-1] = reading.current_price
    lo, hi = min(k.low for k in cs), max(k.high for k in cs)
    lo, hi = min(lo, min(closes)), max(hi, max(closes))
    if hi <= lo:
        hi = lo + 1e-9
    top, bot = y0 + 10, y0 + h - 10

    def yv(v: float) -> float:
        return top + (hi - v) / (hi - lo) * (bot - top)

    n = len(closes)
    pts = [(x0 + (x1 - x0) * i / max(1, n - 1), yv(v)) for i, v in enumerate(closes)]
    up = closes[-1] >= cs[0].open
    col = GREEN if up else RED
    dec = reading.scale.decimals if reading.scale else 5
    # linha pontilhada no preço de abertura da janela
    yr = yv(cs[0].open)
    xx = x0
    while xx < x1:
        c.d.ellipse([c.px(xx), c.px(yr) - c.px(0.8), c.px(xx) + c.px(1.6), c.px(yr) + c.px(0.8)], fill=FAINT)
        xx += 5
    # máxima e mínima à direita
    c.at(W - PAD, top - 2, f"{hi:.{dec}f}", 12, DIM, anchor="ra")
    c.at(W - PAD, bot + 2, f"{lo:.{dec}f}", 12, DIM, anchor="rs")
    # linha
    curve = [(c.px(a), c.px(b)) for a, b in _smooth(pts)]
    c.d.line(curve, fill=col, width=c.px(2.4), joint="curve")
    # ponto atual com halo
    ex_, ey = pts[-1]
    r1, r2 = 7, 3.6
    halo = tuple(int(v * 0.35) for v in col)
    c.d.ellipse([c.px(ex_ - r1), c.px(ey - r1), c.px(ex_ + r1), c.px(ey + r1)], fill=halo)
    c.d.ellipse([c.px(ex_ - r2), c.px(ey - r2), c.px(ex_ + r2), c.px(ey + r2)], fill=col)
    c.y = c.px(y0 + h + 4)


def _chart_empty(c: _Canvas, msg: str, h: float = 150) -> None:
    y0 = c.u
    c.rrect(PAD, y0 + 8, W - PAD, y0 + h - 8, R_CARD, outline=LINE, width=1)
    c.at(W / 2, y0 + h / 2, msg, 13, DIM, anchor="mm")
    c.y = c.px(y0 + h + 4)


def _expiry_tabs(c: _Canvas, d: Decision) -> None:
    """Pílulas das expirações (como 1D 1W 1M da referência); a sugerida fica acesa."""
    if not d.horizons:
        return
    y0 = c.u
    h = 40
    c.rrect(PAD, y0, W - PAD, y0 + h, h / 2, fill=CARD)
    n = len(d.horizons)
    cw = (W - 2 * PAD - 8) / n
    for i, hz in enumerate(d.horizons):
        x = PAD + 4 + i * cw
        sel = d.best is not None and hz.candles == d.best.candles and d.action != INCERTA
        if sel:
            c.rrect(x, y0 + 4, x + cw, y0 + h - 4, (h - 8) / 2, fill=CARD_3)
        lab = fmt_duration(hz.candles * d.timeframe_s).replace(" ", "") if d.timeframe_s else f"{hz.candles}v"
        c.at(x + cw / 2, y0 + h / 2, lab, 14, FG if sel else DIM, "semibold" if sel else "medium", anchor="mm")
    c.y = c.px(y0 + h + GAP)


def _gauge(c: _Canvas, cx: float, cy: float, r: float, st: Optional[float]) -> None:
    """Medidor semicircular: vermelho (venda) · cinza · verde (compra), com ponteiro."""
    box = [c.px(cx - r), c.px(cy - r), c.px(cx + r), c.px(cy + r)]
    wd = c.px(5)
    active = st is not None
    segs = [(180, 128, RED), (120, 60, GREY), (52, 0, GREEN)]
    for a0, a1, col in segs:
        if not active:
            col = LINE
        c.d.arc(box, start=360 - a0, end=360 - a1, fill=col, width=wd)
    # marquinha no topo
    c.d.line([(c.px(cx), c.px(cy - r - 4)), (c.px(cx), c.px(cy - r + 7))], fill=DIM if active else LINE, width=c.px(2))
    if active:
        ang = math.radians(90 - 90 * st)
        L = r - 9
        nx, ny = cx + L * math.cos(ang), cy - L * math.sin(ang)
        # ponteiro em forma de gota
        px_, py_ = -math.sin(ang), -math.cos(ang)
        base = 3.2
        poly = [(cx + px_ * base, cy + py_ * base), (nx, ny), (cx - px_ * base, cy - py_ * base)]
        c.d.polygon([(c.px(a), c.px(b)) for a, b in poly], fill=FG)
        c.d.ellipse([c.px(cx - 4), c.px(cy - 4), c.px(cx + 4), c.px(cy + 4)], fill=FG)


def _live_result(t: dict, price: Optional[float]) -> tuple[Optional[float], Optional[str]]:
    """Resultado da operação aberta se ela vencesse agora: (lucro, 'ganhando'|'perdendo'|'empatando')."""
    entry, side = t.get("entry_price"), t.get("direction")
    if entry is None or price is None or side not in (COMPRA, VENDA):
        return None, None
    diff = price - entry if side == COMPRA else entry - price
    amount = float(t.get("amount") or 0.0)
    if abs(diff) < 1e-12:
        return 0.0, "empatando"
    if diff > 0:
        return amount * (t.get("payout") or 0) / 100.0, "ganhando"
    return -amount, "perdendo"


def _wallet_strip(c: _Canvas, ex: PanelExtras) -> None:
    """Carteira de hoje no topo, numa linha: resultado, entradas e a conta (REAL/TREINO)."""
    if not (ex.wallet_line or ex.wallet_n is not None):
        return
    y0 = c.u
    h = 56
    c.card(y0, y0 + h)
    cy = y0 + h / 2
    _ring(c, PAD + 14 + 15, cy, 15, ex.wallet_n, ex.wallet_wins or 0, ex.wallet_losses or 0, size=13)
    x = PAD + 14 + 30 + 12
    if ex.wallet_n is not None:
        pr = ex.wallet_profit or 0.0
        col = GREEN if pr > 0 else RED if pr < 0 else FG
        c.at(x, cy - 9, "Hoje", 12, DIM, anchor="lm")
        c.at(x, cy + 9, _money(pr), 17, col, "semibold", anchor="lm")
        n, g, p = ex.wallet_n or 0, ex.wallet_wins or 0, ex.wallet_losses or 0
        x2 = x + max(86, c.tw(_money(pr), 17, "semibold") + 14)
        c.at(x2, cy - 9, f"{n} entrada{'s' if n != 1 else ''}", 12, DIM, anchor="lm")
        c.at(x2, cy + 9, f"{g} ganho{'s' if g != 1 else ''} · {p} perda{'s' if p != 1 else ''}", 13, FG, anchor="lm")
    else:
        c.at(x, cy, ex.wallet_line or "", 13, DIM, anchor="lm")
    if ex.wallet_account in ("real", "treino"):
        lab = "REAL" if ex.wallet_account == "real" else "TREINO"
        bc = GREEN if ex.wallet_account == "real" else ORANGE
        bw = c.tw(lab, 11, "bold") + 16
        c.rrect(W - PAD - 14 - bw, cy - 10, W - PAD - 14, cy + 10, 10, outline=bc, width=1.2)
        c.at(W - PAD - 14 - bw / 2, cy, lab, 11, bc, "bold", anchor="mm")
    c.y = c.px(y0 + h + GAP)


def _open_trade_card(c: _Canvas, ex: PanelExtras, reading: Optional[ChartReading]) -> None:
    """A operação aberta, ao vivo: ganhando ou perdendo agora e quanto falta."""
    t = ex.open_trade
    if not t:
        if ex.open_line:
            y0 = c.u
            c.card(y0, y0 + 44)
            c.at(PAD + 18, y0 + 22, ex.open_line, 13, ORANGE, "medium", anchor="lm")
            c.y = c.px(y0 + 44 + GAP)
        return
    price = reading.current_price if reading is not None and reading.ok else None
    profit, state = _live_result(t, price)
    col = GREEN if state == "ganhando" else RED if state == "perdendo" else FG
    y0 = c.u
    h = 112
    c.card(y0, y0 + h)
    c.rrect(PAD, y0, W - PAD, y0 + h, R_CARD, outline=col if state else LINE, width=1.5)
    inner = PAD + 18
    side = {COMPRA: "COMPRA ▲", VENDA: "VENDA ▼"}.get(t.get("direction"), "direção não lida")
    c.at(inner, y0 + 16, f"SUA OPERAÇÃO · {side} · {_money(float(t.get('amount') or 0)).lstrip('+')}", 12, DIM, "semibold")
    if profit is not None:
        c.at(inner, y0 + 58, _money(profit), 32, col, "bold", anchor="ls")
        c.at(inner + c.tw(_money(profit), 32, "bold") + 10, y0 + 56, state, 15, col, "semibold", anchor="ls")
    else:
        c.at(inner, y0 + 58, f"{_money(float(t.get('amount') or 0)).lstrip('+')} em jogo", 26, FG, "bold", anchor="ls")
    left = t.get("seconds_left")
    if left is not None:
        c.at(W - PAD - 18, y0 + 58, _mmss(max(0, int(left))), 26, FG, "bold", anchor="rs")
        c.at(W - PAD - 18, y0 + 16, "vence em", 12, DIM, anchor="ra")
    entry = t.get("entry_price")
    dec = reading.scale.decimals if reading is not None and reading.scale else 5
    if entry is not None and price is not None:
        diff = price - entry
        txt = f"entrada {entry:.{dec}f} → agora {price:.{dec}f} ({'+' if diff >= 0 else '-'}{abs(diff):.{dec}f})"
    elif entry is not None:
        txt = f"entrada {entry:.{dec}f} · preço atual não lido"
    else:
        txt = "preço de entrada não lido"
    c.at(inner, y0 + 74, txt, 12, DIM)
    if profit is None and entry is not None and price is not None:
        c.at(inner, y0 + 92, "a direção só é conhecida quando você segue um sinal do app", 11, FAINT)
    c.y = c.px(y0 + h + GAP)


def _action(d: Decision, ex: PanelExtras) -> tuple[str, str, tuple, tuple, list]:
    """A ordem final: (título, subtítulo, cor de fundo, cor do texto, linhas de apoio)."""
    lines: list[tuple[str, str, tuple, tuple]] = []
    t = ex.open_trade
    if ex.exit_alert:
        return ("SAIR AGORA", ex.exit_alert + ". Venda a operação na IQ Option.", (150, 28, 32), WHITE, lines)
    if d.action == INCERTA:
        why = d.reasons[0] if d.reasons else "a leitura da tela não fechou"
        lines.append(("dot", why, FG, ORANGE))
        return ("NÃO ENTRE", "leitura incerta: sem sinal até a leitura voltar", CARD, ORANGE, lines)
    if t:
        lado = {COMPRA: "compra", VENDA: "venda"}.get(t.get("direction"), "operação")
        return ("SEGURE", f"mantenha a {lado} até o vencimento; o app avisa se for para sair", CARD, FG, lines)
    if d.best is not None:
        lado = "alta" if d.best.p_up >= 0.5 else "baixa"
        lines.append(("up" if lado == "alta" else "down", f"{_pct(d.best.prob)} de chance de {lado} em {d.expiry_text()}",
                      FG, GREEN if lado == "alta" else RED))
    if d.seconds_to_enter is not None:
        fase = f" · {d.phase} da vela" if d.phase else ""
        lines.append(("clock", f"Clique em até {_mmss(d.seconds_to_enter)} (já conta o seu atraso){fase}", FG, DIM))
    if d.threshold is not None:
        be = f" · empate {_pct(d.breakeven)}" if d.breakeven is not None else ""
        lines.append(("target", f"Mínimo para entrar: {_pct(d.threshold)}{be}", DIM, DIM))
    if d.expiry_hint:
        lines.append(("bars", d.expiry_hint, ORANGE if "sugerida é" in d.expiry_hint else DIM, DIM))
    if d.signal:
        if d.action == COMPRA:
            return ("COMPRE AGORA", f"clique em HIGHER · expiração {d.expiry_text()}", GREEN, BLACK, lines)
        return ("VENDA AGORA", f"clique em LOWER · expiração {d.expiry_text()}", RED, WHITE, lines)
    motivo = d.reasons[0] if d.reasons else "sem vantagem agora"
    return ("AGUARDE", motivo, CARD, FG, lines)


def _action_card(c: _Canvas, d: Decision, ex: PanelExtras) -> None:
    """Cartão principal: a ordem final (comprar, vender, aguardar, segurar ou sair) e a força em %."""
    title, sub, bg, tcol, lines = _action(d, ex)
    lit = bg not in (CARD,)
    y0 = c.u
    inner = PAD + 18
    sub_col = (tcol if lit else DIM)
    fs = c.font(14, "medium")
    sub_lines = _wrap(c.d, sub, fs, c.px(W - 2 * inner - (78 if not lit else 0)))[:3]
    text_w = W - PAD - 18 - (inner + 24)
    f = c.font(14)
    hl = sum(len(_wrap(c.d, txt, f, c.px(text_w))) for _, txt, _, _ in lines)
    head_h = 18 + 34 + 8 + 19 * len(sub_lines) + 14
    body_h = (hl * 20 + 24) if lines else 0
    h = head_h + body_h
    # parte de cima colorida com a ordem; parte de baixo com os números
    c.card(y0, y0 + h)
    if lit:
        c.rrect(PAD, y0, W - PAD, y0 + head_h, R_CARD, fill=bg)
        if body_h:
            c.d.rectangle([c.px(PAD), c.px(y0 + head_h - R_CARD), c.px(W - PAD), c.px(y0 + head_h)], fill=bg)
    tsize = 32
    room = W - 2 * inner - (78 if not lit else 0)
    while tsize > 20 and c.tw(title, tsize, "bold") > room:
        tsize -= 2
    c.at(inner, y0 + 18 + 32, title, tsize, tcol, "bold", anchor="ls")
    for i, ln in enumerate(sub_lines):
        c.at(inner, y0 + 18 + 40 + 19 * i, ln, 14, sub_col, "medium")
    if not lit:  # medidor de força ao lado quando não há ordem acesa
        _gauge(c, W - PAD - 18 - 32, y0 + 18 + 32, 28, None if d.action == INCERTA else d.strength)
    c.y = c.px(y0 + head_h + 12)
    for kind, txt, col, icol in lines:
        _icon_line(c, kind, txt, col, icol, 14, x=inner, width=text_w)
    c.y = c.px(y0 + h + GAP)


def _side_pills(c: _Canvas, d: Decision) -> None:
    """Força da compra x da venda em %, como os botões Buy / Sell da referência (só mostram, nunca clicam)."""
    if d.best is None or d.action == INCERTA:
        return
    y0 = c.u
    h = 50
    gap = 10
    pw = (W - 2 * PAD - gap) / 2
    p_up = d.best.p_up
    for i, (name, p, col) in enumerate((("Compra", p_up, GREEN), ("Venda", 1 - p_up, RED))):
        x = PAD + i * (pw + gap)
        lit = d.signal and ((i == 0) == (d.action == COMPRA))
        if lit:
            fill, fg, pc = col, (BLACK if i == 0 else WHITE), (BLACK if i == 0 else WHITE)
        else:
            fill, fg, pc = CARD_2, FG, (col if p > 0.5 else DIM)
        c.rrect(x, y0, x + pw, y0 + h, h / 2, fill=fill)
        t1, t2 = name, _pct0(p)
        w1, w2 = c.tw(t1, 16, "semibold"), c.tw(t2, 16, "bold")
        sx = x + pw / 2 - (w1 + 8 + w2) / 2
        c.at(sx, y0 + h / 2, t1, 16, fg, "semibold", anchor="lm")
        c.at(sx + w1 + 8, y0 + h / 2, t2, 16, pc, "bold", anchor="lm")
    c.y = c.px(y0 + h + GAP)


def _ring(c: _Canvas, cx: float, cy: float, r: float, n: Optional[int], wins: int, losses: int,
          size: float = 18) -> None:
    box = [c.px(cx - r), c.px(cy - r), c.px(cx + r), c.px(cy + r)]
    wd = c.px(4)
    c.d.arc(box, 0, 360, fill=CARD_2, width=wd)
    done = wins + losses
    if done:
        a = 360 * wins / done
        if wins:
            c.d.arc(box, -90, -90 + a, fill=GREEN, width=wd)
        if losses:
            c.d.arc(box, -90 + a, 270, fill=RED, width=wd)
    c.at(cx, cy, str(n if n is not None else 0), size, FG, "semibold", anchor="mm")


def _modules_card(c: _Canvas, d: Decision) -> None:
    if not d.modules:
        return
    y0 = c.u
    inner = PAD + 18
    f = c.font(12)
    tw = W - 2 * inner
    rows = [(m, _wrap(c.d, m.explanation, f, c.px(tw))) for m in d.modules]
    h = 18 + 22 + sum(24 + 17 * len(ls) + 8 for _, ls in rows) + 6
    c.card(y0, y0 + h)
    c.at(inner, y0 + 18, "Por que o app pensa assim", 14, FG, "semibold")
    yy = y0 + 18 + 26
    for m, ls in rows:
        c.at(inner, yy, m.name, 13, FG, "medium")
        bx, bw = inner + 150, W - inner - (inner + 150)
        by = yy + 6
        c.rrect(bx, by, bx + bw, by + 6, 3, fill=CARD_2)
        mid = bx + bw / 2
        end = mid + m.score * m.confidence * bw / 2
        if abs(end - mid) > 0.5:
            col = GREEN if m.score > 0 else RED
            c.rrect(min(mid, end), by, max(mid, end), by + 6, 3, fill=col)
        c.d.line([(c.px(mid), c.px(by - 3)), (c.px(mid), c.px(by + 9))], fill=DIM, width=c.px(1))
        yy += 22
        for ln in ls:
            c.at(inner, yy, ln, 12, DIM)
            yy += 17
        yy += 8
    c.y = c.px(y0 + h + GAP)


def _footer(c: _Canvas, d: Decision, reading: Optional[ChartReading]) -> None:
    c.space(2)
    st = "leitura ok" if reading is None or reading.ok else "leitura incerta"
    if reading is not None and reading.timings_ms:
        st += f" · {sum(reading.timings_ms.values()):.0f} ms"
    st += " · " + ("probabilidade corrigida pelo placar" if d.calibrated else "probabilidade por regras")
    c.para(st, 11, FAINT, gap=4)
    c.para("Só lê a tela: nunca clica e nunca opera.", 11, FAINT, gap=4)
    c.para("Teclas: c carteira · d detalhes · g gráfico · q fechar", 11, FAINT, gap=4)


def render(decision: Decision, reading: Optional[ChartReading] = None, scale: float = 1.0,
           extras: Optional[PanelExtras] = None) -> np.ndarray:
    """Imagem BGR do painel."""
    d = decision
    ex = extras or PanelExtras(details=True, buttons=False)
    c = _Canvas(scale, height=1700 if ex.details else 1100)

    _header(c, reading, ex)
    _wallet_strip(c, ex)          # carteira no topo
    _open_trade_card(c, ex, reading)  # operação aberta, ao vivo
    _action_card(c, d, ex)        # a ordem final: comprar, vender, aguardar, segurar ou sair
    _side_pills(c, d)             # força da compra x venda em %

    cs = []
    if reading is not None and reading.candles:
        cs = [k for k in reading.candles if k.high is not None and k.low is not None][-40:]
    if reading is not None and reading.ok and cs:
        _price(c, reading, cs)
    if ex.chart:
        if reading is not None and reading.ok and len(cs) >= 3:
            _chart(c, reading, cs, h=120)
        else:
            _chart_empty(c, "aguardando a leitura do gráfico", h=100)
    _expiry_tabs(c, d)
    if ex.details and d.action != INCERTA:
        _modules_card(c, d)
    _footer(c, d, reading)
    return c.crop()


class PanelWindow:
    """Janela do painel via OpenCV, pedindo para ficar sempre por cima."""

    NAME = "CandleSense"

    def __init__(self, scale: float = 1.0) -> None:
        import cv2

        self.cv2 = cv2
        self.scale = scale
        cv2.namedWindow(self.NAME, cv2.WINDOW_AUTOSIZE)
        self._topmost()
        self._seen_open = False
        self._placed = False
        self._click: Optional[str] = None
        self._frames = 0
        try:
            cv2.setMouseCallback(self.NAME, self._on_mouse)
        except Exception:
            pass
        self.place: Optional[tuple[int, int]] = None  # canto de cima à esquerda, em pontos da tela

    def _topmost(self) -> None:
        try:
            self.cv2.setWindowProperty(self.NAME, self.cv2.WND_PROP_TOPMOST, 1)
        except Exception:  # backend sem suporte: a janela fica normal
            pass

    def show(self, decision: Decision, reading: Optional[ChartReading] = None,
             extras: Optional[PanelExtras] = None, overlay: Optional[np.ndarray] = None) -> None:
        """Mostra o painel; ``overlay`` (a carteira) troca o conteúdo na mesma janela."""
        img = overlay if overlay is not None else render(decision, reading, self.scale, extras)
        self.cv2.imshow(self.NAME, img)
        self._frames += 1
        if self._frames in (1, 3, 10) or self._frames % 50 == 0:
            from ..platform.appmode import window_buttons

            window_buttons(self.NAME)  # barra normal do Mac: fechar e minimizar
        if not self._placed and self.place is not None:
            try:
                self.cv2.moveWindow(self.NAME, *self.place)
            except Exception:
                pass
            self._placed = True

    def _on_mouse(self, event, x, y, flags, param) -> None:
        cv2 = self.cv2
        over = hit_button(x, y, self.scale)
        before = BUTTON_STATE.get("carteira")
        if event == cv2.EVENT_LBUTTONDOWN:
            now = "down" if over else None
        elif event == cv2.EVENT_LBUTTONUP:
            self._click = over
            now = "hover" if over else None
        elif event == cv2.EVENT_MOUSEMOVE:
            pressed = bool(flags & cv2.EVENT_FLAG_LBUTTON)
            now = ("down" if pressed and before == "down" else "hover") if over else None
        else:
            return
        if now != before:
            BUTTON_STATE["carteira"] = now
            self._hover_changed = True

    def pop_hover_change(self) -> bool:
        """O mouse entrou, saiu ou apertou um botão desde a última vez? (pede redesenho)"""
        ch, self._hover_changed = getattr(self, "_hover_changed", False), False
        return ch

    def pop_click(self) -> Optional[str]:
        """O botão clicado desde a última vez ('carteira') ou None."""
        c, self._click = self._click, None
        return c

    def is_open(self) -> bool:
        """Falso só se a janela já foi vista aberta e agora sumiu (bolinha vermelha).

        Em alguns sistemas o OpenCV não informa a propriedade (devolve -1 sempre);
        nesse caso a janela é considerada aberta e o painel fecha só com q/Esc.
        """
        from ..platform.appmode import was_closed

        if was_closed(self.NAME):
            return False
        try:
            v = self.cv2.getWindowProperty(self.NAME, self.cv2.WND_PROP_VISIBLE)
        except Exception:
            return True
        if v >= 1:
            self._seen_open = True
            return True
        return not self._seen_open

    def close(self) -> None:
        try:
            self.cv2.destroyWindow(self.NAME)
        except Exception:
            pass
