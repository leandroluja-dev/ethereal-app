"""Painel do CandleSense: uma janela pequena, sempre por cima, com a decisão.

Desenhado com Pillow (acentos e fontes do sistema) e mostrado pelo OpenCV,
que já é usado na janela de debug. A mesma função gera a imagem para testes
e para o modo ``debug`` com capturas salvas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..engine.rules import COMPRA, INCERTA, NAO_ENTRAR, VENDA, Decision
from ..models import ChartReading

W = 440
PAD = 14

BG = (24, 26, 31)
FG = (232, 234, 238)
DIM = (150, 156, 166)
LINE = (52, 56, 64)
GREEN = (46, 170, 98)
RED = (222, 76, 64)
GREY = (88, 94, 104)
ORANGE = (214, 140, 40)

HEAD_COLOR = {COMPRA: GREEN, VENDA: RED, NAO_ENTRAR: GREY, INCERTA: ORANGE}

_FONTS = {
    False: [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    True: [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
}


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False):
    for path in _FONTS[bold]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=1 if bold and path.endswith(".ttc") else 0)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow antigo
        return ImageFont.load_default()


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%".replace(".", ",")


def _mmss(s: int) -> str:
    return f"{s // 60:02d}:{s % 60:02d}"


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
    def __init__(self, scale: float, height: int = 1400) -> None:
        self.s = scale
        self.w = int(W * scale)
        self.img = Image.new("RGB", (self.w, int(height * scale)), BG)
        self.d = ImageDraw.Draw(self.img)
        self.y = 0

    def px(self, v: float) -> int:
        return int(round(v * self.s))

    def text(self, x: float, txt: str, size: int, color=FG, bold: bool = False, dy: float = 0) -> None:
        self.d.text((self.px(x), self.y + self.px(dy)), txt, font=_font(self.px(size), bold), fill=color)

    def para(self, txt: str, size: int, color=FG, x: float = PAD, width: Optional[float] = None,
             bold: bool = False, gap: float = 3) -> None:
        f = _font(self.px(size), bold)
        width = width if width is not None else W - x - PAD
        for ln in _wrap(self.d, txt, f, self.px(width)):
            self.d.text((self.px(x), self.y), ln, font=f, fill=color)
            self.y += self.px(size + gap)

    def space(self, v: float) -> None:
        self.y += self.px(v)

    def rule(self) -> None:
        self.space(6)
        self.d.line([(self.px(PAD), self.y), (self.w - self.px(PAD), self.y)], fill=LINE, width=max(1, self.px(1)))
        self.space(8)

    def crop(self) -> np.ndarray:
        img = self.img.crop((0, 0, self.w, self.y + self.px(PAD)))
        return np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR (OpenCV)


@dataclass
class PanelExtras:
    """O que o painel mostra além da decisão: carteira, operação aberta e alerta de saída."""

    wallet_line: Optional[str] = None
    wallet_profit: Optional[float] = None
    open_line: Optional[str] = None
    exit_alert: Optional[str] = None
    details: bool = False       # explicação dos módulos (tecla d)
    chart: bool = True          # mini gráfico ao vivo


def _mini_chart(c: "_Canvas", reading: ChartReading, n: int = 36, h: float = 92) -> None:
    """As velas que o app está lendo, redesenhadas: confira se batem com a IQ Option."""
    cs = [k for k in (reading.candles or []) if k.high is not None and k.low is not None][-n:]
    if len(cs) < 3:
        return
    lo = min(k.low for k in cs)
    hi = max(k.high for k in cs)
    if reading.current_price is not None:
        lo, hi = min(lo, reading.current_price), max(hi, reading.current_price)
    if hi <= lo:
        return
    x0, x1 = PAD, W - PAD - 58
    top = c.y / c.s
    step = (x1 - x0) / n
    bw = max(2.0, step * 0.62)

    def ypx(v: float) -> int:
        return c.px(top + 4 + (hi - v) / (hi - lo) * (h - 8))

    c.d.rectangle([c.px(x0), c.px(top), c.px(W - PAD), c.px(top + h)], fill=(30, 33, 39))
    for i, k in enumerate(cs):
        cx = x0 + (n - len(cs) + i + 0.5) * step
        up = k.close >= k.open
        col = GREEN if up else RED
        c.d.line([(c.px(cx), ypx(k.high)), (c.px(cx), ypx(k.low))], fill=col, width=max(1, c.px(1)))
        y_a, y_b = sorted((ypx(k.open), ypx(k.close)))
        c.d.rectangle([c.px(cx - bw / 2), y_a, c.px(cx + bw / 2), max(y_b, y_a + 1)],
                      fill=col if not k.forming else None, outline=col)
    if reading.current_price is not None:
        y = ypx(reading.current_price)
        c.d.line([(c.px(x0), y), (c.px(x1 + 4), y)], fill=(200, 200, 210), width=1)
        dec = reading.scale.decimals if reading.scale else 5
        c.d.text((c.px(x1 + 8), y - c.px(7)), f"{reading.current_price:.{dec}f}",
                 font=_font(c.px(11)), fill=FG)
    c.y = c.px(top + h + 4)
    c.text(PAD, "Leitura ao vivo das velas (confira se bate com a IQ Option)", 11, DIM)
    c.space(16)


def _gauge(c: "_Canvas", d: Decision) -> None:
    import math

    st = d.strength or 0.0
    r = 70
    cx, cy = PAD + r + 6, c.y / c.s + r + 6  # em unidades do painel
    box = [c.px(cx - r), c.px(cy - r), c.px(cx + r), c.px(cy + r)]
    # Cinco faixas de 180° a 0° (esquerda = venda, direita = compra)
    segs = [(180, 144, RED), (144, 112, (150, 70, 64)), (112, 68, GREY), (68, 36, (60, 130, 90)), (36, 0, GREEN)]
    for a0, a1, col in segs:
        # PIL mede o ângulo no sentido horário a partir das 3h: 180..360 é a metade de cima
        c.d.arc(box, start=360 - a0, end=360 - a1, fill=col, width=max(2, c.px(12)))
    ang = math.radians(90 - 90 * st)  # st=-1 -> 180°, st=+1 -> 0°
    nx, ny = cx + (r - 18) * math.cos(ang), cy - (r - 18) * math.sin(ang)
    c.d.line([(c.px(cx), c.px(cy)), (c.px(nx), c.px(ny))], fill=FG, width=max(2, c.px(4)))
    c.d.ellipse([c.px(cx - 6), c.px(cy - 6), c.px(cx + 6), c.px(cy + 6)], fill=FG)
    # Texto à direita do medidor
    tx = cx + r + 22
    y0 = c.y
    label = d.strength_label
    col = GREEN if st > 0.2 else RED if st < -0.2 else FG
    c.y = y0 + c.px(8)
    c.text(tx, "Força agora", 13, DIM)
    c.y = y0 + c.px(26)
    c.text(tx, label, 24, col, bold=True)
    c.y = y0 + c.px(60)
    lado = "alta" if d.best.p_up >= 0.5 else "baixa"
    c.text(tx, f"{_pct(d.best.prob)} de {lado} em {d.expiry_text()}", 14, FG)
    c.y = y0 + c.px(80)
    if d.threshold is not None:
        c.text(tx, f"forte a partir de {_pct(d.threshold)}", 12, DIM)
    c.y = y0 + c.px(r + 22)
    c.rule()


def render(decision: Decision, reading: Optional[ChartReading] = None, scale: float = 1.0,
           extras: Optional[PanelExtras] = None) -> np.ndarray:
    """Imagem BGR do painel."""
    c = _Canvas(scale)
    d = decision
    ex = extras or PanelExtras(details=True, chart=False)
    head = HEAD_COLOR.get(d.action, GREY)

    # Faixa principal
    hh = 86
    c.d.rectangle([0, 0, c.w, c.px(hh)], fill=head)
    c.y = c.px(10)
    c.text(PAD, d.action, 34, (255, 255, 255), bold=True)
    if d.signal and d.best is not None:
        c.text(PAD, f"{_pct(d.best.prob)}  ·  {d.expiry_text()}", 18, (255, 255, 255), dy=46)
    elif d.action == INCERTA:
        c.text(PAD, "não sinaliza enquanto a leitura não fecha", 16, (255, 255, 255), dy=48)
    elif d.best is not None:
        lado = "alta" if d.best.direction == COMPRA else "baixa"
        c.text(PAD, f"mais provável: {lado} {_pct(d.best.prob)} em {d.expiry_text()}", 16, (255, 255, 255), dy=48)
    c.y = c.px(hh)

    # Alerta de saída: faixa vermelha logo abaixo da decisão
    if ex.exit_alert:
        c.d.rectangle([0, c.y, c.w, c.y + c.px(30)], fill=(170, 30, 30))
        c.space(6)
        c.text(PAD, "SAIR? " + ex.exit_alert, 14, (255, 255, 255), bold=True)
        c.space(24)
    c.space(10)

    # Medidor de força (venda forte ... compra forte)
    if d.strength is not None:
        _gauge(c, d)

    if ex.chart and reading is not None and reading.ok:
        _mini_chart(c, reading)

    # Carteira (deduzida do saldo da tela)
    if ex.wallet_line or ex.open_line:
        if ex.open_line:
            c.para(ex.open_line, 14, ORANGE, bold=True)
        if ex.wallet_line:
            col = FG if not ex.wallet_profit else GREEN if ex.wallet_profit > 0 else RED
            c.para(ex.wallet_line, 13, col)
        c.para("c: carteira e placar  ·  d: detalhes  ·  q: fechar", 11, DIM)
        c.rule()

    # Tempo para entrar e fase da vela
    if d.action != INCERTA:
        left = "—" if d.seconds_to_enter is None else _mmss(d.seconds_to_enter)
        c.text(PAD, "Clique em até", 13, DIM)
        c.text(PAD + 150, "Fase da vela", 13, DIM)
        c.text(PAD + 290, "Mínimo p/ sinal", 13, DIM)
        c.space(17)
        c.text(PAD, left, 22, FG, bold=True)
        c.text(PAD + 150, d.phase or "—", 22, FG, bold=True)
        c.text(PAD + 290, _pct(d.threshold) if d.threshold else "—", 22, FG, bold=True)
        c.space(30)
        if d.best is not None:
            fmt = f"{{:.{d.decimals}f}}"
            alvo = fmt.format(d.target) if d.target is not None else "—"
            c.para(f"Expiração: {d.expiry_text()} · alvo de referência {alvo}", 14, FG)
            c.para(f"Saída: {d.exit_mode}", 14, FG)
            if d.expiry_hint:
                c.para(d.expiry_hint, 13, ORANGE if "sugerida é" in d.expiry_hint else DIM)
        c.rule()

        # Probabilidade por expiração
        if d.horizons:
            c.text(PAD, "Probabilidade de alta por expiração", 13, DIM)
            c.space(19)
            cw = (W - 2 * PAD) / len(d.horizons)
            for i, h in enumerate(d.horizons):
                x = PAD + i * cw
                lab = f"{h.candles} vela" + ("s" if h.candles > 1 else "")
                c.text(x, lab, 12, DIM)
                col = GREEN if h.p_up > 0.5 else RED if h.p_up < 0.5 else FG
                c.text(x, _pct(h.p_up), 18, col, bold=True, dy=15)
            c.space(42)
            if d.breakeven is not None:
                c.para(f"Equilíbrio do payout: {_pct(d.breakeven)} (abaixo disso perde dinheiro no longo prazo)", 12, DIM)
            c.rule()

        # Módulos
        for m in (d.modules if ex.details else []):
            c.text(PAD, m.name, 14, FG, bold=True)
            bx, bw, by = PAD + 190, W - 2 * PAD - 190, c.y + c.px(6)
            c.d.rectangle([c.px(bx), by, c.px(bx + bw), by + c.px(6)], fill=LINE)
            mid = bx + bw / 2
            end = mid + m.score * m.confidence * bw / 2
            col = GREEN if m.score > 0 else RED
            c.d.rectangle([c.px(min(mid, end)), by, c.px(max(mid, end)), by + c.px(6)], fill=col)
            c.d.line([(c.px(mid), by - c.px(3)), (c.px(mid), by + c.px(9))], fill=DIM, width=max(1, c.px(1)))
            c.space(19)
            c.para(m.explanation, 12, DIM)
            c.space(4)
        if ex.details:
            c.rule()

    # Por que não entrar / motivo da incerteza
    if d.reasons:
        c.text(PAD, "Por que não entrar agora" if d.action != INCERTA else "Motivo", 13, DIM)
        c.space(18)
        for r in d.reasons[:4]:
            c.para("• " + r, 13, FG)
        c.rule()

    status = "leitura OK" if reading is None or reading.ok else "leitura incerta"
    if reading is not None and reading.timings_ms:
        status += f" · {sum(reading.timings_ms.values()):.0f} ms"
    c.para(status, 12, DIM)
    if d.calibrated:
        c.para("Probabilidade corrigida pelo placar (acertos e erros já conferidos). "
               "O CandleSense só lê a tela: nunca clica e nunca opera.", 11, DIM)
    else:
        c.para("Probabilidade por regras, ainda sem correção: o placar está juntando dados. "
               "O CandleSense só lê a tela: nunca clica e nunca opera.", 11, DIM)
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
        self.place: Optional[tuple[int, int]] = None  # canto de cima à esquerda, em pontos da tela

    def _topmost(self) -> None:
        try:
            self.cv2.setWindowProperty(self.NAME, self.cv2.WND_PROP_TOPMOST, 1)
        except Exception:  # backend sem suporte: a janela fica normal
            pass

    def show(self, decision: Decision, reading: Optional[ChartReading] = None,
             extras: Optional[PanelExtras] = None) -> None:
        self.cv2.imshow(self.NAME, render(decision, reading, self.scale, extras))
        if not self._placed and self.place is not None:
            try:
                self.cv2.moveWindow(self.NAME, *self.place)
            except Exception:
                pass
            self._placed = True

    def is_open(self) -> bool:
        """Falso só se a janela já foi vista aberta e agora sumiu (bolinha vermelha).

        Em alguns sistemas o OpenCV não informa a propriedade (devolve -1 sempre);
        nesse caso a janela é considerada aberta e o painel fecha só com q/Esc.
        """
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
