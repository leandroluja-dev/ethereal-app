"""Painel do CandleSense: uma janela pequena, sempre por cima, com a decisão.

Desenhado com Pillow (acentos e fontes do sistema) e mostrado pelo OpenCV,
que já é usado na janela de debug. A mesma função gera a imagem para testes
e para o modo ``debug`` com capturas salvas.
"""

from __future__ import annotations

import os
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
    def __init__(self, scale: float) -> None:
        self.s = scale
        self.w = int(W * scale)
        self.img = Image.new("RGB", (self.w, int(1400 * scale)), BG)
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


def render(decision: Decision, reading: Optional[ChartReading] = None, scale: float = 1.0) -> np.ndarray:
    """Imagem BGR do painel."""
    c = _Canvas(scale)
    d = decision
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
    c.y = c.px(hh + 10)

    # Tempo para entrar e fase da vela
    if d.action != INCERTA:
        left = "—" if d.seconds_to_enter is None else _mmss(d.seconds_to_enter)
        c.text(PAD, "Entrar em até", 13, DIM)
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
        for m in d.modules:
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
    c.para("Probabilidade por regras, ainda sem calibração (o placar vem na Fase 3). "
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

    def _topmost(self) -> None:
        try:
            self.cv2.setWindowProperty(self.NAME, self.cv2.WND_PROP_TOPMOST, 1)
        except Exception:  # backend sem suporte: a janela fica normal
            pass

    def show(self, decision: Decision, reading: Optional[ChartReading] = None) -> None:
        self.cv2.imshow(self.NAME, render(decision, reading, self.scale))

    def is_open(self) -> bool:
        try:
            # -1 = a janela não existe mais (fechada pela bolinha vermelha).
            return self.cv2.getWindowProperty(self.NAME, self.cv2.WND_PROP_AUTOSIZE) >= 0
        except Exception:
            return True

    def close(self) -> None:
        try:
            self.cv2.destroyWindow(self.NAME)
        except Exception:
            pass
