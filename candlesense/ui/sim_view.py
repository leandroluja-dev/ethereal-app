"""Telas 2 e 3 ao lado do painel: as entradas do simulador (carteira do app).

* Tela 2, "Entradas do app": entradas virtuais abertas ao vivo (valor, quanto
  ganha se acertar, ganhando ou perdendo agora, quanto falta), o aviso de
  GANHOU / PERDEU e as últimas entradas.
* Tela 3, "Resultados do dia": resultado, acerto, por par, pela chance que a
  entrada mostrava e a lista de todas as entradas de hoje.

As entradas são de mentira: o app nunca clica nem opera na IQ Option.
Mesmo visual do painel (fundo preto, cartões arredondados, fonte Inter).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .panel import (BLACK, CARD, CARD_2, DIM, FAINT, FG, GAP, GREEN, LINE, ORANGE, PAD, R_CARD, RED, W, WHITE,
                    _Canvas, _icon, _mmss)

GAP_PX = 6  # faixa preta entre as telas (pixels de saída, antes da escala)
FLASH_S = 25  # o aviso de GANHOU / PERDEU fica no topo por este tempo
WEEK = ("segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo")
BANDS = ((0.62, 0.64), (0.64, 0.66), (0.66, 0.70), (0.70, 1.01))


@dataclass
class SimState:
    """O que as telas mostram. ``entries`` vem de StrongMode.entries (mais antiga primeiro)."""

    entries: list = field(default_factory=list)
    now: float = 0.0
    asset: Optional[str] = None          # par aberto na IQ agora
    price: Optional[float] = None        # preço atual lido (só se a leitura está ok)
    decimals: int = 5
    hist_min: Optional[int] = None       # minutos seguidos de gráfico de 1 min deste par
    hist_need: int = 135
    note: Optional[str] = None           # por que o app não mede agora (OTC, outro tempo de vela...)
    available: bool = True               # simulador ligado
    stake: float = 10.0
    threshold: float = 0.62
    last_p: Optional[float] = None       # última chance de subir calculada pelo modelo neste par
    last_ts: Optional[float] = None

    def key(self) -> tuple:
        """Muda quando algo na tela muda (para não redesenhar à toa)."""
        live = tuple((e["start"], e["status"], e.get("seen_at")) for e in self.entries)
        return (live, int(self.now), self.asset, self.price, self.hist_min, self.note, self.available, self.last_ts)


# ---------------------------------------------------------------------------
# Formatos
# ---------------------------------------------------------------------------

def brl(v: float, sign: bool = True) -> str:
    s = "+" if v > 0 and sign else "-" if v < 0 else ""
    txt = f"{abs(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s}R$ {txt}"


def _pc(p: float) -> str:
    return f"{p * 100:.0f}%"


def _hhmm(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def live_result(e: dict, asset: Optional[str], price: Optional[float]) -> tuple[Optional[float], Optional[str]]:
    """Entrada aberta se vencesse agora: (lucro, 'ganhando'|'perdendo'|'empatando'). Só com o par na tela."""
    if price is None or asset != e["asset"]:
        return None, None
    diff = price - e["price"] if e["side"] == "COMPRA" else e["price"] - price
    if abs(diff) < 1e-12:
        return 0.0, "empatando"
    return (e["gain"], "ganhando") if diff > 0 else (-e["stake"], "perdendo")


def day_summary(entries: list) -> dict:
    done = [e for e in entries if e["status"] in ("ganhou", "perdeu", "empate")]
    wins = sum(e["status"] == "ganhou" for e in done)
    losses = sum(e["status"] == "perdeu" for e in done)
    return {"n": len(entries), "done": len(done), "wins": wins, "losses": losses,
            "ties": sum(e["status"] == "empate" for e in done),
            "open": sum(e["status"] == "aberta" for e in entries),
            "profit": sum(e["profit"] or 0.0 for e in done),
            "acc": wins / (wins + losses) if wins + losses else None}


def groups(entries: list) -> tuple[list, list]:
    """Resumo por par e por faixa de chance: [(nome, resumo)]."""
    pares: dict[str, list] = {}
    for e in entries:
        pares.setdefault(e["asset"], []).append(e)
    by_pair = sorted(((a, day_summary(es)) for a, es in pares.items()), key=lambda x: -x[1]["n"])
    by_band = []
    for lo, hi in BANDS:
        es = [e for e in entries if lo <= e["chance"] < hi]
        if es:
            name = f"{lo * 100:.0f}–{hi * 100:.0f}%" if hi <= 1 else f"{lo * 100:.0f}% ou mais"
            by_band.append((name, day_summary(es)))
    return by_pair, by_band


# ---------------------------------------------------------------------------
# Peças
# ---------------------------------------------------------------------------

def _badge(c: _Canvas, right: float, cy: float, lab: str, col) -> None:
    bw = c.tw(lab, 11, "bold") + 16
    c.rrect(right - bw, cy - 9, right, cy + 9, 9, outline=col, width=1.2)
    c.at(right - bw / 2, cy, lab, 11, col, "bold", anchor="mm")


def _head(c: _Canvas, title: str, sub: str, badge: Optional[str] = None) -> None:
    c.at(PAD, 16, title, 19, FG, "semibold")
    c.at(PAD, 41, sub, 13, DIM)
    if badge:
        _badge(c, W - PAD, 32, badge, ORANGE)
    c.y = c.px(70)


def _label(c: _Canvas, txt: str) -> None:
    c.at(PAD + 4, c.u, txt, 11, DIM, "semibold")
    c.space(20)


def _col(v: float):
    return GREEN if v > 0 else RED if v < 0 else DIM


def _ring_pct(c: _Canvas, cx: float, cy: float, r: float, wins: int, losses: int) -> None:
    box = [c.px(cx - r), c.px(cy - r), c.px(cx + r), c.px(cy + r)]
    wd = c.px(5)
    c.d.arc(box, 0, 360, fill=CARD_2, width=wd)
    done = wins + losses
    if done:
        a = 360 * wins / done
        if wins:
            c.d.arc(box, -90, -90 + a, fill=GREEN, width=wd)
        if losses:
            c.d.arc(box, -90 + a, 270, fill=RED, width=wd)
        c.at(cx, cy - 3, _pc(wins / done), 15, FG, "bold", anchor="mm")
        c.at(cx, cy + 12, "acerto", 10, DIM, anchor="mm")
    else:
        c.at(cx, cy, "—", 16, DIM, "bold", anchor="mm")


def _today_strip(c: _Canvas, s: dict) -> None:
    """Carteira do app hoje: resultado e placar."""
    y0, h = c.u, 64
    c.card(y0, y0 + h)
    inner, right, cy = PAD + 18, W - PAD - 16, y0 + h / 2
    c.at(inner, cy - 11, "Carteira do app hoje", 12, DIM, anchor="lm")
    c.at(inner, cy + 10, brl(s["profit"]) if s["done"] else "R$ 0,00", 22, _col(s["profit"]), "bold", anchor="lm")
    c.at(right, cy - 11, f"{s['wins']} ganhos · {s['losses']} perdas", 12, DIM, anchor="rm")
    acc = f"acerto {_pc(s['acc'])}" if s["acc"] is not None else "sem resultado ainda"
    c.at(right, cy + 10, acc, 14, FG, "semibold", anchor="rm")
    c.y = c.px(y0 + h + GAP)


def _flash(c: _Canvas, e: dict) -> None:
    """Aviso grande logo depois do resultado."""
    y0, h = c.u, 96
    won = e["status"] == "ganhou"
    tie = e["status"] == "empate"
    bg, fg = (GREEN, BLACK) if won else ((CARD_2, FG) if tie else (RED, WHITE))
    c.rrect(PAD, y0, W - PAD, y0 + h, R_CARD, fill=bg)
    inner, right = PAD + 20, W - PAD - 20
    title = "GANHOU" if won else "EMPATE" if tie else "PERDEU"
    c.at(inner, y0 + 46, title, 30, fg, "bold", anchor="ls")
    c.at(right, y0 + 46, brl(e["profit"] or 0.0), 26, fg, "bold", anchor="rs")
    arrow = "▲" if e["side"] == "COMPRA" else "▼"
    c.at(inner, y0 + 74, f"{e['side']} {arrow} {e['asset']} · chance {_pc(e['chance'])} · {_hhmm(e['ts'])}",
         13, fg, "medium", anchor="ls")
    c.y = c.px(y0 + h + GAP)


def _open_card(c: _Canvas, e: dict, st: SimState) -> None:
    """Entrada aberta ao vivo: valor, quanto ganha, ganhando/perdendo agora e quanto falta."""
    profit, state = live_result(e, st.asset, st.price)
    col = GREEN if state == "ganhando" else RED if state == "perdendo" else FG
    y0, h = c.u, 128
    c.card(y0, y0 + h)
    c.rrect(PAD, y0, W - PAD, y0 + h, R_CARD, outline=col if state else LINE, width=1.5)
    inner, right = PAD + 18, W - PAD - 18
    buy = e["side"] == "COMPRA"
    _icon(c, "up" if buy else "down", inner, y0 + 15, GREEN if buy else RED, 13)
    c.at(inner + 18, y0 + 14, f"{e['side']} · {e['asset']} · chance {_pc(e['chance'])}", 12, FG, "semibold")
    left = max(0, int(e["expiry_ts"] - st.now))
    c.at(right, y0 + 14, "vence em", 12, DIM, anchor="ra")
    c.at(right, y0 + 58, _mmss(left), 26, FG, "bold", anchor="rs")
    if profit is not None:
        c.at(inner, y0 + 58, brl(profit), 30, col, "bold", anchor="ls")
        c.at(inner + c.tw(brl(profit), 30, "bold") + 10, y0 + 56, state, 14, col, "semibold", anchor="ls")
    else:
        c.at(inner, y0 + 58, "em jogo", 26, FG, "bold", anchor="ls")
    c.at(inner, y0 + 76, f"Entrou {brl(e['stake'], False)} · ganha {brl(e['gain'], False)} se acertar",
         13, FG, "medium")
    d = st.decimals
    if st.price is not None and st.asset == e["asset"]:
        diff = st.price - e["price"]
        txt = f"entrada {e['price']:.{d}f} → agora {st.price:.{d}f} ({'+' if diff >= 0 else '-'}{abs(diff):.{d}f})"
    else:
        txt = f"entrada {e['price']:.{d}f} · abra {e['asset']} para ver ao vivo"
    c.at(inner, y0 + 98, txt, 12, DIM)
    c.y = c.px(y0 + h + GAP)


def _waiting_card(c: _Canvas, st: SimState) -> None:
    y0, h = c.u, 116
    c.card(y0, y0 + h)
    inner = PAD + 18
    c.at(inner, y0 + 18, "Nenhuma entrada aberta", 17, FG, "semibold")
    c.y = c.px(y0 + 46)
    if not st.available:
        msg = "Simulador indisponível nesta versão."
    elif st.note:
        msg = st.note
    elif st.hist_min is not None and st.hist_min < st.hist_need:
        msg = (f"Juntando histórico do gráfico de 1 min: {st.hist_min} de {st.hist_need} min. "
               "Deixe o gráfico aberto neste par.")
    else:
        msg = f"O app entra sozinho quando a chance chega a {_pc(st.threshold)}. Expiração de 5 min."
    fresh = (st.available and not st.note and st.last_p is not None and st.last_ts is not None
             and st.now - st.last_ts < 180)
    if fresh:
        side = "compra" if st.last_p >= 0.5 else "venda"
        ch = max(st.last_p, 1 - st.last_p)
        msg = (f"Última conta do modelo ({_hhmm(st.last_ts)}): {side} {_pc(ch)}. "
               f"Ele entra a partir de {_pc(st.threshold)}.")
    c.para(msg, 13, DIM, x=inner, width=W - 2 * inner, gap=6)
    if fresh:
        # barra de 50% a 75%: onde a última conta ficou e a marca do corte
        by = y0 + h - 20
        lo, hi = 0.5, 0.75
        span = W - 2 * inner
        c.rrect(inner, by, W - inner, by + 6, 3, fill=CARD_2)
        f = max(0.0, min(1.0, (ch - lo) / (hi - lo)))
        if f > 0:
            c.rrect(inner, by, inner + span * f, by + 6, 3, fill=GREEN if ch >= st.threshold else DIM)
        mx = inner + span * (st.threshold - lo) / (hi - lo)
        c.d.line([(c.px(mx), c.px(by - 4)), (c.px(mx), c.px(by + 10))], fill=FG, width=c.px(1.5))
        c.y = c.px(y0 + h + GAP)
        return
    if st.available and not st.note and st.hist_min is not None and st.hist_min < st.hist_need:
        by = y0 + h - 20
        c.rrect(inner, by, W - inner, by + 6, 3, fill=CARD_2)
        f = max(0.0, min(1.0, st.hist_min / st.hist_need))
        if f > 0:
            c.rrect(inner, by, inner + (W - 2 * inner) * f, by + 6, 3, fill=ORANGE)
    c.y = c.px(y0 + h + GAP)


def _row(c: _Canvas, y: float, e: dict, inner: float, right: float, show_side: bool = True) -> None:
    """Linha de uma entrada: hora e par, lado e chance, resultado em R$."""
    stt = e["status"]
    col = GREEN if stt == "ganhou" else RED if stt == "perdeu" else ORANGE if stt == "aberta" else DIM
    c.d.ellipse([c.px(inner), c.px(y + 7), c.px(inner + 8), c.px(y + 15)], fill=col)
    c.at(inner + 16, y + 2, f"{_hhmm(e['ts'])}  {e['asset']}", 13, FG, "medium")
    arrow = "▲" if e["side"] == "COMPRA" else "▼"
    c.at(inner + 16, y + 20, f"{e['side'].lower()} {arrow} · mostrava {_pc(e['chance'])}", 12, DIM)
    if stt == "aberta":
        val, sub = f"{brl(e['stake'], False)}", f"ganha {brl(e['gain'], False)}"
    elif stt == "sem leitura":
        val, sub = "—", "sem leitura"
    else:
        val, sub = brl(e["profit"] or 0.0), stt
    c.at(right, y + 2, val, 14, col if stt in ("ganhou", "perdeu") else FG, "semibold", anchor="ra")
    c.at(right, y + 20, sub, 12, DIM, anchor="ra")


def _list_card(c: _Canvas, rows: list, bottom: float, empty: str) -> int:
    """Cartão com linhas de entradas até ``bottom``. Devolve quantas couberam."""
    y0 = c.u
    row_h = 44
    room = int((bottom - y0 - 20) // row_h)
    k = max(0, min(len(rows), room))
    h = 20 + max(1, k) * row_h
    if bottom - y0 < 64:
        return 0
    c.card(y0, y0 + h)
    inner, right = PAD + 16, W - PAD - 16
    if not rows:
        c.at(W / 2, y0 + h / 2, empty, 13, DIM, anchor="mm")
    for i, e in enumerate(rows[:k]):
        y = y0 + 12 + i * row_h
        if i:
            c.d.line([(c.px(inner), c.px(y - 6)), (c.px(right), c.px(y - 6))], fill=LINE, width=c.px(1))
        _row(c, y, e, inner, right)
    c.y = c.px(y0 + h + GAP)
    return k


def _footer(c: _Canvas, height: float, txt: str) -> float:
    """Rodapé colado no fim da tela; devolve o y onde o conteúdo deve parar."""
    c.at(PAD, height - PAD - 12, txt, 11, FAINT)
    return height - PAD - 22


# ---------------------------------------------------------------------------
# Telas
# ---------------------------------------------------------------------------

def render_live(st: SimState, height: float, scale: float = 1.0) -> np.ndarray:
    """Tela 2: entradas em tempo real."""
    c = _Canvas(scale, height=int(height) + 40)
    _head(c, "Entradas do app", f"carteira do app · {brl(st.stake, False)} por entrada", "SIMULAÇÃO")
    s = day_summary(st.entries)
    _today_strip(c, s)
    # o aviso aparece quando a conferência chega (``seen_at``, marcado pelo app) e fica FLASH_S segundos
    flash = [e for e in st.entries if e["status"] in ("ganhou", "perdeu", "empate")
             and e.get("seen_at") is not None and st.now - e["seen_at"] <= FLASH_S]
    if flash:
        _flash(c, flash[-1])
    bottom = _footer(c, height, "Conta de mentira: o app nunca clica nem opera.")
    opened = [e for e in st.entries if e["status"] == "aberta"]
    _label(c, "AO VIVO")
    if opened:
        for e in opened[-2:][::-1]:
            if c.u + 128 > bottom:
                break
            _open_card(c, e, st)
    else:
        _waiting_card(c, st)
    done = [e for e in st.entries if e["status"] != "aberta"][::-1]
    if bottom - c.u > 90:
        _label(c, "ÚLTIMAS ENTRADAS")
        _list_card(c, done, bottom, "nenhuma entrada conferida hoje")
    return _fit(c, height)


def render_day(st: SimState, height: float, scale: float = 1.0) -> np.ndarray:
    """Tela 3: resultados do dia, onde ganhou e onde perdeu, e a chance que cada entrada mostrava."""
    c = _Canvas(scale, height=int(height) + 40)
    lt = time.localtime(st.now or time.time())
    _head(c, "Resultados do dia", f"{WEEK[lt.tm_wday]}, {lt.tm_mday:02d}/{lt.tm_mon:02d} · desde 00:00")
    s = day_summary(st.entries)
    # resumo
    y0, h = c.u, 112
    c.card(y0, y0 + h)
    inner = PAD + 18
    c.at(inner, y0 + 18, "Resultado", 12, DIM)
    c.at(inner, y0 + 66, brl(s["profit"]) if s["done"] else "R$ 0,00", 32, _col(s["profit"]), "bold", anchor="ls")
    parts = [f"{s['n']} entradas", f"{s['wins']} ganhos", f"{s['losses']} perdas"]
    if s["ties"]:
        parts.append(f"{s['ties']} empates")
    c.at(inner, y0 + 82, " · ".join(parts), 12, DIM)
    _ring_pct(c, W - PAD - 52, y0 + h / 2, 34, s["wins"], s["losses"])
    c.y = c.px(y0 + h + GAP)
    bottom = _footer(c, height, "Conta de mentira: o app nunca clica nem opera.")
    by_pair, by_band = groups(st.entries)
    for title, rows in (("POR PAR", by_pair), ("PELA CHANCE QUE A ENTRADA MOSTRAVA", by_band)):
        if not rows or bottom - c.u < 90:
            continue
        _label(c, title)
        y0 = c.u
        row_h = 30
        k = min(len(rows), max(1, int((bottom - y0 - 70) // row_h)))
        h = 16 + k * row_h
        c.card(y0, y0 + h)
        right = W - PAD - 16
        for i, (name, g) in enumerate(rows[:k]):
            y = y0 + 8 + i * row_h + row_h / 2
            c.at(PAD + 16, y, name, 13, FG, "medium", anchor="lm")
            acc = _pc(g["acc"]) if g["acc"] is not None else "—"
            c.at(PAD + 168, y, f"{g['wins']}G {g['losses']}P · {acc}", 12, DIM, anchor="lm")
            c.at(right, y, brl(g["profit"]) if g["done"] else "aberta", 13, _col(g["profit"]), "semibold",
                 anchor="rm")
        c.y = c.px(y0 + h + GAP)
    if bottom - c.u > 90:
        _label(c, "ENTRADA POR ENTRADA")
        _list_card(c, st.entries[::-1], bottom, "nenhuma entrada hoje")
    return _fit(c, height)


def _fit(c: _Canvas, height: float) -> np.ndarray:
    c.y = c.px(height - PAD)
    return c.crop(min_h=height)


def side_screens(st: SimState, height_px: int, scale: float = 1.0) -> np.ndarray:
    """As duas telas novas lado a lado, na altura exata do painel (pixels de saída)."""
    h_units = height_px / scale
    a = render_live(st, h_units, scale)
    b = render_day(st, h_units, scale)
    gap = np.zeros((height_px, max(1, int(round(GAP_PX * scale))), 3), np.uint8)
    imgs = []
    for im in (a, b):
        if im.shape[0] != height_px:
            if im.shape[0] > height_px:
                im = im[:height_px]
            else:
                im = np.vstack([im, np.zeros((height_px - im.shape[0], im.shape[1], 3), np.uint8)])
        imgs.append(im)
    return np.hstack([gap, imgs[0], gap, imgs[1]])


class SideScreens:
    """Guarda a última imagem e só redesenha quando o conteúdo muda (no máximo 1 vez por segundo)."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.state: Optional[SimState] = None
        self.visible = True
        self._key = None
        self._img: Optional[np.ndarray] = None

    def image(self, height_px: int) -> Optional[np.ndarray]:
        if not self.visible or self.state is None:
            return None
        key = (self.state.key(), height_px)
        if key != self._key or self._img is None:
            self._img = side_screens(self.state, height_px, self.scale)
            self._key = key
        return self._img


__all__ = ["SimState", "SideScreens", "side_screens", "render_live", "render_day", "brl", "day_summary",
           "groups", "live_result", "CARD"]
