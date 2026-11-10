"""Telas ao lado do painel: as carteiras do simulador (entradas de mentira).

* Tela do meio, "Carteiras do app": cinco carteiras empilhadas (A 62%, B 58% e três de risco a 55%: vela de 5 min, vela de 1 min e 30 s), cada uma com cor e
  letra próprias. Mostra a entrada aberta ao vivo (valor, quanto ganha, ganhando ou
  perdendo, quanto falta), o aviso de GANHOU / PERDEU, o resultado de hoje e as
  últimas entradas.
    A  Segura         entra a partir de 62%, vence em 5 min
    B  Vela de 5 min  experimento: entra durante a vela de 5 min (60%), vence no fim dela
    C  Moderada       entra a partir de 58%, vence em 5 min
* Tela da direita, "Resultados do dia": todas comparadas, pela chance que a entrada
  mostrava e a lista de todas as entradas de hoje com a letra da carteira.

O app nunca clica nem opera na IQ Option. Mesmo visual do painel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .panel import (BLACK, CARD, CARD_2, DIM, FAINT, FG, GAP, GREEN, LINE, ORANGE, PAD, R_CARD, RED, W, WHITE,
                    _Canvas, _icon, _mmss)

BLUE = (88, 160, 255)
GAP_PX = 6        # faixa preta entre as telas (pixels de saída, antes da escala)
FLASH_S = 25      # o aviso de GANHOU / PERDEU fica este tempo
MIN_HEIGHT = 1000 # altura mínima da janela com as carteiras (unidades do painel)
WEEK = ("segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo")
BANDS = ((0.58, 0.60), (0.60, 0.62), (0.62, 0.64), (0.64, 0.66), (0.66, 0.70), (0.70, 1.01))


@dataclass(frozen=True)
class WalletSpec:
    key: str
    name: str
    cut: float
    color: tuple
    desc: str
    tag: str
    mode: str = "5min"  # 5min, candle5, 1min ou 30s (ver StrongMode.entries)

    @property
    def candle5(self) -> bool:
        return self.mode == "candle5"


PURPLE = (176, 122, 255)
PINK = (255, 112, 166)
RISK_CUT = 0.55

WALLETS = (
    WalletSpec("A", "Segura", 0.62, GREEN, "a partir de 62% · vence em 5 min", "62%"),
    WalletSpec("B", "Moderada", 0.58, BLUE, "a partir de 58% · vence em 5 min", "58%"),
    WalletSpec("C", "Vela de 5 min", RISK_CUT, ORANGE, "55% · vence no fim da vela de 5 min", "RISCO",
               mode="candle5"),
    WalletSpec("D", "Vela de 1 min", RISK_CUT, PURPLE, "55% · vence no fim da próxima vela", "RISCO", mode="1min"),
    WalletSpec("E", "Rápida 30 s", RISK_CUT, PINK, "55% · entra e sai em 30 segundos", "RISCO", mode="30s"),
)


@dataclass
class SimState:
    """O que as telas mostram. ``entries`` = {letra da carteira: entradas (mais antiga primeiro)}."""

    entries: dict = field(default_factory=dict)
    now: float = 0.0
    asset: Optional[str] = None          # par aberto na IQ agora
    price: Optional[float] = None        # preço atual lido (só se a leitura está ok)
    decimals: int = 5
    hist_min: Optional[int] = None       # minutos seguidos de gráfico de 1 min deste par
    hist_need: int = 135
    note: Optional[str] = None           # por que o app não mede agora (OTC, outro tempo de vela...)
    available: bool = True               # simulador ligado
    stake: float = 10.0
    last_p: Optional[float] = None       # última chance de subir calculada pelo modelo neste par
    last_ts: Optional[float] = None

    def get(self, key: str) -> list:
        return self.entries.get(key, [])

    def key(self) -> tuple:
        """Muda quando algo na tela muda (para não redesenhar à toa)."""
        live = tuple((k, e["start"], e["status"], e.get("seen_at")) for k, es in sorted(self.entries.items())
                     for e in es)
        return (live, int(self.now), self.asset, self.price, self.hist_min, self.note, self.available,
                self.last_ts)


# ---------------------------------------------------------------------------
# Formatos e contas
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


def by_band(entries: list) -> list:
    """Resumo pela chance que a entrada mostrava: [(faixa, resumo)], só faixas com entradas."""
    out = []
    for lo, hi in BANDS:
        es = [e for e in entries if lo <= e["chance"] < hi]
        if es:
            name = f"{lo * 100:.0f}–{hi * 100:.0f}%" if hi <= 1 else f"{lo * 100:.0f}% ou mais"
            out.append((name, day_summary(es)))
    return out


def merged(st: SimState) -> list:
    """Todas as entradas de hoje, uma linha por conta do modelo, com as letras das carteiras que entraram."""
    rows: dict = {}
    for spec in WALLETS:
        for e in st.get(spec.key):
            r = rows.setdefault((e["asset"], e["start"], spec.mode), dict(e, wallets=[]))
            r["wallets"].append(spec)
    return sorted(rows.values(), key=lambda e: e["ts"])


# ---------------------------------------------------------------------------
# Peças
# ---------------------------------------------------------------------------

def _col(v: float):
    return GREEN if v > 0 else RED if v < 0 else DIM


def _chip(c: _Canvas, x: float, cy: float, lab: str, col, anchor_right: bool = False) -> float:
    bw = c.tw(lab, 10, "bold") + 12
    x1 = x - bw if anchor_right else x
    c.rrect(x1, cy - 8, x1 + bw, cy + 8, 8, outline=col, width=1.1)
    c.at(x1 + bw / 2, cy, lab, 10, col, "bold", anchor="mm")
    return bw


def _letter(c: _Canvas, x: float, cy: float, spec: WalletSpec, r: float = 10) -> None:
    c.d.ellipse([c.px(x), c.px(cy - r), c.px(x + 2 * r), c.px(cy + r)], fill=spec.color)
    c.at(x + r, cy, spec.key, r * 1.2, BLACK, "bold", anchor="mm")


def _head(c: _Canvas, title: str, sub: str) -> None:
    c.at(PAD, 16, title, 19, FG, "semibold")
    c.at(PAD, 41, sub, 13, DIM)
    c.y = c.px(70)


def _label(c: _Canvas, txt: str) -> None:
    c.at(PAD + 4, c.u, txt, 11, DIM, "semibold")
    c.space(20)


def _footer(c: _Canvas, height: float, txt: str) -> float:
    c.at(PAD, height - PAD - 12, txt, 11, FAINT)
    return height - PAD - 22


def _bar(c: _Canvas, x1: float, x2: float, y: float, frac: float, col) -> None:
    c.rrect(x1, y, x2, y + 6, 3, fill=CARD_2)
    frac = max(0.0, min(1.0, frac))
    if frac > 0:
        c.rrect(x1, y, x1 + (x2 - x1) * frac, y + 6, 3, fill=col)


def _wallet_band(c: _Canvas, spec: WalletSpec, st: SimState, y0: float, h: float) -> None:
    """Uma carteira: nome, resultado de hoje, entrada ao vivo (ou o que falta) e as últimas."""
    es = st.get(spec.key)
    s = day_summary(es)
    c.rrect(PAD, y0, W - PAD, y0 + h, R_CARD, fill=CARD, outline=tuple(int(v * 0.55) for v in spec.color), width=1.2)
    inner, right = PAD + 16, W - PAD - 16
    # cabeçalho: letra, nome, etiqueta, resultado
    _letter(c, inner, y0 + 24, spec)
    c.at(inner + 28, y0 + 24, spec.name, 16, FG, "semibold", anchor="lm")
    _chip(c, inner + 28 + c.tw(spec.name, 16, "semibold") + 8, y0 + 24, spec.tag, spec.color)
    c.at(right, y0 + 24, brl(s["profit"]) if s["done"] else "R$ 0,00", 17, _col(s["profit"]), "bold", anchor="rm")
    c.at(inner, y0 + 46, spec.desc, 12, DIM, anchor="lm")
    stats = f"{s['wins']}G {s['losses']}P" + (f" · {_pc(s['acc'])}" if s["acc"] is not None else "")
    c.at(right, y0 + 46, stats, 12, DIM, anchor="rm")
    c.d.line([(c.px(inner), c.px(y0 + 62)), (c.px(right), c.px(y0 + 62))], fill=LINE, width=c.px(1))
    # ao vivo
    ly = y0 + 70
    live_h = 84
    flash = [e for e in es if e["status"] in ("ganhou", "perdeu", "empate")
             and e.get("seen_at") is not None and st.now - e["seen_at"] <= FLASH_S]
    opened = [e for e in es if e["status"] == "aberta"]
    if flash:
        e = flash[-1]
        won, tie = e["status"] == "ganhou", e["status"] == "empate"
        bg, fg = (GREEN, BLACK) if won else ((CARD_2, FG) if tie else (RED, WHITE))
        c.rrect(inner - 6, ly, right + 6, ly + live_h - 6, 14, fill=bg)
        c.at(inner + 6, ly + 34, "GANHOU" if won else "EMPATE" if tie else "PERDEU", 26, fg, "bold", anchor="ls")
        c.at(right - 6, ly + 34, brl(e["profit"] or 0.0), 22, fg, "bold", anchor="rs")
        arrow = "▲" if e["side"] == "COMPRA" else "▼"
        c.at(inner + 6, ly + 58, f"{e['side']} {arrow} {e['asset']} · mostrava {_pc(e['chance'])}", 12, fg, "medium",
             anchor="ls")
    elif opened:
        e = opened[-1]
        profit, state = live_result(e, st.asset, st.price)
        col = GREEN if state == "ganhando" else RED if state == "perdendo" else FG
        buy = e["side"] == "COMPRA"
        _icon(c, "up" if buy else "down", inner, ly + 1, GREEN if buy else RED, 12)
        more = f" · +{len(opened) - 1} aberta" if len(opened) > 1 else ""
        c.at(inner + 16, ly, f"{e['side']} {e['asset']} · {_pc(e['chance'])}{more}", 12, FG, "semibold")
        c.at(right, ly, "vence em", 11, DIM, anchor="ra")
        if profit is not None:
            c.at(inner, ly + 46, brl(profit), 24, col, "bold", anchor="ls")
            c.at(inner + c.tw(brl(profit), 24, "bold") + 8, ly + 44, state, 13, col, "semibold", anchor="ls")
        else:
            c.at(inner, ly + 46, "em jogo", 22, FG, "bold", anchor="ls")
        c.at(right, ly + 46, _mmss(max(0, int(e["expiry_ts"] - st.now))), 22, FG, "bold", anchor="rs")
        c.at(inner, ly + 56, f"entrou {brl(e['stake'], False)} · ganha {brl(e['gain'], False)} se acertar", 12, DIM)
    else:
        c.at(inner, ly, "Aguardando entrada", 14, FG, "semibold")
        fresh = (st.last_p is not None and st.last_ts is not None and st.now - st.last_ts < 180)
        if not st.available:
            msg, frac, bc = "Simulador indisponível nesta versão.", None, DIM
        elif st.note:
            msg, frac, bc = st.note, None, DIM
        elif st.hist_min is not None and st.hist_min < st.hist_need:
            msg = f"Juntando histórico de 1 min: {st.hist_min} de {st.hist_need} min"
            frac, bc = st.hist_min / st.hist_need, ORANGE
        elif fresh:
            ch = max(st.last_p, 1 - st.last_p)
            lado = "compra" if st.last_p >= 0.5 else "venda"
            falta = spec.cut - ch
            msg = (f"Última conta ({_hhmm(st.last_ts)}): {lado} {_pc(ch)} · "
                   + (f"falta {falta * 100:.0f}%" if falta > 0 else "passou do corte"))
            frac, bc = (ch - 0.5) / (spec.cut - 0.5), spec.color
        else:
            msg, frac, bc = f"Entra sozinho quando a chance chegar a {_pc(spec.cut)}.", None, DIM
        f = c.font(12)
        from .panel import _wrap

        for i, ln in enumerate(_wrap(c.d, msg, f, c.px(right - inner))[:2]):
            c.at(inner, ly + 24 + 16 * i, ln, 12, DIM)
        if frac is not None:
            _bar(c, inner, right, ly + 62, frac, bc)
    # últimas entradas (as que cabem)
    done = [e for e in es if e["status"] != "aberta"][::-1]
    ry = y0 + 70 + live_h + 4
    k = int((y0 + h - 10 - ry) // 24)
    if k > 0 and not done:
        c.at(inner, ry + 2, "nenhuma entrada conferida hoje", 12, FAINT)
    for i, e in enumerate(done[:max(0, k)]):
        y = ry + i * 24
        stt = e["status"]
        col = GREEN if stt == "ganhou" else RED if stt == "perdeu" else DIM
        c.d.ellipse([c.px(inner), c.px(y + 5), c.px(inner + 7), c.px(y + 12)], fill=col)
        arrow = "▲" if e["side"] == "COMPRA" else "▼"
        c.at(inner + 14, y, f"{_hhmm(e['ts'])}  {e['asset']}  {arrow} {_pc(e['chance'])}", 12, FG)
        val = brl(e["profit"]) if e["profit"] is not None else "sem leitura"
        c.at(right, y, val, 12, col if stt in ("ganhou", "perdeu") else DIM, "semibold", anchor="ra")


def _fit(c: _Canvas, height: float) -> np.ndarray:
    c.y = c.px(height - PAD)
    return c.crop(min_h=height)


# ---------------------------------------------------------------------------
# Telas
# ---------------------------------------------------------------------------

def render_wallets(st: SimState, height: float, scale: float = 1.0) -> np.ndarray:
    """Tela do meio: as carteiras, uma embaixo da outra (as de risco por último)."""
    c = _Canvas(scale, height=int(height) + 40)
    _head(c, "Carteiras do app", f"{brl(st.stake, False)} por entrada · conta de mentira")
    bottom = _footer(c, height, "Simulação: o app nunca clica nem opera.")
    top = c.u
    bh = (bottom - top - (len(WALLETS) - 1) * GAP) / len(WALLETS)
    for i, spec in enumerate(WALLETS):
        _wallet_band(c, spec, st, top + i * (bh + GAP), bh)
    return _fit(c, height)


def render_day(st: SimState, height: float, scale: float = 1.0) -> np.ndarray:
    """Tela da direita: as carteiras comparadas, pela chance mostrada e entrada por entrada."""
    c = _Canvas(scale, height=int(height) + 40)
    lt = time.localtime(st.now or time.time())
    _head(c, "Resultados do dia", f"{WEEK[lt.tm_wday]}, {lt.tm_mday:02d}/{lt.tm_mon:02d} · desde 00:00")
    bottom = _footer(c, height, "Simulação: o app nunca clica nem opera.")
    inner, right = PAD + 16, W - PAD - 16
    # as carteiras
    y0, rh = c.u, 46
    c.card(y0, y0 + len(WALLETS) * rh + 8)
    for i, spec in enumerate(WALLETS):
        s = day_summary(st.get(spec.key))
        y = y0 + 4 + i * rh
        if i:
            c.d.line([(c.px(inner), c.px(y)), (c.px(right), c.px(y))], fill=LINE, width=c.px(1))
        _letter(c, inner, y + rh / 2, spec, 11)
        c.at(inner + 30, y + rh / 2 - 9, spec.name, 14, FG, "semibold", anchor="lm")
        acc = f"acerto {_pc(s['acc'])}" if s["acc"] is not None else "sem resultado"
        c.at(inner + 30, y + rh / 2 + 10, f"{s['n']} entradas · {acc}", 12, DIM, anchor="lm")
        c.at(right, y + rh / 2, brl(s["profit"]) if s["done"] else "R$ 0,00", 18, _col(s["profit"]), "bold",
             anchor="rm")
    c.y = c.px(y0 + len(WALLETS) * rh + 8 + GAP)
    # pela chance (entradas de 5 min: a carteira B contém as da A)
    bands = by_band(st.get("B"))
    if bands and bottom - c.u > 120:
        _label(c, "PELA CHANCE QUE A ENTRADA MOSTRAVA")
        y0, row_h = c.u, 28
        k = min(len(bands), max(1, int((bottom - y0 - 110) // row_h)))
        c.card(y0, y0 + 12 + k * row_h)
        for i, (name, g) in enumerate(bands[:k]):
            y = y0 + 6 + i * row_h + row_h / 2
            c.at(inner, y, name, 13, FG, "medium", anchor="lm")
            acc = _pc(g["acc"]) if g["acc"] is not None else "—"
            c.at(inner + 120, y, f"{g['wins']}G {g['losses']}P · {acc}", 12, DIM, anchor="lm")
            c.at(right, y, brl(g["profit"]) if g["done"] else "aberta", 13, _col(g["profit"]), "semibold",
                 anchor="rm")
        c.y = c.px(y0 + 12 + k * row_h + GAP)
    # entrada por entrada
    rows = merged(st)[::-1]
    if bottom - c.u > 80:
        _label(c, "ENTRADA POR ENTRADA")
        y0, row_h = c.u, 42
        k = max(0, min(len(rows), int((bottom - y0 - 16) // row_h)))
        c.card(y0, y0 + 16 + max(1, k) * row_h)
        if not rows:
            c.at(W / 2, y0 + 8 + row_h / 2, "nenhuma entrada hoje", 13, DIM, anchor="mm")
        for i, e in enumerate(rows[:k]):
            y = y0 + 10 + i * row_h
            if i:
                c.d.line([(c.px(inner), c.px(y - 5)), (c.px(right), c.px(y - 5))], fill=LINE, width=c.px(1))
            x = inner
            for spec in e["wallets"]:
                _letter(c, x, y + 9, spec, 7)
                x += 17
            c.at(x + 4, y + 2, f"{_hhmm(e['ts'])}  {e['asset']}", 13, FG, "medium")
            arrow = "▲" if e["side"] == "COMPRA" else "▼"
            c.at(inner, y + 20, f"{e['side'].lower()} {arrow} · mostrava {_pc(e['chance'])}", 12, DIM)
            stt = e["status"]
            col = GREEN if stt == "ganhou" else RED if stt == "perdeu" else ORANGE if stt == "aberta" else DIM
            val = brl(e["stake"], False) if stt == "aberta" else ("—" if e["profit"] is None else brl(e["profit"]))
            c.at(right, y + 2, val, 14, col if stt in ("ganhou", "perdeu") else FG, "semibold", anchor="ra")
            c.at(right, y + 20, stt, 12, DIM, anchor="ra")
    return _fit(c, height)


def side_screens(st: SimState, height_px: int, scale: float = 1.0) -> np.ndarray:
    """As duas telas lado a lado, na altura exata pedida (pixels de saída)."""
    h_units = height_px / scale
    gap = np.zeros((height_px, max(1, int(round(GAP_PX * scale))), 3), np.uint8)
    imgs = []
    for im in (render_wallets(st, h_units, scale), render_day(st, h_units, scale)):
        if im.shape[0] > height_px:
            im = im[:height_px]
        elif im.shape[0] < height_px:
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

    def height(self, panel_px: int) -> int:
        """Altura da janela com as carteiras: a do painel ou a mínima das carteiras."""
        return max(panel_px, int(round(MIN_HEIGHT * self.scale)))

    def image(self, height_px: int) -> Optional[np.ndarray]:
        if not self.visible or self.state is None:
            return None
        key = (self.state.key(), height_px)
        if key != self._key or self._img is None:
            self._img = side_screens(self.state, height_px, self.scale)
            self._key = key
        return self._img
