"""Telas ao lado do painel: as carteiras do simulador (entradas de mentira).

* Tela do meio, "Carteiras do app": cinco carteiras empilhadas (A 62%, B 58% e três de risco a 58% (55% até 10/11/2026 18h): vela de 5 min, vela de 1 min e 30 s), cada uma com cor e
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

from .panel import (bg_block, BLACK, CARD, CARD_2, CARD_3, DIM, FAINT, FG, GAP, GREEN, LINE, ORANGE, PAD, R_CARD, RED, W, WHITE,
                    _Canvas, _icon, _mmss)

BLUE = (88, 160, 255)
GAP_PX = 6        # faixa preta entre as telas (pixels de saída, antes da escala)
FLASH_S = 25      # o aviso de GANHOU / PERDEU fica este tempo
MIN_HEIGHT = 1050 # altura mínima da janela com as carteiras (unidades do painel)
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
RISK_CUT = 0.58
OLD_RISK_CUT = 0.55  # corte das carteiras de risco até RISK_CUT_SINCE (as entradas antigas continuam valendo)
RISK_CUT_SINCE = __import__("time").mktime((2026, 11, 10, 18, 5, 0, 0, 0, -1))

WALLETS = (
    WalletSpec("A", "Segura", 0.62, GREEN, "62% · vence em 5 min", "62%"),
    WalletSpec("B", "Moderada", 0.58, BLUE, "58% · vence em 5 min", "58%"),
    WalletSpec("C", "Vela de 5 min", RISK_CUT, ORANGE, "58% · fim da vela de 5 min", "RISCO",
               mode="candle5"),
    WalletSpec("D", "Vela de 1 min", RISK_CUT, PURPLE, "58% · fim da vela de 1 min", "RISCO", mode="1min"),
    WalletSpec("E", "Rápida 30 s", RISK_CUT, PINK, "58% · sai em 30 s", "RISCO", mode="30s"),
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
    past: list = field(default_factory=list)  # dias anteriores (SimBook.past_days)

    def get(self, key: str) -> list:
        return self.entries.get(key, [])

    def key(self) -> tuple:
        """Muda quando algo na tela muda (para não redesenhar à toa)."""
        live = tuple((k, e["start"], e["status"], e.get("seen_at")) for k, es in sorted(self.entries.items())
                     for e in es)
        return (live, int(self.now), self.asset, self.price, self.hist_min, self.note, self.available,
                self.last_ts, len(self.past))

    def today_profit(self) -> dict:
        return {w.key: day_summary(self.get(w.key))["profit"] for w in WALLETS}

    def balances(self) -> dict:
        from ..history.simbook import balances

        return balances(self.past, self.today_profit(), [w.key for w in WALLETS])

    def app_wallet(self) -> dict:
        """Uma carteira só do app: R$ 10.000 + o resultado de todas as entradas de todas as carteiras."""
        from ..history.simbook import START_BALANCE

        bal = self.balances()
        done = [e for es in self.entries.values() for e in es]
        s = day_summary(done)
        return {"balance": START_BALANCE + sum(v - START_BALANCE for v in bal.values()),
                "day": s["profit"], "wins": s["wins"], "losses": s["losses"], "n": s["n"]}

    def days(self) -> list:
        """Todos os dias com entradas, do mais antigo para hoje: [{day, label, w}]."""
        from ..history.simbook import day_label

        out = list(self.past)
        today = {}
        for w in WALLETS:
            sm = day_summary(self.get(w.key))
            if sm["n"]:
                today[w.key] = sm
        if today:
            d = time.strftime("%Y-%m-%d", time.localtime(self.now or time.time()))
            out.append({"day": d, "label": "hoje", "w": today})
        return out


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
    bal = st.balances().get(spec.key)
    if bal is not None:
        c.at(right, y0 + 24, brl(bal, False), 17, FG, "bold", anchor="rm")
    c.at(inner, y0 + 46, spec.desc, 12, DIM, anchor="lm")
    day = brl(s["profit"]) if s["done"] else "R$ 0,00"
    stats = f" · {s['wins']}G {s['losses']}P"
    c.at(right, y0 + 46, stats, 12, DIM, anchor="rm")
    c.at(right - c.tw(stats, 12), y0 + 46, "hoje " + day, 12, _col(s["profit"]), "semibold", anchor="rm")
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
        bg, fg = (GREEN, WHITE) if won else ((CARD_2, FG) if tie else (RED, WHITE))
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
    _head(c, "Carteiras do app", f"{brl(st.stake, False)} por entrada · cada uma começou com R$ 10 mil")
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
    # total do dia (todas as carteiras juntas)
    tot = day_summary([e for w in WALLETS for e in st.get(w.key)])
    y0 = c.u
    c.card(y0, y0 + 76)
    c.at(inner, y0 + 22, "Total do dia", 12, DIM, anchor="lm")
    tp = tot["profit"] if tot["done"] else 0.0
    c.at(inner, y0 + 50, brl(tp) if tot["done"] else "R$ 0,00", 26, _col(tp), "bold", anchor="lm")
    if tot["done"]:
        lab = "GREEN" if tp > 0 else "LOSS" if tp < 0 else "ZERO"
        bx = inner + c.tw(brl(tp), 26, "bold") + 10
        bw = c.tw(lab, 11, "bold") + 16
        c.rrect(bx, y0 + 41, bx + bw, y0 + 59, 9, fill=_col(tp) if tp else None, outline=None if tp else DIM,
                width=1.2)
        c.at(bx + bw / 2, y0 + 50, lab, 11, BLACK if tp else DIM, "bold", anchor="mm")
    acc = f"acerto {_pc(tot['acc'])}" if tot["acc"] is not None else "sem resultado"
    c.at(right, y0 + 22, f"{tot['n']} entradas · {tot['wins']}G {tot['losses']}P · {acc}", 12, DIM, anchor="rm")
    c.y = c.px(y0 + 76 + GAP)
    # cada carteira
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


# ---------------------------------------------------------------------------
# Histórico
# ---------------------------------------------------------------------------

def render_hist_wallets(st: SimState, height: float, scale: float = 1.0) -> np.ndarray:
    """Histórico, esquerda: saldo de cada carteira desde o começo e a evolução do saldo somado."""
    from ..history.simbook import START_BALANCE, START_TS

    c = _Canvas(scale, height=int(height) + 40)
    _head(c, "Saldo das carteiras", f"cada uma começou com {brl(START_BALANCE, False)} em "
                                    f"{time.strftime('%d/%m', time.localtime(START_TS))}")
    bottom = _footer(c, height, "Simulação: o app nunca clica nem opera.")
    inner, right = PAD + 16, W - PAD - 16
    days = st.days()
    bals = st.balances()
    y0, rh = c.u, 62
    c.card(y0, y0 + len(WALLETS) * rh + 8)
    for i, spec in enumerate(WALLETS):
        y = y0 + 4 + i * rh
        if i:
            c.d.line([(c.px(inner), c.px(y)), (c.px(right), c.px(y))], fill=LINE, width=c.px(1))
        n = sum(d["w"].get(spec.key, {}).get("n", 0) for d in days)
        wins = sum(d["w"].get(spec.key, {}).get("wins", 0) for d in days)
        losses = sum(d["w"].get(spec.key, {}).get("losses", 0) for d in days)
        total = bals[spec.key] - START_BALANCE
        _letter(c, inner, y + rh / 2, spec, 11)
        c.at(inner + 30, y + rh / 2 - 10, spec.name, 14, FG, "semibold", anchor="lm")
        acc = f" · {_pc(wins / (wins + losses))} de acerto" if wins + losses else ""
        c.at(inner + 30, y + rh / 2 + 10, f"{n} entradas{acc}", 12, DIM, anchor="lm")
        c.at(right, y + rh / 2 - 10, brl(bals[spec.key], False), 16, FG, "bold", anchor="rm")
        c.at(right, y + rh / 2 + 10, brl(total) if n else "sem entradas", 12,
             _col(total) if n else DIM, "semibold" if n else None, anchor="rm")
    c.y = c.px(y0 + len(WALLETS) * rh + 8 + GAP)
    # evolução: saldo de cada carteira no fim de cada dia
    if bottom - c.u > 200:
        _label(c, "SALDO NO FIM DE CADA DIA")
        y0 = c.u
        h = min(420.0, bottom - y0 - 4)
        c.card(y0, y0 + h)
        x1, x2, top, bot = inner + 4, right - 4, y0 + 18, y0 + h - 34
        series = {}
        for spec in WALLETS:
            v, pts = START_BALANCE, [START_BALANCE]
            for d in days:
                v += d["w"].get(spec.key, {}).get("profit", 0.0)
                pts.append(v)
            series[spec.key] = pts
        allv = [v for pts in series.values() for v in pts]
        lo, hi = min(allv), max(allv)
        pad = max(20.0, (hi - lo) * 0.15)
        lo, hi = lo - pad, hi + pad
        yb = top + (hi - START_BALANCE) / (hi - lo) * (bot - top)
        c.d.line([(c.px(x1), c.px(yb)), (c.px(x2), c.px(yb))], fill=LINE, width=c.px(1))
        c.at(x2, yb - 4, "início " + brl(START_BALANCE, False), 10, FAINT, anchor="rs")
        n = len(days) + 1
        for spec in WALLETS:
            pts = series[spec.key]
            xy = [(c.px(x1 + (x2 - x1) * (i / max(1, n - 1))), c.px(top + (hi - v) / (hi - lo) * (bot - top)))
                  for i, v in enumerate(pts)]
            if len(xy) > 1:
                c.d.line(xy, fill=spec.color, width=c.px(2), joint="curve")
            xe, ye = xy[-1]
            c.d.ellipse([xe - c.px(3), ye - c.px(3), xe + c.px(3), ye + c.px(3)], fill=spec.color)
        c.at(x1, y0 + h - 14, "começo", 11, FAINT, anchor="lm")
        if days:
            c.at(x2, y0 + h - 14, days[-1]["label"], 11, FAINT, anchor="rm")
        c.y = c.px(y0 + h + GAP)
    return _fit(c, height)


def render_hist_days(st: SimState, height: float, scale: float = 1.0) -> np.ndarray:
    """Histórico, direita: dia a dia, o total e o resultado de cada carteira."""
    c = _Canvas(scale, height=int(height) + 40)
    days = st.days()[::-1]
    _head(c, "Dia a dia", f"{len(days)} dia(s) com entradas" if days else "nenhum dia com entradas ainda")
    bottom = _footer(c, height, "Simulação: o app nunca clica nem opera.")
    inner, right = PAD + 16, W - PAD - 16
    dh = 92
    y = c.u
    if not days:
        c.card(y, y + 60)
        c.at(W / 2, y + 30, "as entradas aparecem aqui conforme o app joga", 13, DIM, anchor="mm")
    for d in days:
        if y + dh > bottom:
            break
        c.card(y, y + dh - 8)
        ws = d["w"]
        tot = sum(v.get("profit", 0.0) for v in ws.values())
        n = sum(v.get("n", 0) for v in ws.values())
        wins = sum(v.get("wins", 0) for v in ws.values())
        losses = sum(v.get("losses", 0) for v in ws.values())
        c.at(inner, y + 20, d["label"], 14, FG, "semibold", anchor="lm")
        acc = f" · {_pc(wins / (wins + losses))}" if wins + losses else ""
        c.at(inner + c.tw(d["label"], 14, "semibold") + 8, y + 20, f"{n} entradas{acc}", 12, DIM, anchor="lm")
        c.at(right, y + 20, brl(tot), 16, _col(tot), "bold", anchor="rm")
        # uma pastilha por carteira
        cw = (right - inner) / len(WALLETS)
        for i, spec in enumerate(WALLETS):
            x = inner + i * cw
            v = ws.get(spec.key)
            _letter(c, x, y + 56, spec, 7)
            txt = brl(v["profit"]).replace("R$ ", "") if v else "—"
            c.at(x + 18, y + 56, txt, 11, _col(v["profit"]) if v else FAINT, "semibold", anchor="lm")
        y += dh
    return _fit(c, height)


# ---------------------------------------------------------------------------
# Hoje (tela larga): total, carteiras lado a lado e a linha do tempo do dia
# ---------------------------------------------------------------------------

WIDE = 2 * W + GAP_PX  # largura das telas laterais juntas (unidades do painel)


def _total_card(c: _Canvas, st: SimState, x1: float, y0: float, x2: float, h: float) -> None:
    tot = day_summary([e for w in WALLETS for e in st.get(w.key)])
    tp = tot["profit"] if tot["done"] else 0.0
    tone = GREEN if tp > 0 else RED if tp < 0 else CARD
    c.rrect(x1, y0, x2, y0 + h, R_CARD, fill=tone)
    fg = WHITE if tp else FG
    sub = (225, 248, 232) if tp > 0 else (255, 225, 225) if tp < 0 else DIM
    c.at(x1 + 18, y0 + 24, "Total do dia · todas as carteiras", 12, sub, "medium", anchor="lm")
    c.at(x2 - 18, y0 + 24, "GREEN" if tp > 0 else "LOSS" if tp < 0 else "ZERO", 12, fg, "bold", anchor="rm")
    c.at(x1 + 16, y0 + 84, brl(tp), 38, fg, "regular", anchor="ls")
    acc = f"{_pc(tot['acc'])} de acerto" if tot["acc"] is not None else "sem resultado"
    c.at(x1 + 18, y0 + h - 18, f"{tot['n']} entradas · {tot['wins']}G {tot['losses']}P · {acc}", 12, sub,
         anchor="ls")


def _model_card(c: _Canvas, st: SimState, x1: float, y0: float, x2: float, h: float) -> None:
    """A última conta do modelo (ou o que falta para ele começar)."""
    c.rrect(x1, y0, x2, y0 + h, R_CARD, fill=CARD)
    inner, right = x1 + 18, x2 - 18
    c.at(inner, y0 + 24, "Modelo agora", 12, DIM, "medium", anchor="lm")
    if st.asset:
        c.at(right, y0 + 24, st.asset, 12, DIM, "medium", anchor="rm")
    fresh = st.last_p is not None and st.last_ts is not None and st.now - st.last_ts < 180
    if not st.available or st.note or (st.hist_min is not None and st.hist_min < st.hist_need) or not fresh:
        if not st.available:
            msg = "Simulador indisponível nesta versão."
        elif st.note:
            msg = st.note
        elif st.hist_min is not None and st.hist_min < st.hist_need:
            msg = f"Juntando histórico: {st.hist_min} de {st.hist_need} min"
        else:
            msg = "Esperando a próxima vela de 1 min fechar."
        from .panel import _wrap

        for i, ln in enumerate(_wrap(c.d, msg, c.font(14), c.px(right - inner))[:3]):
            c.at(inner, y0 + 56 + 20 * i, ln, 14, FG)
        return
    ch = max(st.last_p, 1 - st.last_p)
    buy = st.last_p >= 0.5
    col = GREEN if buy else RED
    _icon(c, "up" if buy else "down", inner, y0 + h - 58, col, 16)
    c.at(inner + 22, y0 + h - 50, "compra" if buy else "venda", 15, col, "semibold", anchor="lm")
    c.at(right, y0 + h - 18, _pc(ch), 40, FG, "regular", anchor="rs")
    c.at(inner, y0 + h - 22, f"conta das {_hhmm(st.last_ts)} · entra a partir de 58%", 11, DIM, anchor="ls")
    _bar(c, inner, right, y0 + 40, (ch - 0.5) / 0.12, col)
    for cut in (0.55, 0.58, 0.62):  # marquinhas dos cortes das carteiras
        x = inner + (right - inner) * (cut - 0.5) / 0.12
        c.d.line([(c.px(x), c.px(y0 + 37)), (c.px(x), c.px(y0 + 49))], fill=FAINT, width=c.px(1))
        c.at(x, y0 + 56, _pc(cut), 9, FAINT, anchor="mm")


def _wallet_tile(c: _Canvas, spec: WalletSpec, st: SimState, x1: float, y0: float, x2: float, h: float) -> None:
    es = st.get(spec.key)
    s = day_summary(es)
    flash = [e for e in es if e["status"] in ("ganhou", "perdeu")
             and e.get("seen_at") is not None and st.now - e["seen_at"] <= FLASH_S]
    tone = CARD
    if flash:
        tone = GREEN if flash[-1]["status"] == "ganhou" else RED
    c.rrect(x1, y0, x2, y0 + h, 16, fill=tone, outline=None if flash else tuple(int(v * 0.5) for v in spec.color))
    inner, right = x1 + 12, x2 - 12
    _letter(c, inner, y0 + 20, spec, 9)
    c.at(inner + 24, y0 + 20, spec.name, 12, FG, "semibold", anchor="lm")
    c.at(inner, y0 + 42, spec.desc, 10, DIM if not flash else WHITE, anchor="lm")
    bal = st.balances().get(spec.key, 0.0)
    c.at(inner, y0 + 70, brl(bal, False).replace("R$ ", "R$"), 17, FG if not flash else WHITE, "medium", anchor="lm")
    opened = [e for e in es if e["status"] == "aberta"]
    if flash:
        e = flash[-1]
        c.at(inner, y0 + h - 16, ("GANHOU " if e["status"] == "ganhou" else "PERDEU ") + brl(e["profit"] or 0),
             12, WHITE, "bold", anchor="ls")
    elif opened:
        e = opened[-1]
        c.at(inner, y0 + h - 16, f"{'▲' if e['side'] == 'COMPRA' else '▼'} aberta · "
                                 f"{_mmss(max(0, int(e['expiry_ts'] - st.now)))}", 11, ORANGE, "semibold",
             anchor="ls")
    else:
        day = brl(s["profit"]) if s["done"] else "R$ 0,00"
        c.at(inner, y0 + h - 16, day, 11, _col(s["profit"]), "semibold", anchor="ls")
        gp = f"{s['wins']}G {s['losses']}P"
        if inner + c.tw(day, 11, "semibold") + 6 < right - c.tw(gp, 10):
            c.at(right, y0 + h - 16, gp, 10, DIM, anchor="rs")


def _timeline(c: _Canvas, st: SimState, x1: float, y0: float, x2: float, y1: float) -> None:
    """Linha do tempo do dia: régua de horário em cima e cada entrada pendurada num cartão."""
    rows = merged(st)
    now = st.now or time.time()
    # janela da régua: das 8 contas mais recentes até agora (mín. 20 min), para os fios ficarem retos
    starts = sorted({(e["asset"], e["start"]): e["ts"] for e in rows}.values())[-8:]
    t_end = now + 2 * 60
    t_start = min(starts[0] - 2 * 60, t_end - 20 * 60) if starts else t_end - 30 * 60
    t_start = max(t_start, t_end - 3 * 3600)
    span = t_end - t_start
    c.rrect(x1, y0, x2, y1, R_CARD, fill=CARD)
    inner, right = x1 + 18, x2 - 18
    c.at(inner, y0 + 22, "Linha do tempo", 14, FG, "semibold", anchor="lm")
    c.at(inner + c.tw("Linha do tempo", 14, "semibold") + 10, y0 + 22, "últimas entradas de hoje", 12, DIM,
         anchor="lm")
    # legenda
    lx = right
    for lab, col in (("aberta", ORANGE), ("perdeu", RED), ("ganhou", GREEN)):
        w = c.tw(lab, 11)
        c.at(lx, y0 + 22, lab, 11, DIM, anchor="rm")
        c.d.ellipse([c.px(lx - w - 14), c.px(y0 + 18), c.px(lx - w - 6), c.px(y0 + 26)], fill=col)
        lx -= w + 26
    ax = y0 + 64  # régua
    ax1, ax2 = inner + 6, right - 6

    def X(t):
        return ax1 + (ax2 - ax1) * (t - t_start) / span

    c.d.line([(c.px(ax1), c.px(ax)), (c.px(ax2), c.px(ax))], fill=LINE, width=c.px(1))
    minor, label = (60, 300) if span <= 40 * 60 else (300, 900) if span <= 2 * 3600 else (600, 1800)
    first = int(t_start // minor + 1) * minor
    for t in range(first, int(t_end), minor):
        big = t % label == 0
        x = X(t)
        c.d.line([(c.px(x), c.px(ax - (6 if big else 3))), (c.px(x), c.px(ax + (6 if big else 3)))],
                 fill=DIM if big else FAINT, width=c.px(1))
        if big and abs(x - X(now)) > 30:
            c.at(x, ax - 16, _hhmm(t), 12, DIM, anchor="mm")
    xn = X(now)  # agora
    c.d.line([(c.px(xn), c.px(ax - 8)), (c.px(xn), c.px(y1 - 14))], fill=(40, 90, 60), width=c.px(1))
    c.at(xn, ax - 16, "agora", 12, GREEN, "semibold", anchor="mm")
    vis = [e for e in rows if e["ts"] >= t_start]
    # todos viram um pontinho na régua
    for e in vis:
        stt = e["status"]
        col = GREEN if stt == "ganhou" else RED if stt == "perdeu" else ORANGE if stt == "aberta" else FAINT
        x = X(e["ts"])
        c.d.ellipse([c.px(x - 3), c.px(ax - 3), c.px(x + 3), c.px(ax + 3)], fill=col)
    # uma conta do modelo = um cartão (com as letras das carteiras que entraram), os mais novos,
    # em zigue-zague de 3 alturas como numa linha do tempo; o fio liga o horário ao cartão
    groups: dict = {}
    for e in vis:
        g = groups.setdefault((e["asset"], e["start"]), {"ts": e["ts"], "side": e["side"], "chance": e["chance"],
                                                        "wallets": [], "profit": 0.0, "open": False,
                                                        "expiry_ts": e["expiry_ts"], "done": False})
        g["wallets"] += e["wallets"]
        n = len(e["wallets"])
        if e["status"] == "aberta":
            g["open"] = True
            g["expiry_ts"] = max(g["expiry_ts"], e["expiry_ts"])
        elif e["profit"] is not None:
            g["profit"] += e["profit"] * n
            g["done"] = True
    cards = sorted(groups.values(), key=lambda g: g["ts"])[-8:]
    k = len(cards)
    cw, chh = 150, 80
    lanes = 3
    top = ax + 40
    lane_h = max(chh + 14, (y1 - 16 - top - chh) / (lanes - 1)) if lanes > 1 else 0
    lane_h = min(lane_h, (y1 - 16 - top - chh) / (lanes - 1))
    ends = [inner - 999.0] * lanes  # onde termina o último cartão de cada altura
    placed = []
    for i, g in enumerate(cards):
        x = X(g["ts"])
        want = min(max(x - cw / 2, inner), right - cw)
        order = [(i + j) % lanes for j in range(lanes)]
        fit = [ln for ln in order if ends[ln] + 8 <= want]
        lane = fit[0] if fit else min(order, key=lambda ln: ends[ln])
        cx1 = max(want, ends[lane] + 8)
        if cx1 > right - cw + 0.5:
            continue  # sem lugar: fica só o pontinho na régua
        ends[lane] = cx1 + cw
        placed.append((g, cx1, top + lane * lane_h, x))
    def _lc(g):
        done = g["done"] and not g["open"]
        return GREEN if done and g["profit"] > 0 else RED if done and g["profit"] < 0 else \
            ORANGE if g["open"] else DIM

    for g, cx1, cy, x in placed:  # fios primeiro, cartões por cima
        lc, mx = _lc(g), cx1 + cw / 2
        c.d.line([(c.px(x), c.px(ax + 3)), (c.px(mx), c.px(cy))], fill=tuple(int(v * 0.75) for v in lc),
                 width=c.px(1))
        c.d.ellipse([c.px(mx - 3), c.px(cy - 3), c.px(mx + 3), c.px(cy + 3)], fill=lc)
    for g, cx1, cy, x in placed:
        net = g["profit"]
        won = g["done"] and not g["open"] and net > 0
        lost = g["done"] and not g["open"] and net < 0
        lc = GREEN if won else RED if lost else ORANGE if g["open"] else DIM
        c.rrect(cx1, cy, cx1 + cw, cy + chh, 16, fill=GREEN if won else RED if lost else CARD_2)
        fg = WHITE if (won or lost) else FG
        sub = (225, 245, 232) if won else (255, 220, 220) if lost else DIM
        arrow = "▲ compra" if g["side"] == "COMPRA" else "▼ venda"
        c.at(cx1 + 12, cy + 18, _hhmm(g["ts"]), 15, fg, "semibold", anchor="lm")
        c.at(cx1 + 12, cy + 38, f"{arrow} · {_pc(g['chance'])}", 10, sub, anchor="lm")
        lx = cx1 + cw - 10
        for spec in sorted(set(g["wallets"]), key=lambda w: w.key, reverse=True):
            _letter(c, lx - 12, cy + 18, spec, 6)
            lx -= 14
        if g["open"]:
            left = int(g["expiry_ts"] - now)
            val, lab = (_mmss(left), "aberta") if left > 0 else ("…", "conferindo")
        elif not g["done"]:
            val, lab = "—", "sem leitura"
        else:
            val, lab = brl(net).replace("R$ ", ""), "ganhou" if net > 0 else "perdeu" if net < 0 else "empate"
        c.at(cx1 + 12, cy + chh - 12, lab, 10, sub, anchor="ls")
        c.at(cx1 + cw - 12, cy + chh - 9, val, 22, fg, "regular", anchor="rs")
    if not vis:
        c.at((x1 + x2) / 2, (ax + y1) / 2, "nenhuma entrada ainda hoje", 13, DIM, anchor="mm")


def render_today(st: SimState, height: float, scale: float = 1.0) -> np.ndarray:
    """Aba Hoje: total do dia e modelo em cima, as cinco carteiras lado a lado e a linha do tempo."""
    c = _Canvas(scale, height=int(height) + 40, width=int(round(WIDE)))
    x1, x2 = PAD, WIDE - PAD
    y = 10
    top_h = 132
    mid = x1 + (x2 - x1) * 0.5
    _total_card(c, st, x1, y, mid - GAP / 2, top_h)
    _model_card(c, st, mid + GAP / 2, y, x2, top_h)
    y += top_h + GAP
    n = len(WALLETS)
    tw = (x2 - x1 - (n - 1) * 10) / n
    th = 116
    for i, spec in enumerate(WALLETS):
        _wallet_tile(c, spec, st, x1 + i * (tw + 10), y, x1 + i * (tw + 10) + tw, th)
    y += th + GAP
    bottom = height - PAD - 20
    _timeline(c, st, x1, y, x2, bottom)
    c.at(x1 + 2, height - PAD - 6, f"Simulação com {brl(st.stake, False)} por entrada: o app nunca clica nem opera.",
         11, FAINT, anchor="ls")
    c.y = c.px(height)
    return c.crop(min_h=height)


# ---------------------------------------------------------------------------
# Abas e montagem
# ---------------------------------------------------------------------------

TABS = (("hoje", "Hoje"), ("historico", "Histórico"))
TAB_H = 46


def _tab_rects(total_w: float) -> list:
    """[(aba, x1, x2)] em unidades do painel, para desenhar e para o clique."""
    out, x = [], PAD + 6
    for key, name in TABS:
        w = 24 + len(name) * 8.6
        out.append((key, x, x + w))
        x += w + 8
    return out


def render_tabs(tab: str, total_w: float, scale: float, st: Optional[SimState] = None) -> np.ndarray:
    c = _Canvas(scale, height=TAB_H + 4, width=int(round(total_w)))
    for key, x1, x2 in _tab_rects(total_w):
        on = key == tab
        c.rrect(x1, 8, x2, 38, 15, fill=CARD_3 if on else CARD_2)
        c.at((x1 + x2) / 2, 23, dict(TABS)[key], 13, FG if on else DIM, "semibold" if on else "medium", anchor="mm")
    c.at(total_w - PAD - 6, 23, "h troca a aba", 11, FAINT, anchor="rm")
    c.y = c.px(TAB_H)
    img = c.crop(min_h=TAB_H)
    return img[: int(round(TAB_H * scale))]


def side_screens(st: SimState, height_px: int, scale: float = 1.0, tab: str = "hoje") -> np.ndarray:
    """As abas e as duas telas lado a lado, na altura exata pedida (pixels de saída)."""
    gw = max(1, int(round(GAP_PX * scale)))
    tabs_px = int(round(TAB_H * scale))
    body_px = max(1, height_px - tabs_px)
    h_units = body_px / scale
    if tab != "historico":
        body = render_today(st, h_units, scale)
        body = body[:body_px] if body.shape[0] >= body_px else \
            np.vstack([body, bg_block(body_px - body.shape[0], body.shape[1], scale, tabs_px + body.shape[0])])
        body = np.hstack([bg_block(body_px, gw, scale, tabs_px), body])
        want_w = gw * 2 + 2 * int(round(W * scale))
        if body.shape[1] < want_w:
            body = np.hstack([body, bg_block(body_px, want_w - body.shape[1], scale, tabs_px)])
        body = body[:, :want_w]
        bar = render_tabs(tab, body.shape[1] / scale, scale, st)
        bar = bar[:, : body.shape[1]]
        if bar.shape[1] < body.shape[1]:
            bar = np.hstack([bar, bg_block(bar.shape[0], body.shape[1] - bar.shape[1], scale)])
        if bar.shape[0] != tabs_px:
            bar = bar[:tabs_px] if bar.shape[0] > tabs_px else \
                np.vstack([bar, bg_block(tabs_px - bar.shape[0], bar.shape[1], scale, bar.shape[0])])
        return np.vstack([bar, body])
    pair = (render_hist_wallets, render_hist_days)
    imgs = []
    for fn in pair:
        im = fn(st, h_units, scale)
        if im.shape[0] > body_px:
            im = im[:body_px]
        elif im.shape[0] < body_px:
            im = np.vstack([im, bg_block(body_px - im.shape[0], im.shape[1], scale, tabs_px + im.shape[0])])
        imgs.append(im)
    gap = bg_block(body_px, gw, scale, tabs_px)
    body = np.hstack([gap, imgs[0], gap, imgs[1]])
    bar = render_tabs(tab, body.shape[1] / scale, scale, st)
    if bar.shape[1] != body.shape[1]:
        bar = np.ascontiguousarray(bar[:, : body.shape[1]]) if bar.shape[1] > body.shape[1] else \
            np.hstack([bar, bg_block(bar.shape[0], body.shape[1] - bar.shape[1], scale)])
    if bar.shape[0] != tabs_px:
        bar = bar[:tabs_px] if bar.shape[0] > tabs_px else \
            np.vstack([bar, bg_block(tabs_px - bar.shape[0], bar.shape[1], scale, bar.shape[0])])
    return np.vstack([bar, body])


class SideScreens:
    """Guarda a última imagem e só redesenha quando o conteúdo muda (no máximo 1 vez por segundo)."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale
        self.state: Optional[SimState] = None
        self.visible = True
        self.tab = "hoje"
        self._key = None
        self._img: Optional[np.ndarray] = None

    def next_tab(self) -> None:
        keys = [k for k, _ in TABS]
        self.tab = keys[(keys.index(self.tab) + 1) % len(keys)]

    def click(self, x_px: float, y_px: float) -> bool:
        """Clique nas telas (x a partir da borda esquerda delas). Devolve True se trocou de aba."""
        x, y = x_px / self.scale, y_px / self.scale
        if not (0 <= y <= TAB_H):
            return False
        for key, x1, x2 in _tab_rects(0):
            if x1 <= x <= x2 and key != self.tab:
                self.tab = key
                return True
        return False

    def height(self, panel_px: int) -> int:
        """Altura da janela com as carteiras: a do painel ou a mínima das carteiras."""
        return max(panel_px, int(round(MIN_HEIGHT * self.scale)))

    def image(self, height_px: int) -> Optional[np.ndarray]:
        if not self.visible or self.state is None:
            return None
        key = (self.state.key(), height_px, self.scale, self.tab)
        if key != self._key or self._img is None:
            self._img = side_screens(self.state, height_px, self.scale, self.tab)
            self._key = key
        return self._img
