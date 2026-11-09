"""Tela da carteira e do placar (tecla c no painel).

* Hoje, por conta (real e treino separadas): entradas, ganhos, perdas, resultado.
* Por hora e por tempo de vela.
* Últimas operações.
* Placar do app: quanto o motor acerta de verdade, por expiração e fase da vela,
  e os sinais que acenderam.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..history.scoreboard import HorizonScore
from ..history.wallet import AccountDay
from ..models import fmt_duration
from .panel import DIM, FG, GREEN, ORANGE, PAD, RED, W, _Canvas, _pct

ACC_NAME = {"real": "Conta REAL", "treino": "Conta de TREINO"}
ACC_COLOR = {"real": GREEN, "treino": ORANGE}
RES_NAME = {"ganho": "ganho", "perda": "perda", "empate": "empate", "vendida": "vendida",
            "sem leitura": "sem leitura", None: "aberta"}


def _money(v: float) -> str:
    sign = "+" if v > 0 else "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def _n(k: int, one: str, many: str) -> str:
    return f"{k} {one if k == 1 else many}"


def _tf(s: int) -> str:
    return fmt_duration(s).replace(" ", "")


def render_wallet(days: dict[str, AccountDay], board: Optional[dict[int, HorizonScore]] = None,
                  signals: Optional[list[dict]] = None, scale: float = 1.0) -> np.ndarray:
    c = _Canvas(scale, height=2400)
    c.y = c.px(10)
    c.text(PAD, "Carteira de hoje", 24, FG, bold=True)
    c.space(32)
    c.para(time.strftime("%d/%m/%Y") + " · deduzida do saldo da tela (o app não acessa a sua conta)", 11, DIM)
    c.rule()
    if not days:
        c.para("Nenhuma entrada hoje. Quando você abrir uma operação, o saldo cai e ela aparece aqui.", 13, FG)
        c.rule()
    for acc in ("real", "treino"):
        a = days.get(acc)
        if a is None:
            continue
        c.text(PAD, ACC_NAME.get(acc, acc), 17, ACC_COLOR.get(acc, FG), bold=True)
        col = GREEN if a.profit > 0 else RED if a.profit < 0 else FG
        c.text(W - PAD - 110, _money(a.profit), 17, col, bold=True)
        c.space(24)
        taxa = f" · acerto {_pct(a.rate)}" if a.rate is not None else ""
        c.para(f"{_n(a.n, 'entrada', 'entradas')} · {_n(a.wins, 'ganho', 'ganhos')} · "
               f"{_n(a.losses, 'perda', 'perdas')}"
               + (f" · {a.ties} empates" if a.ties else "") + (f" · {a.sold} vendidas" if a.sold else "")
               + taxa, 13, FG)
        c.para(f"Total investido: ${a.invested:,.2f}", 12, DIM)
        if a.followed[0]:
            c.para(f"Seguindo o sinal do app: {_n(a.followed[0], 'entrada', 'entradas')}, "
                   f"{_n(a.followed[1], 'ganho', 'ganhos')}", 12, DIM)
        if a.by_tf:
            c.para("Por tempo de vela: " + "  ·  ".join(
                f"{_tf(tf)}: {v[0]} ({v[1]}G) {_money(v[2])}" for tf, v in sorted(a.by_tf.items())), 12, FG)
        if a.by_hour:
            c.para("Por hora: " + "  ·  ".join(
                f"{h:02d}h: {v[0]} {_money(v[1])}" for h, v in sorted(a.by_hour.items())), 12, FG)
        c.space(4)
        c.text(PAD, "Últimas", 12, DIM)
        c.space(16)
        for t in a.last:
            hora = time.strftime("%H:%M:%S", time.localtime(t["opened"]))
            dire = {"COMPRA": "alta", "VENDA": "baixa"}.get(t.get("direction") or "", "")
            res = RES_NAME.get(t.get("result"), t.get("result"))
            p = t.get("profit")
            col = GREEN if (p or 0) > 0 else RED if (p or 0) < 0 else FG
            tf = _tf(int(t["tf"])) if t.get("tf") else "?"
            ativo = t.get("asset") if t.get("asset") not in (None, "?") else ""
            c.text(PAD, "  ".join(x for x in (hora, ativo, f"vela {tf}", f"${t['amount']:,.2f}", dire) if x), 12, FG)
            c.text(W - PAD - 130, f"{res} {_money(p) if p is not None else ''}", 12, col)
            c.space(17)
        c.rule()

    # Placar do motor
    c.text(PAD, "Placar do app (previsões conferidas)", 17, FG, bold=True)
    c.space(26)
    if not board or not any(s.n for s in board.values()):
        c.para("Ainda juntando dados: cada vela que fecha confere as previsões feitas nela.", 13, FG)
    else:
        for h, s in sorted(board.items()):
            if not s.n:
                continue
            fases = "  ".join(f"{f}: {_pct(v[1] / v[0])} ({v[0]})" for f, v in s.by_phase.items() if v[0])
            c.text(PAD, f"{h} vela{'s' if h > 1 else ''}", 13, FG, bold=True)
            c.text(PAD + 80, f"acerto {_pct(s.rate)} em {s.n}", 13, FG)
            c.space(18)
            if fases:
                c.para(fases, 11, DIM, x=PAD + 80)
            if s.signals:
                col = GREEN if s.signal_profit > 0 else RED
                c.para(f"sinais: {s.signal_hits}/{s.signals} ({_pct(s.signal_rate)}), "
                       f"resultado simulado {s.signal_profit:+.1f} entradas", 11, col, x=PAD + 80)
            c.space(3)
    if signals:
        done = [x for x in signals if x.get("outcome") in (0, 1)]
        hits = sum(x["outcome"] for x in done)
        c.space(4)
        c.para(f"Sinais de hoje: {len(signals)} (conferidos {len(done)}, acertos {hits})", 13, FG)
    c.rule()
    c.para("Acerto abaixo do equilíbrio do payout perde dinheiro no longo prazo. "
           "O app corrige a própria probabilidade com esse placar.", 11, DIM)
    return c.crop()


class WalletWindow:
    NAME = "CandleSense - Carteira"

    def __init__(self, scale: float = 1.0) -> None:
        import cv2

        self.cv2 = cv2
        self.scale = scale
        self.visible = False
        self._shown = False

    def toggle(self) -> None:
        if self.visible:
            self.hide()
        else:
            self.visible = True

    def show(self, img: np.ndarray) -> None:
        if not self.visible:
            return
        if self._shown:
            try:  # fechada pela bolinha vermelha
                if self.cv2.getWindowProperty(self.NAME, self.cv2.WND_PROP_VISIBLE) < 1:
                    self.visible = self._shown = False
                    return
            except Exception:
                pass
        self.cv2.imshow(self.NAME, img)
        self._shown = True
        try:
            self.cv2.setWindowProperty(self.NAME, self.cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    def hide(self) -> None:
        self.visible = self._shown = False
        try:
            self.cv2.destroyWindow(self.NAME)
        except Exception:
            pass
