"""Carteira: entradas, ganhos e perdas, deduzidos só do saldo que aparece na tela.

Como funciona (sem clicar em nada e sem API da IQ Option):

* Ao abrir uma operação, a IQ Option desconta o valor do saldo na hora.
  Uma queda do saldo, confirmada em 2 leituras seguidas, vira uma entrada
  com esse valor.
* Ao fechar, o saldo sobe: valor x (1 + payout) = ganho, o próprio valor =
  empate (devolução). Subida antes da expiração = operação vendida antes.
  Sem subida até 20 s depois da expiração = perda.
* A direção (alta ou baixa) é deduzida no fechamento pelo resultado e pelo
  movimento do preço entre a entrada e a expiração.
* Conta real (saldo verde) e de treino (saldo laranja) ficam separadas. Trocar
  de conta não conta como entrada.

Tudo é estimativa a partir da tela e fica marcado como tal: se a leitura do
saldo falha, nada é registrado.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .store import Store

CONFIRM_READS = 2          # leituras iguais para aceitar um saldo novo
LOSS_GRACE_S = 20          # depois da expiração, sem subida = perda
SIGNAL_MEMORY_S = 20       # sinal do app que ainda vale como "seguiu o app"
DEFAULT_EXPIRY_S = 60


@dataclass
class OpenTrade:
    id: int
    opened: float
    account: str
    amount: float
    expiry_ts: float
    entry_price: Optional[float]
    payout: Optional[int]
    app_action: Optional[str]
    asset: str = "?"
    tf: Optional[int] = None


@dataclass
class WalletEvent:
    kind: str                 # "entrada" | "ganho" | "perda" | "empate" | "vendida"
    account: str
    amount: float
    profit: float = 0.0
    direction: Optional[str] = None


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


class WalletTracker:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.balance: dict[str, float] = {}     # saldo confirmado por conta
        self._pending: Optional[tuple[str, float, int]] = None  # (conta, valor, quantas vezes visto)
        self.open: list[OpenTrade] = []
        self._prices: deque[tuple[float, float]] = deque(maxlen=1200)
        self._last_signal: Optional[tuple[float, str, Optional[float]]] = None
        self.account: Optional[str] = None
        self._restore()

    # --------------------------------------------------------------- estado

    def _restore(self) -> None:
        """Operações que ficaram abertas (o app foi fechado no meio): fecha como 'sem leitura'."""
        for t in self.store.trades(since=time.time() - 86400):
            if t["closed"] is None:
                self.store.close_trade(t["id"], closed=time.time(), result="sem leitura", profit=None)

    def note_signal(self, action: Optional[str], prob: Optional[float], now: Optional[float] = None) -> None:
        """Lembra o último sinal que acendeu, para saber se a entrada seguiu o app."""
        if action in ("COMPRA", "VENDA"):
            self._last_signal = (time.time() if now is None else now, action, prob)

    def _price_at(self, ts: float) -> Optional[float]:
        best = None
        for t, p in self._prices:
            if best is None or abs(t - ts) < abs(best[0] - ts):
                best = (t, p)
        return best[1] if best is not None and abs(best[0] - ts) <= 5 else None

    # --------------------------------------------------------------- leitura

    def update(self, account: Optional[str], balance: Optional[float], price: Optional[float] = None,
               asset: str = "?", tf: Optional[int] = None, payout: Optional[int] = None,
               expiry_seconds: Optional[int] = None, invest: Optional[float] = None,
               now: Optional[float] = None) -> list[WalletEvent]:
        now = time.time() if now is None else now
        if price is not None:
            self._prices.append((now, price))
        events: list[WalletEvent] = []
        events += self._expire(now)
        if account is None or balance is None:
            return events
        if account != self.account:
            # Trocou de conta (ou começou agora): só guarda o saldo de partida.
            self.account = account
            self._pending = None
            if account not in self.balance:
                self.balance[account] = balance
                return events
        prev = self.balance.get(account)
        if prev is None:
            self.balance[account] = balance
            return events
        if _close(balance, prev, 0.005):
            self._pending = None
            return events
        # Valor novo: precisa aparecer igual em leituras seguidas.
        if self._pending and self._pending[0] == account and _close(self._pending[1], balance, 0.005):
            self._pending = (account, balance, self._pending[2] + 1)
        else:
            self._pending = (account, balance, 1)
        if self._pending[2] < CONFIRM_READS:
            return events
        self._pending = None
        self.balance[account] = balance
        delta = round(balance - prev, 2)
        if delta < 0:
            events.append(self._open(account, -delta, price, asset, tf, payout, expiry_seconds, invest, now))
        else:
            events += self._credit(account, delta, now)
        return events

    def _open(self, account, amount, price, asset, tf, payout, expiry_seconds, invest, now) -> WalletEvent:
        sig = self._last_signal
        app_action, app_prob = (sig[1], sig[2]) if sig and now - sig[0] <= SIGNAL_MEMORY_S else (None, None)
        exp = expiry_seconds if expiry_seconds and expiry_seconds > 0 else (tf or DEFAULT_EXPIRY_S)
        tid = self.store.open_trade(
            opened=now, account=account, amount=amount, asset=asset, tf=tf, entry_price=price,
            expiry_ts=now + exp, payout=payout, app_action=app_action, app_prob=app_prob)
        self.open.append(OpenTrade(tid, now, account, amount, now + exp, price, payout, app_action, asset, tf))
        return WalletEvent("entrada", account, amount, direction=app_action)

    def _credit(self, account: str, credit: float, now: float) -> list[WalletEvent]:
        """Uma subida do saldo: casa com a operação aberta que explica esse valor."""
        mine = [t for t in self.open if t.account == account]
        if not mine:
            return []  # depósito, recarga da conta de treino etc.
        tol = max(0.02, 0.01 * credit)
        for t in sorted(mine, key=lambda t: t.expiry_ts):
            win_val = t.amount * (1 + (t.payout or 0) / 100.0)
            if t.payout and _close(credit, win_val, tol):
                return [self._finish(t, "ganho", credit - t.amount, now)]
            if _close(credit, t.amount, tol) and now >= t.expiry_ts - 2:
                return [self._finish(t, "empate", 0.0, now)]
        # Não bateu com a fórmula: atribui à mais antiga que já poderia ter fechado.
        cands = [t for t in mine if credit <= 2.5 * t.amount]  # payout nunca passa de 100%
        if not cands:
            return []
        t = min(cands, key=lambda t: t.expiry_ts)
        kind = "vendida" if now < t.expiry_ts - 2 else ("ganho" if credit > t.amount else "empate")
        return [self._finish(t, kind, credit - t.amount, now)]

    def _expire(self, now: float) -> list[WalletEvent]:
        out = []
        for t in list(self.open):
            if now > t.expiry_ts + LOSS_GRACE_S:
                out.append(self._finish(t, "perda", -t.amount, now))
        return out

    def _finish(self, t: OpenTrade, kind: str, profit: float, now: float) -> WalletEvent:
        self.open.remove(t)
        exit_price = self._price_at(min(now, t.expiry_ts))
        direction = None
        if t.entry_price is not None and exit_price is not None and exit_price != t.entry_price:
            up = exit_price > t.entry_price
            if kind == "ganho":
                direction = "COMPRA" if up else "VENDA"
            elif kind == "perda":
                direction = "VENDA" if up else "COMPRA"
        if direction is None and kind in ("ganho", "perda", "vendida"):
            direction = t.app_action  # melhor palpite: seguiu o sinal do app
        self.store.close_trade(t.id, closed=now, result=kind, profit=round(profit, 2),
                               direction=direction, exit_price=exit_price)
        return WalletEvent(kind, t.account, t.amount, round(profit, 2), direction)

    # ------------------------------------------------------- alerta de saída

    def exit_alert(self, strength: Optional[float], now: Optional[float] = None,
                   threshold: float = 0.45) -> Optional[str]:
        """Operação aberta com direção conhecida e o mercado virando forte contra ela."""
        if strength is None:
            return None
        now = time.time() if now is None else now
        for t in self.open:
            if t.app_action is None or now > t.expiry_ts - 3:
                continue
            against = -strength if t.app_action == "COMPRA" else strength
            if against >= threshold:
                return f"o mercado virou contra a sua {t.app_action.lower()}"
        return None


# ------------------------------------------------------------------ resumo

@dataclass
class AccountDay:
    account: str
    n: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    sold: int = 0
    profit: float = 0.0
    invested: float = 0.0
    by_hour: dict[int, list] = field(default_factory=dict)   # hora -> [n, lucro]
    by_tf: dict[int, list] = field(default_factory=dict)     # tf (s) -> [n, ganhos, lucro]
    followed: list = field(default_factory=lambda: [0, 0])   # [n, ganhos] seguindo o app
    last: list[dict] = field(default_factory=list)

    @property
    def rate(self) -> Optional[float]:
        done = self.wins + self.losses
        return self.wins / done if done else None


def day_start(now: Optional[float] = None) -> float:
    lt = time.localtime(time.time() if now is None else now)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))


def summarize(trades: list[dict]) -> dict[str, AccountDay]:
    out: dict[str, AccountDay] = {}
    for t in trades:
        a = out.setdefault(t["account"], AccountDay(t["account"]))
        a.n += 1
        a.invested += t["amount"] or 0
        res = t.get("result")
        p = t.get("profit") or 0.0
        if res == "ganho":
            a.wins += 1
        elif res == "perda":
            a.losses += 1
        elif res == "empate":
            a.ties += 1
        elif res == "vendida":
            a.sold += 1
        if res is not None:
            a.profit += p
        h = time.localtime(t["opened"]).tm_hour
        bh = a.by_hour.setdefault(h, [0, 0.0])
        bh[0] += 1
        bh[1] += p
        if t.get("tf"):
            bt = a.by_tf.setdefault(int(t["tf"]), [0, 0, 0.0])
            bt[0] += 1
            bt[1] += int(res == "ganho")
            bt[2] += p
        if t.get("app_action"):
            a.followed[0] += 1
            a.followed[1] += int(res == "ganho")
    for a in out.values():
        a.last = list(reversed(trades_of(trades, a.account)[-8:]))
    return out


def trades_of(trades: list[dict], account: str) -> list[dict]:
    return [t for t in trades if t["account"] == account]
