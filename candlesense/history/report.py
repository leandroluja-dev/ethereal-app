"""Carteira em números, pronta para qualquer tela (painel, tela da carteira, exportação).

``wallet_report`` junta tudo o que a pessoa pediu sobre a carteira, por conta
(real e treino nunca se misturam):

* saldo atual e operações abertas (com o tempo que falta);
* hoje: entradas, ganhos, perdas, empates, resultado, por hora, por tempo de
  vela e por expiração, e as últimas operações;
* histórico por dia (com saldo de abertura e fechamento) e o total de sempre;
* placar do app: sinais que ele deu hoje e quantos acertou, e o acerto das
  entradas que seguiram o app;
* qual tempo de vela (1, 2, 5 min...) tem dado mais certo (``timeframes``).

Só lê o SQLite e o estado do ``WalletTracker``: não desenha nada.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .store import Store
from .timeframes import TimeframeAdvice, compare
from .wallet import AccountDay, WalletTracker, day_start, summarize


@dataclass
class OpenInfo:
    amount: float
    seconds_left: int
    app_action: Optional[str]   # "COMPRA"/"VENDA" se a entrada seguiu um sinal do app
    asset: str


@dataclass
class DayLine:
    date: str                   # "09/11/2026"
    start: float                # meia-noite (segundos desde 1970)
    n: int
    wins: int
    losses: int
    profit: float
    balance_open: Optional[float]
    balance_close: Optional[float]


@dataclass
class AppScore:
    signals: int = 0            # sinais que acenderam no período
    resolved: int = 0           # já conferidos com a vela de expiração
    hits: int = 0

    @property
    def rate(self) -> Optional[float]:
        return self.hits / self.resolved if self.resolved else None


@dataclass
class WalletReport:
    account: Optional[str]                       # "real" | "treino" | None (saldo ainda não lido)
    balance: Optional[float]
    open: list[OpenInfo] = field(default_factory=list)
    today: Optional[AccountDay] = None
    days: list[DayLine] = field(default_factory=list)   # mais recente primeiro
    all_time: Optional[AccountDay] = None
    app_today: AppScore = field(default_factory=AppScore)
    app_7d: AppScore = field(default_factory=AppScore)
    timeframes: Optional[TimeframeAdvice] = None         # melhor tempo de vela pelo histórico (30 dias)

    @property
    def label(self) -> str:
        return {"real": "Conta REAL", "treino": "Conta de TREINO"}.get(self.account or "", "Conta ?")


def _app_score(signals: list[dict]) -> AppScore:
    s = AppScore(signals=len(signals))
    for x in signals:
        if x.get("outcome") in (0, 1):
            s.resolved += 1
            s.hits += int(x["outcome"] == 1)
    return s


def _days(store: Store, account: str, trades: list[dict], n_days: int, now: float) -> list[DayLine]:
    by_day: dict[float, list[dict]] = {}
    for t in trades:
        by_day.setdefault(day_start(t["opened"]), []).append(t)
    bals = store.balances(since=day_start(now) - n_days * 86400, account=account)
    out = []
    for start in sorted(by_day, reverse=True)[:n_days]:
        a = summarize(by_day[start]).get(account) or AccountDay(account)
        inside = [b for ts, _, b in bals if start <= ts < start + 86400 + 3600]
        before = [b for ts, _, b in bals if ts < start]
        out.append(DayLine(
            date=time.strftime("%d/%m/%Y", time.localtime(start + 3600)), start=start,
            n=a.n, wins=a.wins, losses=a.losses, profit=round(a.profit, 2),
            balance_open=before[-1] if before else (inside[0] if inside else None),
            balance_close=inside[-1] if inside else None))
    return out


def wallet_report(store: Store, wallet: Optional[WalletTracker], account: Optional[str] = None,
                  now: Optional[float] = None, n_days: int = 30) -> WalletReport:
    now = time.time() if now is None else now
    acc = account or (wallet.account if wallet else None)
    if acc is None:
        last = store.balances()
        acc = last[-1][1] if last else None
    rep = WalletReport(account=acc, balance=(wallet.balance.get(acc) if wallet and acc else None))
    month = now - 30 * 86400
    rep.timeframes = compare(store.signals(since=month), store.predictions(since=month),
                             store.trades(since=month))
    if acc is None:
        return rep
    if rep.balance is None:
        last = store.balances(account=acc)
        rep.balance = last[-1][2] if last else None
    if wallet:
        rep.open = [OpenInfo(t.amount, max(0, int(t.expiry_ts - now)), t.app_action, t.asset)
                    for t in wallet.open if t.account == acc]
    trades = store.trades(account=acc)
    today0 = day_start(now)
    rep.today = summarize([t for t in trades if t["opened"] >= today0]).get(acc) or AccountDay(acc)
    rep.all_time = summarize(trades).get(acc) or AccountDay(acc)
    rep.days = _days(store, acc, trades, n_days, now)
    rep.app_today = _app_score(store.signals(since=today0))
    rep.app_7d = _app_score(store.signals(since=now - 7 * 86400))
    return rep
