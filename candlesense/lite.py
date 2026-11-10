"""Modo leve: o app roda sem ler a tela da IQ, só com as velas de 1 min do Twelve Data.

Liga quando existe o arquivo ``dados/modo_leve``. Por padrão roda das 9h às 21h no
horário do Mac (configurável no próprio arquivo, ex.: ``9-21``), de segunda a sexta,
com um pedido por minuto ao Twelve (720 por dia, dentro do plano grátis). Calcula o
modelo, joga nas carteiras de mentira, guarda tudo na agenda e mostra na página.

Diferenças para o modo normal: o preço é o do Twelve (não o da IQ), o payout é fixo
(``dados/payout.txt``, padrão 89%) e não há janelinha nem sons. Nada aqui clica ou opera.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

ASSET = "GBP/USD"
FETCH_AT_S = 12  # segundos depois da virada do minuto: a vela que fechou já está no Twelve
DEFAULT_HOURS = (9, 21)
DEFAULT_PAYOUT = 89


def lite_file(dados) -> Path:
    return Path(dados) / "modo_leve"


def hours(dados) -> tuple[int, int]:
    """Horário do arquivo modo_leve ("9-21"); vazio ou inválido = 9h às 21h."""
    try:
        a, b = lite_file(dados).read_text().strip().split("-")
        a, b = int(a), int(b)
        if 0 <= a < b <= 24:
            return a, b
    except (OSError, ValueError):
        pass
    return DEFAULT_HOURS


def payout(dados) -> int:
    try:
        v = int((Path(dados) / "payout.txt").read_text().strip().rstrip("%"))
        if 50 <= v <= 100:
            return v
    except (OSError, ValueError):
        pass
    return DEFAULT_PAYOUT


def in_window(now: float, span: tuple[int, int]) -> bool:
    t = time.localtime(now)
    return t.tm_wday < 5 and span[0] <= t.tm_hour < span[1]


class _Holder:
    """Faz o papel das telas laterais para o _sim_state (só guarda o estado)."""

    state = None


def fetch(db, sym: str, asset: str, now: float, first: bool, dados) -> int:
    """Pede as velas recentes e grava as fechadas na fita (substitui: aqui não há tela)."""
    from .feeds.twelvedata import series

    bars = series(sym, "1min", outputsize=160 if first else 8, dados=dados)
    if not bars:
        return 0
    cutoff = int(now // 60) * 60
    rows = [(asset, s, o, h, lo, c, now) for s, o, h, lo, c in bars if s < cutoff]
    db.executemany("INSERT OR REPLACE INTO fita VALUES (?,?,?,?,?,?,?)", rows)
    db.commit()
    return len(rows)


def run(args, web=None) -> int:
    from .__main__ import _day_source, _sim_state
    from .engine.strong import StrongMode
    from .feeds.twelvedata import ask_key, load_key, symbol_of
    from .history.store import Store
    from .ui.panel import PanelExtras

    dados = args.dados
    if not load_key(dados) and ask_key(dados) is None:
        print("modo leve: sem chave do Twelve Data, não dá para rodar", flush=True)
        return 2
    span, pay = hours(dados), payout(dados)
    print(f"modo leve: {ASSET} pelo Twelve Data, {span[0]}h às {span[1]}h, payout {pay}%, sem ler a tela", flush=True)
    store = Store(Path(dados) / "candlesense.db")
    strong = StrongMode(store.db)
    if web is not None:
        web.day_source = _day_source(Path(dados) / "candlesense.db")
    updater = None
    try:
        from .platform.selfupdate import Updater

        updater = Updater(Path(__file__).resolve().parents[1]).start()
    except Exception as e:
        print(f"atualização automática indisponível: {e!r}", flush=True)
    sym = symbol_of(ASSET)
    holder, extras = _Holder(), PanelExtras()
    first, last_fetch_min, last_view = True, None, 0.0
    while True:
        now = time.time()
        if updater is not None and updater.ready.is_set():
            from .platform.selfupdate import restart

            print("reabrindo com a versão nova", flush=True)
            store.close()
            restart()
            return 0
        span = hours(dados)
        open_now = in_window(now, span)
        minute = int(now // 60)
        if open_now and minute != last_fetch_min and now - minute * 60 >= FETCH_AT_S:
            last_fetch_min = minute
            try:
                n = fetch(store.db, sym, ASSET, now, first, dados)
                if n:
                    first = False
                    p = strong.measure_closed(ASSET, now, pay)
                    if p is not None and max(p, 1 - p) >= 0.58:
                        print(f"[{time.strftime('%H:%M:%S')}] modelo {'COMPRA' if p >= .5 else 'VENDA'} "
                              f"{max(p, 1 - p):.0%}", flush=True)
                strong.resolve()
                store.db.commit()
            except Exception as e:  # rede fora, banco ocupado: tenta no próximo minuto
                print(f"modo leve: {e!r}", flush=True)
        if now - last_view >= 2:
            last_view = now
            try:
                row = store.db.execute("SELECT close FROM fita WHERE asset=? ORDER BY start DESC LIMIT 1",
                                       (ASSET,)).fetchone()
                r = SimpleNamespace(ok=row is not None, timeframe_s=60, current_price=row[0] if row else None,
                                    scale=None)
                _sim_state(strong, holder, r, ASSET, play_sound=False)
                st = holder.state
                if not open_now:
                    st.note = f"Modo leve: roda das {span[0]}h às {span[1]}h, de segunda a sexta."
                extras.app_wallet = st.app_wallet()
                if web is not None:
                    text = "Modo leve" if open_now else "Fora do horário"
                    from .web.server import build_state

                    state = build_state(st, extras, text, "", url=getattr(web, "phone", None))
                    state["live"] = {"ok": row is not None, "price": st.price, "decimals": 5,
                                     "ticks": web.tick(now, ASSET, st.price if open_now else None),
                                     "market": [], "market_note": "modo leve: preço do Twelve, 1 vez por minuto",
                                     "lite": True}
                    web.publish(state)
            except Exception as e:
                print(f"modo leve (tela): {e!r}", flush=True)
        time.sleep(1.0)
