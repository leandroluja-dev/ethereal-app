"""Modo nuvem: o simulador roda no GitHub, sem o Mac ligado, só com as velas do Twelve Data.

Roda a cada 15 min (GitHub Actions). Em cada rodada pede as últimas ~5 h de velas de 1 min
do GBP/USD (1 pedido), calcula o modelo em cada minuto que ainda não foi calculado, joga
nas carteiras de mentira e grava a página com os dados em ``saida/``:

    saida/index.html          a mesma página do app
    saida/api/state           o estado (o que a página pede)
    saida/api/dias/AAAA-MM-DD.json   as entradas de cada dia (agenda)
    saida/candlesense.db      o banco (volta na rodada seguinte)

Como as contas são feitas depois, minuto a minuto, cada entrada vale como se fosse feita no
fechamento da vela, sem atraso: um pouco melhor do que alguém clicando na hora conseguiria.
O preço é o do Twelve (não o da IQ) e o payout é fixo (PAYOUT). Nada aqui opera.

Uso: TWELVEDATA_KEY=... TZ=America/Sao_Paulo python -m candlesense.cloud saida
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ASSET = "GBP/USD"
PAYOUT = 89
FETCH = 320  # velas por pedido: 135 de histórico + folga para rodadas atrasadas


def measure_all(strong, asset: str, payout: int, now: float) -> int:
    """Calcula o modelo em todo minuto fechado que tem histórico e ainda não foi calculado."""
    done = {s for (s,) in strong.db.execute("SELECT start FROM forte WHERE asset=?", (asset,))}
    starts = [s for (s,) in strong.db.execute("SELECT start FROM fita WHERE asset=? AND start<? ORDER BY start",
                                              (asset, int(now // 60) * 60 - 60 + 1))]
    n = 0
    for s in starts:
        if s in done:
            continue
        # vale como se fosse calculado logo depois do fechamento da vela
        if strong._measure(asset, s, s + 60 + 3, payout) is not None:
            n += 1
    return n


def run(out: Path, key: str, now: float | None = None, bars=None, first_fetch: int = 900) -> dict:
    from .__main__ import _sim_state
    from .engine.strong import StrongMode
    from .feeds.twelvedata import series, symbol_of
    from .history.simbook import day_entries
    from .ui.panel import PanelExtras
    from .web.server import _group, build_state

    now = time.time() if now is None else now
    out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(out / "candlesense.db"))
    strong = StrongMode(db, log=lambda *a, **k: None)
    if bars is None:
        empty = db.execute("SELECT COUNT(*) FROM fita").fetchone()[0] == 0
        # primeira rodada: pega ~15 h para refazer o que o app teria feito desde a tarde anterior
        bars = series(symbol_of(ASSET), "1min", outputsize=first_fetch if empty else FETCH, key=key) or []
    cutoff = int(now // 60) * 60
    rows = [(ASSET, s, o, h, lo, c, now) for s, o, h, lo, c in bars if s < cutoff]
    db.executemany("INSERT OR REPLACE INTO fita VALUES (?,?,?,?,?,?,?)", rows)
    db.commit()
    n = measure_all(strong, ASSET, PAYOUT, now)
    strong.resolve()
    db.commit()

    class Holder:
        state = None

    holder = Holder()
    last = db.execute("SELECT close FROM fita WHERE asset=? ORDER BY start DESC LIMIT 1", (ASSET,)).fetchone()
    r = SimpleNamespace(ok=last is not None, timeframe_s=60, current_price=last[0] if last else None, scale=None)
    _sim_state(strong, holder, r, ASSET, play_sound=False, now=now)
    st = holder.state
    extras = PanelExtras()
    extras.app_wallet = st.app_wallet()
    state = build_state(st, extras, "Nuvem", "")
    ticks = [[s + 60, c] for (s, c) in db.execute(
        "SELECT start, close FROM fita WHERE asset=? AND start>=? ORDER BY start", (ASSET, cutoff - 3600))]
    state["live"] = {"ok": last is not None, "price": r.current_price, "decimals": 5, "ticks": ticks,
                     "market": [], "market_note": "nuvem: velas do Twelve, conferidas a cada 15 min",
                     "cloud": True, "bars_at": rows[-1][1] + 60 if rows else None}
    api = out / "api"
    (api / "dias").mkdir(parents=True, exist_ok=True)
    (api / "state").write_text(json.dumps(state, ensure_ascii=False))
    for d in state["days"]:
        by: dict = {}
        for e in day_entries(db, d["day"]):
            by.setdefault(e.pop("wallet"), []).append(e)
        (api / "dias" / f"{d['day']}.json").write_text(
            json.dumps({"day": d["day"], "entries": _group(by)}, ensure_ascii=False))
    shutil.copy(Path(__file__).parent / "web" / "index.html", out / "index.html")
    (out / ".nojekyll").write_text("")
    db.close()
    return {"velas": len(rows), "medidas": n, "carteira": extras.app_wallet}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = Path(argv[0] if argv else "saida")
    key = os.environ.get("TWELVEDATA_KEY", "").strip()
    if not key:
        print("falta o segredo TWELVEDATA_KEY no GitHub", flush=True)
        return 2
    db_file = out / "candlesense.db"
    if not db_file.exists():
        # primeira rodada: ~41 h de velas, e as carteiras refeitas dia a dia (elas contam o dia de "agora")
        from .feeds.twelvedata import series, symbol_of

        bars = series(symbol_of(ASSET), "1min", outputsize=2500, key=key) or []
        today = time.strftime("%Y-%m-%d")
        days = sorted({time.strftime("%Y-%m-%d", time.localtime(b[0])) for b in bars} - {today})
        for d in days:
            t = time.strptime(d, "%Y-%m-%d")
            end = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 23, 59, 30, 0, 0, -1))
            print(f"nuvem: refazendo {d}: {run(out, key, now=end, bars=bars)}", flush=True)
        res = run(out, key, bars=bars)
    else:
        res = run(out, key)
    print(f"nuvem: {res}", flush=True)
    return 0 if res["velas"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
