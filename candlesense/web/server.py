"""Página do CandleSense servida pelo próprio Mac (tela cheia no navegador e no celular).

O app continua lendo só a tela da IQ e o Twelve Data; aqui ele apenas mostra o que já
calculou. A página (index.html) pede ``/api/state`` a cada 1 s. Nada aqui clica, opera
ou recebe ordens: o servidor só responde leituras (GET).

No celular: mesmo Wi-Fi do Mac, abrir o endereço que aparece na página e no log
(ex.: http://192.168.0.12:8765).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

PORT = 8765
PAGE = Path(__file__).with_name("index.html")
STALE_S = 15  # sem novidade do app há mais que isso: a página avisa


def lan_ip() -> Optional[str]:
    """IP do Mac na rede de casa (não envia nada: só pergunta ao sistema a rota)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None
    finally:
        s.close()


def _group(entries: dict) -> list:
    """Junta as entradas das carteiras que começaram no mesmo minuto num cartão só."""
    by: dict = {}
    for key, es in entries.items():
        for e in es:
            g = by.setdefault((e["asset"], e["start"]), {
                "ts": e["ts"], "side": e["side"], "chance": e["chance"], "asset": e["asset"],
                "wallets": [], "_st": [], "_profit": 0.0, "_open": []})
            g["wallets"].append(key)
            g["_st"].append(e["status"])
            if e["status"] == "aberta":
                g["_open"].append(e["expiry_ts"])
            elif e["status"] in ("ganhou", "perdeu", "empate"):
                g["_profit"] += e["profit"] or 0.0
    out = []
    for g in sorted(by.values(), key=lambda g: g["ts"]):
        st, opened = g.pop("_st"), g.pop("_open")
        profit = round(g.pop("_profit"), 2)
        if opened:
            g.update(status="aberta", profit=None, expiry_ts=min(opened))
        elif any(s in ("ganhou", "perdeu", "empate") for s in st):
            g.update(status="ganhou" if profit > 0 else "perdeu" if profit < 0 else "empate", profit=profit)
        else:
            g.update(status=st[0], profit=None)  # "sem leitura": o app não viu o preço do fim
        g["wallets"].sort()
        out.append(g)
    return out


def build_state(st, extras=None, decision_text: str = "", decision_action: str = "",
                url: Optional[str] = None) -> dict:
    """O que a página mostra, a partir do estado das telas do simulador (SimState)."""
    from ..ui.sim_view import WALLETS, day_summary
    from ..history.simbook import START_BALANCE

    bal = st.balances()
    wallets = []
    for w in WALLETS:
        s = day_summary(st.get(w.key))
        wallets.append({"key": w.key, "name": w.name, "cut": w.cut, "desc": w.desc,
                        "tag": "RISCO" if w.tag == "RISCO" else "", "balance": round(bal[w.key], 2),
                        "today": {"n": s["n"], "wins": s["wins"], "losses": s["losses"],
                                  "profit": round(s["profit"], 2)},
                        "open": s["open"]})
    ready = st.last_p is not None and st.last_ts is not None and st.now - st.last_ts < 180
    need = None
    if not ready:
        if st.note:
            need = st.note
        elif st.hist_min is not None and st.hist_min < st.hist_need:
            need = f"juntando histórico: {st.hist_min} de {st.hist_need} min"
    account = {}
    if extras is not None and extras.wallet_account:
        account = {"type": extras.wallet_account, "balance": extras.wallet_balance, "day": extras.wallet_day}
    return {
        "now": st.now or time.time(), "asset": st.asset, "price": st.price, "stake": st.stake,
        "start_balance": START_BALANCE, "url": url,
        "model": {"p_up": st.last_p, "ts": st.last_ts, "ready": ready, "note": need},
        "decision": {"action": decision_action, "text": decision_text},
        "account": account, "wallets": wallets, "entries": _group(st.entries), "days": st.days(),
    }


class WebServer:
    """Servidor da página, numa linha de execução à parte. ``publish`` troca o estado mostrado."""

    def __init__(self, port: int = PORT, host: str = "0.0.0.0") -> None:
        self._lock = threading.Lock()
        self._body = b'{"waiting": true}'
        self._at = 0.0
        self.last_hit = 0.0  # última vez que alguma página aberta pediu o estado
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 (nome da biblioteca)
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    body, ctype = PAGE.read_bytes(), "text/html; charset=utf-8"
                elif path == "/api/state":
                    srv.last_hit = time.time()
                    body, ctype = srv.body(), "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):  # sem uma linha no log a cada segundo
                pass

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="pagina", daemon=True)
        self.thread.start()

    def publish(self, state: dict) -> None:
        body = json.dumps(state, ensure_ascii=False, default=float).encode()
        with self._lock:
            self._body, self._at = body, time.time()

    def body(self) -> bytes:
        with self._lock:
            body, at = self._body, self._at
        if at and time.time() - at > STALE_S:  # o app parou de ler (Mac dormiu, IQ fechada...)
            d = json.loads(body)
            d["stale"] = int(time.time() - at)
            return json.dumps(d, ensure_ascii=False).encode()
        return body

    def urls(self) -> tuple[str, Optional[str]]:
        ip = lan_ip()
        return f"http://localhost:{self.port}", (f"http://{ip}:{self.port}" if ip else None)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
