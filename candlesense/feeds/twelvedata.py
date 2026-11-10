"""Velas de forex pela Twelve Data (API oficial, plano grátis com chave da pessoa).

Usado só para completar o histórico de velas de 1 min que o modelo precisa
(2h15 seguidas), sem esperar a tela juntar tudo. A tela continua sendo a fonte
do payout, do cronômetro, do preço ao vivo e do OTC (que só existe na IQ).

* A chave fica em dados/twelvedata.key, só no Mac. Na primeira vez o app pede numa
  janelinha; se a pessoa cancelar, não pergunta de novo (apague o arquivo para voltar).
* Plano grátis: 8 pedidos por minuto e 800 por dia. ``series`` respeita os dois limites
  para todo o app (outros módulos chamam a mesma função).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

API = "https://api.twelvedata.com/time_series"
PER_MIN = 8
PER_DAY = 780  # folga abaixo dos 800 do plano grátis
SKIP = "-"     # conteúdo do arquivo quando a pessoa não quis informar a chave

_lock = threading.Lock()
_recent: deque = deque()
_day = {"date": "", "n": 0}
_key_path: Optional[Path] = None
_ask_lock = threading.Lock()  # uma janelinha só, mesmo pedida por dois lugares


def key_path(dados: str | Path = "dados") -> Path:
    return Path(dados) / "twelvedata.key"


def load_key(dados: str | Path = "dados") -> Optional[str]:
    try:
        k = key_path(dados).read_text().strip()
    except OSError:
        return None
    return k if k and k != SKIP else None


def asked_before(dados: str | Path = "dados") -> bool:
    return key_path(dados).exists()


def ask_key(dados: str | Path = "dados") -> Optional[str]:
    """Pede a chave numa janelinha do Mac (uma vez). Grava em dados/twelvedata.key."""
    if sys.platform != "darwin":
        return None
    with _ask_lock:
        if asked_before(dados):  # outra parte do app já perguntou enquanto esta esperava
            return load_key(dados)
        return _ask_dialog(dados)


def _ask_dialog(dados: str | Path) -> Optional[str]:
    script = ('display dialog "Para o CandleSense ter o histórico na hora, cole aqui a API key da '
              'Twelve Data (twelvedata.com, conta grátis).\\n\\nEla fica guardada só neste Mac. '
              'Cancelar = o app junta o histórico pela tela." default answer "" with hidden answer '
              'with title "CandleSense" buttons {"Cancelar", "Salvar"} default button "Salvar"')
    try:
        p = subprocess.run(["osascript", "-e", script, "-e", "text returned of result"],
                           capture_output=True, text=True, timeout=900)
    except Exception:
        return None
    key = p.stdout.strip() if p.returncode == 0 else ""
    try:
        path = key_path(dados)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key or SKIP)
        path.chmod(0o600)
    except OSError:
        pass
    return key or None


def _allow() -> bool:
    """Reserva um pedido dentro dos limites do plano grátis (por minuto e por dia)."""
    with _lock:
        now = time.time()
        today = time.strftime("%Y-%m-%d", time.gmtime(now))
        if _day["date"] != today:
            _day["date"], _day["n"] = today, 0
        while _recent and now - _recent[0] > 60:
            _recent.popleft()
        if len(_recent) >= PER_MIN or _day["n"] >= PER_DAY:
            return False
        _recent.append(now)
        _day["n"] += 1
        return True


def symbol_of(asset: str) -> Optional[str]:
    """'GBP/USD' -> 'GBP/USD'. OTC e nomes estranhos ficam de fora."""
    if not asset or "OTC" in asset.upper():
        return None
    a = asset.strip().upper().split()[0]
    if len(a) == 7 and a[3] == "/" and a.replace("/", "").isalpha():
        return a
    return None


def _get_json(url: str, timeout: float) -> dict:
    """GET com o Python; se faltar o certificado (Python do python.org no Mac), usa o curl do sistema.

    A URL vai ao curl pela entrada padrão, nunca na linha de comando (lá apareceria a chave).
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" not in str(e) or not Path("/usr/bin/curl").exists():
            raise
    cfg = f'url = "{url}"\nsilent\nshow-error\nmax-time = {int(timeout)}\n'
    p = subprocess.run(["/usr/bin/curl", "-K", "-"], input=cfg, capture_output=True, text=True,
                       timeout=timeout + 5)
    if p.returncode != 0:
        raise OSError(f"curl {p.returncode}: {p.stderr.strip()[:200]}")
    return json.loads(p.stdout)


def series(symbol: str, interval: str = "1min", outputsize: int = 30, key: Optional[str] = None,
           dados: str | Path = "dados", timeout: float = 15) -> Optional[list[tuple[int, float, float, float, float]]]:
    """Velas [(início em UTC, open, high, low, close)], da mais antiga para a mais nova.

    Devolve None se não houver chave, se o limite do plano já foi usado ou se der erro.
    Inclui a vela ainda em formação (quem chama decide se usa).
    """
    key = key or load_key(dados)
    if not key or not _allow():
        return None
    q = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "outputsize": int(outputsize),
                                "timezone": "UTC", "apikey": key, "order": "asc"})
    try:
        data = _get_json(f"{API}?{q}", timeout)
    except Exception as e:
        print(f"twelvedata: {e!r}", flush=True)
        return None
    if data.get("status") != "ok":
        print(f"twelvedata: {str(data.get('message') or data)[:200]}", flush=True)
        return None
    out = []
    for v in data.get("values", []):
        try:
            dt = datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S" if " " in v["datetime"] else "%Y-%m-%d")
            start = int(dt.replace(tzinfo=timezone.utc).timestamp())
            out.append((start, float(v["open"]), float(v["high"]), float(v["low"]), float(v["close"])))
        except (KeyError, ValueError):
            continue
    out.sort()
    return out


class FitaFiller:
    """Completa a fita de 1 min (tabela do modo medição) em segundo plano.

    Ao abrir e quando faltar vela nas últimas 2h30, pede o pedaço que falta; fora isso,
    confere a cada 3 min (≈ 480 pedidos por dia, dentro do plano grátis). Usa uma conexão
    própria ao banco: nunca trava a leitura da tela.
    """

    def __init__(self, db_path: str | Path, dados: str | Path = "dados", need: int = 150,
                 every_s: float = 180.0) -> None:
        self.db_path = str(db_path)
        self.dados = dados
        self.need = need
        self.every_s = every_s
        self.asset: Optional[str] = None  # par na tela agora (o laço principal atualiza)
        self.status: Optional[str] = None
        self._stop = threading.Event()
        self._last: dict[str, float] = {}

    def start(self) -> "FitaFiller":
        threading.Thread(target=self._run, name="twelvedata", daemon=True).start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        import sqlite3

        if not load_key(self.dados):
            if ask_key(self.dados) is None:
                self.status = "sem chave"
                return
        db = sqlite3.connect(self.db_path, timeout=10)
        while not self._stop.is_set():
            try:
                self.tick(db)
            except Exception as e:  # nunca derruba o app
                print(f"twelvedata: {e!r}", flush=True)
            self._stop.wait(10)

    def missing(self, db, asset: str, now: float) -> int:
        """Quantas velas fechadas faltam na fita entre agora e ``need`` minutos atrás."""
        last_closed = int(now // 60) * 60 - 60
        lo = last_closed - (self.need - 1) * 60
        n = db.execute("SELECT COUNT(*) FROM fita WHERE asset=? AND start BETWEEN ? AND ?",
                       (asset, lo, last_closed)).fetchone()[0]
        return self.need - n

    def tick(self, db, now: Optional[float] = None) -> int:
        """Uma rodada: pede velas se for a hora. Devolve quantas gravou."""
        now = time.time() if now is None else now
        asset = self.asset
        sym = symbol_of(asset or "")
        if sym is None:
            return 0
        if db.execute("SELECT name FROM sqlite_master WHERE name='fita'").fetchone() is None:
            return 0
        miss = self.missing(db, asset, now)
        last = self._last.get(asset, 0.0)
        if miss <= 1 and now - last < self.every_s:
            return 0
        if miss > 1 and now - last < 60:
            return 0  # pediu agora há pouco: espera um minuto
        self._last[asset] = now
        bars = series(sym, "1min", outputsize=self.need + 5 if miss > 1 else 10, dados=self.dados)
        if not bars:
            return 0
        cutoff = int(now // 60) * 60  # vela em formação fica de fora
        rows = [(asset, s, o, h, lo, c, now) for s, o, h, lo, c in bars if s < cutoff]
        # A tela tem a palavra final nas velas que ela leu: aqui só entra o que falta.
        db.executemany("INSERT OR IGNORE INTO fita VALUES (?,?,?,?,?,?,?)", rows)
        db.commit()
        self.status = f"ok ({len(rows)} velas)"
        if miss > 1:
            print(f"twelvedata: {asset} completou {miss} velas que faltavam", flush=True)
        return len(rows)
