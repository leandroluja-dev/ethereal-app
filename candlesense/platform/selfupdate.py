"""Atualização sozinha, com o app aberto.

A cada minuto, em segundo plano, confere se há versão nova na branch (git
fetch). Se houver, baixa (git pull --ff-only), instala dependências se o
pyproject mudou e avisa o laço principal, que mostra "Atualizando..." e
reabre o próprio processo com o código novo. A pessoa não precisa fechar.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional


def _git(root: Path, *args: str, timeout: float = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:  # sem git, sem rede...
        return 1, repr(e)


def _deps_hash(root: Path) -> str:
    try:
        return hashlib.sha1((root / "pyproject.toml").read_bytes()).hexdigest()
    except OSError:
        return ""


class Updater:
    def __init__(self, root: Path, every_s: float = 60.0) -> None:
        self.root = root
        self.every_s = every_s
        self.ready = threading.Event()   # código novo já baixado: é só reabrir
        self.status: Optional[str] = None
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def start(self) -> "Updater":
        if not (self.root / ".git").exists():
            return self
        self._t = threading.Thread(target=self._run, name="atualizacao", daemon=True)
        self._t.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        self._stop.wait(20)  # deixa o app abrir primeiro
        while not self._stop.is_set() and not self.ready.is_set():
            try:
                self.check()
            except Exception as e:
                print(f"atualização: {e!r}", flush=True)
            self._stop.wait(self.every_s)

    def check(self) -> bool:
        rc, _ = _git(self.root, "fetch", "-q", "origin")
        if rc != 0:
            return False
        rc, head = _git(self.root, "rev-parse", "HEAD")
        rc2, up = _git(self.root, "rev-parse", "@{u}")
        if rc or rc2 or head == up:
            return False
        rc, _ = _git(self.root, "merge-base", "--is-ancestor", "HEAD", "@{u}")
        if rc != 0:
            return False  # histórico diferente: não mexe (o lançador resolve)
        self.status = "baixando"
        before = _deps_hash(self.root)
        rc, out = _git(self.root, "pull", "--ff-only", "-q", timeout=120)
        if rc != 0:
            print(f"atualização: git pull falhou: {out[:200]}", flush=True)
            self.status = None
            return False
        after = _deps_hash(self.root)
        venv_mark = self.root / ".venv" / ".candlesense_deps"
        if after and after != before:
            self.status = "instalando"
            pip = [sys.executable, "-m", "pip", "install", "-q", "-e", str(self.root)]
            p = subprocess.run(pip, cwd=self.root, capture_output=True, text=True, timeout=900)
            if p.returncode == 0:
                try:
                    venv_mark.write_text(after)
                except OSError:
                    pass
            else:
                print(f"atualização: pip falhou: {p.stderr[-300:]}", flush=True)
        _, new = _git(self.root, "log", "--oneline", "-1")
        print(f"atualização baixada: {new}", flush=True)
        self.status = "pronta"
        self.ready.set()
        return True


def _launcher(root: Path) -> Optional[Path]:
    """O CandleSense.app da pasta do projeto (o mesmo do Dock), se existir."""
    for app in (root / "CandleSense.app", Path.home() / "ethereal-app" / "CandleSense.app"):
        if (app / "Contents" / "MacOS" / "CandleSense").exists():
            return app
    return None


def restart() -> None:
    """Reabre o app com o código novo.

    No Mac, abre de novo o CandleSense.app (como um clique no Dock) e sai: o
    lançador fecha esta cópia, atualiza e abre a nova com a tela liberada. Reabrir
    o próprio processo (execv) deixava o painel travado no Mac, sem log.
    Fora do Mac, ou sem o .app, reabre o processo.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    app = _launcher(Path(__file__).resolve().parents[2]) if sys.platform == "darwin" else None
    if app is not None:
        try:
            subprocess.Popen(["open", "-n", str(app)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
            print("reabrindo pelo CandleSense.app", flush=True)
            time.sleep(0.5)
            os._exit(0)
        except Exception as e:
            print(f"não consegui reabrir pelo .app: {e!r}", flush=True)
    os.execv(sys.executable, [sys.executable, "-m", "candlesense", *sys.argv[1:]])


def loading_frame(text: str = "Atualizando o CandleSense...", w: int = 420, h: int = 140,
                  sub: str = "o app reabre sozinho em instantes"):
    """Quadro simples de 'carregando' para mostrar no painel antes de reabrir."""
    import cv2
    import numpy as np

    img = np.zeros((h, w, 3), np.uint8)
    t = time.monotonic()
    cv2.putText(img, text, (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sub, (24, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (152, 152, 159), 1, cv2.LINE_AA)
    x = int((t * 160) % (w - 80))
    cv2.rectangle(img, (24 + x // 2, 116), (24 + x // 2 + 60, 120), (140, 199, 52), -1)
    return img
