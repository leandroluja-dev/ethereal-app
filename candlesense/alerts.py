"""Avisos sonoros (macOS). Nunca bloqueiam a leitura."""

from __future__ import annotations

import subprocess
import sys

SOUNDS = {
    "entrar": "/System/Library/Sounds/Glass.aiff",
    "sair": "/System/Library/Sounds/Basso.aiff",
    "resultado": "/System/Library/Sounds/Pop.aiff",
}


def play(kind: str, voice: str | None = None) -> None:
    if sys.platform != "darwin":
        return
    try:
        path = SOUNDS.get(kind)
        if path:
            subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if voice:
            subprocess.Popen(["say", "-r", "210", voice], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
