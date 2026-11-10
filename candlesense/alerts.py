"""Avisos sonoros (macOS). Nunca bloqueiam a leitura.

Os sons do simulador (ganhou / perdeu) são gerados pelo próprio app na
primeira vez, em arquivos .wav curtos, e tocados com o afplay do Mac.
"""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

SOUNDS = {
    "entrar": "/System/Library/Sounds/Glass.aiff",
    "sair": "/System/Library/Sounds/Basso.aiff",
    "resultado": "/System/Library/Sounds/Pop.aiff",
}

RATE = 44100


def _env(t: float, attack: float, decay: float) -> float:
    """Envelope: sobe em ``attack`` s e cai exponencialmente."""
    if t < 0:
        return 0.0
    a = min(1.0, t / attack) if attack > 0 else 1.0
    return a * math.exp(-t / decay)


def money_samples() -> list[float]:
    """"Ca-tching" de caixa registradora: estalo curto e dois sinos agudos."""
    n = int(RATE * 0.9)
    out = [0.0] * n
    seed = 12345
    for i in range(int(RATE * 0.025)):  # estalo (ruído que some rápido)
        seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
        out[i] += 0.35 * ((seed / 0x7FFFFFFF) * 2 - 1) * math.exp(-i / (RATE * 0.004))
    # sino: fundamental + parciais inarmônicas, como metal
    for t0, f, amp in ((0.04, 1568.0, 0.32), (0.13, 2093.0, 0.38)):
        for mult, a2, dec in ((1.0, 1.0, 0.28), (2.76, 0.35, 0.12), (5.40, 0.15, 0.06)):
            for i in range(int(t0 * RATE), n):
                t = i / RATE - t0
                out[i] += amp * a2 * _env(t, 0.002, dec) * math.sin(2 * math.pi * f * mult * t)
    return out


def error_samples() -> list[float]:
    """Dois tons baixos e curtos, descendo. Discreto."""
    n = int(RATE * 0.42)
    out = [0.0] * n
    for t0, f in ((0.0, 392.0), (0.16, 311.1)):
        for i in range(int(t0 * RATE), n):
            t = i / RATE - t0
            if t > 0.22:
                break
            e = min(1.0, t / 0.012) * min(1.0, (0.22 - t) / 0.06)
            out[i] += 0.22 * e * (math.sin(2 * math.pi * f * t) + 0.25 * math.sin(4 * math.pi * f * t))
    return out


def write_wav(path: Path, samples: list[float]) -> Path:
    peak = max(1e-9, max(abs(s) for s in samples))
    k = min(1.0, 0.9 / peak)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1, min(1, s * k)) * 32767)) for s in samples))
    return path


_GENERATED = {"ganhou": money_samples, "perdeu": error_samples}


def _sound_path(kind: str) -> str | None:
    if kind in SOUNDS:
        return SOUNDS[kind]
    gen = _GENERATED.get(kind)
    if gen is None:
        return None
    d = Path(tempfile.gettempdir()) / "candlesense_sons"
    p = d / f"{kind}_v1.wav"
    if not p.exists():
        d.mkdir(parents=True, exist_ok=True)
        write_wav(p, gen())
    return str(p)


def play(kind: str, voice: str | None = None) -> None:
    if sys.platform != "darwin":
        return
    try:
        path = _sound_path(kind)
        if path:
            subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if voice:
            subprocess.Popen(["say", "-r", "210", voice], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
