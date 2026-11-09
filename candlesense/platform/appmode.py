"""Modo app: o CandleSense aberto pelo ícone do Dock, sem Terminal.

Mensagens viram avisos do macOS (osascript) e o processo do Python não ganha
um ícone próprio no Dock: o único ícone é o do CandleSense.app.
"""

from __future__ import annotations

import subprocess
import sys


def _osa(script: str, wait: bool) -> None:
    if sys.platform != "darwin":
        return
    try:
        if wait:
            subprocess.run(["osascript", "-e", script], check=False, timeout=600)
        else:
            subprocess.Popen(["osascript", "-e", script])
    except Exception:
        pass


def _q(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def notify(text: str, title: str = "CandleSense") -> None:
    """Aviso discreto no canto da tela."""
    _osa(f"display notification {_q(text)} with title {_q(title)}", wait=False)


def dialog(text: str, title: str = "CandleSense") -> None:
    """Janela com botão OK (espera a pessoa fechar)."""
    _osa(f'display dialog {_q(text)} with title {_q(title)} buttons {{"OK"}} default button "OK"', wait=True)


def hide_python_dock_icon() -> None:
    """Some com o foguete do Python no Dock (as janelas continuam aparecendo)."""
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

        NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except Exception:
        pass
