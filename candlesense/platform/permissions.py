"""Permissões do macOS: Gravação de Tela (obrigatória) e Acessibilidade (opcional).

Gravação de Tela é o que permite capturar a janela da IQ Option.
Acessibilidade só é usada para ajustar o tamanho/posição da janela pela API
AXUIElement, quando você pedir. O CandleSense nunca clica, nunca move o mouse e
nunca envia teclas.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass

SETTINGS_SCREEN = "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
SETTINGS_ACCESSIBILITY = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"


@dataclass
class PermissionStatus:
    screen_recording: bool
    accessibility: bool


def is_macos() -> bool:
    return sys.platform == "darwin"


def screen_recording_granted() -> bool:
    if not is_macos():
        return False
    import Quartz

    try:
        return bool(Quartz.CGPreflightScreenCaptureAccess())
    except AttributeError:  # macOS < 10.15 não exige permissão
        return True


def request_screen_recording() -> bool:
    """Mostra o pedido do sistema (só aparece na primeira vez)."""
    if not is_macos():
        return False
    import Quartz

    try:
        return bool(Quartz.CGRequestScreenCaptureAccess())
    except AttributeError:
        return True


def accessibility_granted(prompt: bool = False) -> bool:
    if not is_macos():
        return False
    try:
        from ApplicationServices import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt

        return bool(AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: bool(prompt)}))
    except Exception:
        from ApplicationServices import AXIsProcessTrusted

        return bool(AXIsProcessTrusted())


def status() -> PermissionStatus:
    return PermissionStatus(screen_recording_granted(), accessibility_granted(False))


def open_settings(url: str) -> None:
    subprocess.run(["open", url], check=False)


def guide(need_accessibility: bool = False, wait_s: float = 180.0, out=print) -> PermissionStatus:
    """Guia de primeira execução. Abre os Ajustes do Sistema e espera a liberação."""
    if not is_macos():
        out("Este passo só existe no macOS.")
        return PermissionStatus(False, False)

    st = status()
    if not st.screen_recording:
        out("")
        out("== Permissão de Gravação de Tela ==")
        out("O CandleSense precisa ver a janela da IQ Option para ler o gráfico.")
        out("1. Vou abrir Ajustes do Sistema > Privacidade e Segurança > Gravação de Tela.")
        out("2. Ative a chave do app que você usa para rodar o CandleSense")
        out("   (o próprio CandleSense.app, ou o Terminal/iTerm se estiver rodando pelo terminal).")
        out("3. O macOS pode pedir para reabrir o app. Depois disso é só rodar de novo.")
        request_screen_recording()
        open_settings(SETTINGS_SCREEN)
        st.screen_recording = _wait(screen_recording_granted, wait_s)
        if st.screen_recording:
            out("Gravação de Tela liberada.")
        else:
            out("Ainda não liberada. Se você já ativou a chave, feche e abra o app de novo.")

    if need_accessibility and not st.accessibility:
        out("")
        out("== Permissão de Acessibilidade (opcional) ==")
        out("Só serve para o CandleSense ajustar o tamanho da janela da IQ Option.")
        out("Ele nunca clica, nunca move o mouse e nunca envia teclas.")
        accessibility_granted(prompt=True)
        open_settings(SETTINGS_ACCESSIBILITY)
        st.accessibility = _wait(lambda: accessibility_granted(False), wait_s)
        out("Acessibilidade liberada." if st.accessibility else "Acessibilidade não liberada (dá para seguir sem ela).")
    return st


def _wait(check, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(1.0)
    return check()
