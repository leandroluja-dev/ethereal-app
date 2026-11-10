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


_CLOSED: set[str] = set()   # janelas fechadas pelo botão vermelho
_TARGET = None
_ACTIVATED = False


def _close_target():
    """Objeto que recebe o clique no botão vermelho (criado uma vez só)."""
    global _TARGET
    if _TARGET is None:
        import objc
        from Foundation import NSObject

        class CandleSenseCloseTarget(NSObject):
            @objc.IBAction
            def fechar_(self, sender):
                w = sender.window()
                if w is not None:
                    _CLOSED.add(str(w.title() or ""))
                    w.orderOut_(None)

        _TARGET = CandleSenseCloseTarget.alloc().init()
    return _TARGET


def was_closed(title: str) -> bool:
    """A pessoa fechou essa janela pelo botão vermelho?"""
    return title in _CLOSED


def reopen(title: str) -> None:
    _CLOSED.discard(title)


def window_buttons(title: str) -> bool:
    """Liga a barra normal do macOS (fechar e minimizar) na janela com esse título.

    A janela do OpenCV aparece com os botões cinza (desativados). Aqui o
    vermelho fecha de verdade e o amarelo minimiza; a janela continua por cima
    das outras. Devolve True quando achou a janela.
    """
    global _ACTIVATED
    if sys.platform != "darwin":
        return False
    try:
        from AppKit import (NSApplication, NSFloatingWindowLevel, NSWindowStyleMaskClosable,
                            NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskTitled)

        app = NSApplication.sharedApplication()
        want = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
        for w in app.windows():
            if str(w.title() or "") != title:
                continue
            if (w.styleMask() & want) != want:
                w.setStyleMask_(w.styleMask() | want)
            w.setLevel_(NSFloatingWindowLevel)
            close_btn = w.standardWindowButton_(0)   # NSWindowCloseButton
            mini_btn = w.standardWindowButton_(1)    # NSWindowMiniaturizeButton
            if close_btn is not None:
                close_btn.setEnabled_(True)
                close_btn.setTarget_(_close_target())
                close_btn.setAction_("fechar:")
            if mini_btn is not None:
                mini_btn.setEnabled_(True)
            # A barra de título precisa ficar dentro da tela para dar para clicar.
            scr = w.screen()
            if scr is not None:
                vis, f = scr.visibleFrame(), w.frame()
                top = vis.origin.y + vis.size.height
                if f.origin.y + f.size.height > top:
                    w.setFrameTopLeftPoint_((f.origin.x, top))
            if not _ACTIVATED:  # uma vez, ao abrir: deixa a janela ativa (botões coloridos)
                _ACTIVATED = True
                app.activateIgnoringOtherApps_(True)
                w.makeKeyAndOrderFront_(None)
            return True
    except Exception:
        pass
    return False
