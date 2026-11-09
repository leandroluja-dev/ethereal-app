"""Ajuste de posição/tamanho da janela da IQ Option pela API de Acessibilidade.

Só roda quando você pede (opção --ajustar-janela). Usa AXUIElement, nunca o
mouse ou o teclado.
"""

from __future__ import annotations

import logging

from .windows import WindowInfo

log = logging.getLogger(__name__)


def set_window_frame(win: WindowInfo, x: float, y: float, w: float, h: float) -> bool:
    try:
        import ApplicationServices as AS
        import Quartz
    except ImportError:
        return False

    app = AS.AXUIElementCreateApplication(win.pid)
    err, windows = AS.AXUIElementCopyAttributeValue(app, AS.kAXWindowsAttribute, None)
    if err != 0 or not windows:
        log.warning("Acessibilidade: não consegui listar as janelas (erro %s)", err)
        return False

    target = None
    for wref in windows:
        e, title = AS.AXUIElementCopyAttributeValue(wref, AS.kAXTitleAttribute, None)
        if e == 0 and title and win.title and str(title) == win.title:
            target = wref
            break
    if target is None:
        target = windows[0]

    pos = AS.AXValueCreate(AS.kAXValueCGPointType, Quartz.CGPoint(x, y))
    size = AS.AXValueCreate(AS.kAXValueCGSizeType, Quartz.CGSize(w, h))
    e1 = AS.AXUIElementSetAttributeValue(target, AS.kAXPositionAttribute, pos)
    e2 = AS.AXUIElementSetAttributeValue(target, AS.kAXSizeAttribute, size)
    if e1 != 0 or e2 != 0:
        log.warning("Acessibilidade: ajuste recusado (erros %s, %s)", e1, e2)
        return False
    return True
