"""Janela de debug: mostra a captura com tudo o que a visão leu desenhado por cima.

Também é onde fica o plano B: selecionar manualmente a área das velas arrastando o mouse
sobre a imagem (dentro desta janela, nunca na IQ Option).
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QRubberBand, QSizePolicy, QVBoxLayout, QWidget

from ..common.geometry import Rect
from ..vision.analyzer import Reading
from ..vision.debug_draw import draw_overlay
from .qt_utils import bgr_to_pixmap


class _ImageLabel(QLabel):
    selected = pyqtSignal(QRect)

    def __init__(self) -> None:
        super().__init__()
        self.select_mode = False
        self._origin: QPoint | None = None
        self._band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(QSize(320, 200))

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:  # noqa: N802
        if ev is not None and self.select_mode:
            self._origin = ev.position().toPoint()
            self._band.setGeometry(QRect(self._origin, QSize()))
            self._band.show()

    def mouseMoveEvent(self, ev: QMouseEvent | None) -> None:  # noqa: N802
        if ev is not None and self._origin is not None:
            self._band.setGeometry(QRect(self._origin, ev.position().toPoint()).normalized())

    def mouseReleaseEvent(self, ev: QMouseEvent | None) -> None:  # noqa: N802
        if ev is not None and self._origin is not None:
            r = QRect(self._origin, ev.position().toPoint()).normalized()
            self._origin = None
            self._band.hide()
            if r.width() > 20 and r.height() > 20:
                self.selected.emit(r)


class DebugWindow(QWidget):
    snapshot_requested = pyqtSignal()
    manual_rect_selected = pyqtSignal(object)  # Rect em coordenadas da janela capturada
    manual_rect_cleared = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CandleSense — Debug da visão")
        self.resize(1100, 680)
        self._img_size: tuple[int, int] | None = None
        self._scale = 1.0
        self.image = _ImageLabel()
        self.image.selected.connect(self._on_selected)
        self.hint = QLabel("Ciano: velas · Amarelo: eixo de preço · Magenta: eixo de tempo · Branco: vela em formação · "
                           "X: vela coberta por indicador")
        self.hint.setWordWrap(True)
        b_snap = QPushButton("Salvar captura")
        b_snap.clicked.connect(self.snapshot_requested.emit)
        self.b_select = QPushButton("Selecionar área das velas (plano B)")
        self.b_select.setCheckable(True)
        self.b_select.toggled.connect(self._toggle_select)
        b_clear = QPushButton("Voltar à detecção automática")
        b_clear.clicked.connect(self.manual_rect_cleared.emit)
        row = QHBoxLayout()
        for b in (b_snap, self.b_select, b_clear):
            row.addWidget(b)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addLayout(row)
        lay.addWidget(self.hint)
        lay.addWidget(self.image, 1)

    def _toggle_select(self, on: bool) -> None:
        self.image.select_mode = on
        self.hint.setText("Arraste sobre a imagem para marcar só a área onde ficam as velas (sem o eixo de preço)."
                          if on else "Ciano: velas · Amarelo: eixo de preço · Magenta: eixo de tempo")

    def _on_selected(self, r: QRect) -> None:
        if not self._img_size or self._scale <= 0:
            return
        s = self._scale
        rect = Rect(int(r.x() / s), int(r.y() / s), int(r.width() / s), int(r.height() / s))
        self.b_select.setChecked(False)
        self.manual_rect_selected.emit(rect)

    def show_frame(self, img: np.ndarray | None, reading: Reading) -> None:
        if img is None or not self.isVisible():
            return
        over = draw_overlay(img, reading)
        pm = bgr_to_pixmap(over)
        avail = self.image.size()
        self._img_size = (img.shape[1], img.shape[0])
        self._scale = min(avail.width() / img.shape[1], avail.height() / img.shape[0], 1.0)
        if self._scale < 1.0:
            pm = pm.scaled(int(img.shape[1] * self._scale), int(img.shape[0] * self._scale),
                           Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image.setPixmap(pm)
