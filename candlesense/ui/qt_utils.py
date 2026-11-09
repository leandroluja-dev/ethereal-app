"""Conversões entre numpy/OpenCV e Qt."""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def bgr_to_pixmap(img: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())
