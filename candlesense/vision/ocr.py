"""OCR: framework Vision da Apple (macOS) com fallback para rapidocr.

Todas as engines recebem uma imagem BGR (numpy) e uma região opcional, e
devolvem tokens com caixas em coordenadas da imagem inteira.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Optional, Protocol

import cv2
import numpy as np

from ..models import Box, OcrToken

log = logging.getLogger(__name__)


class OcrEngine(Protocol):
    name: str

    def read(self, bgr: np.ndarray, roi: Optional[Box] = None) -> list[OcrToken]: ...


def _crop(bgr: np.ndarray, roi: Optional[Box]) -> tuple[np.ndarray, int, int]:
    if roi is None:
        return bgr, 0, 0
    return bgr[roi.y : roi.y2, roi.x : roi.x2], roi.x, roi.y


def _upscale_factor(img: np.ndarray, roi: Optional[Box]) -> int:
    # Recortes pequenos (um rótulo, o cronômetro) leem melhor ampliados.
    if roi is None:
        return 1
    return 2 if min(img.shape[:2]) < 120 else 1


class RapidOcrEngine:
    """OCR multiplataforma (ONNX). Mais lento que o Vision, mas roda em qualquer lugar."""

    name = "rapidocr"

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR  # import tardio: é pesado

        self._eng = RapidOCR()

    def read(self, bgr: np.ndarray, roi: Optional[Box] = None) -> list[OcrToken]:
        img, ox, oy = _crop(bgr, roi)
        if img.size == 0:
            return []
        f = _upscale_factor(img, roi)
        if f != 1:
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        # Borda escura ajuda a detecção quando o texto encosta no recorte.
        pad = 8
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        try:
            result, _ = self._eng(img, use_cls=False)
        except TypeError:
            result, _ = self._eng(img)
        tokens: list[OcrToken] = []
        for item in result or []:
            pts, text, score = item[0], str(item[1]), float(item[2])
            arr = np.asarray(pts, dtype=float)
            x1, y1 = (arr.min(axis=0) - pad) / f
            x2, y2 = (arr.max(axis=0) - pad) / f
            box = Box(int(round(x1)) + ox, int(round(y1)) + oy, int(round(x2 - x1)), int(round(y2 - y1)))
            tokens.append(OcrToken(text.strip(), box, score))
        return tokens


class VisionOcrEngine:
    """OCR nativo do macOS (Vision). Rápido e preciso em números."""

    name = "vision"

    def __init__(self, accurate: bool = True) -> None:
        import Quartz  # noqa: F401
        import Vision

        self._Vision = Vision
        self._accurate = accurate

    def read(self, bgr: np.ndarray, roi: Optional[Box] = None) -> list[OcrToken]:
        import Quartz

        Vision = self._Vision
        img, ox, oy = _crop(bgr, roi)
        if img.size == 0:
            return []
        f = _upscale_factor(img, roi)
        if f != 1:
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        data = rgba.tobytes()
        provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
        cs = Quartz.CGColorSpaceCreateDeviceRGB()
        cg = Quartz.CGImageCreate(
            w, h, 8, 32, w * 4, cs,
            Quartz.kCGImageAlphaNoneSkipLast | Quartz.kCGBitmapByteOrderDefault,
            provider, None, False, Quartz.kCGRenderingIntentDefault,
        )
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(
            Vision.VNRequestTextRecognitionLevelAccurate if self._accurate
            else Vision.VNRequestTextRecognitionLevelFast
        )
        req.setUsesLanguageCorrection_(False)
        try:
            req.setRecognitionLanguages_(["en-US", "pt-BR"])
        except Exception:  # idiomas indisponíveis: usa o padrão
            pass
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
        ok, err = handler.performRequests_error_([req], None)
        if not ok:
            log.warning("Vision falhou: %s", err)
            return []
        tokens: list[OcrToken] = []
        for obs in req.results() or []:
            cands = obs.topCandidates_(1)
            if not cands:
                continue
            c = cands[0]
            bb = obs.boundingBox()  # normalizado, origem no canto inferior esquerdo
            bx = bb.origin.x * w
            by = (1.0 - bb.origin.y - bb.size.height) * h
            bw = bb.size.width * w
            bh = bb.size.height * h
            box = Box(
                int(round(bx / f)) + ox, int(round(by / f)) + oy,
                int(round(bw / f)), int(round(bh / f)),
            )
            tokens.append(OcrToken(str(c.string()).strip(), box, float(c.confidence())))
        return tokens


def best_engine(prefer: str = "auto") -> OcrEngine:
    """Vision no macOS; rapidocr se o Vision não estiver disponível."""
    if prefer in ("auto", "vision") and sys.platform == "darwin":
        try:
            return VisionOcrEngine()
        except Exception as e:  # pyobjc ausente ou macOS antigo
            log.warning("Vision indisponível (%s); usando rapidocr", e)
    return RapidOcrEngine()


# ---------------------------------------------------------------- parsing ---

_CONFUSABLE = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "l": "1", "I": "1",
                             "|": "1", "S": "5", "s": "5", "B": "8", ",": ".", " ": ""})
_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_TIME_RE = re.compile(r"(\d{1,2})\s*[:;.]\s*(\d{2})(?!\d)")
_PCT_RE = re.compile(r"\+\s*(\d{1,3})\s*%|(\d{1,3})\s*%")


def parse_number(text: str) -> Optional[tuple[float, int]]:
    """'1.08345' -> (1.08345, 5). Devolve None se não for um número limpo."""
    t = text.strip().translate(_CONFUSABLE)
    if not _NUM_RE.match(t):
        return None
    dec = len(t.split(".")[1]) if "." in t else 0
    try:
        return float(t), dec
    except ValueError:
        return None


def parse_timer(text: str) -> Optional[int]:
    """'00:23' -> 23 segundos; '1:05' -> 65."""
    m = _TIME_RE.search(text.strip())
    if not m:
        return None
    mm, ss = int(m.group(1)), int(m.group(2))
    if ss >= 60:
        return None
    return mm * 60 + ss


def parse_payout(text: str) -> Optional[int]:
    """'Lucro +87%' -> 87."""
    m = _PCT_RE.search(text.replace("O", "0"))
    if not m:
        return None
    v = int(m.group(1) or m.group(2))
    return v if 1 <= v <= 100 else None


def is_clock_text(text: str) -> bool:
    return bool(re.fullmatch(r"\s*\d{1,2}\s*:\s*\d{2}\s*", text))
