"""OCR com RapidOCR (padrão) e Tesseract (alternativa). Nunca lança exceção para o chamador."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from ..common.geometry import Rect
from ..config import OcrConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrToken:
    text: str
    conf: float
    box: Rect


class OcrEngine(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    def read(self, img: np.ndarray) -> list[OcrToken]:
        """Detecta e reconhece todos os textos da imagem."""
        ...

    def read_lines(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Reconhece uma linha de texto por recorte (sem detecção, rápido)."""
        ...


def prepare_line_crop(crop: np.ndarray, min_height: int = 24, pad: int = 4) -> np.ndarray:
    """Amplia recortes pequenos e adiciona margem: melhora muito o reconhecimento de rótulos de 10-12px."""
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    h = crop.shape[0]
    if 0 < h < min_height:
        f = min_height / h
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    border = [int(v) for v in np.median(crop.reshape(-1, 3), axis=0)]
    return cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border)


class NullOcr:
    name = "nenhum"

    @property
    def available(self) -> bool:
        return False

    def read(self, img: np.ndarray) -> list[OcrToken]:
        return []

    def read_lines(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        return [("", 0.0) for _ in crops]


class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self, cfg: OcrConfig) -> None:
        self._cfg = cfg
        self._engine = None
        self._lock = threading.Lock()
        self._failed = False

    def _get(self):  # type: ignore[no-untyped-def]
        if self._engine is None and not self._failed:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._engine = RapidOCR()
            except Exception:  # noqa: BLE001
                log.exception("RapidOCR indisponível")
                self._failed = True
        return self._engine

    @property
    def available(self) -> bool:
        return self._get() is not None

    def read(self, img: np.ndarray) -> list[OcrToken]:
        eng = self._get()
        if eng is None or img.size == 0:
            return []
        try:
            with self._lock:
                result, _ = eng(img)
        except Exception:  # noqa: BLE001
            log.exception("falha no OCR")
            return []
        tokens: list[OcrToken] = []
        for item in result or []:
            try:
                poly, text, score = item[0], str(item[1]), float(item[2])
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                tokens.append(OcrToken(text.strip(), score, Rect.from_ltrb(min(xs), min(ys), max(xs), max(ys))))
            except (IndexError, TypeError, ValueError):
                continue
        return tokens

    def read_lines(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        eng = self._get()
        if eng is None or not crops:
            return [("", 0.0) for _ in crops]
        prepared = [prepare_line_crop(c, self._cfg.min_text_height_px) for c in crops]
        try:
            with self._lock:
                res = eng.text_rec(prepared)
            rec = res[0] if isinstance(res, tuple) else res
            out: list[tuple[str, float]] = []
            for r in rec:
                out.append((str(r[0]).strip(), float(r[1])))
            if len(out) == len(crops):
                return out
        except Exception:  # noqa: BLE001
            log.debug("reconhecimento em lote falhou; tentando um a um", exc_info=True)
        out = []
        for c in prepared:
            try:
                with self._lock:
                    r, _ = eng(c, use_det=False, use_cls=False, use_rec=True)
                out.append((str(r[0][0]).strip(), float(r[0][1])) if r else ("", 0.0))
            except Exception:  # noqa: BLE001
                out.append(("", 0.0))
        return out


class TesseractEngine:
    name = "tesseract"

    def __init__(self, cfg: OcrConfig) -> None:
        self._cfg = cfg
        self._ok: bool | None = None

    @property
    def available(self) -> bool:
        if self._ok is None:
            try:
                import pytesseract

                if self._cfg.tesseract_cmd:
                    pytesseract.pytesseract.tesseract_cmd = self._cfg.tesseract_cmd
                pytesseract.get_tesseract_version()
                self._ok = True
            except Exception:  # noqa: BLE001
                self._ok = False
        return self._ok

    @staticmethod
    def _prep(img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if float(np.median(g)) < 128:
            g = 255 - g  # texto claro em fundo escuro -> texto escuro em fundo claro
        return g

    def read(self, img: np.ndarray) -> list[OcrToken]:
        if not self.available:
            return []
        import pytesseract

        try:
            g = cv2.resize(self._prep(img), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            d = pytesseract.image_to_data(g, config="--psm 11", output_type=pytesseract.Output.DICT)
        except Exception:  # noqa: BLE001
            log.exception("falha no Tesseract")
            return []
        out: list[OcrToken] = []
        for i, text in enumerate(d.get("text", [])):
            text = (text or "").strip()
            conf = float(d["conf"][i]) if str(d["conf"][i]) not in ("", "-1") else -1.0
            if not text or conf < 0:
                continue
            box = Rect(d["left"][i] // 2, d["top"][i] // 2, d["width"][i] // 2, d["height"][i] // 2)
            out.append(OcrToken(text, conf / 100.0, box))
        return out

    def read_lines(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        if not self.available:
            return [("", 0.0) for _ in crops]
        import pytesseract

        out = []
        for c in crops:
            try:
                g = self._prep(prepare_line_crop(c, self._cfg.min_text_height_px * 2))
                t = pytesseract.image_to_string(g, config="--psm 7").strip()
                out.append((t, 0.8 if t else 0.0))
            except Exception:  # noqa: BLE001
                out.append(("", 0.0))
        return out


def create_ocr(cfg: OcrConfig) -> OcrEngine:
    order = {"rapidocr": [RapidOcrEngine], "tesseract": [TesseractEngine]}.get(
        cfg.backend, [RapidOcrEngine, TesseractEngine]
    )
    for cls in order:
        eng = cls(cfg)
        if eng.available:
            log.info("OCR: usando %s", eng.name)
            return eng
    log.error("Nenhum OCR disponível: instale rapidocr-onnxruntime")
    return NullOcr()
