"""Thread de análise: captura -> visão -> envia a leitura para a interface, no fps configurado."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from ..capture.frame import Frame, FrameSource
from ..config import Config, app_dir
from ..replay.recorder import SessionRecorder
from ..vision.analyzer import FrameAnalyzer, Reading, Status
from ..vision.debug_draw import draw_overlay
from ..vision.ocr import create_ocr

log = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    reading_ready = pyqtSignal(object, object)  # Reading, imagem BGR (ou None)
    info = pyqtSignal(str)

    def __init__(self, cfg: Config, source_factory: Callable[[], FrameSource]) -> None:
        super().__init__()
        self.cfg = cfg
        self._factory = source_factory
        self.source: FrameSource | None = None
        self.analyzer: FrameAnalyzer | None = None
        self._stop = threading.Event()
        self._paused = False
        self.want_frames = False
        self._recorder: SessionRecorder | None = None
        self._snapshot = False
        self._lock = threading.Lock()

    # ---- controles chamados pela interface ----
    def stop(self) -> None:
        self._stop.set()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    @property
    def paused(self) -> bool:
        return self._paused

    def request_relayout(self) -> None:
        if self.analyzer is not None:
            self.analyzer.invalidate_layout()

    def request_snapshot(self) -> None:
        self._snapshot = True

    def toggle_recording(self) -> bool:
        with self._lock:
            if self._recorder is None:
                base = Path(self.cfg.replay.dir)
                if not base.is_absolute():
                    base = app_dir() / base
                self._recorder = SessionRecorder(base, self.cfg.replay.record_fps, self.cfg.replay.image_format)
                self.info.emit(f"Gravando em {self._recorder.dir}")
                return True
            rec, self._recorder = self._recorder, None
        rec.close()
        self.info.emit(f"Gravação salva: {rec.count} quadros em {rec.dir}")
        return False

    # ---- laço principal ----
    def run(self) -> None:
        try:
            self.info.emit("Carregando OCR…")
            ocr = create_ocr(self.cfg.ocr)
            self.analyzer = FrameAnalyzer(self.cfg, ocr)
            self.source = self._factory()
            self.info.emit(f"OCR: {ocr.name}")
        except Exception as e:  # noqa: BLE001
            log.exception("falha ao iniciar a análise")
            self.info.emit(f"Erro ao iniciar: {e}")
            return
        period = 1.0 / max(0.5, self.cfg.capture.fps)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                if self._paused:
                    self.reading_ready.emit(Reading(time.time(), Status.PAUSED, "Pausado"), None)
                    self._stop.wait(0.3)
                    continue
                frame = self.source.grab()
                reading = self.analyzer.analyze(frame)
                with self._lock:
                    if self._recorder is not None:
                        self._recorder.offer(frame, reading.summary())
                if self._snapshot:
                    self._snapshot = False
                    self._save_snapshot(frame, reading)
                self.reading_ready.emit(reading, frame.image if self.want_frames else None)
            except Exception:  # noqa: BLE001 - o laço nunca morre
                log.exception("erro no laço de análise")
            dt = time.monotonic() - t0
            self._stop.wait(max(0.0, period - dt))
        with self._lock:
            if self._recorder is not None:
                self._recorder.close()
        if self.source is not None:
            self.source.close()

    def _save_snapshot(self, frame: Frame, reading: Reading) -> None:
        if frame.image is None:
            self.info.emit("Nada para salvar: sem captura")
            return
        out = app_dir() / "captures"
        out.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(out / f"{stem}_raw.png"), frame.image)
        cv2.imwrite(str(out / f"{stem}_debug.png"), draw_overlay(frame.image, reading))
        (out / f"{stem}_leitura.json").write_text(json.dumps(reading.summary(), indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
        self.info.emit(f"Captura salva em {out / stem}_*.png")
