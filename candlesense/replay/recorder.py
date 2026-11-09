"""Grava sessões de tela (quadros PNG sem perdas + metadados) em segundo plano, sem atrasar a leitura."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import cv2

from ..capture.frame import Frame

log = logging.getLogger(__name__)


class SessionRecorder:
    def __init__(self, base_dir: str | Path, fps: float = 2.0, image_format: str = "png") -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = Path(base_dir) / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(exist_ok=True)
        self._interval = 1.0 / max(0.1, fps)
        self._ext = "." + image_format.lstrip(".")
        self._last = -1e9
        self._n = 0
        self._q: queue.Queue[tuple[int, Frame, dict | None] | None] = queue.Queue(maxsize=30)
        self._meta = open(self.dir / "meta.jsonl", "a", encoding="utf-8")  # noqa: SIM115
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        log.info("gravando sessão em %s", self.dir)

    @property
    def count(self) -> int:
        return self._n

    def offer(self, frame: Frame, reading_summary: dict | None = None) -> None:
        """Entrega um quadro; respeita o fps de gravação e descarta se o disco estiver lento."""
        if frame.image is None or frame.ts - self._last < self._interval:
            return
        self._last = frame.ts
        self._n += 1
        try:
            self._q.put_nowait((self._n, frame, reading_summary))
        except queue.Full:
            log.warning("gravação atrasada: quadro descartado")

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            n, frame, summary = item
            name = f"{n:06d}{self._ext}"
            try:
                assert frame.image is not None
                cv2.imwrite(str(self.dir / "frames" / name), frame.image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                meta = {
                    "n": n, "file": name, "ts": frame.ts,
                    "window": None if frame.window is None else {
                        "title": frame.window.title, "kind": frame.window.kind, "rect": frame.window.rect.to_list()},
                    "reading": summary,
                }
                self._meta.write(json.dumps(meta, ensure_ascii=False) + "\n")
                self._meta.flush()
            except Exception:  # noqa: BLE001
                log.exception("falha ao gravar quadro")

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=10)
        self._meta.close()
        log.info("gravação encerrada: %d quadros em %s", self._n, self.dir)
