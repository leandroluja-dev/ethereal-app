"""Monta e roda o app: fonte de quadros + thread de análise + painel + janela de debug."""

from __future__ import annotations

import logging
import sys

from .capture.frame import FrameSource
from .common.geometry import Rect
from .config import Config, load_config, save_user_override
from .logging_setup import setup_logging

log = logging.getLogger(__name__)


def _source_factory(cfg: Config, mode: str, source_arg: str | None, theme: str, occluders):  # type: ignore[no-untyped-def]
    def make() -> FrameSource:
        if mode == "demo":
            from .capture.grabber import SyntheticSource

            return SyntheticSource(theme=theme)
        if mode == "replay":
            from .replay.player import ReplaySource

            assert source_arg
            return ReplaySource(source_arg, realtime=True, loop=True)
        if mode == "image":
            from .capture.grabber import ImageSource

            assert source_arg
            return ImageSource(source_arg)
        from .capture.grabber import LiveWindowSource

        return LiveWindowSource(cfg.capture, occluders=occluders)

    return make


def run_app(config_path: str | None = None, mode: str = "live", source_arg: str | None = None,
            theme: str = "dark") -> int:
    cfg = load_config(config_path)
    log_path = setup_logging(cfg)
    log.info("CandleSense iniciando (modo %s, config %s, log %s)", mode, cfg.source_path or "padrão", log_path)

    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from .capture.window import exclude_from_capture, move_native_window, native_window_rect
    from .ui.debug_view import DebugWindow
    from .ui.panel import Panel
    from .ui.worker import AnalysisWorker

    # O Qt 6 já deixa o processo DPI-aware por monitor (pixels físicos na captura e no SetWindowPos).
    app = QApplication.instance() or QApplication(sys.argv)
    panel = Panel(cfg.ui.always_on_top)
    panel.setWindowOpacity(cfg.ui.opacity)
    debug = DebugWindow()

    # Estado lido pela thread de captura: só números, nunca widgets (Qt não é thread-safe).
    own: dict[str, dict] = {"panel": {"hwnd": 0, "visible": True, "hidden": False},
                            "debug": {"hwnd": 0, "visible": False, "hidden": False}}

    def occluders() -> list[Rect]:
        """Janelas nossas que aparecem na captura (não puderam ser escondidas)."""
        out: list[Rect] = []
        for st in own.values():
            if st["hwnd"] and st["visible"] and not st["hidden"]:
                r = native_window_rect(st["hwnd"])
                if r is not None:
                    out.append(r)
        return out

    worker = AnalysisWorker(cfg, _source_factory(cfg, mode, source_arg, theme, occluders))

    def hide_from_capture() -> None:
        own["panel"]["hwnd"] = int(panel.winId())
        own["debug"]["hwnd"] = int(debug.winId())
        if cfg.capture.exclude_panel_from_capture:
            own["panel"]["hidden"] = exclude_from_capture(own["panel"]["hwnd"])
            own["debug"]["hidden"] = exclude_from_capture(own["debug"]["hwnd"])
            log.info("painel escondido da captura: %s; debug: %s", own["panel"]["hidden"], own["debug"]["hidden"])

    def on_reading(reading, image) -> None:  # type: ignore[no-untyped-def]
        panel.update_reading(reading)
        if debug.isVisible():
            debug.show_frame(image, reading)

    def on_adjust() -> None:
        src = worker.source
        adjust = getattr(src, "adjust_window", None)
        if adjust is None:
            panel.set_info("Ajustar janela só funciona no modo ao vivo (Windows).")
            return
        rect = adjust(cfg.capture.panel_width)
        if rect is None:
            panel.set_info("IQ Option não encontrada para ajustar.")
            return
        move_native_window(int(panel.winId()), rect)
        worker.request_relayout()
        panel.set_info("Janela ajustada.")

    def on_debug(on: bool) -> None:
        worker.want_frames = on
        own["debug"]["visible"] = on
        if on:
            debug.show()
            hide_from_capture()
        else:
            debug.hide()

    def on_manual(rect: Rect) -> None:
        cfg.vision.manual_plot_rect = rect.to_list()
        try:
            save_user_override("vision", "manual_plot_rect", rect.to_list())
        except OSError:
            log.exception("não foi possível salvar a área manual")
        worker.request_relayout()
        panel.set_info(f"Área manual salva: {rect.to_list()}")

    def on_manual_clear() -> None:
        cfg.vision.manual_plot_rect = None
        try:
            save_user_override("vision", "manual_plot_rect", None)
        except OSError:
            pass
        worker.request_relayout()
        panel.set_info("Voltando à detecção automática.")

    def on_record() -> None:
        recording = worker.toggle_recording()
        panel.b_record.setChecked(recording)
        panel.b_record.setText("Parar gravação" if recording else "Gravar sessão")

    worker.reading_ready.connect(on_reading)
    worker.info.connect(panel.set_info)
    panel.adjust_clicked.connect(on_adjust)
    panel.debug_toggled.connect(on_debug)
    panel.pause_toggled.connect(worker.set_paused)
    panel.record_clicked.connect(on_record)
    panel.snapshot_clicked.connect(worker.request_snapshot)
    panel.relayout_clicked.connect(worker.request_relayout)
    debug.snapshot_requested.connect(worker.request_snapshot)
    debug.manual_rect_selected.connect(on_manual)
    debug.manual_rect_cleared.connect(on_manual_clear)

    panel.show()
    QTimer.singleShot(0, hide_from_capture)
    if cfg.ui.debug_on_start or mode in ("demo", "image"):
        panel.b_debug.setChecked(True)
    worker.start()

    def shutdown() -> None:
        worker.stop()
        worker.wait(5000)

    app.aboutToQuit.connect(shutdown)
    debug_close = debug.closeEvent

    def on_debug_close(ev):  # type: ignore[no-untyped-def]
        panel.b_debug.setChecked(False)
        debug_close(ev)

    debug.closeEvent = on_debug_close  # type: ignore[method-assign]
    return app.exec()
