"""Captura (regras de título/layout), gravação/reprodução e configuração."""

import time

import numpy as np
import yaml

from candlesense.capture.frame import Frame, WindowInfo
from candlesense.capture.grabber import ImageSource
from candlesense.capture.window import choose_window, classify_title, standard_layout, title_matches
from candlesense.common.geometry import Rect
from candlesense.config import CaptureConfig, load_config
from candlesense.replay.player import ReplaySource
from candlesense.replay.recorder import SessionRecorder

PAT = CaptureConfig().title_patterns
EXC = CaptureConfig().exclude_title_patterns


def test_title_matching_desktop_and_browser():
    assert title_matches("IQ Option", PAT, EXC)
    assert title_matches("IQOption", PAT, EXC)
    assert title_matches("Trading | IQ Option - Google Chrome", PAT, EXC)
    assert title_matches("iqoption.com/traderoom - Microsoft​ Edge", PAT, EXC)
    assert not title_matches("CandleSense — IQ Option leitor", PAT, EXC)  # nosso próprio painel
    assert not title_matches("YouTube - Google Chrome", PAT, EXC)
    assert not title_matches("", PAT, EXC)


def test_classify_title():
    assert classify_title("IQ Option") == "desktop"
    assert classify_title("IQ Option - Google Chrome") == "browser"
    assert classify_title("IQ Option — Mozilla Firefox") == "browser"


def test_choose_window_prefers_visible_and_large():
    small = WindowInfo(1, "IQ Option", "desktop", Rect(0, 0, 800, 600))
    big = WindowInfo(2, "IQ Option - Chrome", "browser", Rect(0, 0, 1900, 1000))
    mini = WindowInfo(3, "IQ Option", "desktop", Rect(0, 0, 2000, 2000), minimized=True)
    assert choose_window([small, big, mini]).hwnd == 2
    assert choose_window([]) is None


def test_standard_layout_side_by_side():
    iq, panel = standard_layout(Rect(0, 0, 1920, 1040), 360)
    assert iq == Rect(0, 0, 1560, 1040)
    assert panel == Rect(1560, 0, 360, 1040)
    assert iq.intersect(panel) is None
    iq2, panel2 = standard_layout(Rect(1920, 0, 1920, 1040), 360)  # segundo monitor
    assert iq2.x == 1920 and panel2.right == 3840


def test_recorder_and_player_roundtrip(tmp_path):
    rec = SessionRecorder(tmp_path, fps=100)
    t0 = time.time()
    for i in range(5):
        img = np.full((60, 80, 3), i * 40, np.uint8)
        rec.offer(Frame(img, t0 + i * 0.05, WindowInfo(0, "IQ Option", "desktop", Rect(10, 20, 80, 60))),
                  {"status": "OK", "i": i})
    rec.close()
    src = ReplaySource(rec.dir, realtime=False)
    frames = list(src)
    assert len(frames) == 5
    assert frames[3].image[0, 0, 0] == 120
    assert frames[0].window.title == "IQ Option"
    assert abs(frames[1].ts - frames[0].ts - 0.05) < 1e-6


def test_replay_source_grab_stops_at_end(tmp_path):
    rec = SessionRecorder(tmp_path, fps=100)
    rec.offer(Frame(np.zeros((10, 10, 3), np.uint8), 1.0))
    rec.close()
    src = ReplaySource(rec.dir, realtime=False)
    src.grab()
    src.grab()
    assert src.finished


def test_image_source(tmp_path):
    import cv2

    p = tmp_path / "x.png"
    cv2.imwrite(str(p), np.zeros((50, 70, 3), np.uint8))
    f = ImageSource(p).grab()
    assert f.image.shape == (50, 70, 3)


def test_config_defaults_and_override(tmp_path):
    cfg = load_config(tmp_path / "nao_existe.yaml")
    assert cfg.capture.fps == 8.0
    assert cfg.vision.assume_timeframe_min is None
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"capture": {"fps": 5}, "vision": {"min_confidence": 0.7, "chave_errada": 1},
                                 "ocr": {"backend": "tesseract"}}), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.capture.fps == 5.0
    assert cfg.vision.min_confidence == 0.7
    assert cfg.ocr.backend == "tesseract"
    assert cfg.capture.panel_width == 360  # não mexido -> padrão


def test_config_bad_file_falls_back_to_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("capture: [isto nao e um bloco", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.capture.fps == 8.0


def test_rect_helpers():
    r = Rect(10, 10, 100, 50)
    assert r.intersect(Rect(50, 0, 100, 30)) == Rect(50, 10, 60, 20)
    assert r.intersect(Rect(500, 500, 1, 1)) is None
    assert r.clip(80, 40) == Rect(10, 10, 70, 30)
    assert Rect.from_ltrb(1, 2, 11, 12) == Rect(1, 2, 10, 10)
