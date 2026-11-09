import json
import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures" / "synthetic"
REAL = ROOT / "tests" / "fixtures" / "real"


def load_fixture(name: str):
    img = cv2.imread(str(FIX / f"{name}.png"), cv2.IMREAD_COLOR)
    meta = json.loads((FIX / f"{name}.json").read_text())
    return img, meta


@pytest.fixture(scope="session")
def ocr():
    """Vision no macOS; rapidocr nos outros sistemas. Pula se nenhum estiver instalado."""
    from candlesense.vision.ocr import best_engine

    try:
        return best_engine()
    except ImportError:
        pytest.skip("nenhum OCR instalado (Vision ou rapidocr)")
