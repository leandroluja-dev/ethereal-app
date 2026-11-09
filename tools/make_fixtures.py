"""Gera as capturas sintéticas salvas em tests/fixtures/synthetic (PNG + gabarito JSON).

    python tools/make_fixtures.py

Capturas reais da sua tela entram em tests/fixtures/real (veja o README de lá).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.synthetic import BG, Scene, render  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "synthetic"

SCENES = {
    "app_1080": (Scene(), True),
    "navegador_1080": (Scene(seed=3, browser_bar=80, pitch=10, body_w=7), True),
    "retina_2x": (Scene(scale=2, width=1440, height=900, seed=5), True),
    "cripto_2casas": (Scene(seed=7, start_price=64250.0, tick=10.0, vol=12.0, decimals=2,
                            payout=82, timer_text="01:05"), True),
    "doji_atual": (Scene(seed=9, doji_every=5), True),
    "linha_compra_colada": (Scene(seed=15, deadline_at=1.0), True),
    "velas_finas": (Scene(seed=11, n_candles=300, pitch=4, body_w=2), False),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (sc, ok) in SCENES.items():
        img, meta = render(sc)
        meta["expect_ok"] = ok
        cv2.imwrite(str(OUT / f"{name}.png"), img)
        (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
        print("ok", name, img.shape)

    # tela sem gráfico (carregando): não pode sinalizar nada
    img, meta = render(Scene(seed=13))
    x1, y1, x2, y2 = meta["plot"]
    img[y1 - 5 : y2 + 40, x1:x2 + 90] = BG[::-1]
    cv2.imwrite(str(OUT / "sem_grafico.png"), img)
    (OUT / "sem_grafico.json").write_text(json.dumps({"expect_ok": False, "candles": []}))
    print("ok sem_grafico")


if __name__ == "__main__":
    main()
