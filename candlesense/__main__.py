"""Linha de comando do CandleSense.

    python -m candlesense                      # app ao vivo (Windows, com a IQ Option aberta)
    python -m candlesense demo                 # app com mercado sintético (não precisa da IQ Option)
    python -m candlesense analyze print.png    # analisa um print e salva a imagem de debug
    python -m candlesense replay recordings/X  # reprocessa uma sessão gravada (offline)
    python -m candlesense replay recordings/X --gui
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_config
from .logging_setup import setup_logging

log = logging.getLogger("candlesense")


def _analyze(args: argparse.Namespace) -> int:
    import cv2

    from .vision.analyzer import FrameAnalyzer, frame_from_image
    from .vision.debug_draw import draw_overlay
    from .vision.ocr import create_ocr

    cfg = load_config(args.config)
    setup_logging(cfg)
    ocr = create_ocr(cfg.ocr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = 0
    for path in args.images:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[erro] não consegui abrir {path}")
            rc = 1
            continue
        analyzer = FrameAnalyzer(cfg, ocr)  # estado novo para cada imagem
        reading = analyzer.analyze(frame_from_image(img))
        # segunda passada: mostra o tempo de um quadro normal (sem detecção de layout)
        reading = analyzer.analyze(frame_from_image(img))
        stem = Path(path).stem
        dbg = out_dir / f"{stem}_debug.png"
        cv2.imwrite(str(dbg), draw_overlay(img, reading))
        summary = reading.summary()
        (out_dir / f"{stem}_leitura.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
        brief = {k: v for k, v in summary.items() if k != "candles"}
        print(json.dumps(brief, indent=2, ensure_ascii=False))
        print(f"imagem de debug: {dbg}")
    return rc


def _replay_headless(args: argparse.Namespace) -> int:
    from .replay.player import ReplaySource
    from .vision.analyzer import FrameAnalyzer
    from .vision.ocr import create_ocr

    cfg = load_config(args.config)
    setup_logging(cfg)
    src = ReplaySource(args.session, realtime=False)
    analyzer = FrameAnalyzer(cfg, create_ocr(cfg.ocr))
    counts: dict[str, int] = {}
    lat: list[float] = []
    out = Path(args.session) / "reprocessado.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for frame in src:
            r = analyzer.analyze(frame)
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
            lat.append(r.timings_ms.get("total", 0.0))
            fh.write(json.dumps(r.summary(), ensure_ascii=False) + "\n")
    lat.sort()
    print(f"{len(src)} quadros reprocessados -> {out}")
    print("status:", counts)
    if lat:
        print(f"latência mediana {lat[len(lat) // 2]:.0f} ms, p95 {lat[int(len(lat) * 0.95) - 1]:.0f} ms")
    return 0


def _gui(args: argparse.Namespace, mode: str) -> int:
    from .app import run_app

    return run_app(config_path=args.config, mode=mode, source_arg=getattr(args, "session", None)
                   or getattr(args, "image", None), theme=getattr(args, "theme", "dark"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="candlesense", description="Lê o gráfico da IQ Option na tela (somente leitura).")
    p.add_argument("--config", help="caminho do config.yaml", default=None)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("run", help="app ao vivo (padrão)")
    d = sub.add_parser("demo", help="app com mercado sintético")
    d.add_argument("--theme", choices=["dark", "light"], default="dark")
    a = sub.add_parser("analyze", help="analisa prints salvos")
    a.add_argument("images", nargs="+")
    a.add_argument("--out", default="captures")
    r = sub.add_parser("replay", help="reprocessa uma sessão gravada")
    r.add_argument("session")
    r.add_argument("--gui", action="store_true", help="mostra no painel em tempo real")
    i = sub.add_parser("image", help="abre o painel lendo um print fixo")
    i.add_argument("image")
    args = p.parse_args(argv)

    if args.cmd == "analyze":
        return _analyze(args)
    if args.cmd == "replay" and not args.gui:
        return _replay_headless(args)
    if args.cmd == "replay":
        return _gui(args, "replay")
    if args.cmd == "demo":
        return _gui(args, "demo")
    if args.cmd == "image":
        return _gui(args, "image")
    return _gui(args, "live")


if __name__ == "__main__":
    sys.exit(main())
