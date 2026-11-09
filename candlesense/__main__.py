"""CandleSense - linha de comando.

    candlesense                    lê o gráfico ao vivo e mostra o painel (macOS)
    candlesense --debug            idem, com janela de debug desenhando a detecção
    candlesense debug IMG...       lê capturas salvas e grava debug + painel (qualquer sistema)
    candlesense capturar           salva capturas da janela da IQ Option (para testes)
    candlesense permissoes         guia das permissões do macOS
    candlesense janelas            mostra as janelas encontradas

O CandleSense só lê pixels da tela. Ele nunca clica, nunca move o mouse,
nunca envia teclas e nunca abre operações.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from .debug.overlay import draw
from .engine.rules import EngineConfig, RuleEngine
from .vision.ocr import best_engine
from .vision.reader import ChartReader


def _reading_json(r) -> dict:
    return {
        "status": r.status,
        "motivos": r.reasons,
        "avisos": r.warnings,
        "preco_atual": r.current_price,
        "cronometro_s": r.timer_seconds,
        "tempo_compra_s": r.purchase_seconds,
        "tempo_vela_s": r.timeframe_s,
        "expiracao_iq": r.expiry_label,
        "expiracao_faltam_s": r.expiry_seconds,
        "payout_pct": r.payout_pct,
        "tempos_ms": {k: round(v, 1) for k, v in r.timings_ms.items()},
        "velas": [asdict(c) for c in r.candles],
    }


def _decision_json(d) -> dict:
    return {
        "decisao": d.action,
        "motivos": d.reasons,
        "melhor": None if d.best is None else {
            "direcao": d.best.direction, "probabilidade": round(d.best.prob, 4), "velas": d.best.candles},
        "prob_alta_por_expiracao": {str(h.candles): round(h.p_up, 4) for h in d.horizons},
        "equilibrio": d.breakeven,
        "minimo_para_sinal": d.threshold,
        "fase_da_vela": d.phase,
        "entrar_em_ate_s": d.seconds_to_enter,
        "alvo": d.target,
        "saida": d.exit_mode,
        "expiracao": d.expiry_hint,
        "modulos": [{"modulo": m.name, "nota": round(m.score, 3), "confianca": round(m.confidence, 3),
                     "explicacao": m.explanation} for m in d.modules],
    }


def _engine(args) -> RuleEngine:
    cfg = EngineConfig(margin=args.margem / 100.0)
    return RuleEngine(cfg)


def cmd_debug(args) -> int:
    ocr = best_engine(args.ocr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    code = 0
    for p in args.images:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"{p}: não consegui abrir a imagem")
            code = 1
            continue
        from .ui.panel import render

        reader = ChartReader(ocr)
        r = reader.read(img)
        eng = _engine(args)
        eng.cfg.stable_readings = 1  # imagem solta: não há leitura anterior para confirmar
        dec = eng.evaluate(r)
        stem = Path(p).stem
        cv2.imwrite(str(out / f"{stem}_debug.png"), draw(img, r))
        cv2.imwrite(str(out / f"{stem}_painel.png"), render(dec, r))
        data = {"leitura": _reading_json(r), "sinal": _decision_json(dec)}
        (out / f"{stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=float))
        print(f"{p}: {r.summary()}")
        print(f"  -> {dec.summary()}")
    print(f"Imagens de debug em {out.resolve()}")
    return code


APP_MODE = False


def _find_window_or_wait(wait: bool):
    from .platform.windows import find_iq_window

    shown = False
    while True:
        win = find_iq_window()
        if win is not None:
            print(f"Janela encontrada: {win.owner} - '{win.title}' ({win.kind}, {int(win.w)}x{int(win.h)})")
            return win
        if not wait:
            return None
        if not shown:
            print("Procurando a janela da IQ Option... abra o app ou a aba da IQ Option no navegador.")
            if APP_MODE:
                from .platform.appmode import notify

                notify("Abra a IQ Option: o painel aparece assim que o gráfico estiver na tela.")
            shown = True
        time.sleep(1.0)


def _check_platform() -> bool:
    if sys.platform != "darwin":
        print("A captura ao vivo só funciona no macOS. Aqui dá para usar: candlesense debug captura.png")
        return False
    from .platform import permissions

    if not permissions.screen_recording_granted():
        if APP_MODE:
            from .platform.appmode import dialog

            permissions.request_screen_recording()
            permissions.open_settings(permissions.SETTINGS_SCREEN)
            dialog("O CandleSense precisa da permissão de Gravação de Tela para ler o gráfico.\n\n"
                   "1. Nos Ajustes que acabaram de abrir, ative a chave do CandleSense.\n"
                   "2. Depois clique de novo no ícone do CandleSense no Dock.\n\n"
                   "Ele só lê a tela: nunca clica e nunca opera.")
            return False
        st = permissions.guide()
        if not st.screen_recording:
            return False
    return True


def cmd_live(args) -> int:
    if APP_MODE:
        from .platform.appmode import hide_python_dock_icon

        hide_python_dock_icon()
    if not _check_platform():
        return 2
    from .platform.capture import open_window_source, timed_grab

    win = _find_window_or_wait(wait=True)
    if args.ajustar_janela:
        from .platform import accessibility, permissions

        if permissions.accessibility_granted(False) or permissions.guide(need_accessibility=True).accessibility:
            w, h = (int(v) for v in args.ajustar_janela.lower().split("x"))
            if accessibility.set_window_frame(win, 0, 25, w, h):
                time.sleep(0.5)
                win = _find_window_or_wait(wait=True)

    src = open_window_source(win)
    print(f"Captura: {src.name}. Ctrl+C para sair.")
    reader = ChartReader(best_engine(args.ocr))
    engine = _engine(args)
    print(f"OCR: {reader.ocr.name}")
    panel = None
    if not args.sem_painel:
        from .ui.panel import PanelWindow

        panel = PanelWindow(scale=args.painel_escala)
        if APP_MODE:
            from .platform.appmode import hide_python_dock_icon

            hide_python_dock_icon()
    period = 1.0 / max(0.5, args.fps)
    save_dir = Path(args.salvar_debug) if args.salvar_debug else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    last_save = 0.0
    last_action = None
    last_log = 0.0
    fails = 0
    try:
        while True:
            t0 = time.perf_counter()
            img, cap_ms = timed_grab(src)
            if img is None:
                fails += 1
                if fails % 10 == 1:
                    print("\nSem imagem da janela (minimizada ou fechada?). Tentando de novo...")
                if fails >= 30:
                    win = _find_window_or_wait(wait=True)
                    src = open_window_source(win)
                    reader.reset()
                    engine.reset()
                    fails = 0
                time.sleep(0.5)
                continue
            fails = 0
            r = reader.read(img)
            r.timings_ms = {"captura": cap_ms, **r.timings_ms}
            dec = engine.evaluate(r)
            total = (time.perf_counter() - t0) * 1000
            line = f"[{time.strftime('%H:%M:%S')}] {dec.summary()} | {r.summary()} | total {total:.0f}ms"
            if APP_MODE:
                if dec.action != last_action or time.monotonic() - last_log >= 60:  # log enxuto
                    last_log = time.monotonic()
                    print(line[:300], flush=True)
                    last_action = dec.action
            else:
                print("\r" + line[:160].ljust(160), end="", flush=True)
            if panel is not None:
                panel.show(dec, r)
                if not (args.debug or save_dir) and cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    print("fechado com q/Esc", flush=True)
                    break
                if not panel.is_open():
                    print("painel fechado pela janela", flush=True)
                    break  # fechou pela bolinha vermelha
            if args.debug or save_dir:
                ov = draw(img, r)
                if args.debug:
                    view = ov
                    if view.shape[1] > 1600:
                        f = 1600 / view.shape[1]
                        view = cv2.resize(view, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                    cv2.imshow("CandleSense debug", view)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                elif panel is not None and cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                if save_dir and time.monotonic() - last_save >= args.intervalo_debug:
                    stamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(str(save_dir / f"{stamp}_debug.png"), ov)
                    cv2.imwrite(str(save_dir / f"{stamp}_captura.png"), img)
                    (save_dir / f"{stamp}_sinal.json").write_text(
                        json.dumps({"leitura": _reading_json(r), "sinal": _decision_json(dec)},
                                   ensure_ascii=False, indent=1, default=float))
                    last_save = time.monotonic()
            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if panel is not None:
            panel.close()
        print()
    return 0


def cmd_capture(args) -> int:
    if not _check_platform():
        return 2
    from .platform.capture import open_window_source

    win = _find_window_or_wait(wait=True)
    src = open_window_source(win)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    while n < args.n:
        img = src.grab()
        if img is not None:
            p = out / f"captura_{time.strftime('%Y%m%d-%H%M%S')}_{n:03d}.png"
            cv2.imwrite(str(p), img)
            print(f"salvo {p} ({img.shape[1]}x{img.shape[0]})")
            n += 1
        time.sleep(args.intervalo)
    return 0


def cmd_permissions(args) -> int:
    from .platform import permissions

    if not permissions.is_macos():
        print("Permissões só se aplicam ao macOS.")
        return 0
    st = permissions.guide(need_accessibility=args.acessibilidade)
    print(f"Gravação de Tela: {'liberada' if st.screen_recording else 'NÃO liberada'}")
    print(f"Acessibilidade: {'liberada' if st.accessibility else 'não liberada (opcional)'}")
    return 0 if st.screen_recording else 1


def cmd_windows(args) -> int:
    if sys.platform != "darwin":
        print("Só no macOS.")
        return 0
    from .platform.windows import is_iq_window, list_windows

    for w in list_windows():
        mark = "  <== IQ Option" if is_iq_window(w) else ""
        print(f"{w.window_id:>6}  {w.owner[:24]:<24} {w.title[:50]:<50} {int(w.w)}x{int(w.h)}{mark}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="candlesense", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ocr", choices=["auto", "vision", "rapidocr"], default="auto")
    ap.add_argument("--debug", action="store_true", help="abre a janela de debug com a detecção desenhada")
    ap.add_argument("--salvar-debug", metavar="PASTA", help="salva captura + debug periodicamente nesta pasta")
    ap.add_argument("--intervalo-debug", type=float, default=5.0, help="segundos entre imagens salvas")
    ap.add_argument("--fps", type=float, default=4.0, help="leituras por segundo (padrão 4)")
    ap.add_argument("--ajustar-janela", metavar="LxA", help="ajusta a janela (ex.: 1920x1080) via Acessibilidade")
    ap.add_argument("--sem-painel", action="store_true", help="não abre o painel (só o terminal)")
    ap.add_argument("--painel-escala", type=float, default=1.0, help="tamanho do painel (ex.: 1.3)")
    ap.add_argument("--margem", type=float, default=2.0,
                    help="pontos percentuais acima do equilíbrio do payout para sinalizar (padrão 2)")
    ap.add_argument("--app", action="store_true", help=argparse.SUPPRESS)  # aberto pelo CandleSense.app
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd")

    d = sub.add_parser("debug", help="lê capturas salvas e grava a imagem de debug")
    d.add_argument("images", nargs="+")
    d.add_argument("--out", default="debug_out")
    d.set_defaults(func=cmd_debug)

    c = sub.add_parser("capturar", help="salva capturas da janela da IQ Option")
    c.add_argument("--out", default="captures")
    c.add_argument("-n", type=int, default=5)
    c.add_argument("--intervalo", type=float, default=2.0)
    c.set_defaults(func=cmd_capture)

    p = sub.add_parser("permissoes", help="guia das permissões do macOS")
    p.add_argument("--acessibilidade", action="store_true", help="também pede Acessibilidade")
    p.set_defaults(func=cmd_permissions)

    w = sub.add_parser("janelas", help="lista as janelas e marca a da IQ Option")
    w.set_defaults(func=cmd_windows)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    global APP_MODE
    APP_MODE = bool(args.app)
    func = getattr(args, "func", cmd_live)
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
