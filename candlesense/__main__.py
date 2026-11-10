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
    cfg = EngineConfig(margin=args.margem / 100.0, reaction_s=int(args.atraso))
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


_diag_t = 0.0


def _diagnostic(img, r, dec) -> None:
    """Modo app: guarda a captura (e o debug) quando a leitura falha, no máximo 1 a cada 30 s.

    Fica em diagnostico/ na pasta do projeto, só as 6 mais recentes, para o
    ajuste remoto sem pedir print à pessoa.
    """
    global _diag_t
    if r.ok or time.monotonic() - _diag_t < 30:
        return
    _diag_t = time.monotonic()
    try:
        d = Path("diagnostico")
        d.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(str(d / f"{stamp}_captura.png"), img)
        cv2.imwrite(str(d / f"{stamp}_debug.png"), draw(img, r))
        (d / f"{stamp}.json").write_text(json.dumps(
            {"leitura": _reading_json(r), "sinal": _decision_json(dec)}, ensure_ascii=False, indent=1, default=float))
        files = sorted(d.glob("*_captura.png"))
        for old in files[:-6]:
            for f in d.glob(old.name.replace("_captura.png", "*")):
                f.unlink(missing_ok=True)
    except Exception as e:
        print(f"diagnóstico não salvo: {e!r}", flush=True)


def _panel_place(win) -> tuple[int, int]:
    """Painel no lado esquerdo da janela da IQ Option, longe dos botões de compra e venda (à direita)."""
    return int(win.x + 70), int(win.y + 110)


_tf_t = -1e9


def _tf_advice(extras, store) -> None:
    """Melhor tempo de vela pelo histórico (consulta 30 dias: no máximo 1 vez por minuto)."""
    global _tf_t
    if time.monotonic() - _tf_t < 60:
        return
    _tf_t = time.monotonic()
    try:
        from .history.timeframes import compare

        month = time.time() - 30 * 86400
        adv = compare(store.signals(since=month), store.predictions(since=month), store.trades(since=month))
        extras.tf_advice = adv.text if adv else None
        extras.tf_best = adv.best.tf if adv and adv.best else None
    except Exception as e:  # o conselho nunca derruba o painel
        print(f"tempo de vela: {e!r}", flush=True)


def _wallet_extras(extras, wallet, store, account, wallet_view) -> None:
    """Atualiza a faixa da carteira no painel e a tela da carteira (se aberta)."""
    from .history.wallet import day_start, summarize

    _tf_advice(extras, store)

    since = day_start()
    days = summarize(store.trades(since=since))
    acc = account or wallet.account
    a = days.get(acc) if acc else None
    nome = {"real": "REAL", "treino": "treino"}.get(acc or "", "?")
    extras.wallet_account = acc
    extras.wallet_balance = wallet.balance.get(acc) if acc else None
    extras.wallet_n = a.n if a is not None else (0 if acc else None)
    extras.wallet_wins = a.wins if a is not None else 0
    extras.wallet_losses = a.losses if a is not None else 0
    if acc is None:
        extras.wallet_line, extras.wallet_profit = "Carteira: saldo não lido ainda", None
    elif a is None:
        extras.wallet_line, extras.wallet_profit = f"Hoje ({nome}): nenhuma entrada", None
    else:
        extras.wallet_line = (f"Hoje ({nome}): {a.n} entradas · {a.wins}G {a.losses}P · "
                              f"{'+' if a.profit >= 0 else '-'}${abs(a.profit):,.2f}")
        extras.wallet_profit = a.profit
    now = time.time()
    op = [t for t in wallet.open if t.account == acc]
    if op:
        t = op[-1]
        falta = max(0, int(t.expiry_ts - now))
        lado = {"COMPRA": " (compra)", "VENDA": " (venda)"}.get(t.app_action or "", "")
        extras.open_line = f"Operação aberta: ${t.amount:,.2f}{lado} · expira em {falta // 60}:{falta % 60:02d}"
        extras.open_trade = {"amount": t.amount, "direction": t.app_action, "entry_price": t.entry_price,
                             "payout": t.payout, "seconds_left": falta}
    else:
        extras.open_line = None
        extras.open_trade = None
    if wallet_view is not None and wallet_view.visible:
        from .history.scoreboard import score
        from .ui.wallet_view import render_wallet

        board = score(store.predictions(since=now - 7 * 86400))
        wallet_view.show(render_wallet(days, board, store.signals(since=since), wallet_view.scale))


def _show_panel(panel, dec, r, extras, wallet_view) -> None:
    """Uma janela só: a carteira aparece dentro do próprio painel."""
    overlay = wallet_view.image if wallet_view is not None and wallet_view.visible else None
    panel.show(dec, r, extras, overlay=overlay)


def _panel_input(panel, wallet_view, extras, poll: bool = True, wait_ms: int = 1):
    """Atende teclas e cliques do painel. Devolve "sair", "redesenhar" ou None."""
    key = (cv2.waitKey(wait_ms) & 0xFF) if poll else 255
    if key in (27, ord("q")):
        print("fechado com q/Esc", flush=True)
        return "sair"
    act = None
    if key == ord("d"):
        extras.details = not extras.details
        act = "redesenhar"
    if key == ord("g"):
        extras.chart = not extras.chart
        act = "redesenhar"
    if getattr(panel, "pop_hover_change", lambda: False)():
        act = "redesenhar"  # mouse sobre o botão: muda a cor na hora
    if (key == ord("c") or panel.pop_click() == "carteira") and wallet_view is not None:
        wallet_view.toggle()
        act = "redesenhar"
    if not panel.is_open():
        print("painel fechado pela janela", flush=True)
        return "sair"
    return act


def _open_source(win, open_window_source):
    """Captura da janela, com o próximo quadro sendo capturado enquanto o atual é lido."""
    from .platform.capture import PrefetchSource

    src = open_window_source(win)
    try:
        return PrefetchSource(src)
    except Exception as e:
        print(f"captura em paralelo indisponível: {e!r}", flush=True)
        return src


def _restart_for_update(panel, src, store) -> None:
    """Versão nova baixada: mostra 'Atualizando...' no painel e reabre o app sozinho."""
    from .platform.selfupdate import loading_frame, restart

    print("reabrindo com a versão nova", flush=True)
    if panel is not None:
        t_end = time.monotonic() + 1.2
        while time.monotonic() < t_end:
            try:
                cv2.imshow(panel.NAME, loading_frame())
                cv2.waitKey(40)
            except Exception:
                break
    for fn in (src.close, store.close):
        try:
            fn()
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:
        pass
    restart()


def cmd_live(args) -> int:
    if APP_MODE:
        from .platform.appmode import close_other_instances, hide_python_dock_icon

        n = close_other_instances()
        if n:
            print(f"fechei {n} cópia(s) antiga(s) do CandleSense", flush=True)
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

    src = _open_source(win, open_window_source)
    print(f"Captura: {src.name}. Ctrl+C para sair.")
    reader = ChartReader(best_engine(args.ocr))
    engine = _engine(args)
    from .history.scoreboard import learn
    from .history.store import Store

    store = Store(Path(args.dados) / "candlesense.db")
    updater = None
    if APP_MODE:
        from .platform.selfupdate import Updater

        updater = Updater(Path(__file__).resolve().parents[1]).start()
    engine.calibration = learn(store.predictions())
    last_learn = time.monotonic()
    from .history.wallet import WalletTracker
    from .ui.panel import PanelExtras

    wallet = WalletTracker(store)
    extras = PanelExtras()
    wallet_view = None
    wallet_dirty = True
    last_wallet = 0.0
    alerted: set[int] = set()
    print(f"OCR: {reader.ocr.name}")
    panel = None
    if not args.sem_painel:
        from .ui.panel import PanelWindow

        panel = PanelWindow(scale=args.painel_escala)
        panel.place = _panel_place(win)
        from .ui.wallet_view import WalletWindow

        wallet_view = WalletWindow(scale=args.painel_escala)
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
    prev_signal = None
    fails = 0
    try:
        while True:
            t0 = time.perf_counter()
            if updater is not None and updater.ready.is_set():
                _restart_for_update(panel, src, store)
            img, cap_ms = timed_grab(src)
            if img is None:
                fails += 1
                if fails % 10 == 1:
                    print("\nSem imagem da janela (minimizada ou fechada?). Tentando de novo...")
                if fails >= 30:
                    win = _find_window_or_wait(wait=True)
                    src.close()
                    src = open_window_source(win)  # sem captura em paralelo, por segurança
                    reader.reset()
                    engine.reset()
                    fails = 0
                time.sleep(0.5)
                continue
            fails = 0
            r = reader.read(img)
            r.timings_ms = {"captura": cap_ms, **r.timings_ms}
            dec = engine.evaluate(r)
            try:
                asset = (reader.layout.asset if reader.layout else None) or "?"
                store.record(r, dec, asset=asset)
                if time.monotonic() - last_learn > 600:  # reaprende a correção a cada 10 min
                    engine.calibration = learn(store.predictions())
                    last_learn = time.monotonic()
            except Exception as e:  # o histórico nunca derruba a leitura
                print(f"histórico: {e!r}", flush=True)
            from .alerts import play

            if dec.signal and dec.action != prev_signal:
                try:
                    store.record_signal(r, dec, asset=asset)
                except Exception as e:
                    print(f"histórico: {e!r}", flush=True)
                if not args.sem_som:
                    play("entrar", voice=("compra" if dec.action == "COMPRA" else "venda") + ", " + dec.expiry_text())
            prev_signal = dec.action if dec.signal else None
            if dec.signal:
                wallet.note_signal(dec.action, dec.best.prob if dec.best else None)
            # Carteira: entradas, ganhos e perdas pelo saldo da tela
            try:
                events = wallet.update(
                    r.account, r.balance, price=r.current_price if r.ok else None, asset=asset,
                    tf=r.timeframe_s, payout=r.payout_pct, expiry_seconds=r.expiry_seconds,
                    invest=r.invest_amount)
            except Exception as e:
                events = []
                print(f"carteira: {e!r}", flush=True)
            for ev in events:
                wallet_dirty = True
                conta = "real" if ev.account == "real" else "treino"
                if ev.kind == "entrada":
                    print(f"carteira: entrada de ${ev.amount:.2f} na conta {conta}", flush=True)
                else:
                    print(f"carteira: {ev.kind} {ev.profit:+.2f} na conta {conta}", flush=True)
                    if not args.sem_som:
                        play("resultado")
            extras.exit_alert = wallet.exit_alert(dec.strength if dec.action != "INCERTA" else None)
            if extras.exit_alert:
                new = [t.id for t in wallet.open if t.id not in alerted]
                if new:
                    alerted.update(new)
                    if not args.sem_som:
                        play("sair", voice="atenção, considere sair")
            if wallet_dirty or time.monotonic() - last_wallet > 1:
                _wallet_extras(extras, wallet, store, r.account, wallet_view)
                wallet_dirty = False
                last_wallet = time.monotonic()
            total = (time.perf_counter() - t0) * 1000
            line = f"[{time.strftime('%H:%M:%S')}] {dec.summary()} | {r.summary()} | total {total:.0f}ms"
            if APP_MODE:
                if dec.action != last_action or time.monotonic() - last_log >= 60:  # log enxuto
                    last_log = time.monotonic()
                    print(line[:300], flush=True)
                    last_action = dec.action
            else:
                print("\r" + line[:160].ljust(160), end="", flush=True)
            if APP_MODE:
                _diagnostic(img, r, dec)
            if panel is not None:
                _show_panel(panel, dec, r, extras, wallet_view)
                act = _panel_input(panel, wallet_view, extras, poll=not (args.debug or save_dir))
                if act == "sair":
                    break
                if act == "redesenhar":
                    wallet_dirty = False
                    last_wallet = time.monotonic()
                    _wallet_extras(extras, wallet, store, r.account, wallet_view)
                    _show_panel(panel, dec, r, extras, wallet_view)
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
            # Espera a próxima leitura atendendo cliques e teclas na hora (antes ficavam
            # parados até a leitura seguinte terminar).
            stop = False
            while time.perf_counter() - t0 < period:
                if panel is None or args.debug or save_dir:
                    time.sleep(max(0.0, period - (time.perf_counter() - t0)))
                    break
                act = _panel_input(panel, wallet_view, extras, poll=True, wait_ms=15)
                if act == "sair":
                    stop = True
                    break
                if act == "redesenhar":
                    _wallet_extras(extras, wallet, store, r.account, wallet_view)
                    _show_panel(panel, dec, r, extras, wallet_view)
            if stop:
                break
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
    ap.add_argument("--atraso", type=float, default=3.0,
                    help="seu tempo de reação em segundos (padrão 3): o aviso vem com essa folga")
    ap.add_argument("--sem-som", action="store_true", help="sem avisos sonoros")
    ap.add_argument("--dados", default="dados", help="pasta do histórico (padrão ./dados)")
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
