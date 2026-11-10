import pytest

from candlesense.engine.rules import COMPRA, INCERTA, NAO_ENTRAR, RuleEngine
from candlesense.ui.panel import W, render
from tests.market import candles_from_path, reading, zigzag_up


@pytest.mark.parametrize("ok", [True, False])
def test_render_panel(ok):
    r = reading(candles_from_path(zigzag_up()), ok=ok)
    d = RuleEngine().evaluate(r)
    img = render(d, r)
    assert img.ndim == 3 and img.shape[1] == W and img.shape[0] > 100
    assert d.action in (INCERTA,) if not ok else d.action in (NAO_ENTRAR, COMPRA)


def test_render_scaled():
    r = reading(candles_from_path(zigzag_up()))
    img = render(RuleEngine().evaluate(r), r, scale=1.5)
    assert img.shape[1] == int(W * 1.5)


def test_wallet_button_hit():
    from candlesense.ui.panel import hit_button

    assert hit_button(320, 30) == "carteira"
    assert hit_button(640, 60, scale=2.0) == "carteira"
    assert hit_button(100, 20) is None
    assert hit_button(320, 300) is None


def test_render_all_blocks():
    """Sinal, alerta de saída, carteira, operação aberta e detalhes cabem no painel."""
    from candlesense.ui.panel import PanelExtras

    r = reading(candles_from_path(zigzag_up()), timeframe=60)
    d = RuleEngine().evaluate(r)
    ex = PanelExtras(wallet_account="treino", wallet_n=5, wallet_wins=3, wallet_losses=2, wallet_profit=12.4,
                     open_line="Operação aberta: $2,00 (compra) · expira em 0:42",
                     exit_alert="a força virou para venda", details=True)
    img = render(d, r, extras=ex)
    assert img.shape[1] == W and 400 < img.shape[0] < 1700


def test_wallet_inside_panel():
    """A carteira não abre outra janela: vira a imagem mostrada no painel."""
    import numpy as np

    from candlesense.ui.wallet_view import WalletWindow, render_wallet

    wv = WalletWindow()
    wv.show(np.zeros((5, 5, 3), np.uint8))
    assert wv.image is None  # escondida: nada para mostrar
    wv.toggle()
    img = render_wallet({}, None, None)
    wv.show(img)
    assert wv.image is img and img.shape[1] == W
    wv.toggle()
    assert not wv.visible and wv.image is None


def test_open_trade_live_result():
    """Operação aberta: ganhando/perdendo pelo preço atual e a direção."""
    from candlesense.ui.panel import PanelExtras, _live_result

    t = {"amount": 2.0, "direction": "COMPRA", "entry_price": 1.1, "payout": 86, "seconds_left": 30}
    assert _live_result(t, 1.1003) == (2.0 * 0.86, "ganhando")
    assert _live_result(t, 1.0990) == (-2.0, "perdendo")
    assert _live_result({**t, "direction": "VENDA"}, 1.0990)[1] == "ganhando"
    assert _live_result({**t, "direction": None}, 1.1003) == (None, None)
    r = reading(candles_from_path(zigzag_up()), timeframe=60)
    img = render(RuleEngine().evaluate(r), r, extras=PanelExtras(wallet_n=1, open_trade=t, exit_alert="virou"))
    assert img.shape[1] == W


def test_panel_input_toggles_right_away(monkeypatch):
    """Clique no botão Carteira pede redesenho na hora, sem esperar a próxima leitura."""
    import candlesense.__main__ as m
    from candlesense.ui.panel import PanelExtras
    from candlesense.ui.wallet_view import WalletWindow

    class FakePanel:
        def __init__(self):
            self.click = "carteira"

        def pop_click(self):
            c, self.click = self.click, None
            return c

        def is_open(self):
            return True

    monkeypatch.setattr(m.cv2, "waitKey", lambda ms: -1)
    p, wv, ex = FakePanel(), WalletWindow(), PanelExtras()
    assert m._panel_input(p, wv, ex) == "redesenhar" and wv.visible
    assert m._panel_input(p, wv, ex) is None
    monkeypatch.setattr(m.cv2, "waitKey", lambda ms: ord("q"))
    assert m._panel_input(p, wv, ex) == "sair"


def test_button_hover_and_press_change_look():
    """Mouse em cima e clique mudam a cor do botão Carteira."""
    import numpy as np

    from candlesense.ui import panel as pn

    r = reading(candles_from_path(zigzag_up()), timeframe=60)
    d = RuleEngine().evaluate(r)
    ex = pn.PanelExtras()
    x1, y1, x2, y2 = pn.BUTTONS["carteira"]
    looks = []
    try:
        for st in (None, "hover", "down"):
            pn.BUTTON_STATE["carteira"] = st
            img = render(d, r, extras=ex)
            looks.append(img[int(y1) + 3:int(y2) - 3, int(x1) + 12:int(x1) + 20].mean())
    finally:
        pn.BUTTON_STATE["carteira"] = None
    assert looks[0] < looks[1] < looks[2]


def test_timeframe_hint_line(monkeypatch):
    """A linha do melhor tempo de vela aparece abaixo do cabeçalho e aumenta o painel."""
    import candlesense.ui.panel as pn
    from candlesense.ui.panel import PanelExtras

    monkeypatch.setattr(pn, "MIN_H", 0)

    r = reading(candles_from_path(zigzag_up()), timeframe=300)
    d = RuleEngine().evaluate(r)
    base = render(d, r, extras=PanelExtras()).shape[0]
    a = render(d, r, extras=PanelExtras(tf_advice="Melhor tempo de vela até agora: 2 min", tf_best=120))
    b = render(d, r, extras=PanelExtras(tf_advice="Ainda poucos sinais para escolher o tempo de vela"))
    assert a.shape[0] > base and b.shape[0] > base


def test_tf_advice_from_store(tmp_path):
    import candlesense.__main__ as m
    from candlesense.history.store import Store
    from candlesense.ui.panel import PanelExtras

    m._tf_t = -1e9
    ex = PanelExtras()
    m._tf_advice(ex, Store(tmp_path / "x.db"))
    assert ex.tf_advice and ex.tf_best is None



def test_fixed_height_and_no_nao_entre():
    """Sem sinal e com leitura incerta o painel tem a mesma altura e mostra só "Aguarde"."""
    from candlesense.ui.panel import MIN_H, PanelExtras

    ok = reading(candles_from_path(zigzag_up()), timeframe=60)
    bad = reading(candles_from_path(zigzag_up()), ok=False, timeframe=60)
    ex = PanelExtras(wallet_balance=923.22, wallet_account="treino", wallet_n=0)
    a = render(RuleEngine().evaluate(ok), ok, extras=ex)
    b = render(RuleEngine().evaluate(bad), bad, extras=ex)
    assert a.shape == b.shape and a.shape[0] == MIN_H


def test_panel_window_keeps_last_good_chart(monkeypatch):
    """Leitura incerta por um instante não troca a tela: o gráfico fica o da última leitura boa."""
    import candlesense.ui.panel as pn

    shown = []
    monkeypatch.setattr(pn, "render", lambda d, r, s, e: shown.append((r, e.stale)) or None)

    class FakeCv2:
        def imshow(self, *a):
            pass

    w = pn.PanelWindow.__new__(pn.PanelWindow)
    w.cv2, w.scale, w._frames, w._placed, w.place = FakeCv2(), 1.0, 99, True, None
    monkeypatch.setattr("candlesense.platform.appmode.window_buttons", lambda name: None)
    ok = reading(candles_from_path(zigzag_up()), timeframe=60)
    bad = reading(candles_from_path(zigzag_up()), ok=False, timeframe=60)
    ex = pn.PanelExtras()
    w.show(None, ok, ex)
    w.show(None, bad, ex)
    assert shown[0] == (ok, False) and shown[1] == (ok, True)
