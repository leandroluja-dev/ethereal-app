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
