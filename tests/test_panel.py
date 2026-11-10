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

    assert hit_button(380, 20) == "carteira"
    assert hit_button(760, 40, scale=2.0) == "carteira"
    assert hit_button(100, 20) is None
    assert hit_button(380, 300) is None
