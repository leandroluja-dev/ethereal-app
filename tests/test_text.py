import pytest

from candlesense.vision.text import (
    asset_score,
    normalize_asset,
    parse_clock,
    parse_number,
    parse_percent,
    repair_decimals,
)


@pytest.mark.parametrize(
    ("text", "value", "decimals"),
    [
        ("1.08452", 1.08452, 5),
        ("1,08452", 1.08452, 5),
        ("62 345.12", 62345.12, 2),
        ("62,345.12", 62345.12, 2),
        ("62.345,12", 62345.12, 2),
        ("150.250", 150.25, 3),
        ("1.O8452", 1.08452, 5),  # O lido no lugar de 0
        ("+87%", 87.0, 0),
    ],
)
def test_parse_number(text, value, decimals):
    n = parse_number(text)
    assert n is not None
    assert n.value == pytest.approx(value)
    assert n.decimals == decimals


@pytest.mark.parametrize("text", ["", "EUR/USD", "ALTA", "15:30", "abc", ".", "1.2.3,4,5x"])
def test_parse_number_rejects(text):
    assert parse_number(text) is None


def test_repair_decimals_reinserts_missing_point():
    nums = [parse_number(t) for t in ["1.08450", "108400", "1.08350", "1.08300"]]
    assert repair_decimals(nums) == pytest.approx([1.0845, 1.084, 1.0835, 1.083])


def test_repair_decimals_discards_incompatible():
    nums = [parse_number(t) for t in ["1.08450", "8400", "1.08350"]]
    vals = repair_decimals(nums)
    assert vals[1] is None


@pytest.mark.parametrize(("text", "v"), [("+87%", 87), ("87 %", 87), ("Lucro +92%", 92), ("1O0%", 100)])
def test_parse_percent(text, v):
    assert parse_percent(text) == v


def test_parse_percent_none():
    assert parse_percent("87") is None


@pytest.mark.parametrize(("text", "v"), [("00:34", (0, 34)), ("15:30", (15, 30)), ("15:30:05", (15, 30, 5)),
                                         ("O0:34", (0, 34))])
def test_parse_clock(text, v):
    assert parse_clock(text) == v


@pytest.mark.parametrize("text", ["1.08452", "99:99", "abc", "12:7"])
def test_parse_clock_rejects(text):
    assert parse_clock(text) is None


@pytest.mark.parametrize(
    ("text", "norm"),
    [
        ("EUR/USD (OTC)", "EUR/USD (OTC)"),
        ("eur/usd(otc)", "EUR/USD (OTC)"),
        ("EUR / USD", "EUR/USD"),
        ("Bitcoin", "BITCOIN"),
        ("Apple (OTC)", "APPLE (OTC)"),
    ],
)
def test_normalize_asset(text, norm):
    assert normalize_asset(text) == norm


@pytest.mark.parametrize("text", ["ALTA", "BAIXA", "87%", "15:30", "Depositar", "x"])
def test_normalize_asset_rejects_ui_text(text):
    assert normalize_asset(text) is None


def test_asset_score_prefers_fx_otc():
    assert asset_score("EUR/USD (OTC)") > asset_score("EUR/USD") > asset_score("Bitcoin") > 0
