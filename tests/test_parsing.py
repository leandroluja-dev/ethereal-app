from candlesense.models import Box, OcrToken
from candlesense.vision.axis import axis_labels, fit_scale
from candlesense.vision.ocr import is_clock_text, parse_number, parse_payout, parse_timer


def test_parse_number():
    assert parse_number("1.08345") == (1.08345, 5)
    assert parse_number("1,08345") == (1.08345, 5)
    assert parse_number("64 250.10") == (64250.10, 2)
    assert parse_number("1.O8345") == (1.08345, 5)  # O lido no lugar de 0
    assert parse_number("EUR/USD") is None
    assert parse_number("") is None


def test_parse_timer():
    assert parse_timer("00:23") == 23
    assert parse_timer("1:05") == 65
    assert parse_timer("00:75") is None
    assert parse_timer("abc") is None
    assert is_clock_text("14:35")
    assert not is_clock_text("Lucro +87%")


def test_parse_payout():
    assert parse_payout("+87%") == 87
    assert parse_payout("Lucro +82 %") == 82
    assert parse_payout("87%") == 87
    assert parse_payout("+870%") is None
    assert parse_payout("87") is None


def _labels(pairs):
    return [(y, p, 5) for y, p in pairs]


def test_fit_scale_rejects_ocr_outlier():
    good = [(100 + 50 * i, 1.0850 - 0.0002 * i) for i in range(8)]
    bad = [(325.0, 1.9999)]  # rótulo lido errado
    s = fit_scale(_labels(good + bad), tol_px=2)
    assert s is not None
    assert s.inliers == 8
    assert abs(s.price(100) - 1.0850) < 1e-9
    assert abs(s.y_of(1.0840) - 350) < 1e-6


def test_fit_scale_needs_three_labels_and_negative_slope():
    assert fit_scale(_labels([(100, 1.0), (200, 0.9)])) is None
    assert fit_scale(_labels([(100, 1.0), (200, 1.1), (300, 1.2)])) is None  # preço subindo para baixo
    assert fit_scale([]) is None


def test_axis_labels_ignores_text():
    toks = [OcrToken("1.08345", Box(0, 10, 50, 10)), OcrToken("Lucro", Box(0, 40, 50, 10))]
    assert axis_labels(toks) == [(15.0, 1.08345, 5)]


def test_parse_clock_and_timeframe():
    from candlesense.vision.ocr import parse_clock, parse_timeframe

    assert parse_clock("18:16") == 18 * 3600 + 16 * 60
    assert parse_clock("① 18:16") == 18 * 3600 + 16 * 60
    assert parse_clock("CURRENT TIME: 9 NOVEMBER, 18:14:37 (UTC-3)", need_seconds=True) == 18 * 3600 + 14 * 60 + 37
    assert parse_clock("18:16", need_seconds=True) is None
    assert parse_clock("25:00") is None
    assert parse_timeframe("5m") == 300
    assert parse_timeframe("M15") == 900
    assert parse_timeframe("1h") == 3600
    assert parse_timeframe("30s") == 30
    assert parse_timeframe("5") is None
    assert parse_timeframe("3h") == 3 * 3600


def test_timeframe_from_axis():
    from candlesense.models import Box, OcrToken
    from candlesense.vision.layout import timeframe_from_axis

    toks = [OcrToken(f"{h}:00:00", Box(100 + 744 * i, 1900, 98, 24)) for i, h in enumerate(range(15, 19))]
    assert timeframe_from_axis(toks, 62.0) == 300
    assert timeframe_from_axis(toks, 12.4) == 60
    assert timeframe_from_axis(toks, 40.0) is None  # fora dos tempos padrão
