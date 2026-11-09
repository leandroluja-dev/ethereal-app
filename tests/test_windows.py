from candlesense.platform.windows import from_cg_dict, is_iq_window, pick_iq_window


def win(num, owner, title, w=1920, h=1080, layer=0):
    return from_cg_dict({
        "kCGWindowNumber": num, "kCGWindowOwnerPID": 100 + num, "kCGWindowOwnerName": owner,
        "kCGWindowName": title, "kCGWindowLayer": layer,
        "kCGWindowBounds": {"X": 0, "Y": 25, "Width": w, "Height": h},
    })


def test_finds_desktop_app():
    ws = [win(1, "Finder", ""), win(2, "IQ Option", "IQ Option")]
    assert pick_iq_window(ws).window_id == 2
    assert pick_iq_window(ws).kind == "app"


def test_finds_browser_tab_by_title():
    ws = [win(1, "Google Chrome", "YouTube"), win(2, "Google Chrome", "IQ Option – Trading Platform"),
          win(3, "Safari", "iqoption.com")]
    got = pick_iq_window(ws)
    assert got.window_id in (2, 3)
    assert got.kind == "navegador"


def test_ignores_small_menus_and_other_tabs():
    assert not is_iq_window(win(1, "IQ Option", "", w=300, h=200))
    assert not is_iq_window(win(2, "IQ Option", "", layer=25))
    assert not is_iq_window(win(3, "Google Chrome", "Gmail"))
    assert not is_iq_window(win(4, "Notes", "anotações IQ Option"))  # não é navegador
    assert pick_iq_window([]) is None


def test_prefers_biggest():
    ws = [win(1, "Google Chrome", "IQ Option", w=900, h=600), win(2, "IQ Option", "IQ Option", w=1920, h=1080)]
    assert pick_iq_window(ws).window_id == 2
