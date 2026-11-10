import json
import urllib.request

from candlesense.ui.sim_view import SimState
from candlesense.web.server import WebServer, _group, build_state


def _e(asset, start, status, profit, expiry=0, side="COMPRA", chance=0.6):
    return {"asset": asset, "start": start, "ts": start + 2, "side": side, "chance": chance,
            "status": status, "profit": profit, "expiry_ts": expiry, "stake": 10.0}


def test_group_joins_wallets_of_same_minute():
    g = _group({"A": [_e("GBP/USD", 60, "ganhou", 8.9)],
                "C": [_e("GBP/USD", 60, "perdeu", -10.0)],
                "E": [_e("GBP/USD", 60, "aberta", None, expiry=200)],
                "B": [_e("GBP/USD", 120, "perdeu", -10.0)]})
    assert [x["wallets"] for x in g] == [["A", "C", "E"], ["B"]]
    assert g[0]["status"] == "aberta" and g[0]["expiry_ts"] == 200
    assert g[1]["status"] == "perdeu" and g[1]["profit"] == -10.0


def test_group_sums_profit_when_all_done():
    g = _group({"A": [_e("X", 60, "ganhou", 8.9)], "B": [_e("X", 60, "ganhou", 8.9)]})
    assert g[0]["status"] == "ganhou" and g[0]["profit"] == 17.8


def test_server_serves_page_and_state():
    web = WebServer(port=0, host="127.0.0.1")
    try:
        st = SimState(entries={"A": [_e("GBP/USD", 60, "ganhou", 8.9)]}, now=100.0, asset="GBP/USD")
        web.publish(build_state(st, None, "Aguarde", ""))
        base = f"http://127.0.0.1:{web.port}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "CandleSense" in page
        d = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert d["asset"] == "GBP/USD" and d["decision"]["text"] == "Aguarde"
        a = next(w for w in d["wallets"] if w["key"] == "A")
        assert a["today"]["wins"] == 1 and abs(a["balance"] - 10008.9) < 1e-6
        assert d["entries"][0]["wallets"] == ["A"]
    finally:
        web.close()


def test_app_wallet_sums_all_wallets():
    st = SimState(entries={"A": [_e("X", 60, "ganhou", 8.9)], "C": [_e("X", 60, "perdeu", -10.0)]}, now=100.0)
    w = st.app_wallet()
    assert abs(w["balance"] - (10_000 - 1.1)) < 1e-6 and abs(w["day"] + 1.1) < 1e-6
    assert w["wins"] == 1 and w["losses"] == 1


def test_day_endpoint_groups_entries():
    web = WebServer(port=0, host="127.0.0.1")
    try:
        web.day_source = lambda d: {"A": [_e("X", 60, "ganhou", 8.9)], "B": [_e("X", 60, "ganhou", 8.9)]}
        base = f"http://127.0.0.1:{web.port}"
        d = json.loads(urllib.request.urlopen(base + "/api/day?d=2026-11-09").read())
        assert d["entries"][0]["wallets"] == ["A", "B"] and d["entries"][0]["profit"] == 17.8
        bad = json.loads(urllib.request.urlopen(base + "/api/day?d=x").read())
        assert bad == {"entries": []}
    finally:
        web.close()
