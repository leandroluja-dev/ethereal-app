import json
import random
import time

from candlesense import cloud


def _bars(now, n=320, seed=3):
    random.seed(seed)
    m = int(now // 60) * 60
    c, out = 1.33, []
    for i in range(n, -1, -1):
        o = c
        c = o + random.gauss(0, 0.00025)
        out.append((m - 60 * i, o, max(o, c) + 1e-4, min(o, c) - 1e-4, c))
    return out


def test_cloud_run_writes_page_and_state(tmp_path):
    now = time.mktime((2026, 11, 11, 15, 0, 12, 0, 0, -1))
    res = cloud.run(tmp_path, "x", now=now, bars=_bars(now))
    assert res["velas"] == 320 and res["medidas"] > 100
    st = json.loads((tmp_path / "api" / "state").read_text())
    assert st["asset"] == "GBP/USD" and st["live"]["cloud"] and [w["key"] for w in st["wallets"]] == ["A", "B", "C"]
    assert (tmp_path / "index.html").exists()
    # a rodada seguinte reaproveita o banco e não mede de novo o que já mediu
    res2 = cloud.run(tmp_path, "x", now=now + 900, bars=_bars(now + 900, seed=3))
    assert res2["medidas"] <= 20
    for d in st["days"]:
        assert (tmp_path / "api" / "dias" / f"{d['day']}.json").exists()
