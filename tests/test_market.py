"""Histórico longo do mercado: gravação na tabela mercado e alta/baixa por período."""
import calendar
import sqlite3

from candlesense.history import market


def test_symbol():
    assert market.symbol("GBP/USD") == "GBP/USD"
    assert market.symbol("EUR/USD (OTC)") is None
    assert market.symbol("?") is None


def _fake(now):
    jan = calendar.timegm((2026, 1, 1, 0, 0, 0))
    today = now - now % 86400
    days = [(d, 1.20 + i * 1e-4, 1.21 + i * 1e-4, 1.19 + i * 1e-4, 1.20 + (i + 1) * 1e-4)
            for i, d in enumerate(range(jan, today, 86400))]
    # ontem (9/nov, meia-noite de Brasília = 03h UTC) caindo; hoje subindo
    y0 = calendar.timegm((2026, 11, 9, 3, 0, 0))
    hours = [(y0 + h * 3600, 1.33 - h * 1e-4, 1.3302 - h * 1e-4, 1.3298 - (h + 1) * 1e-4, 1.33 - (h + 1) * 1e-4)
             for h in range(24)]
    hours += [(y0 + (24 + h) * 3600, 1.3276 + h * 1e-4, 1.3278 + (h + 1) * 1e-4, 1.3275 + h * 1e-4,
               1.3276 + (h + 1) * 1e-4) for h in range(9)]
    calls = []

    def fetch(sym, interval, size):
        calls.append(interval)
        return days if interval == "1day" else hours
    return fetch, calls


def test_update_and_context(tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", "America/Sao_Paulo")
    import time
    time.tzset()
    now = calendar.timegm((2026, 11, 10, 15, 30, 0))
    fetch, calls = _fake(now)
    mh = market.MarketHistory(tmp_path / "c.db", log=lambda *a, **k: None, fetch=fetch)
    assert mh.update("GBP/USD", now=now) > 0
    assert mh.update("GBP/USD", now=now + 60) == 0          # nada venceu: não gasta pedido
    assert calls == ["1day", "1h"]
    mh.update("GBP/USD", now=now + 3700)                     # 1 hora depois: só as velas de 1 h
    assert calls == ["1day", "1h", "1h"]
    db = sqlite3.connect(tmp_path / "c.db")
    per = {p.name: p for p in market.context(db, "GBP/USD", price=1.3290, now=now)}
    assert per["ontem"].trend == "baixa"
    assert per["hoje"].trend == "alta"
    assert per["ano"].trend == "alta"
    assert "Ano ▲" in market.context_text(list(per.values()))


def test_sem_chave_nao_quebra(tmp_path):
    mh = market.MarketHistory(tmp_path / "c.db", log=lambda *a, **k: None, fetch=lambda *a: None)
    assert mh.update("GBP/USD", now=1_800_000_000) == 0
    assert mh.update("EUR/USD (OTC)") == 0
