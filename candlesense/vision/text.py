"""Interpretação dos textos lidos pelo OCR: preços, payout, relógio e nome do ativo."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_NUMERIC_CONFUSIONS = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1", "i": "1",
                                     "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "G": "6"})


@dataclass(frozen=True)
class ParsedNumber:
    value: float
    decimals: int
    digits: str
    had_separator: bool


def parse_number(text: str) -> ParsedNumber | None:
    """Converte '1.08452', '1,08452', '62 345.12', '62,345.12' em número.

    Devolve None se o texto não for claramente numérico.
    """
    s = text.strip().replace(" ", "").replace(" ", "").replace(" ", "")
    s = s.lstrip("+$€£R").rstrip("%")
    if not s:
        return None
    if re.fullmatch(r"-?[0-9OoDQIl|iSsBZzG.,]+", s) is None:
        return None
    if sum(ch.isdigit() for ch in s) < max(1, len(s.replace(".", "").replace(",", "")) - 1):
        # Muitas letras "corrigidas": provavelmente não é um número.
        return None
    s = s.translate(_NUMERIC_CONFUSIONS)
    neg = s.startswith("-")
    s = s.lstrip("-")
    if not s or not s[0].isdigit() or not s[-1].isdigit():
        return None
    dots, commas = s.count("."), s.count(",")
    had_sep = dots + commas > 0
    if dots and commas:
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        thou = "," if dec == "." else "."
        s = s.replace(thou, "")
        if s.count(dec) > 1:
            return None
        s = s.replace(dec, ".")
    elif dots > 1:
        s = s.replace(".", "")
    elif commas > 1:
        s = s.replace(",", "")
    elif commas == 1:
        s = s.replace(",", ".")
    try:
        value = float(s)
    except ValueError:
        return None
    decimals = len(s.split(".")[1]) if "." in s else 0
    digits = s.replace(".", "")
    return ParsedNumber(-value if neg else value, decimals, digits, had_sep)


def repair_decimals(numbers: list[ParsedNumber | None]) -> list[float | None]:
    """Uniformiza as casas decimais dos rótulos do eixo.

    Os rótulos de um eixo têm o mesmo formato. Se o OCR perdeu o ponto em um deles
    ("108452" em vez de "1.08452"), reinsere o ponto na posição usada pela maioria.
    """
    with_sep = [n.decimals for n in numbers if n is not None and n.had_separator]
    if not with_sep:
        return [n.value if n is not None else None for n in numbers]
    majority = Counter(with_sep).most_common(1)[0][0]
    int_lens = [len(n.digits) - n.decimals for n in numbers if n is not None and n.had_separator]
    int_len = Counter(int_lens).most_common(1)[0][0] if int_lens else None
    out: list[float | None] = []
    for n in numbers:
        if n is None:
            out.append(None)
        elif n.had_separator or majority == 0:
            out.append(n.value)
        elif int_len is not None and len(n.digits) == int_len + majority:
            out.append(int(n.digits) / (10**majority))
        else:
            out.append(None)  # formato incompatível: descarta em vez de chutar
    return out


_PERCENT = re.compile(r"([+]?)\s*(\d{1,3})\s*%")


def parse_percent(text: str) -> int | None:
    m = _PERCENT.search(text.replace("O", "0").replace("o", "0"))
    if not m:
        return None
    v = int(m.group(2))
    return v if 0 < v <= 1000 else None


_CLOCK = re.compile(r"^\s*(\d{1,2})\s*[:.]\s*(\d{2})(?:\s*[:.]\s*(\d{2}))?\s*$")


def parse_clock(text: str) -> tuple[int, ...] | None:
    """'00:34' -> (0, 34); '15:30:00' -> (15, 30, 0). Só aceita formatos de relógio."""
    t = text.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1"}))
    m = _CLOCK.match(t)
    if not m or ":" not in t:
        return None
    parts = tuple(int(g) for g in m.groups() if g is not None)
    if any(p >= 60 for p in parts[1:]):
        return None
    return parts


_FX = re.compile(r"\b([A-Z]{3,5})\s*/\s*([A-Z]{3,5})\b")
_UI_WORDS = {
    "IQ OPTION", "IQOPTION", "DEPOSITAR", "DEPOSIT", "SACAR", "WITHDRAW", "CONTA", "ACCOUNT", "TREINAMENTO",
    "PRACTICE", "REAL", "SALDO", "BALANCE", "LUCRO", "PROFIT", "VALOR", "AMOUNT", "EXPIRAÇÃO", "EXPIRACAO",
    "EXPIRATION", "ALTA", "BAIXA", "HIGHER", "LOWER", "CALL", "PUT", "TOTAL", "PORTFÓLIO", "PORTFOLIO",
    "MERCADO", "MARKET", "CHAT", "SUPORTE", "SUPPORT", "HISTÓRICO", "HISTORY", "BINÁRIA", "BINARY",
    "DIGITAL", "OPÇÕES", "OPTIONS", "MAIS", "MORE", "TUTORIAIS", "TUTORIALS", "NOTÍCIAS", "NEWS",
}


def normalize_asset(text: str) -> str | None:
    """Normaliza o nome do ativo ('eur/usd (otc)' -> 'EUR/USD (OTC)'). None se parecer texto de interface."""
    t = re.sub(r"\s+", " ", text.strip()).upper()
    t = t.replace("( ", "(").replace(" )", ")")
    if len(t) < 3 or not re.search(r"[A-Z]", t):
        return None
    if t in _UI_WORDS or parse_percent(t) is not None or parse_clock(t) is not None:
        return None
    otc = "OTC" in t
    m = _FX.search(t.replace(" ", "") if "/" in t else t)
    if m:
        base = f"{m.group(1)}/{m.group(2)}"
        return f"{base} (OTC)" if otc else base
    t = re.sub(r"\(?OTC\)?", "", t).strip(" -·")
    if len(t) < 3 or t in _UI_WORDS:
        return None
    return f"{t} (OTC)" if otc else t


def asset_score(text: str) -> float:
    """Quão provável é o texto ser um nome de ativo (0 a 1)."""
    norm = normalize_asset(text)
    if norm is None:
        return 0.0
    score = 0.3
    if _FX.search(norm):
        score += 0.5
    if "OTC" in norm:
        score += 0.2
    return min(score, 1.0)
