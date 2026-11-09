"""Detecta troca de ativo ou de timeframe (confirmada em leituras seguidas, para ignorar falhas de OCR)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Tracked:
    value: object = None
    pending: object = None
    pending_count: int = 0


class ChangeDetector:
    def __init__(self, confirm: int = 2) -> None:
        self._confirm = confirm
        self._items: dict[str, _Tracked] = {"ativo": _Tracked(), "timeframe": _Tracked()}
        self.reset_ts: float | None = None

    def update(self, now: float, asset: str | None, timeframe: int | None) -> list[str]:
        events: list[str] = []
        for key, value in (("ativo", asset), ("timeframe", timeframe)):
            if value is None:
                continue
            t = self._items[key]
            if t.value is None:
                t.value = value
                if self.reset_ts is None:
                    self.reset_ts = now
                continue
            if value == t.value:
                t.pending, t.pending_count = None, 0
                continue
            if value == t.pending:
                t.pending_count += 1
            else:
                t.pending, t.pending_count = value, 1
            if t.pending_count >= self._confirm:
                events.append(f"troca de {key}: {t.value} -> {value}")
                t.value, t.pending, t.pending_count = value, None, 0
                self.reset_ts = now
        return events

    def seconds_since_reset(self, now: float) -> float | None:
        return None if self.reset_ts is None else now - self.reset_ts
