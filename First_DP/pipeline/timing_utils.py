#!/usr/bin/env python3
"""Cronômetro com ETA para treinos longos."""

from __future__ import annotations

import time


def format_duration(seconds: float) -> str:
    """Formata segundos como '2h 15m 30s', '45m 12s' ou '38s'."""
    total = int(round(max(0.0, seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


class RunTimer:
    """Cronômetro com progresso e ETA linear."""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self._t0 = time.perf_counter()
        self._done = 0
        self._total: int | None = None
        self._last_step_duration: float | None = None

    def reset(self) -> None:
        self._t0 = time.perf_counter()
        self._done = 0
        self._total = None
        self._last_step_duration = None

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def set_total(self, total: int) -> None:
        self._total = max(0, int(total))

    def step_done(self, n: int = 1) -> None:
        self._done += n

    def mark_step_start(self) -> None:
        self._step_t0 = time.perf_counter()

    def mark_step_end(self) -> float:
        dur = time.perf_counter() - getattr(self, "_step_t0", self._t0)
        self._last_step_duration = dur
        return dur

    def eta(self) -> float | None:
        if self._total is None or self._done <= 0:
            return None
        remaining = self._total - self._done
        if remaining <= 0:
            return 0.0
        return (self.elapsed() / self._done) * remaining

    def status(self, *, extra: str = "") -> str:
        parts = [f"⏱ {format_duration(self.elapsed())}"]
        if self.label:
            parts[0] = f"⏱ [{self.label}] {format_duration(self.elapsed())}"
        if self._total is not None:
            parts.append(f"{self._done}/{self._total}")
        eta = self.eta()
        if eta is not None:
            parts.append(f"ETA {format_duration(eta)}")
        if self._last_step_duration is not None:
            parts.append(f"último {format_duration(self._last_step_duration)}")
        if extra:
            parts.append(extra)
        return " | ".join(parts)
