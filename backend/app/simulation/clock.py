"""
Simulated clock — §4 of BACKEND_IMPLEMENTATION_PLAN.md.

One tick = 5 simulated minutes.
Derives hour, day_of_week, month, and shift from the simulated timestamp.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.core.config import BASE_DATETIME, SHIFTS, TICK_SIMULATED_MINUTES


class SimulatedClock:
    """Tracks simulated time, advancing by TICK_SIMULATED_MINUTES per tick."""

    def __init__(self, start: datetime | None = None):
        self._current: datetime = start or BASE_DATETIME

    # ── public interface ──────────────────────────────────────────────────

    @property
    def current(self) -> datetime:
        return self._current

    def advance(self) -> datetime:
        """Advance one tick and return the new simulated timestamp."""
        self._current += timedelta(minutes=TICK_SIMULATED_MINUTES)
        return self._current

    def reset(self, start: datetime | None = None) -> None:
        self._current = start or BASE_DATETIME

    # ── derived fields ────────────────────────────────────────────────────

    @property
    def hour(self) -> int:
        return self._current.hour

    @property
    def day_of_week(self) -> int:
        """Monday = 0 … Sunday = 6."""
        return self._current.weekday()

    @property
    def month(self) -> int:
        return self._current.month

    @property
    def shift(self) -> str:
        """Return the current shift name (A, B, or C)."""
        h = self._current.hour
        for s in SHIFTS:
            start, end = s["start_hour"], s["end_hour"]
            if start < end:
                # Normal range (e.g. 06–14)
                if start <= h < end:
                    return s["name"]
            else:
                # Wraps midnight (e.g. 22–06)
                if h >= start or h < end:
                    return s["name"]
        # Fallback (should not happen with default shifts covering 24 h)
        return "C"

    def snapshot(self) -> dict:
        """Return a dict of all time-derived fields for a tick."""
        return {
            "timestamp": self._current.isoformat(),
            "hour": self.hour,
            "day_of_week": self.day_of_week,
            "month": self.month,
            "shift": self.shift,
        }
