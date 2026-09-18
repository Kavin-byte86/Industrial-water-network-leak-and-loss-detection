"""
In-memory session store — §5 of BACKEND_IMPLEMENTATION_PLAN.md.

Holds:
    • Current machine states and production percentages.
    • Current tap states.
    • A ring buffer of the last N tick snapshots.

No database — everything is in-process memory.
"""

from __future__ import annotations

from collections import deque

from app.core.config import HISTORY_BUFFER_SIZE
from app.network.topology import MACHINES, TAPS


class SessionStore:
    """Single-instance in-memory state + history manager."""

    def __init__(self) -> None:
        self.machines: dict[str, dict] = {}
        self.taps: dict[str, dict] = {}
        self.history: deque[dict] = deque(maxlen=HISTORY_BUFFER_SIZE)
        self._latest: dict | None = None
        self.reset()

    # ── reset to initial conditions ───────────────────────────────────────

    def reset(self) -> None:
        """
        Reset all machines to OFF / 0 % production and all taps to CLOSED.
        Clears history.
        """
        self.machines = {
            mid: {"production_pct": 0.0, "state": "OFF"} for mid in MACHINES
        }
        self.taps = {tid: {"state": "CLOSED"} for tid in TAPS}
        self.history.clear()
        self._latest = None

    # ── snapshot management ───────────────────────────────────────────────

    def push_snapshot(self, snapshot: dict) -> None:
        """Append a tick snapshot to the history ring buffer."""
        self.history.append(snapshot)
        self._latest = snapshot

    @property
    def latest(self) -> dict | None:
        return self._latest

    def get_history(self, limit: int = 100) -> list[dict]:
        """Return the last *limit* snapshots, oldest first."""
        items = list(self.history)
        return items[-limit:]

    # ── control helpers ───────────────────────────────────────────────────

    def update_machine(
        self,
        machine_id: str,
        production_pct: float | None = None,
        state: str | None = None,
    ) -> dict:
        """Partially update a machine's production_pct and/or state."""
        m = self.machines[machine_id]
        if production_pct is not None:
            m["production_pct"] = production_pct
        if state is not None:
            m["state"] = state
        return m

    def update_tap(self, tap_id: str, state: str) -> dict:
        """Update a tap's state."""
        t = self.taps[tap_id]
        t["state"] = state
        return t
