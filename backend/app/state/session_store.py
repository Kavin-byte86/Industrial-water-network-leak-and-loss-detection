"""
In-memory session store — §5 of BACKEND_IMPLEMENTATION_PLAN.md.

Holds:
    • Current machine states and production percentages.
    • Current tap states.
    • A ring buffer of the last N tick snapshots.

Also writes each tick to Postgres asynchronously (background thread) so
historical data survives Render free-tier spin-downs.
"""

from __future__ import annotations

import logging
import threading
from collections import deque

from app.core.config import HISTORY_BUFFER_SIZE
from app.leak_extension_point import LeakRegistry
from app.network.topology import MACHINES, TAPS

logger = logging.getLogger(__name__)


def _flatten_snapshot_for_db(snap: dict) -> dict:
    """
    Convert a nested tick snapshot into the flat column dict that matches
    the TickRecord ORM model (same shape as datasink._flatten_snapshot).
    """
    flat: dict = {
        "timestamp": snap["timestamp"],
        "hour": snap["hour"],
        "day_of_week": snap["day_of_week"],
        "month": snap["month"],
        "shift": snap["shift"],
    }

    for jid, val in snap.get("flows", {}).items():
        flat[f"flow_{jid}"] = val

    for jid, val in snap.get("pressures", {}).items():
        flat[f"pressure_{jid}"] = val

    for mid, minfo in snap.get("machines", {}).items():
        flat[f"production_{mid}"] = minfo["production_pct"]
        flat[f"machine_status_{mid}"] = minfo["state"]

    for tid, tinfo in snap.get("taps", {}).items():
        flat[f"tap_status_{tid}"] = tinfo["state"]

    injected = snap.get("injected_leaks") or {}
    flat["leak"] = 1 if injected else 0
    flat["leak_rate"] = round(sum(injected.values()), 2)
    flat["leak_zone"] = (
        f"ZONE_{max(injected, key=injected.get)}" if injected else None
    )

    return flat


def _write_tick_to_db(flat: dict) -> None:
    """Synchronous DB write — runs in a daemon thread."""
    try:
        from app.db.database import get_session
        from app.db.models import TickRecord

        session = get_session()
        try:
            record = TickRecord(**flat)
            session.add(record)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    except Exception as exc:
        logger.error("DB write failed (tick continues in-memory): %s", exc)


class SessionStore:
    """Single-instance in-memory state + history manager."""

    def __init__(self) -> None:
        self.machines: dict[str, dict] = {}
        self.taps: dict[str, dict] = {}
        self.history: deque[dict] = deque(maxlen=HISTORY_BUFFER_SIZE)
        self._latest: dict | None = None
        # Injected leaks and manual sensor overrides, both driven by the test
        # bench UI. Overrides map a sensor key ("flow_J3", "pressure_J3") to a
        # forced reading, letting an operator fake a sensor without changing the
        # underlying simulation.
        self.leaks = LeakRegistry()
        self.overrides: dict[str, float] = {}
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
        self.leaks.clear_all()
        self.overrides.clear()
        self.history.clear()
        self._latest = None

    # ── snapshot management ───────────────────────────────────────────────

    def push_snapshot(self, snapshot: dict) -> None:
        """Append a tick snapshot to the history ring buffer and persist to DB."""
        self.history.append(snapshot)
        self._latest = snapshot

        # Fire-and-forget DB write in a background thread so the tick loop
        # is never blocked by a slow or failed Postgres round-trip.
        try:
            flat = _flatten_snapshot_for_db(snapshot)
            t = threading.Thread(target=_write_tick_to_db, args=(flat,), daemon=True)
            t.start()
        except Exception as exc:
            logger.error("Could not dispatch DB write thread: %s", exc)

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

    # ── sensor overrides ──────────────────────────────────────────────────

    def set_override(self, sensor: str, value: float) -> None:
        """Force a sensor reading (e.g. "flow_J3") to a fixed value."""
        self.overrides[sensor] = float(value)

    def clear_override(self, sensor: str) -> None:
        self.overrides.pop(sensor, None)

    def clear_overrides(self) -> None:
        self.overrides.clear()

