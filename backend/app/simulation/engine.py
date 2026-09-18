"""
Simulation engine — §5 of BACKEND_IMPLEMENTATION_PLAN.md.

Orchestrates the full tick pipeline:
    1. Advance the simulated clock.
    2. Compute per-endpoint flows (production model).
    3. Aggregate junction flows bottom-up.
    4. Derive pressures.
    5. Build a tick snapshot and push it to the session store.

Also manages the background auto-tick loop (start / pause / resume).
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.config import TICK_INTERVAL_SECONDS
from app.network.topology import MACHINES, TAPS
from app.simulation.aggregation import aggregate_flows
from app.simulation.clock import SimulatedClock
from app.simulation.pressure_model import compute_pressures
from app.simulation.production_model import compute_machine_flow, compute_tap_flow


class SimulationEngine:
    """Central simulation coordinator."""

    def __init__(self, store: Any):
        """
        Parameters
        ----------
        store : SessionStore
            The in-memory state & history manager.
        """
        self.store = store
        self.clock = SimulatedClock()
        self._task: asyncio.Task | None = None
        self._paused = False
        self._running = False

    # ── single tick ───────────────────────────────────────────────────────

    def tick(self) -> dict:
        """Execute one simulation tick and return the snapshot."""
        # 1. Advance clock
        self.clock.advance()
        time_info = self.clock.snapshot()

        # 2. Compute endpoint flows
        endpoint_flows: dict[str, float] = {}
        machines_snapshot: dict[str, dict] = {}

        for mid in MACHINES:
            m = self.store.machines[mid]
            flow = compute_machine_flow(mid, m["production_pct"], m["state"])
            endpoint_flows[mid] = flow
            machines_snapshot[mid] = {
                "production_pct": m["production_pct"],
                "state": m["state"],
                "flow_lpm": round(flow, 2),
            }

        taps_snapshot: dict[str, dict] = {}
        for tid in TAPS:
            t = self.store.taps[tid]
            flow = compute_tap_flow(tid, t["state"])
            endpoint_flows[tid] = flow
            taps_snapshot[tid] = {
                "state": t["state"],
                "flow_lpm": round(flow, 2),
            }

        # 3. Aggregate junction flows
        junction_flows = aggregate_flows(endpoint_flows)
        flows_rounded = {k: round(v, 2) for k, v in junction_flows.items()}

        # 4. Derive pressures
        pressures = compute_pressures(junction_flows)
        pressures_rounded = {k: round(v, 4) for k, v in pressures.items()}

        # 5. Build snapshot
        snapshot: dict = {
            **time_info,
            "flows": flows_rounded,
            "pressures": pressures_rounded,
            "machines": machines_snapshot,
            "taps": taps_snapshot,
        }

        # 6. Push to store
        self.store.push_snapshot(snapshot)

        return snapshot

    # ── background loop ───────────────────────────────────────────────────

    async def start_background_loop(self) -> None:
        """Start the auto-tick background task."""
        if self._running:
            return
        self._running = True
        self._paused = False
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while self._running:
            if not self._paused:
                self.tick()
            await asyncio.sleep(TICK_INTERVAL_SECONDS)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    def reset(self) -> None:
        """Reset the entire simulation to initial state."""
        self.clock.reset()
        self.store.reset()

    async def stop(self) -> None:
        """Stop the background loop entirely."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
