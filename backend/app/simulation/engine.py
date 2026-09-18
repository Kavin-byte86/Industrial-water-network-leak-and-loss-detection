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
import threading
from typing import Any

from app.core.config import TICK_INTERVAL_SECONDS
from app.network.topology import MACHINES, TAPS
from app.simulation.aggregation import aggregate_flows
from app.simulation.clock import SimulatedClock
from app.simulation.detection import detect
from app.simulation.expected import reconcile
from app.simulation.loss_tracker import LossTracker
from app import ml_detector
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
        # Accumulates volume and cost per leak event across ticks.
        self.losses = LossTracker()
        # FastAPI runs sync endpoints in a worker thread, so POST /simulation/step
        # can land while the asyncio auto-tick loop is mid-tick. Without this lock
        # the two interleave: the clock advances twice, and one tick can read
        # endpoint flows that belong to the other, producing a snapshot whose
        # detection result does not match any real plant state.
        self._lock = threading.Lock()
        self._task: asyncio.Task | None = None
        self._paused = False
        self._running = False

    # ── single tick ───────────────────────────────────────────────────────

    def tick(self) -> dict:
        """Execute one simulation tick and return the snapshot.

        Serialised: concurrent callers queue rather than interleave.
        """
        with self._lock:
            return self._tick_locked()

    def _tick_locked(self) -> dict:
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

        # 3. Aggregate junction flows, including any injected leaks
        leak_rates = self.store.leaks.get_active_leaks()
        junction_flows = aggregate_flows(endpoint_flows, leak_rates)

        # 4. Derive pressures from the (leak-inflated) flows
        pressures = compute_pressures(junction_flows)

        # 5. Apply manual sensor overrides. These are applied last so a forced
        # reading is what both the UI and the detector see, exactly as a faulty
        # or tampered sensor would behave.
        overrides = dict(self.store.overrides)
        for sensor, value in overrides.items():
            kind, _, jid = sensor.partition("_")
            if kind == "flow" and jid in junction_flows:
                junction_flows[jid] = value
            elif kind == "pressure" and jid in pressures:
                pressures[jid] = value

        flows_rounded = {k: round(v, 2) for k, v in junction_flows.items()}
        pressures_rounded = {k: round(v, 4) for k, v in pressures.items()}

        # 6. Reconcile measured flow against what production should be drawing.
        #    This is what separates "the plant got busy" from "water is being
        #    lost": a production ramp moves both numbers together.
        production = reconcile(junction_flows, machines_snapshot, taps_snapshot)

        # 7. Detect leaks from mass balance, corroborated by pressure and by
        #    each sensor's own history.
        detection = detect(
            junction_flows,
            endpoint_flows,
            pressures=pressures,
            history=list(self.store.history),
        )

        # 8. Age the leak events and price the water lost.
        detection["leaks"] = self.losses.update(
            detection["leaks"], time_info["timestamp"]
        )
        if detection["leaks"]:
            primary = detection["leaks"][0]
            detection["volume_lost_litres"] = primary["volume_lost_litres"]
            detection["cost_so_far"] = primary["cost_so_far"]
        loss_summary = self.losses.summary(detection["leaks"])

        # The trained 4-stage classifier, run on the same history. This is the
        # project's actual model; mass balance above stays as an independent
        # physics cross-check, so the UI can show whether the two agree.
        ml_result = ml_detector.predict_from_history(list(self.store.history))

        # 9. Build snapshot
        snapshot: dict = {
            **time_info,
            "flows": flows_rounded,
            "pressures": pressures_rounded,
            "machines": machines_snapshot,
            "taps": taps_snapshot,
            "detection": detection,
            "ml": ml_result,
            "production": production,
            "losses": loss_summary,
            # Ground truth, so the UI can show what was actually injected next
            # to what the detector found.
            "injected_leaks": {k: round(v, 2) for k, v in leak_rates.items()},
            "overrides": overrides,
        }

        # 9. Push to store
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
        self.losses.reset()

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
