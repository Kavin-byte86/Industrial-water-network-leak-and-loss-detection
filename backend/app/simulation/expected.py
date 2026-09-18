"""
Production-aware expected consumption — the answer to the key challenge.

Total facility flow moves with the production schedule, so raw flow says
nothing on its own: at 200% production the plant legitimately draws roughly
twice the water it draws at 100%. Comparing measured flow against a *fixed*
threshold therefore raises an alarm every time the plant gets busy.

This module computes what the plant *should* be drawing right now, given each
machine's production rate and state and each tap's state. The difference

    unexplained = measured - expected_from_production

is the only part that can be a loss. A production ramp moves `measured` and
`expected` together and leaves `unexplained` at zero; a leak moves `measured`
alone.
"""

from __future__ import annotations

from app.simulation.production_model import (
    expected_machine_flow,
    expected_tap_flow,
)
from app.network.topology import (
    ALL_JUNCTION_IDS,
    JUNCTION_TO_ENDPOINT,
    MACHINES,
    NODES,
    TAPS,
)

def expected_endpoint_demands(machines: dict, taps: dict) -> dict:
    """Endpoint id -> expected demand (L/min) from the production schedule."""
    demands = {}
    for mid in MACHINES:
        m = machines.get(mid) or {}
        demands[mid] = expected_machine_flow(
            mid, m.get("production_pct", 0.0), m.get("state", "OFF")
        )
    for tid in TAPS:
        t = taps.get(tid) or {}
        demands[tid] = expected_tap_flow(tid, t.get("state", "CLOSED"))
    return demands


def expected_junction_flows(endpoint_demands: dict) -> dict:
    """Roll expected endpoint demand up the tree to every junction."""
    expected: dict = {}

    def resolve(jid: str) -> float:
        if jid in expected:
            return expected[jid]
        node = NODES[jid]
        if node.children:
            value = sum(resolve(child) for child in node.children)
        else:
            ep = JUNCTION_TO_ENDPOINT.get(jid)
            value = endpoint_demands.get(ep, 0.0) if ep else 0.0
        expected[jid] = value
        return value

    for jid in ALL_JUNCTION_IDS:
        resolve(jid)
    return expected


def reconcile(flows: dict, machines: dict, taps: dict) -> dict:
    """
    Compare measured flow against production-driven expected consumption.

    Returns the plant-level reconciliation plus per-junction expected values,
    which is what lets the UI say "flow is up, and here is how much of that
    production actually accounts for".
    """
    endpoint_demands = expected_endpoint_demands(machines, taps)
    expected = expected_junction_flows(endpoint_demands)

    measured_total = flows.get("J1", 0.0)
    expected_total = expected.get("J1", 0.0)
    unexplained = measured_total - expected_total

    # Share of the water actually drawn that production accounts for.
    explained_pct = (
        100.0 if measured_total <= 0.0
        else max(0.0, min(100.0, (expected_total / measured_total) * 100.0))
    )

    # Aggregate production load, so the UI can show that a flow rise tracks a
    # production rise rather than a fault.
    running = [
        (machines.get(mid) or {}).get("production_pct", 0.0)
        for mid in MACHINES
        if (machines.get(mid) or {}).get("state") not in (None, "OFF", "MAINTENANCE")
    ]
    production_load = sum(running) / len(MACHINES) if MACHINES else 0.0

    return {
        "measured_total_lpm": round(measured_total, 2),
        "expected_total_lpm": round(expected_total, 2),
        "unexplained_lpm": round(unexplained, 2),
        "explained_pct": round(explained_pct, 1),
        "production_load_pct": round(production_load, 1),
        "machines_running": len(running),
        "expected_flows": {k: round(v, 2) for k, v in expected.items()},
        "endpoint_demands": {k: round(v, 2) for k, v in endpoint_demands.items()},
    }
