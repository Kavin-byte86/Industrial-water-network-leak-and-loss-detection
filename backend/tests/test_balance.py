"""
Test flow-balance equations — §10.4 of BACKEND_IMPLEMENTATION_PLAN.md.

Asserts:
    J1  ≈ J2  + J3  + J4  + J7
    J2  ≈ J5  + J6
    J3  ≈ J8  + J9  + J10
    J4  ≈ J11 + J12
    J7  ≈ J13 + J14 + J15 + J16

across many random ticks with varying machine/tap states.
"""

from __future__ import annotations

import random
import sys
import os

# Ensure the backend directory is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.simulation.production_model import compute_machine_flow, compute_tap_flow
from app.simulation.aggregation import aggregate_flows
from app.network.topology import MACHINES, TAPS


BALANCE_EQUATIONS = [
    ("J1", ["J2", "J3", "J4", "J7"]),
    ("J2", ["J5", "J6"]),
    ("J3", ["J8", "J9", "J10"]),
    ("J4", ["J11", "J12"]),
    ("J7", ["J13", "J14", "J15", "J16"]),
]

# Tolerance: 5% of the parent's flow (generous because of additive noise at
# each junction level).
RELATIVE_TOLERANCE = 0.05
ABSOLUTE_TOLERANCE = 5.0  # L/min — covers cases where flows are very small


def _random_endpoint_flows() -> dict[str, float]:
    """Generate one tick's worth of random endpoint flows."""
    states = ["OFF", "RUNNING", "STARTING", "STOPPING", "MAINTENANCE"]
    tap_states = ["OPEN", "CLOSED"]
    flows: dict[str, float] = {}
    for mid in MACHINES:
        pct = random.uniform(0, 200)
        st = random.choice(states)
        flows[mid] = compute_machine_flow(mid, pct, st)
    for tid in TAPS:
        st = random.choice(tap_states)
        flows[tid] = compute_tap_flow(tid, st)
    return flows


def test_balance_equations_hold():
    """Balance equations must hold within noise tolerance across 200 random ticks."""
    for _ in range(200):
        ep_flows = _random_endpoint_flows()
        jf = aggregate_flows(ep_flows)

        for parent, children in BALANCE_EQUATIONS:
            parent_flow = jf[parent]
            children_sum = sum(jf[c] for c in children)
            diff = abs(parent_flow - children_sum)
            tol = max(ABSOLUTE_TOLERANCE, abs(parent_flow) * RELATIVE_TOLERANCE)
            assert diff < tol, (
                f"Balance violated: {parent}={parent_flow:.2f} vs "
                f"sum({children})={children_sum:.2f}, diff={diff:.2f}, tol={tol:.2f}"
            )


def test_j1_approx_1000_at_full_production():
    """
    All machines at 100% RUNNING, all taps CLOSED → J1 ≈ 1000 L/min.
    (8 machines × 125 LPM = 1000, ± noise)
    """
    flows: dict[str, float] = {}
    for mid in MACHINES:
        flows[mid] = compute_machine_flow(mid, 100.0, "RUNNING")
    for tid in TAPS:
        flows[tid] = compute_tap_flow(tid, "CLOSED")

    jf = aggregate_flows(flows)
    # With 1% noise at each level, J1 should be within ~10% of 1000
    assert 850 < jf["J1"] < 1150, f"J1={jf['J1']:.2f}, expected ~1000"
