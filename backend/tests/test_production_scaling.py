"""
Test production scaling — §10.3 of BACKEND_IMPLEMENTATION_PLAN.md.

Setting all machines to 200% RUNNING should scale flows roughly linearly
compared to 100% (within noise tolerance). No leak fields exist yet.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.simulation.production_model import compute_machine_flow, compute_tap_flow
from app.simulation.aggregation import aggregate_flows
from app.network.topology import MACHINES, TAPS


def _flows_at_production(pct: float) -> dict[str, float]:
    """Compute a full set of junction flows at a given production %."""
    ep: dict[str, float] = {}
    for mid in MACHINES:
        ep[mid] = compute_machine_flow(mid, pct, "RUNNING")
    for tid in TAPS:
        ep[tid] = compute_tap_flow(tid, "CLOSED")
    return aggregate_flows(ep)


def test_200pct_scales_roughly_double():
    """200% production should yield ~2× the flow of 100%, ±20% tolerance."""
    # Average over several runs to smooth noise
    runs = 50
    j1_at_100 = sum(_flows_at_production(100.0)["J1"] for _ in range(runs)) / runs
    j1_at_200 = sum(_flows_at_production(200.0)["J1"] for _ in range(runs)) / runs

    ratio = j1_at_200 / j1_at_100
    assert 1.7 < ratio < 2.3, (
        f"Expected ratio ~2.0, got {ratio:.3f} "
        f"(J1@100%={j1_at_100:.1f}, J1@200%={j1_at_200:.1f})"
    )


def test_various_production_levels_scale_linearly():
    """Flow should scale approximately linearly across a range of production %."""
    runs = 30
    levels = [50, 100, 150, 200]
    avg_flows = {}
    for pct in levels:
        avg_flows[pct] = sum(
            _flows_at_production(pct)["J1"] for _ in range(runs)
        ) / runs

    # Check each pair of adjacent levels
    for i in range(len(levels) - 1):
        low, high = levels[i], levels[i + 1]
        expected_ratio = high / low
        actual_ratio = avg_flows[high] / avg_flows[low] if avg_flows[low] > 0 else 0
        assert abs(actual_ratio - expected_ratio) < 0.3 * expected_ratio, (
            f"Non-linear scaling between {low}% and {high}%: "
            f"expected ratio ~{expected_ratio:.2f}, got {actual_ratio:.2f}"
        )
