"""
Production model — §3.1 / §3.2 of BACKEND_IMPLEMENTATION_PLAN.md.

Computes per-tick flow contributions for every machine and tap based on
their current state and production rate.
"""

from __future__ import annotations

import random

from app.core.config import (
    MACHINE_BASE_FLOWS,
    MACHINE_TRANSITION_FACTOR,
    NOISE_SIGMA_FRACTION,
    TAP_BASE_FLOWS,
    TAP_VARIATION_HIGH,
    TAP_VARIATION_LOW,
)


def _add_noise(value: float, sigma_frac: float = NOISE_SIGMA_FRACTION) -> float:
    """Add Gaussian measurement noise proportional to the value."""
    if value == 0.0:
        return 0.0
    sigma = abs(value) * sigma_frac
    return value + random.gauss(0, sigma)


# ---------------------------------------------------------------------------
# Machine flow
# ---------------------------------------------------------------------------

def compute_machine_flow(
    machine_id: str,
    production_pct: float,
    state: str,
) -> float:
    """
    Return the flow contribution (L/min) for a single machine this tick.

    States:
        OFF / MAINTENANCE  → 0
        RUNNING            → base_flow × (production_pct / 100) + noise
        STARTING / STOPPING → RUNNING value × transition factor + noise
    """
    if state in ("OFF", "MAINTENANCE"):
        return 0.0

    base = MACHINE_BASE_FLOWS.get(machine_id, 125.0)
    running_flow = base * (production_pct / 100.0)

    if state in ("STARTING", "STOPPING"):
        running_flow *= MACHINE_TRANSITION_FACTOR

    return _add_noise(running_flow)


# ---------------------------------------------------------------------------
# Tap flow
# ---------------------------------------------------------------------------

def compute_tap_flow(
    tap_id: str,
    state: str,
) -> float:
    """
    Return the flow contribution (L/min) for a single tap this tick.

    CLOSED → 0
    OPEN   → base_flow × random_factor(0.6–1.4) + noise
    """
    if state == "CLOSED":
        return 0.0

    base = TAP_BASE_FLOWS.get(tap_id, 20.0)
    random_factor = random.uniform(TAP_VARIATION_LOW, TAP_VARIATION_HIGH)
    return _add_noise(base * random_factor)
