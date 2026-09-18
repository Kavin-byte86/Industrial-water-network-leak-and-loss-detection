"""
Pressure model — §3.4 of BACKEND_IMPLEMENTATION_PLAN.md.

Simple derived pressure per junction:

    pressure = BASE_PRESSURE − (k × total_downstream_flow) + noise

This is NOT hydraulically rigorous — it provides a plausible, consistent
signal so the classifier's pressure-based features have something to work
with.  Replace with a real hydraulic model (e.g. EPANET/WNTR) if needed.
"""

from __future__ import annotations

import random

from app.core.config import BASE_PRESSURE, NOISE_SIGMA_FRACTION, PRESSURE_K
from app.network.topology import ALL_JUNCTION_IDS, NODES


def compute_pressures(
    junction_flows: dict[str, float],
) -> dict[str, float]:
    """
    Compute a pressure value at every junction based on downstream flow.

    Parameters
    ----------
    junction_flows : dict
        Junction id → flow (L/min) as produced by the aggregation step.

    Returns
    -------
    dict
        Junction id → pressure (bar).
    """
    pressures: dict[str, float] = {}

    for jid in ALL_JUNCTION_IDS:
        downstream = _total_downstream_flow(jid, junction_flows)
        raw_pressure = BASE_PRESSURE - (PRESSURE_K * downstream)
        # Clamp to non-negative (pressure can't go below zero in reality)
        raw_pressure = max(raw_pressure, 0.0)
        # Add small noise
        noise = random.gauss(0, abs(raw_pressure) * NOISE_SIGMA_FRACTION) if raw_pressure > 0 else 0.0
        pressures[jid] = round(raw_pressure + noise, 4)

    return pressures


def _total_downstream_flow(
    jid: str,
    junction_flows: dict[str, float],
) -> float:
    """
    Return the total flow at and below junction `jid`.

    For a leaf junction this is just its own flow.
    For an intermediate junction, it equals its own measured flow
    (which already includes children, per aggregation).
    """
    return junction_flows.get(jid, 0.0)
