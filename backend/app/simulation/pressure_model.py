"""
Pressure model — header pressures derived from flow.

Uses the hydraulics from `classifier/config.py` so live pressure readings sit in
the same range as the training data:

    P(J1)      = inlet_base - source_friction * flow(J1)
    P(branch)  = P(J1)      - branch_friction * flow(branch)

The previous model (6.0 bar minus 0.002 x flow) produced ~4.0 bar at the main
under full load where the training data sits near 4.95, so every pressure
feature the classifier had learned was shifted.

Pressure is reported at the five header sensors the classifier knows about.
Downstream junctions are still published for the UI, derived from their parent.
"""

from __future__ import annotations

import random

from app.classifier_config import (
    PRESSURE_CONFIG,
    PRESSURE_NOISE_BAR,
    PRESSURE_SENSORS,
)
from app.network.topology import ALL_JUNCTION_IDS, NODES

_MIN_P = PRESSURE_CONFIG["min_plausible_pressure"]
_MAX_P = PRESSURE_CONFIG["max_plausible_pressure"]


def _noise() -> float:
    return random.gauss(0, PRESSURE_NOISE_BAR)


def compute_pressures(junction_flows: dict) -> dict:
    """
    Compute pressure (bar) at every junction from the current flows.

    Parameters
    ----------
    junction_flows : dict
        Junction id → flow (L/min) after aggregation, leaks included. A leak
        raises the flow through its ancestors, so it pulls their pressure down —
        which is what makes a pressure drop corroborating evidence.
    """
    pressures: dict = {}

    inlet_flow = junction_flows.get("J1", 0.0)
    p_inlet = (
        PRESSURE_CONFIG["inlet_base_pressure"]
        - PRESSURE_CONFIG["source_flow_friction"] * inlet_flow
    )
    pressures["J1"] = p_inlet

    friction = PRESSURE_CONFIG["pipe_friction_loss"]
    for jid in ("J2", "J3", "J4", "J7"):
        pressures[jid] = p_inlet - friction.get(jid, 0.0003) * junction_flows.get(jid, 0.0)

    # Leaf junctions inherit their parent header's pressure, less a small drop
    # for their own flow. Not metered in the training data, but the UI shows them.
    for jid in ALL_JUNCTION_IDS:
        if jid in pressures:
            continue
        parent = NODES[jid].parent_id
        parent_p = pressures.get(parent, p_inlet)
        pressures[jid] = parent_p - 0.00025 * junction_flows.get(jid, 0.0)

    for jid, value in pressures.items():
        noisy = value + _noise()
        pressures[jid] = max(_MIN_P, min(_MAX_P, noisy))

    return pressures


def header_pressures(pressures: dict) -> dict:
    """The five headers the classifier was trained on."""
    return {jid: pressures[jid] for jid in PRESSURE_SENSORS if jid in pressures}
