"""
Production model — per-machine water demand.

The flow curves come from `classifier/config.py`, the same file that generated
the dataset the ML models were trained on. That is deliberate: the backend used
to carry its own simplified physics (every machine a flat 125 L/min, a linear
pressure model), which put live telemetry in a different distribution from the
training data and made the trained models useless against it.

Each machine follows

    flow = base_flow + alpha * prod + beta * prod ** power

with per-machine coefficients, a startup multiplier while STARTING, and a small
standby draw when OFF.
"""

from __future__ import annotations

import random

from app.core.config import NOISE_SIGMA_FRACTION
from app.classifier_config import MACHINE_SPECS, TAP_SPECS, FLOW_NOISE_FRACTION


def _add_noise(value: float, sigma_frac: float | None = None) -> float:
    """Gaussian measurement noise proportional to the reading."""
    if value == 0.0:
        return 0.0
    frac = FLOW_NOISE_FRACTION if sigma_frac is None else sigma_frac
    return value + random.gauss(0, abs(value) * frac)


def running_flow(spec: dict, production_pct: float) -> float:
    """The machine's demand curve at a given production rate."""
    prod = max(0.0, production_pct)
    return (
        spec["base_flow"]
        + spec["alpha"] * prod
        + spec["beta"] * (prod ** spec["power"])
    )


def compute_machine_flow(
    machine_id: str,
    production_pct: float,
    state: str,
) -> float:
    """
    Return the flow contribution (L/min) for a single machine this tick.

    State handling matches `classifier/feature_engineering.py` exactly, so the
    expected-flow model sees the relationship it was fitted on:

        OFF          -> standby draw
        MAINTENANCE  -> standby * 4 + 5
        STOPPING     -> half of base flow
        STARTING     -> running flow * startup factor (startup draws MORE)
        RUNNING      -> the demand curve
    """
    spec = MACHINE_SPECS.get(machine_id)
    if spec is None:
        return 0.0

    if state == "OFF":
        demand = spec["standby_flow"]
    elif state == "MAINTENANCE":
        demand = spec["standby_flow"] * 4.0 + 5.0
    elif state == "STOPPING":
        demand = spec["base_flow"] * 0.5
    elif state == "STARTING":
        demand = running_flow(spec, production_pct) * spec["startup_factor"]
    else:  # RUNNING
        demand = running_flow(spec, production_pct)

    demand = max(spec.get("min_flow", 0.0), min(demand, spec.get("max_flow", 1e9)))
    return _add_noise(demand)


def compute_tap_flow(tap_id: str, state: str) -> float:
    """Tap draw: nominal when OPEN with its own spread, zero when CLOSED."""
    spec = TAP_SPECS.get(tap_id)
    if spec is None or state != "OPEN":
        return 0.0
    nominal = spec["nominal_flow"]
    value = random.gauss(nominal, spec["flow_std"])
    return max(0.0, _add_noise(value))


def expected_machine_flow(machine_id: str, production_pct: float, state: str) -> float:
    """Noise-free demand — what the machine *should* draw. Used by reconciliation."""
    spec = MACHINE_SPECS.get(machine_id)
    if spec is None:
        return 0.0
    if state == "OFF":
        return spec["standby_flow"]
    if state == "MAINTENANCE":
        return spec["standby_flow"] * 4.0 + 5.0
    if state == "STOPPING":
        return spec["base_flow"] * 0.5
    if state == "STARTING":
        return running_flow(spec, production_pct) * spec["startup_factor"]
    return running_flow(spec, production_pct)


def expected_tap_flow(tap_id: str, state: str) -> float:
    spec = TAP_SPECS.get(tap_id)
    if spec is None or state != "OPEN":
        return 0.0
    return spec["nominal_flow"]
