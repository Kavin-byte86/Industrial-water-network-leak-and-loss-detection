"""
Core configuration — env-driven constants for the simulation.

All defaults match BACKEND_IMPLEMENTATION_PLAN.md §3–§4.
Override any value via environment variable of the same name (uppercase).
"""

from __future__ import annotations

import os
from datetime import datetime


# ---------------------------------------------------------------------------
# Tick / time
# ---------------------------------------------------------------------------
TICK_INTERVAL_SECONDS: float = float(os.getenv("TICK_INTERVAL_SECONDS", "2"))
TICK_SIMULATED_MINUTES: int = 5  # one tick = 5 simulated minutes
BASE_DATETIME: datetime = datetime.fromisoformat(
    os.getenv("BASE_DATETIME", "2026-01-01T00:00:00")
)

# ---------------------------------------------------------------------------
# Shift definitions (hour ranges, inclusive start, exclusive end)
# Shift A: 06:00–14:00 | Shift B: 14:00–22:00 | Shift C: 22:00–06:00
# ---------------------------------------------------------------------------
SHIFTS: list[dict] = [
    {"name": "A", "start_hour": 6, "end_hour": 14},
    {"name": "B", "start_hour": 14, "end_hour": 22},
    {"name": "C", "start_hour": 22, "end_hour": 6},  # wraps midnight
]

# ---------------------------------------------------------------------------
# Machine defaults
# ---------------------------------------------------------------------------
# Base flow at 100 % production while RUNNING (L/min per machine).
# Total ~1 000 L/min at full plant (8 × 125).
MACHINE_BASE_FLOWS: dict[str, float] = {
    f"M{i}": float(os.getenv(f"BASE_FLOW_M{i}", "125.0")) for i in range(1, 9)
}
MACHINE_TRANSITION_FACTOR: float = float(
    os.getenv("MACHINE_TRANSITION_FACTOR", "0.5")
)

# ---------------------------------------------------------------------------
# Tap defaults
# ---------------------------------------------------------------------------
TAP_BASE_FLOWS: dict[str, float] = {
    f"T{i}": float(os.getenv(f"BASE_FLOW_T{i}", "20.0")) for i in range(1, 4)
}
TAP_VARIATION_LOW: float = 0.6
TAP_VARIATION_HIGH: float = 1.4

# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
# Gaussian measurement noise σ as a fraction of the expected value
NOISE_SIGMA_FRACTION: float = float(os.getenv("NOISE_SIGMA_FRACTION", "0.01"))

# ---------------------------------------------------------------------------
# Pressure model  (§3.4)
# pressure = BASE_PRESSURE - (PRESSURE_K * downstream_flow) + noise
# ---------------------------------------------------------------------------
BASE_PRESSURE: float = float(os.getenv("BASE_PRESSURE", "6.0"))  # bar
PRESSURE_K: float = float(os.getenv("PRESSURE_K", "0.002"))

# ---------------------------------------------------------------------------
# History ring-buffer size
# ---------------------------------------------------------------------------
HISTORY_BUFFER_SIZE: int = int(os.getenv("HISTORY_BUFFER_SIZE", "1000"))

# ---------------------------------------------------------------------------
# Water tariff — used to convert a loss rate into financial impact.
# Default is a mid-range Indian industrial water tariff; override per site.
# ---------------------------------------------------------------------------
WATER_COST_PER_M3: float = float(os.getenv("WATER_COST_PER_M3", "45.0"))
CURRENCY: str = os.getenv("CURRENCY", "INR")
CURRENCY_SYMBOL: str = os.getenv("CURRENCY_SYMBOL", "\u20b9")

# Effluent handling often costs more than the intake itself: leaked process
# water is usually charged twice (supply + treatment). Set to 0 to count
# supply only.
EFFLUENT_COST_PER_M3: float = float(os.getenv("EFFLUENT_COST_PER_M3", "25.0"))

# ---------------------------------------------------------------------------
# Historical baseline — how many recent ticks define "normal" for a sensor.
# 72 ticks x 5 min = 6 hours, long enough to span a shift's usage pattern.
# ---------------------------------------------------------------------------
BASELINE_WINDOW_TICKS: int = int(os.getenv("BASELINE_WINDOW_TICKS", "72"))
BASELINE_MIN_TICKS: int = int(os.getenv("BASELINE_MIN_TICKS", "12"))
# How many standard deviations above its own baseline a junction must sit
# before its consumption counts as abnormal.
BASELINE_SIGMAS: float = float(os.getenv("BASELINE_SIGMAS", "3.0"))
