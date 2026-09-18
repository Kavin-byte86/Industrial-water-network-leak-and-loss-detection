"""
Core configuration — env-driven constants for the simulation.

All defaults match BACKEND_IMPLEMENTATION_PLAN.md §3–§4.
Override any value via environment variable of the same name (uppercase).
"""

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
