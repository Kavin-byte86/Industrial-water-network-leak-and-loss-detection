"""
Bridge to the classifier's physics constants.

The dataset the ML models were trained on was generated from
`classifier/config.py`. If the live simulator invents its own machine curves and
pressure model, live telemetry lands in a different distribution from the
training data and the models cannot work against it — which is exactly what was
wrong before.

So the simulator reads its physics from that same file. One definition, used by
both the generator that trained the models and the service that serves them.

If the classifier package is unavailable (a backend-only deployment), the module
falls back to equivalent literals so the API still starts; `USING_CLASSIFIER_CONFIG`
reports which path was taken.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CLASSIFIER_DIR = os.path.join(_REPO_ROOT, "classifier")
for _path in (_REPO_ROOT, _CLASSIFIER_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

USING_CLASSIFIER_CONFIG = False

try:
    from classifier import config as _cfg  # type: ignore

    MACHINE_SPECS = _cfg.MACHINE_SPECS
    TAP_SPECS = _cfg.TAP_SPECS
    PRESSURE_CONFIG = _cfg.PRESSURE_CONFIG
    FLOW_NOISE_PERCENT = _cfg.FLOW_NOISE_PERCENT
    PRESSURE_NOISE_BAR = _cfg.PRESSURE_NOISE_BAR
    SAMPLING_MINUTES = _cfg.SAMPLING_MINUTES
    USING_CLASSIFIER_CONFIG = True
except Exception:  # pragma: no cover - fallback for backend-only installs
    MACHINE_SPECS = {
        "M1": {"name": "Scouring machine", "sensor": "J5", "base_flow": 22.0,
               "standby_flow": 0.2, "startup_factor": 1.35, "alpha": 0.95,
               "beta": 0.00065, "power": 1.35, "min_flow": 0.0, "max_flow": 300.0},
        "M2": {"name": "Bleaching machine", "sensor": "J6", "base_flow": 26.0,
               "standby_flow": 0.3, "startup_factor": 1.30, "alpha": 1.05,
               "beta": 0.0007, "power": 1.30, "min_flow": 0.0, "max_flow": 330.0},
        "M3": {"name": "Dyeing machine 1", "sensor": "J8", "base_flow": 28.0,
               "standby_flow": 0.4, "startup_factor": 1.45, "alpha": 1.15,
               "beta": 0.0009, "power": 1.38, "min_flow": 0.0, "max_flow": 380.0},
        "M4": {"name": "Dyeing machine 2", "sensor": "J9", "base_flow": 25.0,
               "standby_flow": 0.3, "startup_factor": 1.40, "alpha": 1.12,
               "beta": 0.00085, "power": 1.36, "min_flow": 0.0, "max_flow": 370.0},
        "M5": {"name": "Washing machine", "sensor": "J10", "base_flow": 20.0,
               "standby_flow": 0.2, "startup_factor": 1.25, "alpha": 0.90,
               "beta": 0.0006, "power": 1.32, "min_flow": 0.0, "max_flow": 290.0},
        "M6": {"name": "Finishing machine", "sensor": "J11", "base_flow": 15.0,
               "standby_flow": 0.1, "startup_factor": 1.20, "alpha": 0.70,
               "beta": 0.00045, "power": 1.30, "min_flow": 0.0, "max_flow": 230.0},
        "M7": {"name": "Washing/rinsing machine", "sensor": "J12", "base_flow": 18.0,
               "standby_flow": 0.2, "startup_factor": 1.30, "alpha": 0.85,
               "beta": 0.00055, "power": 1.32, "min_flow": 0.0, "max_flow": 270.0},
        "M8": {"name": "Utility/process machine", "sensor": "J13", "base_flow": 10.0,
               "standby_flow": 0.2, "startup_factor": 1.15, "alpha": 0.45,
               "beta": 0.0003, "power": 1.25, "min_flow": 0.0, "max_flow": 150.0},
    }
    TAP_SPECS = {
        "T1": {"sensor": "J14", "nominal_flow": 22.0, "flow_std": 2.5},
        "T2": {"sensor": "J15", "nominal_flow": 18.0, "flow_std": 2.0},
        "T3": {"sensor": "J16", "nominal_flow": 14.0, "flow_std": 1.5},
    }
    PRESSURE_CONFIG = {
        "inlet_base_pressure": 5.4,
        "source_flow_friction": 0.00045,
        "pipe_friction_loss": {"J2": 0.00032, "J3": 0.00035,
                               "J4": 0.00030, "J7": 0.00028},
        "min_plausible_pressure": 2.5,
        "max_plausible_pressure": 6.2,
    }
    FLOW_NOISE_PERCENT = 2.0
    PRESSURE_NOISE_BAR = 0.08
    SAMPLING_MINUTES = 5

FLOW_NOISE_FRACTION = FLOW_NOISE_PERCENT / 100.0

# Machine/tap id -> the junction that meters it.
ENDPOINT_SENSOR = {mid: spec["sensor"] for mid, spec in MACHINE_SPECS.items()}
ENDPOINT_SENSOR.update({tid: spec["sensor"] for tid, spec in TAP_SPECS.items()})

# The classifier only has pressure sensors at the headers.
PRESSURE_SENSORS = ["J1", "J2", "J3", "J4", "J7"]
