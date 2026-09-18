"""
config.py
Configuration parameters for the Synthetic Industrial Water Network Dataset Generator.
Textile Wet-Processing Manufacturing Plant Simulation.
"""

from pathlib import Path

# ==============================================================================
# 1. SIMULATION TIMEFRAME & RESOLUTION
# ==============================================================================
START_DATE = "2025-01-01 00:00:00"
END_DATE = "2025-12-31 23:55:00"
SAMPLING_MINUTES = 5
RANDOM_SEED = 42

# Total expected timestamps for 365 days at 5-minute sampling: 365 * 24 * 12 = 105,120
DAYS = 365

# ==============================================================================
# 2. OUTPUT DIRECTORIES & PATHS
# ==============================================================================
BASE_OUTPUT_DIR = Path("synthetic_textile_water_dataset")
SENSORS_DIR = BASE_OUTPUT_DIR / "sensors"
MACHINES_DIR = BASE_OUTPUT_DIR / "machines"
TAPS_DIR = BASE_OUTPUT_DIR / "taps"
PLOTS_DIR = BASE_OUTPUT_DIR / "plots"

# Key output filenames
NETWORK_MASTER_CSV = BASE_OUTPUT_DIR / "network_master.csv"
ML_TRAINING_CSV = BASE_OUTPUT_DIR / "ml_training_dataset.csv"
LEAK_EVENTS_CSV = BASE_OUTPUT_DIR / "leak_events.csv"
TOPOLOGY_JSON = BASE_OUTPUT_DIR / "network_topology.json"
SENSOR_METADATA_CSV = BASE_OUTPUT_DIR / "sensor_metadata.csv"
DATASET_README_MD = BASE_OUTPUT_DIR / "README.md"

# ==============================================================================
# 3. NOISE & IMPERFECTION PARAMETERS
# ==============================================================================
FLOW_NOISE_PERCENT = 2.0         # Normal measurement noise: ~ +/- 1-3%
PRESSURE_NOISE_BAR = 0.08        # Pressure measurement noise: ~ +/- 0.05-0.15 bar
MISSING_DATA_PERCENT = 0.5       # Percentage of missing sensor readings (NaN): 0.1% - 1%
OUTLIER_PERCENT = 0.3            # Percentage of normal operational spikes/outliers (leak = 0)

# ==============================================================================
# 4. TEXTILE MACHINES SPECIFICATIONS (8 MACHINES)
# Calibrated so that at 100% total production, Main Flow (J1) ≈ 1000 L/min,
# and at 200% production, Main Flow (J1) ≈ 2000 L/min.
# ==============================================================================
MACHINE_SPECS = {
    "M1": {
        "name": "Scouring machine",
        "sensor": "J5",
        "parent_junction": "J2",
        "base_flow": 22.0,           # Flow when running at near 0%
        "standby_flow": 0.2,         # Flow when OFF
        "nominal_flow_100": 130.0,   # Expected flow at 100% production (L/min)
        "max_flow_200": 240.0,       # Expected flow at 200% production (L/min)
        "startup_factor": 1.35,      # Multiplier during STARTING phase
        "alpha": 0.95,               # Linear production coefficient
        "beta": 0.00065,             # Nonlinear production coefficient (prod^1.5)
        "power": 1.35,
        "min_flow": 0.0,
        "max_flow": 300.0,
    },
    "M2": {
        "name": "Bleaching machine",
        "sensor": "J6",
        "parent_junction": "J2",
        "base_flow": 26.0,
        "standby_flow": 0.3,
        "nominal_flow_100": 145.0,
        "max_flow_200": 265.0,
        "startup_factor": 1.30,
        "alpha": 1.05,
        "beta": 0.0007,
        "power": 1.30,
        "min_flow": 0.0,
        "max_flow": 330.0,
    },
    "M3": {
        "name": "Dyeing machine 1",
        "sensor": "J8",
        "parent_junction": "J3",
        "base_flow": 28.0,
        "standby_flow": 0.4,
        "nominal_flow_100": 160.0,
        "max_flow_200": 305.0,
        "startup_factor": 1.45,
        "alpha": 1.15,
        "beta": 0.0009,
        "power": 1.38,
        "min_flow": 0.0,
        "max_flow": 380.0,
    },
    "M4": {
        "name": "Dyeing machine 2",
        "sensor": "J9",
        "parent_junction": "J3",
        "base_flow": 25.0,
        "standby_flow": 0.3,
        "nominal_flow_100": 155.0,
        "max_flow_200": 295.0,
        "startup_factor": 1.40,
        "alpha": 1.12,
        "beta": 0.00085,
        "power": 1.36,
        "min_flow": 0.0,
        "max_flow": 370.0,
    },
    "M5": {
        "name": "Washing machine",
        "sensor": "J10",
        "parent_junction": "J3",
        "base_flow": 20.0,
        "standby_flow": 0.2,
        "nominal_flow_100": 125.0,
        "max_flow_200": 230.0,
        "startup_factor": 1.25,
        "alpha": 0.90,
        "beta": 0.0006,
        "power": 1.32,
        "min_flow": 0.0,
        "max_flow": 290.0,
    },
    "M6": {
        "name": "Finishing machine",
        "sensor": "J11",
        "parent_junction": "J4",
        "base_flow": 15.0,
        "standby_flow": 0.1,
        "nominal_flow_100": 95.0,
        "max_flow_200": 180.0,
        "startup_factor": 1.20,
        "alpha": 0.70,
        "beta": 0.00045,
        "power": 1.30,
        "min_flow": 0.0,
        "max_flow": 230.0,
    },
    "M7": {
        "name": "Washing/rinsing machine",
        "sensor": "J12",
        "parent_junction": "J4",
        "base_flow": 18.0,
        "standby_flow": 0.2,
        "nominal_flow_100": 115.0,
        "max_flow_200": 215.0,
        "startup_factor": 1.30,
        "alpha": 0.85,
        "beta": 0.00055,
        "power": 1.32,
        "min_flow": 0.0,
        "max_flow": 270.0,
    },
    "M8": {
        "name": "Utility/process machine",
        "sensor": "J13",
        "parent_junction": "J7",
        "base_flow": 10.0,
        "standby_flow": 0.2,
        "nominal_flow_100": 60.0,
        "max_flow_200": 110.0,
        "startup_factor": 1.15,
        "alpha": 0.45,
        "beta": 0.0003,
        "power": 1.25,
        "min_flow": 0.0,
        "max_flow": 150.0,
    },
}

# ==============================================================================
# 5. WATER TAPS SPECIFICATIONS (3 TAPS)
# Taps are independent of machine production.
# ==============================================================================
TAP_SPECS = {
    "T1": {
        "name": "Utility Washdown Tap 1",
        "sensor": "J14",
        "parent_junction": "J7",
        "nominal_flow": 22.0,    # Flow when OPEN (L/min)
        "closed_flow": 0.0,      # Flow when CLOSED
        "flow_std": 2.5,
    },
    "T2": {
        "name": "Equipment Cleaning Tap 2",
        "sensor": "J15",
        "parent_junction": "J7",
        "nominal_flow": 18.0,
        "closed_flow": 0.0,
        "flow_std": 2.0,
    },
    "T3": {
        "name": "Sampling & Dilution Tap 3",
        "sensor": "J16",
        "parent_junction": "J7",
        "nominal_flow": 14.0,
        "closed_flow": 0.0,
        "flow_std": 1.5,
    },
}

# ==============================================================================
# 6. HYDRAULIC PRESSURE PARAMETERS (bar)
# Realistic prototype values: 3.0 to 6.0 bar.
# ==============================================================================
PRESSURE_CONFIG = {
    "inlet_base_pressure": 5.4,      # Main inlet pressure under zero-flow (bar)
    "source_flow_friction": 0.00045, # Pressure drop at inlet per L/min
    "pipe_friction_loss": {          # Friction drop coefficient per branch
        "J2": 0.00032,
        "J3": 0.00035,
        "J4": 0.00030,
        "J7": 0.00028,
    },
    "min_plausible_pressure": 2.5,
    "max_plausible_pressure": 6.2,
}

# ==============================================================================
# 7. LEAK EVENT INJECTION CONFIGURATION
# 30 to 60 leak events per year, distributed realistically across zones & types.
# ==============================================================================
LEAK_CONFIG = {
    "min_events": 35,
    "max_events": 50,
    "target_leak_percentage_min": 0.05,  # 5% of total time
    "target_leak_percentage_max": 0.12,  # 12% of total time
    "severity_levels": {
        "Small": {"flow_min": 10.0, "flow_max": 50.0},
        "Medium": {"flow_min": 50.0, "flow_max": 150.0},
        "Large": {"flow_min": 150.0, "flow_max": 480.0},
    },
    "types": [
        "Small gradual leak",
        "Small sudden leak",
        "Medium leak",
        "Large leak",
        "Progressive leak",
        "Intermittent leak",
    ],
    "duration_hours": {
        "min": 1.0,
        "max": 36.0,
    },
}
