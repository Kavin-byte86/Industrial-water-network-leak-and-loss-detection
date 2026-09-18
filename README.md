# Industrial Water Network Leak & Loss Detection

**End-to-end system for detecting, localising, and quantifying water leaks in industrial distribution networks.**

This repository delivers three integrated pillars:

1. **Synthetic Dataset Generator** — a physically correlated, production-aware engine that simulates a 1-year telemetry stream for a textile wet-processing plant's water network.
2. **4-Stage AI/ML Classifier** — a trained pipeline that detects leaks from flow residuals, localises the zone, and estimates the loss rate.
3. **Live Simulation Backend + Dashboard** — a FastAPI simulation engine with physics-based and ML-based detection running concurrently, connected to a React frontend for real-time monitoring and a test bench for leak injection.

---

## Table of Contents

- [1. Project Objective & Core Philosophy](#1-project-objective--core-philosophy)
- [2. Network Topology & Hydraulic Architecture](#2-network-topology--hydraulic-architecture)
- [3. Sensor Definitions](#3-sensor-definitions)
- [4. Textile Equipment & Consumption Models](#4-textile-equipment--consumption-models)
- [5. Leak Generation & Physics Propagation](#5-leak-generation--physics-propagation)
- [6. Synthetic Dataset Generator](#6-synthetic-dataset-generator)
- [7. 4-Stage AI/ML Classifier Pipeline](#7-4-stage-aiml-classifier-pipeline)
- [8. Backend Simulation Engine](#8-backend-simulation-engine)
- [9. Frontend Dashboard](#9-frontend-dashboard)
- [10. Running the Full Stack](#10-running-the-full-stack)
- [11. Test Suite](#11-test-suite)
- [12. External Validation (LeakDB)](#12-external-validation-leakdb)
- [13. Repository Structure](#13-repository-structure)
- [14. Known Limitations & Roadmap](#14-known-limitations--roadmap)

---

## 1. Project Objective & Core Philosophy

The system trains and serves a multivariate machine learning model capable of:

1. **Detecting** whether a water leak is occurring in the industrial water network.
2. **Quantifying** the leak rate (L/min).
3. **Localising** the probable leak zone (`ZONE_J1` through `ZONE_J16`).

### Critical Requirement: High Flow ≠ Leak

In an industrial textile wet-processing facility, water demand is strongly governed by the production schedules of heavy-duty processing machines.

- At **100% aggregate production**: expected main incoming flow ≈ 1 000 L/min.
- At **200% aggregate production**: expected main incoming flow ≈ 2 000 L/min.

An increase from 1 000 to 2 000 L/min due to elevated production **must not** be labelled as a leak. A leak is flagged only when observed flow significantly exceeds the expected consumption calculated from current machine production rates, machine states, and tap usage — or when localised mass-balance and pressure anomalies occur across junction headers.

This production-awareness is the project's central design constraint and is enforced at every layer: dataset generation, feature engineering, model training, and live detection.

---

## 2. Network Topology & Hydraulic Architecture

The facility water network is a tree-structured hydraulic graph branching from a single main inlet (J1) into four primary distribution manifolds and terminal units.

```
                                [ J1: Main Water Inlet ]
                                           │
       ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
       ▼                   ▼                               ▼                   ▼
 [ J2: Branch A ]    [ J3: Branch B ]                [ J4: Branch C ]    [ J7: Branch D ]
    ├── J5  (M1)        ├── J8  (M3)                    ├── J11 (M6)        ├── J13 (M8)
    └── J6  (M2)        ├── J9  (M4)                    └── J12 (M7)        ├── J14 (Tap 1)
                        └── J10 (M5)                                        ├── J15 (Tap 2)
                                                                            └── J16 (Tap 3)
```

### Conservation of Mass

At all times, the network adheres to hydraulic conservation laws:

- `J1 ≈ J2 + J3 + J4 + J7`
- `J2 ≈ J5 + J6`
- `J3 ≈ J8 + J9 + J10`
- `J4 ≈ J11 + J12`
- `J7 ≈ J13 + J14 + J15 + J16`

During normal operations, mass-balance residuals remain near zero (subject only to minor measurement noise). When a leak occurs downstream of any junction, the upstream sensors reflect the added flow, creating measurable localised mass-balance deficits that the detection system exploits.

> **Note on scalability:** The current topology has 16 junctions, 8 machines, and 3 taps. This is a representative configuration for one factory section. The topology definition (`backend/app/network/topology.py`, `classifier/topology.py`) is data-driven — adding machines, taps, or entire branch manifolds requires only configuration changes, not code changes.

---

## 3. Sensor Definitions

### 16 Flow Measurement Points

| Sensor | Location | Parent | Downstream | Unit |
|--------|----------|--------|------------|------|
| **J1** | Main Inlet Header | Source | J2, J3, J4, J7 | L/min |
| **J2** | Branch A Manifold | J1 | J5, J6 | L/min |
| **J3** | Branch B Manifold | J1 | J8, J9, J10 | L/min |
| **J4** | Branch C Manifold | J1 | J11, J12 | L/min |
| **J5** | Machine M1 Feed | J2 | M1 | L/min |
| **J6** | Machine M2 Feed | J2 | M2 | L/min |
| **J7** | Branch D Utility | J1 | J13, J14, J15, J16 | L/min |
| **J8** | Machine M3 Feed | J3 | M3 | L/min |
| **J9** | Machine M4 Feed | J3 | M4 | L/min |
| **J10** | Machine M5 Feed | J3 | M5 | L/min |
| **J11** | Machine M6 Feed | J4 | M6 | L/min |
| **J12** | Machine M7 Feed | J4 | M7 | L/min |
| **J13** | Machine M8 Feed | J7 | M8 | L/min |
| **J14** | Utility Washdown Tap 1 | J7 | Tap 1 | L/min |
| **J15** | Equipment Cleaning Tap 2 | J7 | Tap 2 | L/min |
| **J16** | Sampling & Dilution Tap 3 | J7 | Tap 3 | L/min |

### Pressure Sensors

Pressure measurements are captured at all 16 junctions. Pressures typically range between 3.0–6.0 bar, dropping with increased flow velocity (Darcy-Weisbach friction loss) and undergoing localised pressure drops during pipe rupture or leak events. The pressure model is per-branch with configurable friction coefficients (`classifier/config.py → PRESSURE_CONFIG`).

---

## 4. Textile Equipment & Consumption Models

### 8 Wet-Processing Machines

Each machine has independent production rates (0–200%) and distinct operational states (`OFF`, `STARTING`, `RUNNING`, `STOPPING`, `MAINTENANCE`).

The flow demand follows a non-linear curve with per-machine coefficients:

```
flow = base_flow + α × production + β × production^γ + ε
```

| Machine | Process | α | Nominal Flow | Base Flow |
|---------|---------|---|-------------|-----------|
| **M1** | Scouring (alkaline wash) | 0.95 | ~130 L/min | 22 L/min |
| **M2** | Bleaching (H₂O₂ line) | 1.05 | ~145 L/min | 26 L/min |
| **M3** | Jet Dyeing (autoclave) | 1.15 | ~160 L/min | 28 L/min |
| **M4** | Overflow Dyeing | 1.12 | ~155 L/min | 25 L/min |
| **M5** | Counter-current Washing | 0.90 | ~125 L/min | 20 L/min |
| **M6** | Chemical Finishing | 0.70 | ~95 L/min | 15 L/min |
| **M7** | Post-dye Rinsing | 0.85 | ~115 L/min | 18 L/min |
| **M8** | Boiler Feed / Cooling | 0.45 | ~60 L/min | 10 L/min |

State-dependent behaviour:
- **OFF** → standby draw only (0.2–0.5 L/min)
- **STARTING** → running flow × startup factor (1.2–1.5×, startup draws *more*)
- **RUNNING** → the demand curve above
- **STOPPING** → half of base flow
- **MAINTENANCE** → standby × 4 + 5 L/min (flushing)

### 3 Utility Taps

Taps operate independently of machine production schedules, representing cleaning routines, shift handovers (06:00, 14:00, 22:00), equipment washdowns, and quality control sampling. Each has a configurable nominal flow and standard deviation.

---

## 5. Leak Generation & Physics Propagation

### Leak Types (Dataset Generator)

- **Frequency**: 30–60 events per year (default: ~46 events, covering 5–12% of the year).
- **Types**:
  1. *Small gradual leak*: ramping up slowly over hours (10–50 L/min)
  2. *Small sudden leak*: abrupt step change (10–50 L/min)
  3. *Medium leak*: continuous pipe defect (50–150 L/min)
  4. *Large leak*: major burst/rupture (150–480 L/min) with pronounced pressure drop
  5. *Progressive leak*: accelerating pipe degradation
  6. *Intermittent leak*: pressure-dependent or cyclical leakage

### Topological Propagation

Leak water is added to the affected node and propagated to all ancestors without inflating downstream machine demand:

```
Leak at J3 → J1 ↑, J3 ↑; downstream J8, J9, J10 unchanged
           → balance_J3 = flow_J3 − (J8 + J9 + J10) = leak_rate
```

This is the signal the mass-balance detector exploits. Concurrent leaks at different nodes produce independent residuals, so multiple leaks are detected and reported separately.

---

## 6. Synthetic Dataset Generator

Located in [`classifier/`](classifier/). Generates a full 1-year telemetry dataset from configurable parameters.

### Quick Start

```bash
cd classifier
pip install -r requirements.txt
python generator.py        # generates the dataset
python validation.py       # runs 16 automated validation checks
```

### Configuration (`classifier/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `START_DATE` / `END_DATE` | 2025-01-01 → 2025-12-31 | Temporal horizon |
| `SAMPLING_MINUTES` | 5 | Sampling resolution |
| `FLOW_NOISE_PERCENT` | 2% | Gaussian sensor noise σ |
| `PRESSURE_NOISE_BAR` | 0.05 | Pressure noise σ |
| `MISSING_DATA_PERCENT` | 0.1% | Data corruption rate |
| `OUTLIER_PERCENT` | 0.05% | Non-leak spike rate |
| `LEAK_CONFIG` | 30–60 events | Event count, severity ranges, duration |
| `MACHINE_SPECS` | per-machine | Flow curves, coefficients, limits |

### Generated Output

```
synthetic_textile_water_dataset/
├── sensors/sensor_J01.csv … sensor_J16.csv
├── machines/machine_M01.csv … machine_M08.csv
├── taps/tap_T01.csv … tap_T03.csv
├── network_master.csv                    # Unified multi-sensor master
├── ml_training_dataset.csv               # Feature-engineered ML training set
├── leak_events.csv                       # Ground-truth leak catalog
├── sensor_metadata.csv                   # Topological metadata
├── network_topology.json                 # Graph representation
└── plots/                                # 7 validation plots
```

---

## 7. 4-Stage AI/ML Classifier Pipeline

The detector is a 4-stage pipeline in [`classifier/`](classifier/):

| Stage | Model | Purpose |
|-------|-------|---------|
| 1 | 16 × Ridge Regression (degree-2 polynomial) | Expected flow per sensor from production rates and machine/tap states. Fit on **normal (`leak == 0`) rows only**. |
| 2 | XGBoost Binary Classifier | Leak/no-leak from Stage-1 residuals plus mass balances, pressures, and rolling statistics. |
| 3 | XGBoost Multiclass Classifier | Which of the 16 zones ruptured. |
| 4 | XGBoost Regressor | Leak rate in L/min. |

**Key design constraint:** Stage 1 never sees a leak during training, keeping its residuals meaningful when one occurs. Splits are strictly chronological (70/15/15) and target columns are excluded from every feature set.

### Training

```bash
# v1 baseline: all four stages
python classifier/training/train_all.py

# v2: adds temporal lag/deviation features, tunes scale_pos_weight and
# the decision threshold, writes reports under classifier/reports/
python classifier/training/optimize_pipeline.py
```

`classifier/features.py` holds the v2 feature block and is imported by **both** training and inference, ensuring the two cannot drift apart.

### Inference

```python
from classifier.predict import WaterNetworkLeakDetector

detector = WaterNetworkLeakDetector(models_dir="classifier/models/v2")

detector.predict(state_dict)        # one network state (~240 ms)
detector.predict_batch(dataframe)   # vectorised (~3 ms/row, 75× faster)
```

`predict_batch` is preferred: it runs each stage once for the whole batch and is also more accurate for v2, whose lag features need surrounding rows.

### Live Stream Replay

```bash
python classifier/predict_stream.py --start 100000 --steps 100
```

### Benchmark Dashboard

```bash
pip install -r classifier/requirements.txt
streamlit run benchmark_app.py
```

Four tabs: methodology, live test-set accuracy (scored on load, not hardcoded), the LeakDB external benchmark, and an interactive leak-injection simulator.

---

## 8. Backend Simulation Engine

A FastAPI-based real-time simulation that reproduces the same physics the dataset generator uses, with two detection engines running concurrently.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SimulationEngine                          │
│                                                             │
│  1. Clock.advance()          → simulated time               │
│  2. ProductionModel          → per-machine/tap flow demand  │
│  3. LeakRegistry             → injected leak rates          │
│  4. Aggregation              → bottom-up junction sums      │
│  5. PressureModel            → per-junction bar values      │
│  6. Sensor Overrides         → manual forcing (testbench)   │
│  7. Expected (reconcile)     → production vs. measured      │
│  8. Detection (mass-balance) → physics-based leak finding   │
│  9. LossTracker              → volume & cost accounting     │
│ 10. ML Detector              → trained 4-stage pipeline     │
│ 11. → Snapshot → SessionStore → API                        │
└─────────────────────────────────────────────────────────────┘
```

### Tick Pipeline (every 2 seconds real-time = 5 simulated minutes)

Each tick executes the following steps atomically (thread-locked):

1. **Clock advance** — increments simulated time by `TICK_SIMULATED_MINUTES`.
2. **Production model** — computes each machine's and tap's water demand using the same non-linear curves from `classifier/config.py` (single source of truth).
3. **Leak injection** — any leaks set via the test-bench API are added at their target nodes.
4. **Bottom-up aggregation** — walks the topology tree from leaves to root, summing child flows and adding Gaussian measurement noise.
5. **Pressure derivation** — per-branch friction-based pressure model using topology-specific coefficients.
6. **Sensor overrides** — manual sensor forcing from the test bench (simulates faulty/tampered meters).
7. **Production reconciliation** — compares measured total flow against what the production schedule accounts for. The difference (`unexplained_lpm`) is the only part that can be a loss.
8. **Mass-balance detection** — deterministic physics-based detector that computes residuals at every junction with noise-aware thresholds, corroborated by pressure drops and historical baselines.
9. **Loss accounting** — each detected leak becomes an event with accumulated duration, volume lost (litres), and financial cost (₹/hour, projected daily/annual).
10. **ML detection** — the trained 4-stage classifier runs on the same tick history, buffered to supply rolling statistics and lag features.
11. **Snapshot** — everything is bundled and pushed to the session store's ring buffer.

### Physics-Based Detection vs. ML Detection

The backend runs **both** detectors in parallel:

| Aspect | Mass-Balance Detector | ML Classifier |
|--------|----------------------|---------------|
| Method | Exact conservation-of-mass residuals | Trained XGBoost on residuals + rolling features |
| Training required | No | Yes (4-stage pipeline) |
| Handles concurrent leaks | Yes (independent residuals) | Single primary leak |
| Confidence source | Signal-to-noise ratio | Model probability |
| History required | ≥12 ticks for baseline | ≥18 ticks for warming up |

The UI can display both results side-by-side for cross-validation.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/network/topology` | GET | Static topology graph |
| `/state/current` | GET | Latest tick snapshot (flows, pressures, detection, ML, production, losses) |
| `/state/history?limit=N` | GET | Last N tick snapshots |
| `/control/machine` | POST | Set machine production % and state |
| `/control/tap` | POST | Open/close a tap |
| `/simulation/step` | POST | Advance one tick manually |
| `/simulation/pause` | POST | Pause auto-tick loop |
| `/simulation/resume` | POST | Resume auto-tick loop |
| `/simulation/reset` | POST | Reset simulation to t=0 |
| `/datasink/latest` | GET | Latest tick in flat ML-schema format |
| `/datasink/export?since_ticks=N` | GET | Batch export in flat ML-schema format |
| `/predict/current` | GET | ML classifier result for current state |
| `/testbench/leak` | POST | Inject a leak at a junction (rate in L/min) |
| `/testbench/leak/{node_id}` | DELETE | Clear a specific leak |
| `/testbench/leaks/clear` | POST | Clear all leaks |
| `/testbench/leaks` | GET | List active injected leaks |
| `/testbench/override` | POST | Force a sensor reading |
| `/testbench/override/{key}` | DELETE | Clear a sensor override |
| `/testbench/overrides/clear` | POST | Clear all overrides |
| `/testbench/endpoint` | POST | Set machine/tap state in one call |
| `/testbench/snapshot` | GET | Combined test-bench state snapshot |

### Loss Accounting

When a leak is detected, the `LossTracker` accumulates:

```
volume_litres += rate_lpm × minutes_per_tick
cost = volume_m³ × (supply_tariff + effluent_tariff)
```

Leaked process water is charged twice — once to buy it (₹45/m³ default), once to treat it as effluent (₹25/m³ default). The API surfaces real-time cost per hour, projected daily/annual cost, and total volume lost since the leak started.

---

## 9. Frontend Dashboard

A React + Vite single-page application in [`frontend/`](frontend/) with two interfaces:

### Main Dashboard (port 5173)

| View | Description |
|------|-------------|
| **Pipeline Skeleton** | SVG P&ID-style schematic with animated flow dashes proportional to rate. Circles = junctions, squares = machines, triangles = taps. |
| **Facility Map** | 2D top-down factory floor layout grouped by process zones (Pre-treatment, Dyeing, Finishing, Utility). Click any machine/tap for inline controls. |
| **Live Flow Data** | Data-dense tables of all 16 junction flows, 8 machine states, 3 tap states with inline SVG sparkline trends. |
| **Historical Data** | Canvas-based multi-series line chart with junction selector, raw tick data table with pagination. |
| **Leak Alerts** | Real-time leak detection overlay with severity, rate, confidence, and cost. |
| **Production Panel** | Production reconciliation: measured vs. expected flow, explained percentage. |
| **Loss Summary** | Financial impact: volume lost, cost so far, projected daily/annual cost. |
| **ML Verdict** | Trained classifier output: leak probability, predicted zone, estimated rate. |

### Test Bench (port 5173/testbench.html)

A separate page for injecting leaks and forcing sensor readings to validate detection:
- Select any junction and set a leak rate (L/min)
- Override individual sensor values
- Set machine/tap states directly
- Watch detection response in real-time

---

## 10. Running the Full Stack

### Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt    # Windows
# .venv/bin/pip install -r requirements.txt      # Linux/macOS
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

Serves on `http://127.0.0.1:8000`. Interactive API docs at `/docs`. The background auto-tick loop starts immediately (2s real-time = 5 simulated minutes).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Serves on `http://localhost:5173`. The Vite proxy forwards `/api/*` to the backend automatically.

### Single-Origin Deployment

Build the frontend and let the backend serve it:

```bash
cd frontend && npm run build
cd ../backend
python -m uvicorn app.main:app --port 8000
# → UI + API both at http://localhost:8000
```

### Docker

```bash
cd backend && docker build -t water-network-backend .
docker run -p 8000:8000 water-network-backend
```

### Environment Variables

All configurable via env vars (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `TICK_INTERVAL_SECONDS` | 2 | Real-time seconds between ticks |
| `BASE_DATETIME` | 2026-01-01T00:00:00 | Simulation start time |
| `NOISE_SIGMA_FRACTION` | 0.01 | Measurement noise σ |
| `BASE_PRESSURE` | 6.0 | Nominal pressure (bar) |
| `WATER_COST_PER_M3` | 45.0 | Supply tariff (₹/m³) |
| `EFFLUENT_COST_PER_M3` | 25.0 | Effluent tariff (₹/m³) |
| `HISTORY_BUFFER_SIZE` | 1000 | Ring buffer capacity |
| `CLASSIFIER_VERSION` | v2 | Model version directory |
| `VITE_BACKEND_URL` | (proxy) | Backend URL for frontend |
| `VITE_POLL_INTERVAL_MS` | 2000 | Frontend polling interval |

---

## 11. Test Suite

```bash
cd backend
python -m pytest tests -q
```

| Test Module | Tests | What It Validates |
|-------------|-------|-------------------|
| `test_balance.py` | Conservation of mass across topology |
| `test_control.py` | Machine/tap control API and state persistence |
| `test_production_scaling.py` | Flow scaling with production % |
| `test_leak_detection.py` | Mass-balance detection accuracy, concurrent leaks, severity bands |
| `test_loss_and_production.py` | Loss accounting, cost calculation, production reconciliation |
| `test_ml_detector.py` | ML detector warmup, schema compatibility, graceful degradation |

---

## 12. External Validation (LeakDB)

The synthetic benchmark partly measures self-consistency, so the methodology is also validated against the public **LeakDB** municipal benchmark (Hanoi CMH):

```bash
git clone https://github.com/KIOS-Research/LeakDB.git external_validation/LeakDB
python external_validation/run_leakdb_test.py
```

This regenerates [`external_validation/leakdb_report.md`](external_validation/leakdb_report.md) entirely from result CSVs.

**Caveat:** LeakDB is a municipal network with no production schedule. A good result there validates the *hydraulic residual methodology*, not the production-aware textile pipeline as a whole.

---

## 13. Repository Structure

```
├── classifier/                          # Dataset generator + ML pipeline
│   ├── config.py                        #   Physics constants (single source of truth)
│   ├── generator.py                     #   1-year synthetic data generator
│   ├── topology.py                      #   Network graph definition
│   ├── feature_engineering.py           #   ML feature pipeline
│   ├── features.py                      #   v2 lag/deviation features
│   ├── predict.py                       #   WaterNetworkLeakDetector inference
│   ├── predict_stream.py               #   SCADA stream replay
│   ├── validation.py                    #   16 automated validation checks
│   ├── training/                        #   train_all.py, optimize_pipeline.py
│   ├── models/                          #   Trained model artifacts (.pkl, gitignored)
│   ├── reports/                         #   Performance reports
│   └── requirements.txt
│
├── backend/                             # FastAPI simulation engine
│   ├── app/
│   │   ├── main.py                      #   App entry, router mounting, SPA serving
│   │   ├── dependencies.py              #   Shared store + engine singletons
│   │   ├── classifier_config.py         #   Bridge to classifier/config.py constants
│   │   ├── ml_detector.py              #   Live 4-stage ML inference on tick history
│   │   ├── leak_extension_point.py     #   LeakRegistry for test-bench injection
│   │   ├── core/config.py              #   Env-driven runtime configuration
│   │   ├── network/topology.py         #   Static topology definition
│   │   ├── simulation/
│   │   │   ├── engine.py               #     Tick pipeline orchestrator
│   │   │   ├── clock.py                #     Simulated time management
│   │   │   ├── production_model.py     #     Per-machine/tap flow curves
│   │   │   ├── aggregation.py          #     Bottom-up junction flow summation
│   │   │   ├── pressure_model.py       #     Per-branch friction pressure
│   │   │   ├── expected.py             #     Production reconciliation
│   │   │   ├── detection.py            #     Mass-balance leak detector
│   │   │   └── loss_tracker.py         #     Volume & cost accumulation
│   │   ├── api/                        #     FastAPI route handlers
│   │   │   ├── network.py, state.py, control.py, simulation.py
│   │   │   ├── datasink.py, predict.py, testbench.py
│   │   └── state/session_store.py      #     In-memory ring buffer
│   ├── tests/                           #   pytest test suite
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                            # React + Vite dashboard
│   ├── src/
│   │   ├── App.jsx                      #   Root component, view routing
│   │   ├── api/client.js               #   API client (all endpoints)
│   │   ├── hooks/                       #   useNetworkState, useHistory, usePolling
│   │   ├── components/
│   │   │   ├── Sidebar/                #     Navigation + simulation controls
│   │   │   ├── PipelineSkeleton/       #     SVG P&ID schematic
│   │   │   ├── FacilityMap/            #     2D factory floor layout
│   │   │   ├── LiveFlowPanel/          #     Data tables + sparklines
│   │   │   ├── HistoryView/            #     Chart + raw data table
│   │   │   ├── LeakAlert/              #     Detection overlay
│   │   │   ├── ProductionPanel/        #     Measured vs. expected
│   │   │   ├── LossSummary/            #     Financial impact
│   │   │   └── MLVerdict/              #     Classifier output
│   │   ├── testbench/                   #   Test-bench UI
│   │   └── styles/theme.css            #   Design system (industrial blue)
│   ├── testbench.html
│   └── package.json
│
├── external_validation/                 # LeakDB municipal benchmark
├── benchmark_app.py                     # Streamlit benchmark dashboard
├── .env.example                         # Environment template
├── .gitignore
└── README.md
```

---

## 14. Known Limitations & Roadmap

### Current Limitations

- **Small leaks are undetectable by construction.** At 2% flow noise on a ~1 000 L/min main, the noise floor is ~20 L/min. Leaks below ~30 L/min (SNR < 1.5) are buried in it. See [`classifier/reports/DETECTABILITY_ANALYSIS.md`](classifier/reports/DETECTABILITY_ANALYSIS.md).
- **ML classifier requires trained models.** The `.pkl` model files are gitignored due to size. Run the training pipeline (`classifier/training/train_all.py`) against a generated dataset to produce them.
- **ML warmup period.** The trained classifier needs ≥18 ticks of history before its rolling features are meaningful. During warmup it reports `warming_up` status.
- **Reports state their scope.** Threshold-sweep numbers are validation-set and optimistic by construction; the test-set table in `MODEL_OPTIMIZATION_REPORT.md` is the honest estimate.

### Verification Cases

| Case | Condition | Flow Behavior | Label | Rationale |
|------|-----------|---------------|-------|-----------|
| 1 | Production = 100% | Main ≈ 900–1 000 L/min | `leak = 0` | Legitimate production demand |
| 2 | Production = 200% | Main ≈ 1 500–2 000 L/min | `leak = 0` | Legitimate doubled production |
| 3 | Production = 200% | Main far exceeds expected | `leak = 1` | Unaccounted water above expected |
| 4 | Constant production | Localised branch imbalance | `leak = 1` | Mass-balance violation |
