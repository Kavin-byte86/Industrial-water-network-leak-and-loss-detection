# Industrial Water Network Leak & Loss Detection
## Realistic 1-Year Synthetic Dataset Generator for AI/ML Leak Detection

This repository contains a physically correlated, production-aware synthetic data generation engine simulating an industrial water distribution network for a representative textile wet-processing manufacturing plant.

---

## 1. Project Objective & Core Philosophy

The primary purpose of this dataset is to train a multivariate machine learning model capable of:
1. **Detecting** whether a water leak is occurring in the industrial water network.
2. **Quantifying** the leak rate (L/min).
3. **Localizing** the probable leak zone (`ZONE_J1` through `ZONE_J16`).

### ⚠️ Critical Requirement: High Flow $\neq$ Leak
In an industrial textile wet-processing facility, water demand is strongly governed by the production schedules of eight heavy-duty processing machines. 
- At **100% aggregate production**: Expected main incoming water flow $\approx 1000\text{ L/min}$.
- At **200% aggregate production**: Expected main incoming water flow $\approx 2000\text{ L/min}$.

An increase from 1000 to approximately 2000 L/min due to elevated production **MUST NOT** be labeled as a leak. 
A leak is flagged (`leak = 1`) **only** when observed water flow significantly exceeds the expected consumption calculated from current machine production, machine states, and tap usage, or when localized mass-balance and pressure anomalies occur across junction headers.

---

## 2. Network Topology & Hydraulic Architecture

The facility water network follows a tree-structured hydraulic graph branching from a single main inlet ($J_1$) into four primary distribution manifolds and terminal units.

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

### Physical Conservation of Mass
At all times, the network adheres to hydraulic conservation laws:
$$J_1 \approx J_2 + J_3 + J_4 + J_7$$
$$J_2 \approx J_5 + J_6$$
$$J_3 \approx J_8 + J_9 + J_{10}$$
$$J_4 \approx J_{11} + J_{12}$$
$$J_7 \approx J_{13} + J_{14} + J_{15} + J_{16}$$

During normal operations, mass-balance residuals remain near zero (subject only to minor measurement noise). When a leak occurs downstream of any junction, the appropriate upstream sensors reflect the added flow, creating measurable localized mass-balance deficits.

---

## 3. Sensor Definitions (16 Flow Measurement Points)

| Sensor ID | Location | Type | Parent Node | Downstream Nodes | Measurement Unit |
|---|---|---|---|---|---|
| **J1** | Main Inlet Header | Ultrasonic Flow | Source | J2, J3, J4, J7 | L/min |
| **J2** | Branch A Manifold | Electromagnetic Flow | J1 | J5, J6 | L/min |
| **J3** | Branch B Manifold | Electromagnetic Flow | J1 | J8, J9, J10 | L/min |
| **J4** | Branch C Manifold | Electromagnetic Flow | J1 | J11, J12 | L/min |
| **J5** | Machine M1 Feed | Electromagnetic Flow | J2 | M1 | L/min |
| **J6** | Machine M2 Feed | Electromagnetic Flow | J2 | M2 | L/min |
| **J7** | Branch D Utility | Electromagnetic Flow | J1 | J13, J14, J15, J16 | L/min |
| **J8** | Machine M3 Feed | Electromagnetic Flow | J3 | M3 | L/min |
| **J9** | Machine M4 Feed | Electromagnetic Flow | J3 | M4 | L/min |
| **J10** | Machine M5 Feed | Electromagnetic Flow | J3 | M5 | L/min |
| **J11** | Machine M6 Feed | Electromagnetic Flow | J4 | M6 | L/min |
| **J12** | Machine M7 Feed | Electromagnetic Flow | J4 | M7 | L/min |
| **J13** | Machine M8 Feed | Electromagnetic Flow | J7 | M8 | L/min |
| **J14** | Utility Washdown Tap 1 | Turbine Flow | J7 | Tap 1 | L/min |
| **J15** | Equipment Cleaning Tap 2 | Turbine Flow | J7 | Tap 2 | L/min |
| **J16** | Sampling & Dilution Tap 3 | Turbine Flow | J7 | Tap 3 | L/min |

### Pressure Sensors
Pressure measurements are captured at the primary headers: `pressure_J1`, `pressure_J2`, `pressure_J3`, `pressure_J4`, and `pressure_J7`. Pressures typically range between **3.0 and 6.0 bar**, dropping with increased flow velocity (Darcy-Weisbach friction loss) and undergoing localized pressure drops during pipe rupture or leak events.

---

## 4. Textile Equipment & Consumption Models

### 8 Wet-Processing Machines
Each machine possesses independent production rates ($0\%$ to $200\%$) and distinct operational states (`OFF`, `STARTING`, `RUNNING`, `STOPPING`, `MAINTENANCE`).

$$\text{flow}_i = \text{base\_flow}_i + \alpha_i \cdot \text{prod}_i + \beta_i \cdot (\text{prod}_i)^{\gamma_i} + \epsilon$$

1. **M1 (Scouring Machine)**: Alkaline wetting and scour. High chemical wash flow ($\alpha=0.95$, nominal: 130 L/min).
2. **M2 (Bleaching Machine)**: Continuous hydrogen peroxide bleach line ($\alpha=1.05$, nominal: 145 L/min).
3. **M3 (Dyeing Machine 1)**: High-temperature jet dyeing autoclave ($\alpha=1.15$, nominal: 160 L/min).
4. **M4 (Dyeing Machine 2)**: Atmospheric overflow dyeing vessel ($\alpha=1.12$, nominal: 155 L/min).
5. **M5 (Washing Machine)**: Multi-stage counter-current wash range ($\alpha=0.90$, nominal: 125 L/min).
6. **M6 (Finishing Machine)**: Chemical finish applicator and stenter padder ($\alpha=0.70$, nominal: 95 L/min).
7. **M7 (Washing/Rinsing Machine)**: Post-dye neutralizer and soaping range ($\alpha=0.85$, nominal: 115 L/min).
8. **M8 (Utility/Process Machine)**: Steam boiler makeup and heat exchange cooling ($\alpha=0.45$, nominal: 60 L/min).

### 3 Utility Taps
Taps operates independently of machine production schedules, representing cleaning routines, shift handovers (06:00, 14:00, 22:00), equipment washdowns, and quality control sampling.

---

## 5. Leak Generation & Physics Propagation

- **Frequency**: 30–60 events per year (default: ~46 events, covering 5–12% of the year).
- **Leak Types**:
  1. *Small gradual leak*: Ramping up slowly over hours (10–50 L/min).
  2. *Small sudden leak*: Abrupt step change (10–50 L/min).
  3. *Medium leak*: Continuous pipe defect (50–150 L/min).
  4. *Large leak*: Major burst / rupture (150–480 L/min) accompanied by pronounced pressure drop.
  5. *Progressive leak*: Accelerating pipe degradation.
  6. *Intermittent leak*: Pressure-dependent or cyclical leakage.
- **Topological Propagation**: Leak water added to upstream parents without inflating downstream machines.
  - Leak at $J_3$ header $\implies J_1 \uparrow$, $J_3 \uparrow$; downstream $J_8, J_9, J_{10}$ remain at machine demand $\implies \text{balance}_{J3} = \text{leak\_rate} > 0$.

---

## 6. Generated Directory & File Structure

```
synthetic_textile_water_dataset/
├── sensors/
│   ├── sensor_J01.csv ... sensor_J16.csv   # Synchronized telemetry per sensor
├── machines/
│   ├── machine_M01.csv ... machine_M08.csv # Production rate, status, water demand
├── taps/
│   ├── tap_T01.csv ... tap_T03.csv         # Tap state and flow
├── network_master.csv                      # Unified multi-sensor master dataset
├── ml_training_dataset.csv                 # Feature-engineered ML training dataset
├── leak_events.csv                         # Ground-truth leak event catalog
├── sensor_metadata.csv                     # Topological metadata
├── network_topology.json                   # Network graph representation
├── README.md                               # Dataset summary document
└── plots/                                  # 7 automated validation plots
    ├── main_flow_vs_production.png
    ├── production_vs_expected_flow.png
    ├── normal_vs_leak_behavior.png
    ├── pressure_during_leak.png
    ├── leak_event_across_sensors.png
    ├── network_flow_balance.png
    └── machine_production_vs_demand.png
```

---

## 7. How to Run & Customizing Simulation Parameters

### Quick Start (Local)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run simulation generator
python generator.py

# 3. Run automated validation checks
python validation.py
```

### Running with Docker (Cross-Device Support)
Build and execute the complete pipeline on any device or cloud container:
```bash
# Build docker container
docker build -t textile-water-leak-detection .

# Run dataset generation and validation
docker run --rm -v $(pwd)/synthetic_textile_water_dataset:/workspace/synthetic_textile_water_dataset textile-water-leak-detection
```

### Customizing Simulation Parameters (`config.py`)
Edit `config.py` to change:
- `START_DATE`, `END_DATE`, `SAMPLING_MINUTES`: Adjust temporal horizon and sampling resolution.
- `FLOW_NOISE_PERCENT`, `PRESSURE_NOISE_BAR`: Fine-tune sensor measurement noise.
- `MISSING_DATA_PERCENT`, `OUTLIER_PERCENT`: Tune data corruption and non-leak spikes.
- `LEAK_CONFIG`: Adjust event count, severity ranges, and duration.
- `MACHINE_SPECS`: Modify machine names, baseline flows, and production power coefficients.

---

## 8. Verification & Demonstration Cases

The system includes automated demonstrations validating production-aware behavior:

| Case | Condition | Flow Behavior | Label | Rationale |
|---|---|---|---|---|
| **Case 1** | Production = 100% | Main Flow $\approx 900-1000\text{ L/min}$ | `leak = 0` | Legitimate production demand |
| **Case 2** | Production = 200% | Main Flow $\approx 1500-2000\text{ L/min}$ | `leak = 0` | Legitimate doubled production |
| **Case 3** | Production = 200% | Main Flow significantly exceeds expected | `leak = 1` | Unaccounted water detected above expected |
| **Case 4** | Constant Production | Localized branch imbalance ($\Delta Q$) | `leak = 1` | Topological mass-balance violation |

All **16 Automated Validation Checks** pass with $0$ errors.

---


---

## 9. Running the Full Stack (Backend + Frontend)

The live demo is two processes: a FastAPI simulation backend and a React (Vite)
frontend that polls it.

### Backend — FastAPI simulator

```bash
cd backend
python -m venv .venv                     # Python 3.8+ works; Docker image uses 3.11
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

Serves on `http://127.0.0.1:8000`; interactive API docs at `/docs`. A background
loop auto-ticks every 2 s (= 5 simulated minutes) as soon as the app starts.

Run the test suite with `.venv/Scripts/python -m pytest tests -q` (17 tests).

### Frontend — React + Vite

```bash
cd frontend
npm install
cp .env.example .env     # optional, see below
npm run dev
```

Serves on `http://localhost:5173`.

### How the two connect

There are two working paths, and the frontend picks one automatically:

| `VITE_BACKEND_URL` | Request path | Mechanism |
|---|---|---|
| set (e.g. `http://127.0.0.1:8000`) | direct to backend | backend CORS allows all origins |
| unset | `/api/*` | Vite dev-server proxy in `vite.config.js` (same-origin, no CORS) |

The proxy path is the better default for local development; set the env var when
the backend runs on another host or port.

Endpoints consumed by the UI ([`frontend/src/api/client.js`](frontend/src/api/client.js)):

| Endpoint | Used by |
|---|---|
| `GET /state/current` | `useNetworkState` — polls every 2 s, drives every view |
| `GET /state/history?limit=N` | `useHistory` — polls every 5 s for the history view |
| `GET /network/topology` | static graph for rendering |
| `POST /control/machine`, `POST /control/tap` | Facility Map controls |
| `POST /simulation/{step,pause,resume,reset}` | Sidebar controls |
| `GET /datasink/latest`, `GET /datasink/export` | flat ML-schema rows |
| `GET /predict/current` | classifier output — **see the caveat below** |

### Docker

```bash
cd backend && docker build -t water-network-backend . && docker run -p 8000:8000 water-network-backend
```

### Known gap: `/predict/current` is not usable yet

The backend simulator and the trained classifier were built to different
specifications, so the live prediction path is wrong in a way that *looks* like it
works. On entirely normal input it returns a high-confidence leak.

1. `/datasink/latest` emits `shift` as `"A"/"B"/"C"`; the classifier was trained on
   `1/2/3`, so the call currently fails with a conversion error.
2. Even with that fixed, the flat row supplies only 31 of the 147 Stage-2 features.
   Stage 1 derives 48 more, leaving ~68 zero-filled — including every rolling
   statistic and all five mass balances, which are the core leak signal.
3. The backend's physics (uniform 125 L/min machines, linear pressure model) does
   not match `classifier/config.py` (per-machine alpha/beta curves, per-branch friction),
   so Stage 1's expected-flow model is calibrated for a different plant.

Fixing this needs a design decision: either drive the backend from
`classifier/config.py`'s physics, or buffer ticks and call `predict_batch` over a
window so the rolling and balance features can actually be computed. Until then the
UI does not surface predictions.

## 10. Machine Learning Pipeline

The detector is a 4-stage pipeline in [`classifier/`](classifier/):

| Stage | Model | Purpose |
|---|---|---|
| 1 | 16 x Ridge (degree-2 polynomial) | Expected flow per sensor from production rates and machine/tap states. Fit on **normal (`leak == 0`) rows only**. |
| 2 | XGBoost binary classifier | Leak / no-leak, from Stage-1 residuals plus mass balances, pressures and rolling statistics. |
| 3 | XGBoost multiclass | Which of the 16 zones ruptured. |
| 4 | XGBoost regressor | Leak rate in L/min. |

Stage 1 never sees a leak during training, which is what keeps its residuals
meaningful when one occurs. Splits are strictly chronological (70/15/15) and the
target columns are excluded from every feature set.

### Training

```bash
# v1 baseline: all four stages
python classifier/training/train_all.py

# v2: adds temporal lag / deviation features, tunes scale_pos_weight and
# the decision threshold, and writes the reports under classifier/reports/
python classifier/training/optimize_pipeline.py
```

`classifier/features.py` holds the v2 feature block and is imported by **both**
training and inference, so the two cannot drift apart.

### Inference

```python
from classifier.predict import WaterNetworkLeakDetector

detector = WaterNetworkLeakDetector(models_dir="classifier/models/v2")

detector.predict(state_dict)        # one network state
detector.predict_batch(dataframe)   # a window of telemetry, vectorised
```

Prefer `predict_batch` wherever possible: it is roughly 75x faster per row (~3 ms
vs ~240 ms) because each stage runs once for the whole batch. It is also the more
accurate call when using v2, whose lag features need surrounding rows to exist.

Replay the dataset as a simulated SCADA feed with:

```bash
python classifier/predict_stream.py --start 100000 --steps 100
```

### Benchmark dashboard

```bash
pip install -r requirements.txt
streamlit run benchmark_app.py
```

Four tabs: methodology, live test-set accuracy (scored on load, not hardcoded),
the LeakDB external benchmark, and an interactive leak-injection simulator.

---

## 11. External Validation (LeakDB)

The synthetic benchmark partly measures self-consistency, so the methodology is
also validated against the public **LeakDB** municipal benchmark (Hanoi CMH). The
benchmark data is large and third-party, so it is not committed - fetch it first:

```bash
git clone https://github.com/KIOS-Research/LeakDB.git external_validation/LeakDB
python external_validation/run_leakdb_test.py
```

This regenerates [`external_validation/leakdb_report.md`](external_validation/leakdb_report.md)
entirely from the result CSVs, so the report cannot drift from the artifacts it
describes.

**Caveat worth stating plainly:** LeakDB is a municipal network with no production
schedule. A good result there validates the *hydraulic residual methodology*, not
the production-aware textile pipeline as a whole.

---

## 12. Known Limitations

- **Small leaks are undetectable by construction.** At 2% flow noise on a
  ~1000 L/min main, the noise floor is ~20 L/min. Leaks below roughly 30 L/min
  (SNR > 1.5) are buried in it; most missed leaks in the test split are in this
  band. See [`classifier/reports/DETECTABILITY_ANALYSIS.md`](classifier/reports/DETECTABILITY_ANALYSIS.md).
- **`backend/` and `frontend/` are scaffolding only.** All working code lives at
  the repository root, in `classifier/` and in `external_validation/`; the
  Streamlit dashboard is the current front end.
- **Reports state their scope.** Threshold-sweep numbers are validation-set and
  optimistic by construction; the test-set table in
  `MODEL_OPTIMIZATION_REPORT.md` is the honest estimate.
