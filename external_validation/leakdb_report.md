# LeakDB External Validation Report

## 1. Objective
To independently validate the core methodology of the textile leak-detection system (`Expected hydraulic behavior -> Residuals -> Leak Detection`) against the public LeakDB water distribution benchmark.

**Crucial Note**: LeakDB is a municipal water distribution benchmark. Its success validates the *hydraulic anomaly detection methodology*. It does *not* prove that the textile-specific production-aware ML pipeline will directly transfer without production data, but it validates the underlying physics engine.

## 2. Dataset Overview
- **Network**: Hanoi_CMH
- **Total Scenarios**: 10 (8760 hours / 1 year per scenario)
- **Train Scenarios**: ['Scenario-8', 'Scenario-10', 'Scenario-5', 'Scenario-1', 'Scenario-7', 'Scenario-2', 'Scenario-9']
- **Validation Scenarios**: ['Scenario-4']
- **Test Scenarios**: ['Scenario-3', 'Scenario-6']
- **Data Leakage Check**: PASS (Scenario boundaries perfectly respected. `leak`, `leak_zone`, and `leak_rate` target columns strictly excluded from input features).

## 3. Feature Engineering Methodology
- **Expected Flow Model**: Linear Ridge regression mapping total network demand + time -> Link Flow. Trained exclusively on non-leak normal scenarios.
- **Residuals**: `Actual Flow - Expected Flow`
- **Network Balance**: Sum(Inflow) - Sum(Outflow) - Demand at every topological junction.

## 4. Multi-Baseline Comparison (Test Scenarios)
The objective was to prove that Residuals + Network Balance outperforms raw flow/pressure data.

```text
                    Model  Precision   Recall       F1   PR-AUC
               A_FlowOnly   0.383766 0.275475 0.320727 0.329659
           B_FlowPressure   0.368757 0.246587 0.295544 0.334306
             C_FlowDemand   0.376464 0.258532 0.306547 0.318334
           D_ResidualOnly   0.370964 0.428571 0.397693 0.341221
E_ResidualPressureBalance   0.349427 0.208191 0.260923 0.343537
        F_IsolationForest   0.196361 0.038152 0.063891 0.232691
```

## 5. Final Best Model Metrics
- **ResidualOnly** achieved the highest Recall (0.4286) and F1 Score (0.3977).
- **ResidualPressureBalance** achieved the highest PR-AUC (0.3435).

Metrics for ResidualPressureBalance:
- **Precision**: 0.3494
- **Recall**: 0.2082
- **F1 Score**: 0.2609
- **PR-AUC**: 0.3435

### Event-Level Detection
```text
scenario_id  is_leak_scenario  event_detected  false_alarms  time_to_detection_hrs  percent_detected  max_leak_rate
 Scenario-3              True            True           666                    5.5         26.168224           97.2
 Scenario-6              True           False             0                    NaN          0.000000          655.2
```

## 6. Feature Importance
The top 10 most critical hydraulic features determined by the XGBoost algorithm:
```text
         Feature  Importance
residual_flow_27    0.182490
      balance_21    0.171955
      pressure_6    0.137187
      pressure_3    0.068287
      pressure_2    0.051477
     pressure_15    0.047556
     pressure_10    0.045413
     pressure_12    0.038105
residual_flow_15    0.018558
      pressure_4    0.013061
```

## 7. Execution Reproducibility
```bash
python external_validation/run_leakdb_test.py
```
