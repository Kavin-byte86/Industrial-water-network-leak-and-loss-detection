# LeakDB External Validation Report
_Generated 2026-09-18 21:45 by `python external_validation/run_leakdb_test.py`._
_Every number below is read from `external_validation/expected_flow_experiment/*.csv`._

## 1. Objective
Independently validate the core methodology of the textile leak-detection system
(`Expected hydraulic behavior -> Residuals -> Leak Detection`) against the public
LeakDB water distribution benchmark.

**Scope caveat**: LeakDB is a *municipal* distribution benchmark with no production
schedule. A good result here validates the *hydraulic residual methodology*. It does
not prove the textile-specific, production-aware pipeline transfers as-is.

## 2. Dataset Overview
- **Network**: Hanoi_CMH
- **Sampling**: native 1-hour data upsampled to **5-minute** resolution to match the textile pipeline
- **Total Scenarios**: 10 (8760 hours / 1 year each)
- **Train**: ['Scenario-8', 'Scenario-10', 'Scenario-5', 'Scenario-1', 'Scenario-7', 'Scenario-2', 'Scenario-9']
- **Validation**: ['Scenario-4']
- **Test**: ['Scenario-3', 'Scenario-6']
- **Leakage check**: PASS - splits fall on whole-scenario boundaries; `leak`,
  `leak_zone` and `leak_rate` are excluded from every feature set.

## 3. Method
- **Expected flow**: per-link regression from total network demand + time features,
  fit exclusively on non-leak scenarios.
- **Residual**: `actual flow - expected flow`.
- **Network balance**: `sum(inflow) - sum(outflow) - demand` at each junction.

## 4. Expected-Flow Model Quality
Per-link regressors (Ridge / RandomForest / XGBoost, best-of-three per link) were fit
on non-leak data only. 15 of 34 links are predicted to within 5%
relative error. The poorly-predicted links are low-magnitude branches where absolute
error stays small but relative error inflates.

## 5. Multi-Baseline Comparison (Test Scenarios)
Does adding residuals and balances beat raw flow/pressure?

```text
                       Model  Precision   Recall       F1  ROC-AUC   PR-AUC  False_Positives  False_Negatives
                  A_FlowOnly   0.370763 0.269320 0.312003 0.601847 0.300377            22499            35967
           B_NewResidualOnly   0.471271 0.808142 0.595357 0.858997 0.746587            44630             9444
       C_NewResidualPressure   0.997573 0.434280 0.605126 0.849790 0.773300               52            27847
D_NewResidualPressureBalance   0.998725 0.524947 0.688177 0.843307 0.778505               33            23384
```

Raw flow alone (`A_FlowOnly`) reaches F1 0.3120. The best residual-based
feature set reaches F1 0.6882 - the residual methodology is what
carries the signal.

## 6. Best Model: `D_NewResidualPressureBalance`
- **Precision**: 0.9987
- **Recall**: 0.5249
- **F1 Score**: 0.6882
- **ROC-AUC**: 0.8433
- **PR-AUC**: 0.7785
- **False Positives**: 33
- **False Negatives**: 23384

## 7. Event-Level Detection (`D_NewResidualPressureBalance`)
2 of 2 leak scenarios detected.

- **Scenario-3**: DETECTED | time-to-detection 0.00 h | 52.8% of leak duration flagged | 9 false alarms before leak onset
- **Scenario-6**: DETECTED | time-to-detection 0.00 h | 51.4% of leak duration flagged | 9 false alarms before leak onset

Full per-model event breakdown:
```text
  Scenario                        Model  Has_Leak  Detected  TTD_Hrs  Pct_Detected  False_Alarms_Before
Scenario-3                   A_FlowOnly      True      True 6.250000     33.851693                 4989
Scenario-3            B_NewResidualOnly      True      True 0.000000     87.860681                11909
Scenario-3        C_NewResidualPressure      True      True 0.000000     51.526990                   14
Scenario-3 D_NewResidualPressureBalance      True      True 0.000000     52.785864                    9
Scenario-6                   A_FlowOnly      True     False      NaN      0.000000                    0
Scenario-6            B_NewResidualOnly      True      True 0.000000     53.388988                   99
Scenario-6        C_NewResidualPressure      True      True 5.166667     11.906182                    0
Scenario-6 D_NewResidualPressureBalance      True      True 0.000000     51.361558                    9
```

## 9. Appendix: Superseded 1-Hour Baseline Run
The original evaluation used raw 1-hour LeakDB sampling and a single global
expected-flow model. Retained for comparison only; the results above supersede it.
```text
                    Model  Precision   Recall       F1   PR-AUC
               A_FlowOnly   0.383766 0.275475 0.320727 0.329659
           B_FlowPressure   0.368757 0.246587 0.295544 0.334306
             C_FlowDemand   0.376464 0.258532 0.306547 0.318334
           D_ResidualOnly   0.370964 0.428571 0.397693 0.341221
E_ResidualPressureBalance   0.349427 0.208191 0.260923 0.343537
        F_IsolationForest   0.196361 0.038152 0.063891 0.232691
```

## 10. Reproducibility
```bash
git clone https://github.com/KIOS-Research/LeakDB.git external_validation/LeakDB
python external_validation/run_leakdb_test.py
```
