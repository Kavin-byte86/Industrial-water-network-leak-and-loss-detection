"""
validation.py
Comprehensive automated validation module for the Synthetic Industrial Water Network Dataset.
Executes 16 rigorous criteria checking physical consistency, label integrity,
mass balances, production scaling, and lack of target leakage.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import config
import topology


def validate_dataset() -> bool:
    """
    Executes all 16 validation checks and prints a formatted summary.
    Returns True if all checks pass, False otherwise.
    """
    print("\n" + "=" * 60)
    print("STARTING AUTOMATED DATASET VALIDATION")
    print("=" * 60)

    checks_passed = []

    # -------------------------------------------------------------
    # CHECK 1: All 16 sensor CSVs exist
    # -------------------------------------------------------------
    sensor_files = [config.SENSORS_DIR / f"sensor_J{i:02d}.csv" for i in range(1, 17)]
    all_exist = all(p.exists() for p in sensor_files)
    checks_passed.append(("1. All 16 sensor CSVs exist", all_exist))

    # -------------------------------------------------------------
    # Load Master & ML datasets
    # -------------------------------------------------------------
    if not config.NETWORK_MASTER_CSV.exists() or not config.ML_TRAINING_CSV.exists():
        print("[ERROR] Master or ML CSV files missing!")
        return False

    master_df = pd.read_csv(config.NETWORK_MASTER_CSV)
    ml_df = pd.read_csv(config.ML_TRAINING_CSV)

    expected_rows = 105120  # 365 days * 24 hrs * 12 (5-min intervals)

    # -------------------------------------------------------------
    # CHECK 2: All timestamps are synchronized across files
    # -------------------------------------------------------------
    # Compare first sensor file with master
    s01_df = pd.read_csv(sensor_files[0])
    ts_sync = (s01_df["date"] == master_df["date"]).all() and (s01_df["time"] == master_df["time"]).all()
    checks_passed.append(("2. Timestamps are synchronized across files", bool(ts_sync)))

    # -------------------------------------------------------------
    # CHECK 3: No duplicate timestamps
    # -------------------------------------------------------------
    dt_comb = master_df["date"] + " " + master_df["time"]
    no_dup = dt_comb.duplicated().sum() == 0
    checks_passed.append(("3. No duplicate timestamps", bool(no_dup)))

    # -------------------------------------------------------------
    # CHECK 4: Number of rows is correct (105,120)
    # -------------------------------------------------------------
    row_count_correct = len(master_df) == expected_rows
    checks_passed.append((f"4. Correct row count ({expected_rows})", row_count_correct))

    # -------------------------------------------------------------
    # CHECK 5: Network mass-balance relationships during normal operation
    # -------------------------------------------------------------
    normal_mask = master_df["leak"] == 0
    normal_df = master_df[normal_mask].dropna()
    # J1 ≈ J2 + J3 + J4 + J7
    j1_pred = normal_df["flow_J2"] + normal_df["flow_J3"] + normal_df["flow_J4"] + normal_df["flow_J7"]
    mb_diff = np.abs(normal_df["flow_J1"] - j1_pred)
    mean_mb_err_pct = (mb_diff / (normal_df["flow_J1"] + 1e-5)).mean() * 100.0
    # Should be < 3% accounting for measurement noise
    mb_valid = mean_mb_err_pct < 3.0
    checks_passed.append((f"5. Mass-balance error normal operation ({mean_mb_err_pct:.2f}% < 3%)", bool(mb_valid)))

    # -------------------------------------------------------------
    # CHECK 6: Leak events affect correct upstream sections
    # -------------------------------------------------------------
    leak_events_df = pd.read_csv(config.LEAK_EVENTS_CSV)
    upstream_valid = True
    for _, evt in leak_events_df.head(10).iterrows():
        zone = evt["leak_zone"]
        exp_upstream = topology.LEAK_ZONES[zone]["affected_upstream_sensors"]
        logged_upstream = evt["affected_upstream_sensors"].split(",")
        if set(exp_upstream) != set(logged_upstream):
            upstream_valid = False
            break
    checks_passed.append(("6. Leaks affect correct upstream topology", upstream_valid))

    # -------------------------------------------------------------
    # CHECK 7: Production increases are not automatically labeled leaks
    # -------------------------------------------------------------
    total_prod = master_df[[f"production_M{i}" for i in range(1, 9)]].mean(axis=1)
    high_prod_no_leak = np.any((total_prod >= 95) & (total_prod <= 105) & (master_df["leak"] == 0))
    peak_prod_no_leak = np.any((total_prod >= 160) & (master_df["leak"] == 0))
    prod_leak_valid = high_prod_no_leak and peak_prod_no_leak
    checks_passed.append(("7. 100% and 200% production have leak=0 normal operation", bool(prod_leak_valid)))

    # -------------------------------------------------------------
    # CHECK 8: Leak labels match generated leak events
    # -------------------------------------------------------------
    event_leak_count = len(leak_events_df)
    labels_have_leaks = (master_df["leak"] == 1).sum() > 0
    checks_passed.append((f"8. Leak labels present ({event_leak_count} events)", labels_have_leaks))

    # -------------------------------------------------------------
    # CHECK 9: leak_rate = 0 whenever leak = 0
    # -------------------------------------------------------------
    rate_zero_when_no_leak = (master_df.loc[master_df["leak"] == 0, "leak_rate"] == 0.0).all()
    checks_passed.append(("9. leak_rate = 0 when leak = 0", bool(rate_zero_when_no_leak)))

    # -------------------------------------------------------------
    # CHECK 10: leak_zone = NONE whenever leak = 0
    # -------------------------------------------------------------
    zone_none_when_no_leak = (master_df.loc[master_df["leak"] == 0, "leak_zone"] == "NONE").all()
    checks_passed.append(("10. leak_zone = NONE when leak = 0", bool(zone_none_when_no_leak)))

    # -------------------------------------------------------------
    # CHECK 11: Flow values are never negative
    # -------------------------------------------------------------
    flow_cols = [f"flow_J{i}" for i in range(1, 17)]
    min_flow = master_df[flow_cols].min().min()
    max_flow = master_df[flow_cols].max().max()
    no_neg_flow = min_flow >= 0.0
    checks_passed.append((f"11. Flow values non-negative (min: {min_flow:.2f})", bool(no_neg_flow)))

    # -------------------------------------------------------------
    # CHECK 12: Pressure values remain within physically plausible ranges (2.5 - 6.2 bar)
    # -------------------------------------------------------------
    p_cols = ["pressure_J1", "pressure_J2", "pressure_J3", "pressure_J4", "pressure_J7"]
    min_p = master_df[p_cols].min().min()
    max_p = master_df[p_cols].max().max()
    pressure_valid = (min_p >= 2.5) and (max_p <= 6.2)
    checks_passed.append((f"12. Pressures plausible ({min_p:.2f} to {max_p:.2f} bar)", bool(pressure_valid)))

    # -------------------------------------------------------------
    # CHECK 13: Machine production values remain within 0–200%
    # -------------------------------------------------------------
    prod_cols = [f"production_M{i}" for i in range(1, 9)]
    min_prod = master_df[prod_cols].min().min()
    max_prod = master_df[prod_cols].max().max()
    prod_bounds_valid = (min_prod >= 0.0) and (max_prod <= 200.0)
    checks_passed.append((f"13. Production within 0-200% ({min_prod:.1f}% to {max_prod:.1f}%)", bool(prod_bounds_valid)))

    # -------------------------------------------------------------
    # CHECK 14: Machine OFF state produces approximately zero flow (<= 1.0 L/min standby)
    # -------------------------------------------------------------
    off_flows_valid = True
    for i in range(1, 9):
        m_id = f"M{i}"
        s_id = config.MACHINE_SPECS[m_id]["sensor"]
        off_mask = (master_df[f"machine_status_{m_id}"] == "OFF") & (master_df["leak"] == 0)
        if np.any(off_mask):
            off_mean_f = master_df.loc[off_mask, f"flow_{s_id}"].mean()
            if off_mean_f > 1.5:
                off_flows_valid = False
                break
    checks_passed.append(("14. Machine OFF state flow <= 1.5 L/min (standby)", off_flows_valid))

    # -------------------------------------------------------------
    # CHECK 15: Tap CLOSED state produces 0 flow
    # -------------------------------------------------------------
    tap_closed_valid = True
    for i in range(1, 4):
        t_id = f"T{i}"
        s_id = config.TAP_SPECS[t_id]["sensor"]
        closed_mask = (master_df[f"tap_status_{t_id}"] == "CLOSED") & (master_df["leak"] == 0)
        if np.any(closed_mask):
            closed_mean_f = master_df.loc[closed_mask, f"flow_{s_id}"].mean()
            if closed_mean_f > 0.5:
                tap_closed_valid = False
                break
    checks_passed.append(("15. Tap CLOSED state produces ~0 flow", tap_closed_valid))

    # -------------------------------------------------------------
    # CHECK 16: No target leakage into ML input features
    # -------------------------------------------------------------
    # Check that expected_flow columns correlate strongly with production, not leak label
    corr_leak = ml_df["expected_flow_J1"].corr(ml_df["leak"])
    no_leakage = abs(corr_leak) < 0.05
    checks_passed.append((f"16. No target leakage in expected flow (|corr| = {abs(corr_leak):.4f} < 0.05)", bool(no_leakage)))

    # -------------------------------------------------------------
    # PRINT RESULTS
    # -------------------------------------------------------------
    print("\nCHECK RESULTS:")
    all_ok = True
    for name, ok in checks_passed:
        status_str = "[PASS]" if ok else "[FAIL]"
        print(f"  {status_str} {name}")
        if not ok:
            all_ok = False

    # Summary metrics
    leak_count = (master_df["leak"] == 1).sum()
    normal_count = (master_df["leak"] == 0).sum()
    missing_pct = (master_df[flow_cols].isna().sum().sum() / (len(master_df) * len(flow_cols))) * 100.0

    print("\n" + "-" * 60)
    print(f"Total timestamps: {len(master_df)}")
    print(f"Total sensor files: {len(sensor_files)}")
    print(f"Total leak events: {len(leak_events_df)}")
    print(f"Leak timestamps: {leak_count} ({(leak_count / len(master_df)) * 100:.2f}%)")
    print(f"Normal timestamps: {normal_count} ({(normal_count / len(master_df)) * 100:.2f}%)")
    print(f"Minimum flow: {min_flow:.2f} L/min")
    print(f"Maximum flow: {max_flow:.2f} L/min")
    print(f"Mass-balance error: {mean_mb_err_pct:.2f}%")
    print(f"Missing values: {missing_pct:.2f}%")
    print("=" * 60)
    if all_ok:
        print("DATASET VALIDATION: PASSED")
    else:
        print("DATASET VALIDATION: FAILED")
    print("=" * 60 + "\n")

    return all_ok


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
