"""
feature_engineering.py
Feature engineering pipeline for ML training dataset generation.
Computes mass balances, temporal metrics, rolling window statistics,
production-normalized expected flows, and residuals without target leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import config

def compute_expected_machine_flow(machine_id: str, prod_rate: float, status: str) -> float:
    """
    Calculates expected baseline water demand for a machine given its production rate and status.
    This does NOT use any leak information (no target leakage).
    """
    spec = config.MACHINE_SPECS[machine_id]
    if status == "OFF":
        return spec["standby_flow"]
    elif status == "MAINTENANCE":
        return spec["standby_flow"] * 4.0 + 5.0
    elif status == "STOPPING":
        return spec["base_flow"] * 0.5
    elif status == "STARTING":
        prod_val = max(0.0, prod_rate)
        nominal_prod_flow = spec["base_flow"] + spec["alpha"] * prod_val + spec["beta"] * (prod_val ** spec["power"])
        return nominal_prod_flow * spec["startup_factor"]
    else:  # RUNNING
        prod_val = max(0.0, prod_rate)
        return spec["base_flow"] + spec["alpha"] * prod_val + spec["beta"] * (prod_val ** spec["power"])

def compute_expected_tap_flow(tap_id: str, status: str) -> float:
    """Expected tap flow given status (independent of machines)."""
    spec = config.TAP_SPECS[tap_id]
    return spec["nominal_flow"] if status == "OPEN" else 0.0

def build_feature_engineered_dataset(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the feature-engineered dataset for ML training.
    Strictly avoids target leakage: input features do not depend on leak labels.
    Uses dictionary batching to prevent DataFrame fragmentation.
    """
    new_cols: Dict[str, Any] = {}

    # 1. Temporal & Shift Features
    dt_series = pd.to_datetime(master_df["date"] + " " + master_df["time"])
    new_cols["hour"] = dt_series.dt.hour
    new_cols["day_of_week"] = dt_series.dt.dayofweek
    # Shifts: Day=1 (06:00-14:00), Evening=2 (14:00-22:00), Night=3 (22:00-06:00)
    new_cols["shift"] = np.where(
        (new_cols["hour"] >= 6) & (new_cols["hour"] < 14), 1,
        np.where((new_cols["hour"] >= 14) & (new_cols["hour"] < 22), 2, 3)
    )

    # 2. Hydraulic Balance Features (Conservation of Mass)
    new_cols["balance_J1"] = master_df["flow_J1"] - (master_df["flow_J2"] + master_df["flow_J3"] + master_df["flow_J4"] + master_df["flow_J7"])
    new_cols["balance_J2"] = master_df["flow_J2"] - (master_df["flow_J5"] + master_df["flow_J6"])
    new_cols["balance_J3"] = master_df["flow_J3"] - (master_df["flow_J8"] + master_df["flow_J9"] + master_df["flow_J10"])
    new_cols["balance_J4"] = master_df["flow_J4"] - (master_df["flow_J11"] + master_df["flow_J12"])
    new_cols["balance_J7"] = master_df["flow_J7"] - (master_df["flow_J13"] + master_df["flow_J14"] + master_df["flow_J15"] + master_df["flow_J16"])

    # 3. Production-Normalized Expected Flows
    for i in range(1, 9):
        m_id = f"M{i}"
        j_sensor = config.MACHINE_SPECS[m_id]["sensor"]
        spec = config.MACHINE_SPECS[m_id]
        
        prod = master_df[f"production_{m_id}"].clip(lower=0.0)
        status = master_df[f"machine_status_{m_id}"]
        running_flow = spec["base_flow"] + spec["alpha"] * prod + spec["beta"] * (prod ** spec["power"])
        
        expected = np.select(
            [
                status == "OFF",
                status == "MAINTENANCE",
                status == "STOPPING",
                status == "STARTING",
                status == "RUNNING"
            ],
            [
                spec["standby_flow"],
                spec["standby_flow"] * 4.0 + 5.0,
                spec["base_flow"] * 0.5,
                running_flow * spec["startup_factor"],
                running_flow
            ],
            default=spec["standby_flow"]
        )
        new_cols[f"expected_flow_{j_sensor}"] = expected

    for i in range(1, 4):
        t_id = f"T{i}"
        j_sensor = config.TAP_SPECS[t_id]["sensor"]
        nom = config.TAP_SPECS[t_id]["nominal_flow"]
        new_cols[f"expected_flow_{j_sensor}"] = np.where(master_df[f"tap_status_{t_id}"] == "OPEN", nom, 0.0)

    # Propagate expected flows upstream
    new_cols["expected_flow_J2"] = new_cols["expected_flow_J5"] + new_cols["expected_flow_J6"]
    new_cols["expected_flow_J3"] = new_cols["expected_flow_J8"] + new_cols["expected_flow_J9"] + new_cols["expected_flow_J10"]
    new_cols["expected_flow_J4"] = new_cols["expected_flow_J11"] + new_cols["expected_flow_J12"]
    new_cols["expected_flow_J7"] = new_cols["expected_flow_J13"] + new_cols["expected_flow_J14"] + new_cols["expected_flow_J15"] + new_cols["expected_flow_J16"]
    new_cols["expected_flow_J1"] = new_cols["expected_flow_J2"] + new_cols["expected_flow_J3"] + new_cols["expected_flow_J4"] + new_cols["expected_flow_J7"]

    # 4. Flow Residuals and Percentage Residuals (Actual - Expected)
    for i in range(1, 17):
        s_id = f"J{i}"
        residual = master_df[f"flow_{s_id}"] - new_cols[f"expected_flow_{s_id}"]
        new_cols[f"flow_residual_{s_id}"] = residual
        new_cols[f"flow_residual_percent_{s_id}"] = (
            (residual / (np.abs(new_cols[f"expected_flow_{s_id}"]) + 1.0)) * 100.0
        ).clip(-1000.0, 1000.0)

    # 5. Dynamics (First differences)
    for i in range(1, 17):
        s_id = f"J{i}"
        new_cols[f"flow_change_{s_id}"] = master_df[f"flow_{s_id}"].diff().fillna(0.0)

    for p_id in ["pressure_J1", "pressure_J2", "pressure_J3", "pressure_J4", "pressure_J7"]:
        if p_id in master_df.columns:
            new_cols[f"{p_id}_change"] = master_df[p_id].diff().fillna(0.0)

    # 6. Rolling Statistics (window of 12 timestamps = 1 hour at 5-min intervals)
    window = 12
    for i in range(1, 17):
        s_id = f"J{i}"
        new_cols[f"rolling_mean_flow_{s_id}"] = master_df[f"flow_{s_id}"].rolling(window=window, min_periods=1).mean()
        new_cols[f"rolling_std_flow_{s_id}"] = master_df[f"flow_{s_id}"].rolling(window=window, min_periods=1).std().fillna(0.0)

    for p_id in ["pressure_J1", "pressure_J2", "pressure_J3", "pressure_J4", "pressure_J7"]:
        if p_id in master_df.columns:
            new_cols[f"rolling_mean_{p_id}"] = master_df[p_id].rolling(window=window, min_periods=1).mean()
            new_cols[f"rolling_std_{p_id}"] = master_df[p_id].rolling(window=window, min_periods=1).std().fillna(0.0)

    # Combine original non-target features, new engineered features, and targets at the end
    target_cols = ["leak", "leak_rate", "leak_zone"]
    base_features = [c for c in master_df.columns if c not in target_cols]
    
    engineered_df = pd.DataFrame(new_cols, index=master_df.index)
    final_df = pd.concat([master_df[base_features], engineered_df, master_df[target_cols]], axis=1)

    return final_df
