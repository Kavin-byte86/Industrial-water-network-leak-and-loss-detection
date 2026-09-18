import os
import sys
import pickle
import pandas as pd
import numpy as np

# config.py is in classifier/ (one level up from training/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from feature_engineering import compute_expected_machine_flow, compute_expected_tap_flow

MODELS_V2_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'v2')

def create_base_state(prod_level, tap_open=False, extra_flow_j1=0.0, is_leak=0):
    row = {"hour": 12, "day_of_week": 2, "shift": 1}
    for i in range(1, 9):
        row[f"production_M{i}"] = float(prod_level)
        row[f"machine_status_M{i}"] = "RUNNING"
    for i in range(1, 4):
        if tap_open and i == 1:
            row[f"tap_status_T{i}"] = "OPEN"
        else:
            row[f"tap_status_T{i}"] = "CLOSED"
    
    expected_flows = {}
    for i in range(1, 9):
        m_id = f"M{i}"
        expected_flows[config.MACHINE_SPECS[m_id]["sensor"]] = compute_expected_machine_flow(m_id, prod_level, "RUNNING")
    for i in range(1, 4):
        t_id = f"T{i}"
        status = "OPEN" if (tap_open and i == 1) else "CLOSED"
        expected_flows[config.TAP_SPECS[t_id]["sensor"]] = compute_expected_tap_flow(t_id, status)
        
    expected_flows["J2"] = expected_flows["J5"] + expected_flows["J6"]
    expected_flows["J3"] = expected_flows["J8"] + expected_flows["J9"] + expected_flows["J10"]
    expected_flows["J4"] = expected_flows["J11"] + expected_flows["J12"]
    expected_flows["J7"] = expected_flows["J13"] + expected_flows["J14"] + expected_flows["J15"] + expected_flows["J16"]
    expected_flows["J1"] = expected_flows["J2"] + expected_flows["J3"] + expected_flows["J4"] + expected_flows["J7"]
    
    for j in range(1, 17):
        row[f"flow_J{j}"] = expected_flows[f"J{j}"] + 0.5
        row[f"rolling_mean_flow_J{j}"] = row[f"flow_J{j}"]
        row[f"rolling_median_flow_J{j}"] = row[f"flow_J{j}"]
        row[f"short_term_dev_flow_J{j}"] = 0.0
        row[f"rolling_std_flow_J{j}"] = 0.5
        row[f"flow_change_J{j}"] = 0.0
        row[f"flow_J{j}_lag1"] = row[f"flow_J{j}"]
        row[f"flow_J{j}_lag3"] = row[f"flow_J{j}"]
        row[f"flow_J{j}_lag6"] = row[f"flow_J{j}"]
        
    for p in ["J1", "J2", "J3", "J4", "J7"]:
        row[f"pressure_{p}"] = 5.0
        row[f"pressure_{p}_change"] = 0.0
        row[f"rolling_mean_pressure_{p}"] = 5.0
        row[f"rolling_std_pressure_{p}"] = 0.01
    for b in ["J1", "J2", "J3", "J4", "J7"]:
        row[f"balance_{b}"] = 0.0
        
    # Inject anomalies
    if extra_flow_j1 > 0:
        row["flow_J1"] += extra_flow_j1
        if is_leak:
            row["balance_J1"] += extra_flow_j1
            row["pressure_J1"] -= 0.5
            row["pressure_J1_change"] = -0.5
            
    row["leak"] = is_leak
    return pd.DataFrame([row])

def run_tests():
    print("=======================================================================")
    print("PRODUCTION INVARIANCE & GENERALIZATION TESTS (v2)")
    print("=======================================================================")
    
    with open(os.path.join(MODELS_V2_DIR, 'stage1_expected_flow_models.pkl'), 'rb') as f:
        s1_data = pickle.load(f)
    with open(os.path.join(MODELS_V2_DIR, 'stage2_leak_classifier.pkl'), 'rb') as f:
        s2_data = pickle.load(f)
        
    s2_model = s2_data["model"]
    s2_features = s2_data["features"]
    threshold = s2_data["threshold"]
    
    cases = [
        ("CASE 1", 50, False, 0.0, 0, "50% normal"),
        ("CASE 2", 100, False, 0.0, 0, "100% normal"),
        ("CASE 3", 150, False, 0.0, 0, "150% normal"),
        ("CASE 4", 200, False, 0.0, 0, "200% normal"),
        ("CASE 5", 200, False, 100.0, 0, "200% abnormal high flow (NO LEAK)"),
        ("CASE 6", 200, False, 250.0, 1, "200% normal flow + genuine leak"),
        ("CASE 7", 100, True, 0.0, 0, "100% + tap opens (NO LEAK)"),
        ("CASE 8", 100, False, 150.0, 1, "100% + pipe leak -> imbalance"),
    ]
    
    for case_name, prod, tap, extra, is_leak, desc in cases:
        df_row = create_base_state(prod, tap, extra, is_leak)
        
        # S1 Prediction
        X_s1 = df_row[s1_data["numeric_features"] + s1_data["categorical_features"]]
        for j in range(1, 17):
            pred = s1_data["models"][f"flow_J{j}"].predict(X_s1)[0]
            df_row[f"ml_expected_flow_J{j}"] = pred
            res = df_row[f"flow_J{j}"] - pred
            df_row[f"ml_flow_residual_J{j}"] = res
            df_row[f"ml_flow_residual_percent_J{j}"] = (res / (np.abs(pred) + 1.0)) * 100.0
            
        # Add dynamic features needed by v2
        total_prod = df_row[[f"production_M{i}" for i in range(1, 9)]].sum(axis=1) + 1.0
        for i in range(1, 17):
            s_id = f"J{i}"
            df_row[f'flow_per_prod_{s_id}'] = df_row[f'flow_{s_id}'] / total_prod
            df_row[f'residual_per_prod_{s_id}'] = df_row[f'ml_flow_residual_{s_id}'] / total_prod
            
        # Stage 2 Prediction
        for col in s2_features:
            if col not in df_row.columns:
                df_row[col] = 0.0
                
        X_s2 = df_row[s2_features]
        prob = s2_model.predict_proba(X_s2)[0, 1]
        pred_label = 1 if prob >= threshold else 0
        
        exp_j1 = df_row["ml_expected_flow_J1"].iloc[0]
        act_j1 = df_row["flow_J1"].iloc[0]
        
        expected_label = is_leak
        if case_name == "CASE 5":
            # We injected extra flow but didn't inject mass imbalance. The model SHOULD recognize
            # that flow - expected > 0, but balance == 0, so it's NOT a pipe leak, it's a sensor anomaly.
            pass
            
        result = "PASS" if pred_label == expected_label else "FAIL"
        
        print(f"\n{case_name}: {desc}")
        print(f"Production: {prod}% | Expected J1: {exp_j1:.1f} | Actual J1: {act_j1:.1f}")
        print(f"Leak Prob: {prob:.4f} (Thresh: {threshold:.2f}) -> Pred: {pred_label} (Expected: {expected_label}) [{result}]")

if __name__ == "__main__":
    run_tests()
