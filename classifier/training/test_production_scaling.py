import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure config can be loaded
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def run_scaling_test():
    print("--- Running Production Invariance Scaling Test ---")
    
    stage1_path = os.path.join(MODELS_DIR, 'stage1_expected_flow_models.pkl')
    stage2_path = os.path.join(MODELS_DIR, 'stage2_leak_classifier.pkl')
    
    if not os.path.exists(stage1_path) or not os.path.exists(stage2_path):
        print("Models not found. Please run train_all.py first.")
        return
        
    with open(stage1_path, 'rb') as f:
        stage1_data = pickle.load(f)
        stage1_models = stage1_data["models"]
        
    with open(stage2_path, 'rb') as f:
        stage2_data = pickle.load(f)
        stage2_model = stage2_data["model"]
        stage2_features = stage2_data["features"]
        
    # Create a synthetic base dataframe for testing
    # We will simulate a state where all machines are RUNNING at 50%, 100%, 150%, 200%
    
    test_cases = []
    
    for prod_level in [50, 100, 150, 200]:
        row = {
            "hour": 12,
            "day_of_week": 2,
            "shift": 1,
        }
        for i in range(1, 9):
            row[f"production_M{i}"] = float(prod_level)
            row[f"machine_status_M{i}"] = "RUNNING"
            
        for i in range(1, 4):
            row[f"tap_status_T{i}"] = "CLOSED"
            
        # We need actual flows. We will use the explicit physics model to calculate them perfectly.
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from feature_engineering import compute_expected_machine_flow, compute_expected_tap_flow
        
        expected_flows = {}
        # Calculate machine branches
        for i in range(1, 9):
            m_id = f"M{i}"
            j_id = config.MACHINE_SPECS[m_id]["sensor"]
            expected_flows[j_id] = compute_expected_machine_flow(m_id, prod_level, "RUNNING")
            
        for i in range(1, 4):
            t_id = f"T{i}"
            j_id = config.TAP_SPECS[t_id]["sensor"]
            expected_flows[j_id] = compute_expected_tap_flow(t_id, "CLOSED")
            
        # Upstream aggregation
        expected_flows["J2"] = expected_flows["J5"] + expected_flows["J6"]
        expected_flows["J3"] = expected_flows["J8"] + expected_flows["J9"] + expected_flows["J10"]
        expected_flows["J4"] = expected_flows["J11"] + expected_flows["J12"]
        expected_flows["J7"] = expected_flows["J13"] + expected_flows["J14"] + expected_flows["J15"] + expected_flows["J16"]
        expected_flows["J1"] = expected_flows["J2"] + expected_flows["J3"] + expected_flows["J4"] + expected_flows["J7"]
        
        # We set actual flow exactly to expected flow (plus tiny noise) to simulate perfect normal behavior
        for j in range(1, 17):
            row[f"flow_J{j}"] = expected_flows[f"J{j}"] + np.random.normal(0, 0.5)
            # Dummy pressure and rolling stats for test since they don't change drastically without leaks
            # but we'll set reasonable values
            row[f"rolling_mean_flow_J{j}"] = row[f"flow_J{j}"]
            row[f"rolling_std_flow_J{j}"] = 0.5
            row[f"flow_change_J{j}"] = 0.0
            
        # Pressure placeholders (simulating normal pressure)
        for p in ["J1", "J2", "J3", "J4", "J7"]:
            row[f"pressure_{p}"] = 5.0
            row[f"pressure_{p}_change"] = 0.0
            row[f"rolling_mean_pressure_{p}"] = 5.0
            row[f"rolling_std_pressure_{p}"] = 0.01
            
        # Balance placeholders (perfect balance)
        for b in ["J1", "J2", "J3", "J4", "J7"]:
            row[f"balance_{b}"] = 0.0
            
        test_cases.append(row)
        
    df_test = pd.DataFrame(test_cases)
    
    # 1. Generate ML Stage 1 Expectations
    num_features = [f"production_M{i}" for i in range(1, 9)] + ["hour", "day_of_week", "shift"]
    cat_features = [f"machine_status_M{i}" for i in range(1, 9)] + [f"tap_status_T{i}" for i in range(1, 4)]
    X_stage1 = df_test[num_features + cat_features]
    
    for i in range(1, 17):
        target = f"flow_J{i}"
        preds = stage1_models[target].predict(X_stage1)
        df_test[f"ml_expected_flow_J{i}"] = preds
        res = df_test[f"flow_J{i}"] - preds
        df_test[f"ml_flow_residual_J{i}"] = res
        df_test[f"ml_flow_residual_percent_J{i}"] = (res / (np.abs(preds) + 1.0)) * 100.0
        
    # 2. Predict with Stage 2
    X_stage2 = df_test[stage2_features]
    probs = stage2_model.predict_proba(X_stage2)[:, 1]
    preds = stage2_model.predict(X_stage2)
    
    print("\n--- Results ---")
    for idx, prod_level in enumerate([50, 100, 150, 200]):
        actual_main_flow = df_test.iloc[idx]['flow_J1']
        predicted_prob = probs[idx]
        is_leak = preds[idx]
        print(f"Production: {prod_level:3d}% | Total Flow J1: {actual_main_flow:7.2f} L/min | Leak Prob: {predicted_prob:.4f} | Prediction: {is_leak}")
        
        if is_leak == 1:
            print(f"  [ERROR] Model falsely detected a leak due to scaling!")
        else:
            print(f"  [PASS] Model ignored the scale increase correctly.")
            
    # Extract Feature Importances
    importances = stage2_model.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({'feature': stage2_features, 'importance': importances})
    feat_imp = feat_imp.sort_values(by='importance', ascending=False).head(20)
    
    print("\nTop 20 Features Driving Stage 2 Classifier:")
    for _, row in feat_imp.iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

if __name__ == "__main__":
    run_scaling_test()
