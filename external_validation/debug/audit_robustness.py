import os
import sys
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def get_features(df):
    flow_cols = [c for c in df.columns if c.startswith("flow_") and "expected" not in c and "residual" not in c]
    press_cols = [c for c in df.columns if c.startswith("pressure_")]
    dem_cols = [c for c in df.columns if c.startswith("demand_")]
    resid_cols = [c for c in df.columns if c.startswith("residual_flow_")]
    bal_cols = [c for c in df.columns if c.startswith("balance_")]
    
    return {
        "A_FlowOnly": flow_cols,
        "D_ResidualOnly": resid_cols,
        "E_ResidualPressureBalance": resid_cols + press_cols + bal_cols
    }

def train_and_evaluate_loso():
    print("--- Robust Train/Test Evaluation (Leave-One-Scenario-Out) ---")
    
    train_df = pd.read_pickle(os.path.join(RESULTS_DIR, "train_df.pkl"))
    val_df = pd.read_pickle(os.path.join(RESULTS_DIR, "val_df.pkl"))
    test_df = pd.read_pickle(os.path.join(RESULTS_DIR, "test_df.pkl"))
    
    master_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    scenarios = master_df["scenario_id"].unique()
    
    feature_sets = get_features(master_df)
    
    results = []
    events = []
    
    for test_scenario in scenarios:
        print(f"\nEvaluating LOSO with Test Set = {test_scenario}")
        
        train_data = master_df[master_df["scenario_id"] != test_scenario]
        test_data = master_df[master_df["scenario_id"] == test_scenario]
        
        y_train = train_data["leak"]
        y_test = test_data["leak"]
        
        neg = sum(y_train == 0)
        pos = sum(y_train == 1)
        spw = neg / pos if pos > 0 else 1.0
        spw = min(spw, 10.0)
        
        for model_name, feats in feature_sets.items():
            X_train = train_data[feats]
            X_test = test_data[feats]
            
            clf = XGBClassifier(
                n_estimators=50,
                max_depth=5,
                scale_pos_weight=spw,
                random_state=42,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            
            probs = clf.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            pr = precision_score(y_test, preds, zero_division=0)
            rc = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            prauc = average_precision_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan
            
            results.append({
                "Test_Scenario": test_scenario,
                "Model": model_name,
                "Precision": pr,
                "Recall": rc,
                "F1": f1,
                "PR-AUC": prauc
            })
            
            # Event-level evaluation
            is_leak_scenario = y_test.sum() > 0
            if is_leak_scenario:
                leak_idx = np.where(y_test == 1)[0]
                start_idx = leak_idx[0]
                detected_idx = np.where((y_test == 1) & (preds == 1))[0]
                
                if len(detected_idx) > 0:
                    first_detection = detected_idx[0]
                    ttd_hrs = (first_detection - start_idx) * 0.5 
                    percent_detected = len(detected_idx) / len(leak_idx) * 100.0
                    detected = True
                else:
                    ttd_hrs = None
                    percent_detected = 0.0
                    detected = False
                    
                false_alarms = preds[:start_idx].sum()
                max_prob = probs[leak_idx].max()
            else:
                detected = False
                ttd_hrs = None
                percent_detected = 0.0
                false_alarms = preds.sum()
                max_prob = probs.max()
                
            events.append({
                "Test_Scenario": test_scenario,
                "Model": model_name,
                "is_leak": is_leak_scenario,
                "detected": detected,
                "ttd_hrs": ttd_hrs,
                "percent_detected": percent_detected,
                "false_alarms": false_alarms,
                "max_prob": max_prob
            })

    res_df = pd.DataFrame(results)
    
    print("\n--- Mean & Std Metrics Across All 10 Scenarios (LOSO) ---")
    agg_df = res_df.groupby("Model").agg({
        "Precision": ["mean", "std"],
        "Recall": ["mean", "std"],
        "F1": ["mean", "std"],
        "PR-AUC": ["mean", "std"]
    }).round(4)
    print(agg_df)
    
    events_df = pd.DataFrame(events)
    events_df.to_csv(os.path.join(os.path.dirname(__file__), "loso_events.csv"), index=False)
    agg_df.to_csv(os.path.join(os.path.dirname(__file__), "loso_metrics.csv"))

if __name__ == "__main__":
    train_and_evaluate_loso()
