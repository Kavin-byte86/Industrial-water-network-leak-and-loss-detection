import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def tune_threshold(model, features, X_val, y_val, model_name):
    print(f"\n--- Tuning Threshold for {model_name} ---")
    probs = model.predict_proba(X_val)[:, 1]
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_t = 0.5
    best_f1 = 0
    
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        pr = precision_score(y_val, preds, zero_division=0)
        rc = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        results.append((t, pr, rc, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    print(f"Optimal Threshold for {model_name}: {best_t} (F1: {best_f1:.4f})")
    
    # Save threshold analysis for best model
    if model_name == "E_ResidualPressureBalance":
        pd.DataFrame(results, columns=["Threshold", "Precision", "Recall", "F1"]).to_csv(
            os.path.join(RESULTS_DIR, "threshold_analysis.csv"), index=False
        )
        
    return best_t

def evaluate_test_set(models_dict, test_df):
    print("\n--- Final Test Set Evaluation ---")
    y_test = test_df["leak"]
    
    metrics = []
    best_model_name = "E_ResidualPressureBalance"
    best_preds = None
    
    for name, data in models_dict.items():
        model = data["model"]
        feats = data["features"]
        threshold = data.get("threshold", 0.5)
        
        X_test = test_df[feats]
        
        if name == "F_IsolationForest":
            # Isolation forest returns -1 for anomaly, 1 for normal
            preds = model.predict(X_test)
            preds = (preds == -1).astype(int)
            probs = preds  # IF doesn't have predict_proba by default
            prauc = average_precision_score(y_test, probs)
        else:
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= threshold).astype(int)
            prauc = average_precision_score(y_test, probs)
            
        pr = precision_score(y_test, preds, zero_division=0)
        rc = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        metrics.append({
            "Model": name,
            "Precision": pr,
            "Recall": rc,
            "F1": f1,
            "PR-AUC": prauc
        })
        
        if name == best_model_name:
            best_preds = preds
            
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    print(metrics_df.to_string(index=False))
    
    return best_model_name, best_preds

def evaluate_events(test_df, best_preds):
    """
    Evaluates detection per leak scenario (Event-level evaluation)
    """
    df = test_df.copy()
    df["pred"] = best_preds
    
    events = []
    
    for sid in df["scenario_id"].unique():
        sdf = df[df["scenario_id"] == sid]
        
        # Is there actually a leak in this scenario?
        if sdf["leak"].sum() == 0:
            # Check false alarms
            false_alarms = sdf["pred"].sum()
            events.append({
                "scenario_id": sid,
                "is_leak_scenario": False,
                "event_detected": False,
                "false_alarms": false_alarms,
                "time_to_detection_hrs": None,
                "percent_detected": 0.0,
                "max_leak_rate": 0.0
            })
            continue
            
        # Scenario has a leak
        leak_idx = np.where(sdf["leak"] == 1)[0]
        start_idx = leak_idx[0]
        
        # Did we detect it?
        detected_idx = np.where((sdf["leak"] == 1) & (sdf["pred"] == 1))[0]
        
        if len(detected_idx) > 0:
            first_detection = detected_idx[0]
            # time to detection in hours (30 min steps)
            ttd_hrs = (first_detection - start_idx) * 0.5 
            percent_detected = len(detected_idx) / len(leak_idx) * 100.0
            detected = True
        else:
            ttd_hrs = None
            percent_detected = 0.0
            detected = False
            
        max_rate = sdf["leak_rate"].max()
        false_alarms_before = sdf.iloc[:start_idx]["pred"].sum()
        
        events.append({
            "scenario_id": sid,
            "is_leak_scenario": True,
            "event_detected": detected,
            "false_alarms": false_alarms_before,
            "time_to_detection_hrs": ttd_hrs,
            "percent_detected": percent_detected,
            "max_leak_rate": max_rate
        })
        
    events_df = pd.DataFrame(events)
    events_df.to_csv(os.path.join(RESULTS_DIR, "scenario_results.csv"), index=False)
    
    print("\n--- Event-Level Detection ---")
    print(events_df.to_string())

def main():
    with open(os.path.join(MODELS_DIR, "leakdb_classifiers.pkl"), "rb") as f:
        models = pickle.load(f)
        
    train_df = pd.read_pickle(os.path.join(RESULTS_DIR, "train_df.pkl"))
    val_df = pd.read_pickle(os.path.join(RESULTS_DIR, "val_df.pkl"))
    test_df = pd.read_pickle(os.path.join(RESULTS_DIR, "test_df.pkl"))
    
    # Tune thresholds
    for name, data in models.items():
        if name == "F_IsolationForest": continue
        t = tune_threshold(data["model"], data["features"], val_df[data["features"]], val_df["leak"], name)
        data["threshold"] = t
        
    best_name, best_preds = evaluate_test_set(models, test_df)
    
    evaluate_events(test_df, best_preds)
    
    # Feature Importance for best model
    best_model = models[best_name]["model"]
    best_feats = models[best_name]["features"]
    
    fi = pd.DataFrame({
        "Feature": best_feats,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    fi.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
    
if __name__ == "__main__":
    main()
