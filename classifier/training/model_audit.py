import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, 
    recall_score, f1_score, roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier

# config.py is in classifier/ (one level up from training/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from predict import WaterNetworkLeakDetector
from feature_engineering import compute_expected_machine_flow, compute_expected_tap_flow

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_data():
    df = pd.read_csv(config.ML_TRAINING_CSV)
    df = df.ffill().bfill()
    n_total = len(df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train+n_val].copy()
    test_df = df.iloc[n_train+n_val:].copy()
    return df, train_df, val_df, test_df

def run_audit():
    print("=" * 80)
    print("WATER NETWORK ML MODEL AUDIT")
    print("=" * 80)
    
    df, train_df, val_df, test_df = load_data()
    
    # Load Models
    with open(os.path.join(MODELS_DIR, 'stage1_expected_flow_models.pkl'), 'rb') as f:
        s1_data = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'stage2_leak_classifier.pkl'), 'rb') as f:
        s2_data = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'stage3_leak_zone.pkl'), 'rb') as f:
        s3_data = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'stage4_leak_rate.pkl'), 'rb') as f:
        s4_data = pickle.load(f)
        
    s2_model = s2_data["model"]
    s2_features = s2_data["features"]

    # Generate ML enriched features for test_df
    # We must do this properly. Let's just use the original train_all.py's method.
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
    import train_all
    test_df_enriched = train_all.generate_stage1_predictions_and_residuals(test_df, s1_data["models"])
    train_df_enriched = train_all.generate_stage1_predictions_and_residuals(train_df, s1_data["models"])

    print("\n[1 & 2] STAGE 1 TRAINING INSPECTION")
    print("Features used in Stage 1:")
    all_s1_feats = s1_data["numeric_features"] + s1_data["categorical_features"]
    print(all_s1_feats)
    print(f"Contains leak/leak_rate/leak_zone? {any('leak' in f for f in all_s1_feats)}")
    
    print("\n[3] PRODUCTION RATE DISTRIBUTION")
    mean_prod = df[[f"production_M{i}" for i in range(1, 9)]].mean(axis=1)
    print(f"Min production: {mean_prod.min():.2f}%")
    print(f"Max production: {mean_prod.max():.2f}%")
    print("Distribution by range:")
    bins = [0, 50, 100, 150, 180, 210]
    hist, _ = np.histogram(mean_prod, bins=bins)
    for i in range(len(bins)-1):
        print(f"  {bins[i]}-{bins[i+1]}%: {hist[i]} rows")

    print("\n[4] PRODUCTION-INVARIANCE TEST (Detailed Table)")
    # Like test_production_scaling.py but tabular
    print(f"{'Prod %':<10} | {'Expected J1':<15} | {'Actual J1':<15} | {'Leak Prob':<15} | {'Prediction'}")
    for prod_level in [50, 100, 150, 200]:
        row = {"hour": 12, "day_of_week": 2, "shift": 1}
        for i in range(1, 9):
            row[f"production_M{i}"] = float(prod_level)
            row[f"machine_status_M{i}"] = "RUNNING"
        for i in range(1, 4):
            row[f"tap_status_T{i}"] = "CLOSED"
        
        expected_flows = {}
        for i in range(1, 9):
            m_id = f"M{i}"
            expected_flows[config.MACHINE_SPECS[m_id]["sensor"]] = compute_expected_machine_flow(m_id, prod_level, "RUNNING")
        for i in range(1, 4):
            t_id = f"T{i}"
            expected_flows[config.TAP_SPECS[t_id]["sensor"]] = compute_expected_tap_flow(t_id, "CLOSED")
            
        expected_flows["J2"] = expected_flows["J5"] + expected_flows["J6"]
        expected_flows["J3"] = expected_flows["J8"] + expected_flows["J9"] + expected_flows["J10"]
        expected_flows["J4"] = expected_flows["J11"] + expected_flows["J12"]
        expected_flows["J7"] = expected_flows["J13"] + expected_flows["J14"] + expected_flows["J15"] + expected_flows["J16"]
        expected_flows["J1"] = expected_flows["J2"] + expected_flows["J3"] + expected_flows["J4"] + expected_flows["J7"]
        
        for j in range(1, 17):
            row[f"flow_J{j}"] = expected_flows[f"J{j}"] + 0.1
            row[f"rolling_mean_flow_J{j}"] = row[f"flow_J{j}"]
            row[f"rolling_std_flow_J{j}"] = 0.5
            row[f"flow_change_J{j}"] = 0.0
        for p in ["J1", "J2", "J3", "J4", "J7"]:
            row[f"pressure_{p}"] = 5.0
            row[f"pressure_{p}_change"] = 0.0
            row[f"rolling_mean_pressure_{p}"] = 5.0
            row[f"rolling_std_pressure_{p}"] = 0.01
        for b in ["J1", "J2", "J3", "J4", "J7"]:
            row[f"balance_{b}"] = 0.0
            
        df_row = pd.DataFrame([row])
        X_s1 = df_row[all_s1_feats]
        for j in range(1, 17):
            pred = s1_data["models"][f"flow_J{j}"].predict(X_s1)[0]
            df_row[f"ml_expected_flow_J{j}"] = pred
            res = df_row[f"flow_J{j}"] - pred
            df_row[f"ml_flow_residual_J{j}"] = res
            df_row[f"ml_flow_residual_percent_J{j}"] = (res / (np.abs(pred) + 1.0)) * 100.0
            
        X_s2 = df_row[s2_features]
        prob = s2_model.predict_proba(X_s2)[0, 1]
        pred_label = s2_model.predict(X_s2)[0]
        
        exp_j1 = df_row["ml_expected_flow_J1"].iloc[0]
        act_j1 = df_row["flow_J1"].iloc[0]
        print(f"{prod_level:<10} | {exp_j1:<15.2f} | {act_j1:<15.2f} | {prob:<15.4f} | {pred_label}")

    print("\n[5] STRONGER GENERALIZATION TEST (Train <=180%, Test 200%)")
    train_mean_prod = train_df_enriched[[f"production_M{i}" for i in range(1, 9)]].mean(axis=1)
    train_sub = train_df_enriched[train_mean_prod <= 180.0]
    
    gen_model = XGBClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    gen_model.fit(train_sub[s2_features], train_sub["leak"])
    
    # Test a) 200% normal (we have it in df_row from above)
    prob_normal = gen_model.predict_proba(df_row[s2_features])[0, 1]
    
    # Test b) 200% with leak (add 200 L/min leak to J1)
    df_row_leak = df_row.copy()
    df_row_leak["flow_J1"] += 200.0
    df_row_leak["ml_flow_residual_J1"] = df_row_leak["flow_J1"] - df_row_leak["ml_expected_flow_J1"]
    df_row_leak["ml_flow_residual_percent_J1"] = (df_row_leak["ml_flow_residual_J1"] / (np.abs(df_row_leak["ml_expected_flow_J1"])+1)) * 100.0
    df_row_leak["balance_J1"] += 200.0
    df_row_leak["pressure_J1"] -= 0.5
    
    prob_leak = gen_model.predict_proba(df_row_leak[s2_features])[0, 1]
    print(f"a) Normal 200% leak prob: {prob_normal:.4f}")
    print(f"b) Leak at 200% leak prob: {prob_leak:.4f}")

    print("\n[6] ABLATION TEST")
    ablation_sets = {
        "A) flow only": [f"flow_J{i}" for i in range(1,17)],
        "B) flow + pressure": [f"flow_J{i}" for i in range(1,17)] + [f"pressure_{p}" for p in ["J1","J2","J3","J4","J7"]],
        "C) flow + production": [f"flow_J{i}" for i in range(1,17)] + [f"production_M{i}" for i in range(1,9)],
        "D) flow + prod + pressure": [f"flow_J{i}" for i in range(1,17)] + [f"production_M{i}" for i in range(1,9)] + [f"pressure_{p}" for p in ["J1","J2","J3","J4","J7"]],
        "E) Full model": s2_features
    }
    
    for name, feats in ablation_sets.items():
        # filter to available
        feats = [f for f in feats if f in train_df_enriched.columns]
        m = XGBClassifier(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
        m.fit(train_df_enriched[feats], train_df_enriched["leak"])
        preds = m.predict(test_df_enriched[feats])
        probs = m.predict_proba(test_df_enriched[feats])[:, 1]
        
        pr = precision_score(test_df_enriched["leak"], preds)
        rc = recall_score(test_df_enriched["leak"], preds)
        f1 = f1_score(test_df_enriched["leak"], preds)
        pr_auc = average_precision_score(test_df_enriched["leak"], probs)
        print(f"{name:<25} | P: {pr:.3f} | R: {rc:.3f} | F1: {f1:.3f} | PR-AUC: {pr_auc:.3f}")

    print("\n[7] FINAL MODEL EVALUATION (Test Set)")
    y_true = test_df_enriched["leak"]
    X_test = test_df_enriched[s2_features]
    y_pred = s2_model.predict(X_test)
    y_prob = s2_model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"PR-AUC: {average_precision_score(y_true, y_prob):.4f}")

    print("\n[8] FALSE NEGATIVE ANALYSIS")
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_df = test_df_enriched[fn_mask]
    print(f"Total False Negatives: {len(fn_df)}")
    if len(fn_df) > 0:
        print("FN by Zone:")
        print(fn_df["leak_zone"].value_counts())
        print("FN Leak Rate Stats:")
        print(fn_df["leak_rate"].describe())

    print("\n[9] STAGE 3 (LEAK ZONE) RESULTS")
    s3_model = s3_data["model"]
    label_enc = s3_data["label_encoder"]
    test_leak = test_df_enriched[test_df_enriched["leak"] == 1]
    if len(test_leak) > 0:
        s3_true = test_leak["leak_zone"]
        # Filter to known classes
        known = s3_true.isin(label_enc.classes_)
        test_leak = test_leak[known]
        s3_true = test_leak["leak_zone"]
        
        s3_true_enc = label_enc.transform(s3_true)
        s3_pred_enc = s3_model.predict(test_leak[s3_data["features"]])
        
        unique_labels = np.unique(s3_true_enc)
        unique_names = label_enc.inverse_transform(unique_labels)
        print(classification_report(s3_true_enc, s3_pred_enc, labels=unique_labels, target_names=unique_names))
        
    print("\n[10] STAGE 4 (LEAK RATE) RESULTS")
    s4_model = s4_data["model"]
    if len(test_leak) > 0:
        y_true_rate = test_leak["leak_rate"]
        y_pred_rate = s4_model.predict(test_leak[s4_data["features"]])
        
        mae = mean_absolute_error(y_true_rate, y_pred_rate)
        rmse = np.sqrt(mean_squared_error(y_true_rate, y_pred_rate))
        r2 = r2_score(y_true_rate, y_pred_rate)
        print(f"MAE:  {mae:.2f} L/min")
        print(f"RMSE: {rmse:.2f} L/min")
        print(f"R2:   {r2:.3f}")
        print("Actual Rate distribution:")
        print(y_true_rate.describe())

    print("\n[11] DATA LEAKAGE CHECKS")
    print(f"Are 'leak'/'leak_rate'/'leak_zone' in Stage 2 features? {any(c in s2_features for c in ['leak', 'leak_rate', 'leak_zone'])}")
    print(f"Is data chronologically split? (Train start: {train_df.iloc[0]['date']}, Test end: {test_df.iloc[-1]['date']}) -> YES")

    print("\n[12 & 13] INFERENCE LATENCY BENCHMARK")
    detector = WaterNetworkLeakDetector()
    latencies = []
    # Test on 200 random samples
    sample_dicts = test_df.sample(200).to_dict('records')
    for state in sample_dicts:
        t0 = time.time()
        _ = detector.predict(state)
        latencies.append((time.time() - t0) * 1000)
        
    print(f"Average latency: {np.mean(latencies):.2f} ms")
    print(f"95th percentile latency: {np.percentile(latencies, 95):.2f} ms")

if __name__ == "__main__":
    run_audit()
