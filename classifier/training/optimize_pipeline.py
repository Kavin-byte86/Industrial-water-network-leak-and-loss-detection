import os
import sys
import json
import pickle
import time
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, 
    recall_score, f1_score, roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# config.py is in classifier/ (one level up from training/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from predict import WaterNetworkLeakDetector
# Single source of truth for the v2 feature block, shared with inference.
from features import add_advanced_features

MODELS_V1_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'v1')
MODELS_V2_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'v2')
REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
os.makedirs(MODELS_V2_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

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

def verify_data_leakage(features, train_start, test_end):
    print("\n=====================================================================")
    print("DATA LEAKAGE AUDIT")
    print("=====================================================================")
    leaky_keywords = ['leak', 'leak_rate', 'leak_zone']
    has_leak = any(k in f for f in features for k in leaky_keywords)
    print(f"Contains target variables as inputs? {'FAIL' if has_leak else 'PASS'}")
    
    chronological = str(train_start) < str(test_end)
    print(f"Chronological Split Maintained? {'PASS' if chronological else 'FAIL'}")
    
    future_leakage = False
    for f in features:
        if 'future' in f or 'next' in f:
            future_leakage = True
    print(f"Contains future-looking features? {'FAIL' if future_leakage else 'PASS'}")
    return not has_leak and chronological and not future_leakage

def analyze_detectability(df):
    noise_percent = config.FLOW_NOISE_PERCENT / 100.0
    nom_flow_j1 = 1000.0
    absolute_noise_j1 = nom_flow_j1 * noise_percent
    
    content = f"""# Sensor Noise & Detectability Analysis
_Derived from `config.py`; generated {datetime.now():%Y-%m-%d %H:%M}._

## Noise Floor Profile
- Configured Flow Noise: {config.FLOW_NOISE_PERCENT}%
- Typical Main Inlet Flow (J1): ~{nom_flow_j1} L/min
- Expected Absolute Noise Floor: ±{absolute_noise_j1:.1f} L/min

## Detectability Conclusion
A synthetic leak of 3-5 L/min represents ~0.3% of the total network flow. This is structurally buried beneath the {config.FLOW_NOISE_PERCENT}% measurement noise. 
We MUST NOT force the ML model to blindly trigger on mathematically imperceptible signals, as this would drastically increase false alarms during normal legitimate demand variations.

Minimum practically detectable leak (Signal-to-Noise Ratio > 1.5): **~{absolute_noise_j1 * 1.5:.1f} L/min**
"""
    with open(os.path.join(REPORTS_DIR, 'DETECTABILITY_ANALYSIS.md'), 'w') as f:
        f.write(content)
    print("Detectability analysis saved.")

def tune_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []

    # Reports from this pipeline are scoped explicitly: the threshold sweep is a
    # v2 VALIDATION-set sweep, while MODEL_OPTIMIZATION_REPORT compares v1 vs v2
    # on the untouched TEST set. Without these labels the two files read as if
    # they contradict each other.
    content = "# Decision Threshold Tuning\n\n"
    content += f"- **Model**: v2 Stage-2 leak classifier\n"
    content += f"- **Split**: validation ({len(y_val):,} rows, {int(y_val.sum()):,} leak)\n"
    content += f"- **Generated**: {datetime.now():%Y-%m-%d %H:%M}\n\n"
    content += "> These are validation-set numbers used only to pick the operating\n"
    content += "> threshold. Headline performance is the test-set figure in\n"
    content += "> `MODEL_OPTIMIZATION_REPORT.md`.\n\n"
    content += "| Threshold | Precision | Recall | F1 | False Positives | False Negatives |\n"
    content += "|-----------|-----------|--------|----|-----------------|-----------------|\n"
    
    best_t = 0.5
    best_f1 = 0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        pr = precision_score(y_val, preds, zero_division=0)
        rc = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        content += f"| {t:.2f} | {pr:.4f} | {rc:.4f} | {f1:.4f} | {fp} | {fn} |\n"
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    with open(os.path.join(REPORTS_DIR, 'THRESHOLD_ANALYSIS.md'), 'w') as f:
        f.write(content)
        
    print(f"Optimal validation threshold identified: {best_t}")
    return best_t

def optimize_model():
    print("=" * 80)
    print("SECOND ML OPTIMIZATION AND RETRAINING PHASE")
    print("=" * 80)
    
    df, train_df, val_df, test_df = load_data()
    analyze_detectability(df)
    
    # Load Stage 1 models from v1 (since they don't need retraining, just physics)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
    import train_all
    with open(os.path.join(MODELS_V1_DIR, 'stage1_expected_flow_models.pkl'), 'rb') as f:
        s1_data = pickle.load(f)
        
    print("Enriching features (Stage 1 + Advanced Temporal/Production)...")
    train_df = train_all.generate_stage1_predictions_and_residuals(train_df, s1_data["models"])
    val_df = train_all.generate_stage1_predictions_and_residuals(val_df, s1_data["models"])
    test_df = train_all.generate_stage1_predictions_and_residuals(test_df, s1_data["models"])
    
    train_df = add_advanced_features(train_df)
    val_df = add_advanced_features(val_df)
    test_df = add_advanced_features(test_df)
    
    # Base features from v1 + new features
    with open(os.path.join(MODELS_V1_DIR, 'stage2_leak_classifier.pkl'), 'rb') as f:
        s2_v1 = pickle.load(f)
    base_features = s2_v1["features"]
    
    new_features = []
    for i in range(1, 17):
        s_id = f"J{i}"
        new_features.extend([f'flow_{s_id}_lag1', f'flow_{s_id}_lag3', f'flow_{s_id}_lag6', 
                             f'rolling_median_flow_{s_id}', f'short_term_dev_flow_{s_id}',
                             f'flow_per_prod_{s_id}'])
                             
        if f'ml_flow_residual_{s_id}' in train_df.columns:
            new_features.append(f'residual_per_prod_{s_id}')
            
    final_features = base_features + new_features
    final_features = [f for f in final_features if f in train_df.columns]
    
    verify_data_leakage(final_features, train_df.iloc[0]['date'], test_df.iloc[-1]['date'])
    
    X_train = train_df[final_features]
    y_train = train_df["leak"]
    X_val = val_df[final_features]
    y_val = val_df["leak"]
    
    # Handle Class Imbalance using scale_pos_weight
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    spw = neg / pos if pos > 0 else 1.0
    # Tone it down slightly so we don't destroy precision
    spw = min(spw, 5.0) 
    
    print(f"Training optimized Stage 2 model (scale_pos_weight={spw:.2f})...")
    model_v2 = XGBClassifier(
        n_estimators=150, 
        max_depth=7, 
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=config.RANDOM_SEED,
        early_stopping_rounds=15,
        n_jobs=-1
    )
    
    model_v2.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    best_thresh = tune_threshold(model_v2, X_val, y_val)
    
    print("Evaluating untouched test set...")
    X_test = test_df[final_features]
    y_test = test_df["leak"]
    probs_v2 = model_v2.predict_proba(X_test)[:, 1]
    preds_v2 = (probs_v2 >= best_thresh).astype(int)
    
    # Old model eval
    model_v1 = s2_v1["model"]
    # Re-extract just the v1 features
    X_test_v1 = test_df[base_features]
    probs_v1 = model_v1.predict_proba(X_test_v1)[:, 1]
    preds_v1 = model_v1.predict(X_test_v1)
    
    rc_v1 = recall_score(y_test, preds_v1)
    rc_v2 = recall_score(y_test, preds_v2)
    f1_v1 = f1_score(y_test, preds_v1)
    f1_v2 = f1_score(y_test, preds_v2)
    prauc_v1 = average_precision_score(y_test, probs_v1)
    prauc_v2 = average_precision_score(y_test, probs_v2)
    
    print("\n====================================================")
    print("FINAL OPTIMIZED MODEL COMPARISON")
    print("====================================================")
    print(f"Old test recall: {rc_v1:.4f}")
    print(f"New test recall: {rc_v2:.4f}")
    print(f"Old test F1: {f1_v1:.4f}")
    print(f"New test F1: {f1_v2:.4f}")
    print(f"Old PR-AUC: {prauc_v1:.4f}")
    print(f"New PR-AUC: {prauc_v2:.4f}")
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds_v2).ravel()
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    
    # Save v2
    with open(os.path.join(MODELS_V2_DIR, 'stage2_leak_classifier.pkl'), 'wb') as f:
        pickle.dump({"model": model_v2, "features": final_features, "threshold": best_thresh}, f)
    # Copy S1, S3, S4 to v2
    import shutil
    shutil.copy(os.path.join(MODELS_V1_DIR, 'stage1_expected_flow_models.pkl'), os.path.join(MODELS_V2_DIR, 'stage1_expected_flow_models.pkl'))
    if os.path.exists(os.path.join(MODELS_V1_DIR, 'stage3_leak_zone.pkl')):
        shutil.copy(os.path.join(MODELS_V1_DIR, 'stage3_leak_zone.pkl'), os.path.join(MODELS_V2_DIR, 'stage3_leak_zone.pkl'))
    if os.path.exists(os.path.join(MODELS_V1_DIR, 'stage4_leak_rate.pkl')):
        shutil.copy(os.path.join(MODELS_V1_DIR, 'stage4_leak_rate.pkl'), os.path.join(MODELS_V2_DIR, 'stage4_leak_rate.pkl'))
        
    print(f"Model saved: classifier/models/v2/")
    
    # False Negative Analysis
    fn_mask = (y_test == 1) & (preds_v2 == 0)
    fn_df = test_df[fn_mask]
    fn_df.to_csv(os.path.join(REPORTS_DIR, 'false_negative_analysis.csv'), index=False)
    with open(os.path.join(REPORTS_DIR, 'FALSE_NEGATIVE_ANALYSIS.md'), 'w') as f:
        f.write("# False Negative Analysis\n\n")
        f.write("- **Model**: v2 Stage-2 leak classifier\n")
        f.write(f"- **Split**: test ({len(test_df):,} rows)\n")
        f.write(f"- **Decision threshold**: {best_thresh:.2f}\n")
        f.write(f"- **Generated**: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"Total Missed Leaks: {len(fn_df)}\n\n")
        if len(fn_df) > 0:
            f.write("### Breakdown by Zone\n```\n")
            f.write(fn_df["leak_zone"].value_counts().to_string())
            f.write("\n```\n### Missed Leak Rates\n```\n")
            f.write(fn_df["leak_rate"].describe().to_string())
            f.write("\n```\n")
            
    # Final Report
    pr_v1 = precision_score(y_test, preds_v1, zero_division=0)
    pr_v2 = precision_score(y_test, preds_v2, zero_division=0)

    with open(os.path.join(REPORTS_DIR, 'MODEL_OPTIMIZATION_REPORT.md'), 'w') as f:
        f.write("# Model Optimization Report (v1 -> v2)\n\n")
        f.write(f"- **Split**: untouched test set ({len(test_df):,} rows, "
                f"{int(y_test.sum()):,} leak)\n")
        f.write(f"- **v2 decision threshold**: {best_thresh:.2f} "
                f"(tuned on validation, see `THRESHOLD_ANALYSIS.md`)\n")
        f.write("- **v1 decision threshold**: 0.50 (default)\n")
        f.write(f"- **Generated**: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write("## Improvements\n")
        f.write("- Added advanced temporal lags and short-term deviations.\n")
        f.write("- Tuned XGBoost `scale_pos_weight` and decision threshold.\n\n")
        f.write("## Test-Set Metrics\n\n")
        f.write("| Metric | v1 | v2 |\n|---|---|---|\n")
        f.write(f"| Precision | {pr_v1:.4f} | {pr_v2:.4f} |\n")
        f.write(f"| Recall | {rc_v1:.4f} | {rc_v2:.4f} |\n")
        f.write(f"| F1 | {f1_v1:.4f} | {f1_v2:.4f} |\n")
        f.write(f"| PR-AUC | {prauc_v1:.4f} | {prauc_v2:.4f} |\n\n")
        f.write(f"False positives (v2): {fp} | False negatives (v2): {fn}\n\n")
        f.write("> Validation-set recall in `THRESHOLD_ANALYSIS.md` is higher than the\n")
        f.write("> test-set recall here. That is expected: the threshold was selected on\n")
        f.write("> validation, so those numbers are optimistic. The table above is the\n")
        f.write("> honest estimate.\n")

    # Machine-readable summary so the dashboard never has to hardcode a claim.
    with open(os.path.join(REPORTS_DIR, 'metrics.json'), 'w') as f:
        json.dump({
            "generated": datetime.now().isoformat(timespec="seconds"),
            "split": "test",
            "test_rows": int(len(test_df)),
            "test_leak_rows": int(y_test.sum()),
            "v2_threshold": float(best_thresh),
            "v1": {"precision": float(pr_v1), "recall": float(rc_v1),
                   "f1": float(f1_v1), "pr_auc": float(prauc_v1)},
            "v2": {"precision": float(pr_v2), "recall": float(rc_v2),
                   "f1": float(f1_v2), "pr_auc": float(prauc_v2),
                   "false_positives": int(fp), "false_negatives": int(fn)},
        }, f, indent=2)


if __name__ == "__main__":
    optimize_model()
