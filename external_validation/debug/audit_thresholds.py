import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__), "threshold_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

def audit_residual_only_and_thresholds():
    print("--- Auditing ResidualOnly Baseline and Thresholds ---")
    
    val_df = pd.read_pickle(os.path.join(RESULTS_DIR, "val_df.pkl"))
    test_df = pd.read_pickle(os.path.join(RESULTS_DIR, "test_df.pkl"))
    
    with open(os.path.join(RESULTS_DIR, "..", "models", "leakdb_classifiers.pkl"), "rb") as f:
        models = pickle.load(f)
        
    model_data = models["D_ResidualOnly"]
    model = model_data["model"]
    feats = model_data["features"]
    
    y_val = val_df["leak"]
    probs_val = model.predict_proba(val_df[feats])[:, 1]
    
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    results = []
    best_t = 0.5
    best_f1 = 0
    
    print("Threshold Sweep on Validation Scenario:")
    for t in thresholds:
        preds = (probs_val >= t).astype(int)
        pr = precision_score(y_val, preds, zero_division=0)
        rc = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        results.append((t, pr, rc, f1))
        
        print(f"  T={t:.2f} | PR={pr:.4f} | RC={rc:.4f} | F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    print(f"\nBest Threshold selected strictly on Validation: {best_t} (F1: {best_f1:.4f})")
    
    # Evaluate untouched test
    y_test = test_df["leak"]
    probs_test = model.predict_proba(test_df[feats])[:, 1]
    preds_test = (probs_test >= best_t).astype(int)
    
    test_pr = precision_score(y_test, preds_test, zero_division=0)
    test_rc = recall_score(y_test, preds_test, zero_division=0)
    test_f1 = f1_score(y_test, preds_test, zero_division=0)
    
    print("\nResidualOnly Test Set Performance (using best validation threshold):")
    print(f"Precision: {test_pr:.4f}")
    print(f"Recall:    {test_rc:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    
    # Plot residual magnitude vs prediction probability for a subset
    subset = val_df.sample(2000, random_state=42)
    # Sum of absolute residuals
    resid_sum = subset[feats].abs().sum(axis=1)
    probs_subset = model.predict_proba(subset[feats])[:, 1]
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(resid_sum, probs_subset, c=subset["leak"], cmap="coolwarm", alpha=0.5)
    plt.colorbar(scatter, label="Actual Leak Label (0/1)")
    plt.axhline(y=best_t, color='black', linestyle='--', label=f"Best Threshold ({best_t})")
    plt.xlabel("Sum of Absolute Residuals (CMH)")
    plt.ylabel("Predicted Leak Probability")
    plt.title("Residual Magnitude vs Predicted Leak Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, "residual_vs_prob.png"))
    plt.close()
    print(f"\nSaved threshold analysis plot to {OUT_DIR}/residual_vs_prob.png")
    
if __name__ == "__main__":
    audit_residual_only_and_thresholds()
