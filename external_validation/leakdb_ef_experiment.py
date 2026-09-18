import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
OUT_DIR = os.path.join(os.path.dirname(__file__), "expected_flow_experiment")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_enhanced_features(df):
    df = df.copy()
    
    demand_cols = [c for c in df.columns if c.startswith("demand_")]
    
    df["total_demand"] = df[demand_cols].sum(axis=1)
    
    # User specified operational features
    df["shift"] = df["hour"] // 8
    
    # Rolling statistics on total demand (normal operating context)
    # 6 hours at 5-minute intervals = 72 time steps
    df["total_demand_rolling_mean_6h"] = df.groupby("scenario_id")["total_demand"].transform(lambda x: x.rolling(72, min_periods=1).mean())
    df["total_demand_rolling_std_6h"] = df.groupby("scenario_id")["total_demand"].transform(lambda x: x.rolling(72, min_periods=1).std().fillna(0))
    
    return df

def run_experiment():
    print("--- 1. LOADING DATA AND ENHANCING FEATURES ---")
    train_df = pd.read_pickle(os.path.join(RESULTS_DIR, "train_df.pkl"))
    val_df = pd.read_pickle(os.path.join(RESULTS_DIR, "val_df.pkl"))
    test_df = pd.read_pickle(os.path.join(RESULTS_DIR, "test_df.pkl"))
    
    train_df = generate_enhanced_features(train_df)
    val_df = generate_enhanced_features(val_df)
    test_df = generate_enhanced_features(test_df)
    
    print("--- 2. TRAINING EXPECTED-FLOW MODELS ---")
    
    train_normal = train_df[train_df["leak"] == 0]
    val_normal = val_df[val_df["leak"] == 0]
    
    # Strict operational features as requested by user
    demand_cols = [c for c in train_normal.columns if c.startswith("demand_")]
    reg_features = ["total_demand", "hour", "day_of_week", "shift", "total_demand_rolling_mean_6h", "total_demand_rolling_std_6h"] + demand_cols
    
    X_train_reg = train_normal[reg_features]
    X_val_reg = val_normal[reg_features]
    
    flow_cols = [c for c in train_df.columns if c.startswith("flow_") and "expected" not in c and "residual" not in c]
    
    reg_results = []
    best_models = {}
    
    for fc in flow_cols:
        y_train = train_normal[fc]
        y_val = val_normal[fc]
        
        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_reg, y_train)
        pred_ridge = ridge.predict(X_val_reg)
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_reg, y_train)
        pred_rf = rf.predict(X_val_reg)
        
        # XGBoost
        xgb = XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        xgb.fit(X_train_reg, y_train)
        pred_xgb = xgb.predict(X_val_reg)
        
        mean_flow = y_val.mean()
        
        def calc_metrics(y_true, y_pred, name):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            rel_err = (mae / (abs(mean_flow) + 1e-5)) * 100
            return mae, rmse, r2, rel_err
            
        mae_r, rmse_r, r2_r, rel_r = calc_metrics(y_val, pred_ridge, "Ridge")
        mae_rf, rmse_rf, r2_rf, rel_rf = calc_metrics(y_val, pred_rf, "RF")
        mae_x, rmse_x, r2_x, rel_x = calc_metrics(y_val, pred_xgb, "XGB")
        
        reg_results.append({
            "Link": fc,
            "Mean_Flow": mean_flow,
            "Ridge_MAE": mae_r, "Ridge_RMSE": rmse_r, "Ridge_R2": r2_r, "Ridge_RelErr": rel_r,
            "RF_MAE": mae_rf, "RF_RMSE": rmse_rf, "RF_R2": r2_rf, "RF_RelErr": rel_rf,
            "XGB_MAE": mae_x, "XGB_RMSE": rmse_x, "XGB_R2": r2_x, "XGB_RelErr": rel_x
        })
        
        # Select best model for this link based on MAE
        best_mae = min(mae_r, mae_rf, mae_x)
        if best_mae == mae_r: best_model = ridge; b_name = "Ridge"
        elif best_mae == mae_rf: best_model = rf; b_name = "RF"
        else: best_model = xgb; b_name = "XGB"
        
        best_models[fc] = best_model
        
    df_reg = pd.DataFrame(reg_results)
    df_reg.to_csv(os.path.join(OUT_DIR, "expected_flow_metrics.csv"), index=False)
    
    print("--- 3. MEASURING EXPECTED-FLOW QUALITY ---")
    def summarize_errors(rel_errors, name):
        lt5 = sum(rel_errors < 5)
        lt10 = sum((rel_errors >= 5) & (rel_errors < 10))
        lt20 = sum((rel_errors >= 10) & (rel_errors < 20))
        gt20 = sum(rel_errors >= 20)
        print(f"{name} Errors: <5%: {lt5} | 5-10%: {lt10} | 10-20%: {lt20} | >20%: {gt20}")
        
    summarize_errors(df_reg["Ridge_RelErr"], "Ridge")
    summarize_errors(df_reg["RF_RelErr"], "RF")
    summarize_errors(df_reg["XGB_RelErr"], "XGB")
    
    print("--- 4. GENERATING NEW RESIDUALS ---")
    
    def apply_new_residuals(df):
        df_new = df.copy()
        X_reg = df_new[reg_features]
        for fc, model in best_models.items():
            expected = model.predict(X_reg)
            df_new[f"new_expected_{fc}"] = expected
            
            res = df_new[fc] - expected
            df_new[f"new_residual_{fc}"] = res
            df_new[f"new_abs_residual_{fc}"] = res.abs()
            df_new[f"new_residual_pct_{fc}"] = (res.abs() / (df_new[fc].abs() + 1e-5)) * 100.0
        return df_new
        
    train_df = apply_new_residuals(train_df)
    val_df = apply_new_residuals(val_df)
    test_df = apply_new_residuals(test_df)
    
    # Merge train and val for classifier training to be robust
    train_master = pd.concat([train_df, val_df], ignore_index=True)
    
    print("--- 5. TESTING LEAK CLASSIFIERS ---")
    
    flow_features = flow_cols
    resid_features = [c for c in train_master.columns if c.startswith("new_residual_flow")]
    press_features = [c for c in train_master.columns if c.startswith("pressure_")]
    bal_features = [c for c in train_master.columns if c.startswith("balance_")]
    
    feature_sets = {
        "A_FlowOnly": flow_features,
        "B_NewResidualOnly": resid_features,
        "C_NewResidualPressure": resid_features + press_features,
        "D_NewResidualPressureBalance": resid_features + press_features + bal_features
    }
    
    y_train = train_master["leak"]
    y_test = test_df["leak"]
    
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    spw = neg / pos if pos > 0 else 1.0
    spw = min(spw, 10.0)
    
    clf_results = []
    classifiers = {}
    
    for name, feats in feature_sets.items():
        X_train = train_master[feats]
        X_test = test_df[feats]
        
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        classifiers[name] = clf
        
        probs = clf.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        
        pr = precision_score(y_test, preds, zero_division=0)
        rc = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan
        prauc = average_precision_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan
        
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        
        clf_results.append({
            "Model": name,
            "Precision": pr, "Recall": rc, "F1": f1,
            "ROC-AUC": roc, "PR-AUC": prauc,
            "False_Positives": fp, "False_Negatives": fn
        })
        
    df_clf = pd.DataFrame(clf_results)
    print(df_clf.to_string(index=False))
    df_clf.to_csv(os.path.join(OUT_DIR, "classifier_metrics.csv"), index=False)
    
    print("--- 6. SCENARIO-LEVEL EVALUATION ---")
    
    event_results = []
    
    for scenario in test_df["scenario_id"].unique():
        s_df = test_df[test_df["scenario_id"] == scenario]
        y_s = s_df["leak"].values
        is_leak = y_s.sum() > 0
        
        for name, feats in feature_sets.items():
            clf = classifiers[name]
            X_s = s_df[feats]
            probs = clf.predict_proba(X_s)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            if is_leak:
                leak_idx = np.where(y_s == 1)[0]
                start_idx = leak_idx[0]
                detected_idx = np.where((y_s == 1) & (preds == 1))[0]
                
                if len(detected_idx) > 0:
                    first = detected_idx[0]
                    # Data is 5-minute sampled, so 1 index = 5 minutes = 5/60 hours
                    ttd = (first - start_idx) * (5.0 / 60.0)
                    pct = len(detected_idx) / len(leak_idx) * 100.0
                    det = True
                else:
                    ttd = None; pct = 0.0; det = False
                    
                fp_before = preds[:start_idx].sum()
            else:
                det = False; ttd = None; pct = 0.0; fp_before = preds.sum()
                
            event_results.append({
                "Scenario": scenario,
                "Model": name,
                "Has_Leak": is_leak,
                "Detected": det,
                "TTD_Hrs": ttd,
                "Pct_Detected": pct,
                "False_Alarms_Before": fp_before
            })
            
    df_evt = pd.DataFrame(event_results)
    print(df_evt[df_evt["Model"] == "D_NewResidualPressureBalance"].to_string(index=False))
    df_evt.to_csv(os.path.join(OUT_DIR, "scenario_events.csv"), index=False)

    print("--- 7. FEATURE IMPORTANCE (best model by F1) ---")
    best_name = df_clf.loc[df_clf["F1"].idxmax(), "Model"]
    best_clf = classifiers[best_name]
    df_fi = pd.DataFrame({
        "Feature": feature_sets[best_name],
        "Importance": best_clf.feature_importances_,
        "Model": best_name,
    }).sort_values("Importance", ascending=False)
    df_fi.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    print(df_fi.head(10).to_string(index=False))

    print(f"\nExperiment complete. Results saved to {OUT_DIR}")

if __name__ == "__main__":
    run_experiment()
