import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def audit_expected_flow():
    print("--- Auditing Expected Flow Models ---")
    
    train_df = pd.read_pickle(os.path.join(RESULTS_DIR, "train_df.pkl"))
    val_df = pd.read_pickle(os.path.join(RESULTS_DIR, "val_df.pkl"))
    
    # Use only normal data for training and evaluating expected flow
    train_normal = train_df[train_df["leak"] == 0].copy()
    val_normal = val_df[val_df["leak"] == 0].copy()
    
    demand_cols = [c for c in train_normal.columns if c.startswith("demand_")]
    train_normal["total_demand"] = train_normal[demand_cols].sum(axis=1)
    val_normal["total_demand"] = val_normal[demand_cols].sum(axis=1)
    
    features = ["total_demand", "hour", "day_of_week"]
    
    X_train = train_normal[features]
    X_val = val_normal[features]
    
    flow_cols = [c for c in train_normal.columns if c.startswith("flow_")]
    
    results = []
    
    for fc in flow_cols:
        y_train = train_normal[fc]
        y_val = val_normal[fc]
        
        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        pred_ridge = ridge.predict(X_val)
        mae_ridge = mean_absolute_error(y_val, pred_ridge)
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_val)
        mae_rf = mean_absolute_error(y_val, pred_rf)
        
        # XGBoost
        xgb = XGBRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_val)
        mae_xgb = mean_absolute_error(y_val, pred_xgb)
        
        mean_flow = y_val.mean()
        
        results.append({
            "Link": fc,
            "Mean_Flow": mean_flow,
            "MAE_Ridge": mae_ridge,
            "MAE_RF": mae_rf,
            "MAE_XGB": mae_xgb,
            "Ridge_Error_%": (mae_ridge / (mean_flow + 1e-5)) * 100
        })
        
    df_res = pd.DataFrame(results).sort_values("Ridge_Error_%", ascending=False)
    
    print("\nExpected Flow Model Comparison (MAE on Normal Validation Data):")
    print(df_res.to_string(index=False))
    
    print("\nSummary:")
    print(f"Mean Ridge MAE: {df_res['MAE_Ridge'].mean():.2f}")
    print(f"Mean RF MAE: {df_res['MAE_RF'].mean():.2f}")
    print(f"Mean XGB MAE: {df_res['MAE_XGB'].mean():.2f}")
    
    # Identify badly modeled links
    bad_links = df_res[df_res["Ridge_Error_%"] > 10.0]
    if len(bad_links) > 0:
        print(f"\nFound {len(bad_links)} links where Ridge Error > 10% of mean flow.")
    else:
        print("\nRidge model appears sufficient (Error < 10% for all links).")

if __name__ == "__main__":
    audit_expected_flow()
