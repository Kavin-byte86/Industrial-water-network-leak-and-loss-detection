import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest

from leakdb_loader import load_all_scenarios

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def temporal_features(df):
    df_out = df.copy()
    df_out["hour"] = df_out["Timestamp"].dt.hour
    df_out["day_of_week"] = df_out["Timestamp"].dt.dayofweek
    return df_out

def split_scenarios(df):
    scenarios = df["scenario_id"].unique()
    np.random.seed(42)
    np.random.shuffle(scenarios)
    
    # 7 train, 1 val, 2 test for a 10 scenario dataset
    train_scenarios = scenarios[:7]
    val_scenarios = scenarios[7:8]
    test_scenarios = scenarios[8:]
    
    print("\n--- LeakDB Scenario Split ---")
    print(f"TRAIN Scenarios: {train_scenarios}")
    print(f"VAL Scenarios: {val_scenarios}")
    print(f"TEST Scenarios: {test_scenarios}")
    
    pd.DataFrame({
        "TRAIN": pd.Series(train_scenarios),
        "VAL": pd.Series(val_scenarios),
        "TEST": pd.Series(test_scenarios)
    }).to_csv(os.path.join(os.path.dirname(__file__), "leakdb_split.csv"), index=False)
    
    return (
        df[df["scenario_id"].isin(train_scenarios)].copy(),
        df[df["scenario_id"].isin(val_scenarios)].copy(),
        df[df["scenario_id"].isin(test_scenarios)].copy()
    )

def train_expected_flow_model(train_df):
    print("\n--- Training Expected Flow Model (LeakDB) ---")
    # Train only on normal conditions
    train_normal = train_df[train_df["leak"] == 0].copy()
    
    # Identify total demand as predictor
    demand_cols = [c for c in train_normal.columns if c.startswith("demand_")]
    train_normal["total_demand"] = train_normal[demand_cols].sum(axis=1)
    
    features = ["total_demand", "hour", "day_of_week"]
    X_train = train_normal[features]
    
    flow_cols = [c for c in train_normal.columns if c.startswith("flow_")]
    models = {}
    
    for fc in flow_cols:
        y_train = train_normal[fc]
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        models[fc] = model
        
    with open(os.path.join(MODELS_DIR, "expected_flow_models.pkl"), "wb") as f:
        pickle.dump({"models": models, "features": features}, f)
        
    return models, features

def apply_expected_flow(df, models, features):
    df_out = df.copy()
    demand_cols = [c for c in df_out.columns if c.startswith("demand_")]
    df_out["total_demand"] = df_out[demand_cols].sum(axis=1)
    X = df_out[features]
    
    flow_cols = [c for c in df_out.columns if c.startswith("flow_")]
    for fc in flow_cols:
        expected = models[fc].predict(X)
        df_out[f"expected_{fc}"] = expected
        df_out[f"residual_{fc}"] = df_out[fc] - expected
        df_out[f"residual_percent_{fc}"] = (df_out[f"residual_{fc}"] / (np.abs(expected) + 1.0)) * 100.0
        
    return df_out

def train_classifiers(train_df, val_df):
    print("\n--- Training Multiple Baselines (LeakDB) ---")
    
    y_train = train_df["leak"]
    y_val = val_df["leak"]
    
    flow_cols = [c for c in train_df.columns if c.startswith("flow_") and "expected" not in c and "residual" not in c]
    press_cols = [c for c in train_df.columns if c.startswith("pressure_")]
    dem_cols = [c for c in train_df.columns if c.startswith("demand_")]
    resid_cols = [c for c in train_df.columns if c.startswith("residual_flow_")]
    bal_cols = [c for c in train_df.columns if c.startswith("balance_")]
    
    feature_sets = {
        "A_FlowOnly": flow_cols,
        "B_FlowPressure": flow_cols + press_cols,
        "C_FlowDemand": flow_cols + dem_cols,
        "D_ResidualOnly": resid_cols,
        "E_ResidualPressureBalance": resid_cols + press_cols + bal_cols
    }
    
    # Calculate scale_pos_weight
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    spw = neg / pos if pos > 0 else 1.0
    spw = min(spw, 10.0)
    
    trained_models = {}
    
    for name, feats in feature_sets.items():
        print(f"Training {name} with {len(feats)} features...")
        X_train = train_df[feats]
        X_val = val_df[feats]
        
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            scale_pos_weight=spw,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        trained_models[name] = {
            "model": clf,
            "features": feats
        }
        
    # Unsupervised IF
    print("Training F_IsolationForest on normal data...")
    normal_feats = resid_cols + press_cols + bal_cols
    X_train_norm = train_df[train_df["leak"] == 0][normal_feats]
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_norm)
    trained_models["F_IsolationForest"] = {
        "model": iso_forest,
        "features": normal_feats
    }
        
    with open(os.path.join(MODELS_DIR, "leakdb_classifiers.pkl"), "wb") as f:
        pickle.dump(trained_models, f)
        
    return trained_models

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "LeakDB", "CCWI-WDSA2018", "Benchmarks", "Hanoi_CMH")
    df = load_all_scenarios(base_dir)
    df = temporal_features(df)
    
    # Backwards fill NaNs just in case
    df = df.bfill().ffill()
    
    train_df, val_df, test_df = split_scenarios(df)
    
    ef_models, ef_feats = train_expected_flow_model(train_df)
    
    train_df = apply_expected_flow(train_df, ef_models, ef_feats)
    val_df = apply_expected_flow(val_df, ef_models, ef_feats)
    test_df = apply_expected_flow(test_df, ef_models, ef_feats)
    
    # Save datasets for evaluation phase
    train_df.to_pickle(os.path.join(RESULTS_DIR, "train_df.pkl"))
    val_df.to_pickle(os.path.join(RESULTS_DIR, "val_df.pkl"))
    test_df.to_pickle(os.path.join(RESULTS_DIR, "test_df.pkl"))
    
    train_classifiers(train_df, val_df)
    print("Training complete. Datasets and models saved.")

if __name__ == "__main__":
    main()
