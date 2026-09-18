import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report
)

# Ensure config can be loaded
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import config

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# STAGE 1: Expected Flow Models (16 Models)
# -------------------------------------------------------------------------
def train_stage_1(train_df: pd.DataFrame, val_df: pd.DataFrame):
    print("\n--- STAGE 1: Training Expected Flow Models (16 Sensors) ---")
    
    # Filter to only normal operations (leak == 0)
    train_normal = train_df[train_df["leak"] == 0].copy()
    val_normal = val_df[val_df["leak"] == 0].copy()
    
    # Features: Operational context only
    numeric_features = [f"production_M{i}" for i in range(1, 9)] + ["hour", "day_of_week", "shift"]
    categorical_features = [f"machine_status_M{i}" for i in range(1, 9)] + [f"tap_status_T{i}" for i in range(1, 4)]
    
    X_train = train_normal[numeric_features + categorical_features]
    X_val = val_normal[numeric_features + categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    models = {}
    
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    
    for i in range(1, 17):
        target = f"flow_J{i}"
        y_train = train_normal[target]
        y_val = val_normal[target]
        
        print(f"Training Stage 1 Model for {target}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', Ridge(alpha=1.0))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Validation
        val_preds = pipeline.predict(X_val)
        mae = mean_absolute_error(y_val, val_preds)
        print(f"  {target} Validation MAE: {mae:.2f} L/min")
        
        models[target] = pipeline
        
    # Save the models
    with open(os.path.join(MODELS_DIR, 'stage1_expected_flow_models.pkl'), 'wb') as f:
        pickle.dump({
            "models": models,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features
        }, f)
        
    return models

# -------------------------------------------------------------------------
# Feature generation helper using Stage 1
# -------------------------------------------------------------------------
def generate_stage1_predictions_and_residuals(df: pd.DataFrame, stage1_models: Dict) -> pd.DataFrame:
    print("Generating Stage 1 predictions and residuals for the dataset...")
    df_out = df.copy()
    
    # Ensure all required features exist
    num_features = [f"production_M{i}" for i in range(1, 9)] + ["hour", "day_of_week", "shift"]
    cat_features = [f"machine_status_M{i}" for i in range(1, 9)] + [f"tap_status_T{i}" for i in range(1, 4)]
    
    X = df_out[num_features + cat_features]
    
    for i in range(1, 17):
        target = f"flow_J{i}"
        pipeline = stage1_models[target]
        preds = pipeline.predict(X)
        
        # Add to df
        df_out[f"ml_expected_flow_J{i}"] = preds
        residual = df_out[f"flow_J{i}"] - preds
        df_out[f"ml_flow_residual_J{i}"] = residual
        df_out[f"ml_flow_residual_percent_J{i}"] = (residual / (np.abs(preds) + 1.0)) * 100.0
        
    return df_out

# -------------------------------------------------------------------------
# STAGE 2: Leak Classifier
# -------------------------------------------------------------------------
def train_stage_2(train_df: pd.DataFrame, val_df: pd.DataFrame):
    print("\n--- STAGE 2: Training Leak Detection Classifier ---")
    
    # We include all rows (leak 0 and 1)
    feature_cols = []
    # Actual flows and ML predictions
    for i in range(1, 17):
        feature_cols.extend([f"flow_J{i}", f"ml_expected_flow_J{i}", f"ml_flow_residual_J{i}", f"ml_flow_residual_percent_J{i}"])
        feature_cols.extend([f"rolling_mean_flow_J{i}", f"rolling_std_flow_J{i}", f"flow_change_J{i}"])
        
    # Pressure
    for p in ["J1", "J2", "J3", "J4", "J7"]:
        feature_cols.extend([f"pressure_{p}", f"pressure_{p}_change", f"rolling_mean_pressure_{p}", f"rolling_std_pressure_{p}"])
        
    # Balances
    for b in ["J1", "J2", "J3", "J4", "J7"]:
        feature_cols.append(f"balance_{b}")
        
    # Operational
    feature_cols.extend([f"production_M{i}" for i in range(1, 9)] + ["hour", "day_of_week"])
    
    # Filter available columns (in case some are missing in dataset)
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    X_train = train_df[feature_cols]
    y_train = train_df["leak"]
    
    X_val = val_df[feature_cols]
    y_val = val_df["leak"]
    
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=config.RANDOM_SEED, n_jobs=-1))
    ])
    
    print("Fitting XGBoost Classifier...")
    pipeline.fit(X_train, y_train)
    
    val_preds = pipeline.predict(X_val)
    print("Stage 2 Validation Metrics:")
    print(classification_report(y_val, val_preds, digits=4))
    
    with open(os.path.join(MODELS_DIR, 'stage2_leak_classifier.pkl'), 'wb') as f:
        pickle.dump({
            "model": pipeline,
            "features": feature_cols
        }, f)
        
    return pipeline, feature_cols

# -------------------------------------------------------------------------
# STAGE 3: Leak Zone Multiclass Classifier
# -------------------------------------------------------------------------
def train_stage_3(train_df: pd.DataFrame, val_df: pd.DataFrame, stage2_features: List[str]):
    print("\n--- STAGE 3: Training Leak Zone Locator ---")
    
    train_leak = train_df[train_df["leak"] == 1].copy()
    val_leak = val_df[val_df["leak"] == 1].copy()
    
    if len(train_leak) == 0 or len(val_leak) == 0:
        print("Warning: Not enough leak events to train Stage 3.")
        return None
        
    X_train = train_leak[stage2_features]
    y_train = train_leak["leak_zone"]
    
    X_val = val_leak[stage2_features]
    y_val = val_leak["leak_zone"]
    
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    
    # Filter val data to only classes seen in train
    seen_classes = set(label_encoder.classes_)
    val_mask = y_val.isin(seen_classes)
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    y_val_enc = label_encoder.transform(y_val)
    
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            objective='multi:softprob', 
            n_estimators=100, 
            max_depth=6, 
            random_state=config.RANDOM_SEED, 
            n_jobs=-1
        ))
    ])
    
    print(f"Fitting XGBoost Multiclass Classifier on {len(X_train)} leak samples...")
    pipeline.fit(X_train, y_train_enc)
    
    val_preds_enc = pipeline.predict(X_val)
    acc = accuracy_score(y_val_enc, val_preds_enc)
    print(f"Stage 3 Validation Accuracy: {acc:.4f}")
    
    with open(os.path.join(MODELS_DIR, 'stage3_leak_zone.pkl'), 'wb') as f:
        pickle.dump({
            "model": pipeline,
            "features": stage2_features,
            "label_encoder": label_encoder
        }, f)

# -------------------------------------------------------------------------
# STAGE 4: Leak Rate Regressor
# -------------------------------------------------------------------------
def train_stage_4(train_df: pd.DataFrame, val_df: pd.DataFrame, stage2_features: List[str]):
    print("\n--- STAGE 4: Training Leak Rate Regressor ---")
    
    train_leak = train_df[train_df["leak"] == 1].copy()
    val_leak = val_df[val_df["leak"] == 1].copy()
    
    if len(train_leak) == 0 or len(val_leak) == 0:
        print("Warning: Not enough leak events to train Stage 4.")
        return None
        
    X_train = train_leak[stage2_features]
    y_train = train_leak["leak_rate"]
    
    X_val = val_leak[stage2_features]
    y_val = val_leak["leak_rate"]
    
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', XGBRegressor(n_estimators=100, max_depth=6, random_state=config.RANDOM_SEED, n_jobs=-1))
    ])
    
    print("Fitting XGBoost Regressor...")
    pipeline.fit(X_train, y_train)
    
    val_preds = pipeline.predict(X_val)
    mae = mean_absolute_error(y_val, val_preds)
    print(f"Stage 4 Validation MAE: {mae:.2f} L/min")
    
    with open(os.path.join(MODELS_DIR, 'stage4_leak_rate.pkl'), 'wb') as f:
        pickle.dump({
            "model": pipeline,
            "features": stage2_features
        }, f)

# -------------------------------------------------------------------------
# MAIN ORCHESTRATION
# -------------------------------------------------------------------------
def main():
    print("Starting Machine Learning Pipeline Training...")
    
    dataset_path = config.ML_TRAINING_CSV
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    print(f"Loading dataset from {dataset_path}...")
    # Load dataset. It is already sorted chronologically by the generator.
    df = pd.read_csv(dataset_path)
    
    # Handle missing values (NaNs) injected by the simulation
    print("Handling missing data with forward and backward fill...")
    df = df.ffill().bfill()
    
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # Chronological Split
    # Train: 70%, Val: 15%, Test: 15%
    n_total = len(df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train+n_val].copy()
    test_df = df.iloc[n_train+n_val:].copy()
    
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # --- Execute Pipeline ---
    t0 = time.time()
    
    # STAGE 1
    stage1_models = train_stage_1(train_df, val_df)
    
    # Enrich datasets with ML predicted residuals
    train_df_enriched = generate_stage1_predictions_and_residuals(train_df, stage1_models)
    val_df_enriched = generate_stage1_predictions_and_residuals(val_df, stage1_models)
    test_df_enriched = generate_stage1_predictions_and_residuals(test_df, stage1_models)
    
    # STAGE 2
    stage2_model, stage2_features = train_stage_2(train_df_enriched, val_df_enriched)
    
    # STAGE 3
    train_stage_3(train_df_enriched, val_df_enriched, stage2_features)
    
    # STAGE 4
    train_stage_4(train_df_enriched, val_df_enriched, stage2_features)
    
    print(f"\nAll models trained and saved successfully in {(time.time() - t0):.1f} seconds!")
    
    # Final Test Set Evaluation (Stage 2)
    print("\n--- Final Test Set Evaluation (Stage 2) ---")
    test_preds = stage2_model.predict(test_df_enriched[stage2_features])
    print(classification_report(test_df_enriched["leak"], test_preds, digits=4))

if __name__ == "__main__":
    main()
