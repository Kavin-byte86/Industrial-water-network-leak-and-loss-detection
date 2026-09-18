import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# config.py is in the same directory (classifier/)
import config

class WaterNetworkLeakDetector:
    """
    Production Inference API for the 4-Stage Water Network Leak Detection ML System.
    """
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        else:
            self.models_dir = models_dir
            
        self.stage1_models = None
        self.stage2_model = None
        self.stage3_model = None
        self.stage4_model = None
        self.stage2_features = None
        self.label_encoder = None
        
        self.load_models()
        
    def load_models(self):
        """Loads all trained models from disk."""
        print(f"Loading models from {self.models_dir}...")
        
        # Stage 1
        with open(os.path.join(self.models_dir, 'stage1_expected_flow_models.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.stage1_models = data["models"]
            self.num_features_s1 = data["numeric_features"]
            self.cat_features_s1 = data["categorical_features"]
            
        # Stage 2
        with open(os.path.join(self.models_dir, 'stage2_leak_classifier.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.stage2_model = data["model"]
            self.stage2_features = data["features"]
            
        # Stage 3 (Optional, might not exist if no leaks were in training)
        s3_path = os.path.join(self.models_dir, 'stage3_leak_zone.pkl')
        if os.path.exists(s3_path):
            with open(s3_path, 'rb') as f:
                data = pickle.load(f)
                self.stage3_model = data["model"]
                self.label_encoder = data["label_encoder"]
                
        # Stage 4 (Optional)
        s4_path = os.path.join(self.models_dir, 'stage4_leak_rate.pkl')
        if os.path.exists(s4_path):
            with open(s4_path, 'rb') as f:
                data = pickle.load(f)
                self.stage4_model = data["model"]
                
        print("Models loaded successfully.")
        
    def predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the 4-stage prediction pipeline on a single network state (dictionary).
        
        The state dict MUST contain all raw sensor readings (flow, pressure), 
        operational context (production rates, statuses), and temporal/rolling features
        as expected by the models.
        """
        df = pd.DataFrame([state])
        
        # 1. Stage 1: Predict Expected Flows and Residuals
        X_stage1 = df[self.num_features_s1 + self.cat_features_s1]
        
        for i in range(1, 17):
            target = f"flow_J{i}"
            pred_flow = self.stage1_models[target].predict(X_stage1)[0]
            
            df[f"ml_expected_flow_J{i}"] = pred_flow
            
            if target in df.columns:
                residual = df[target].iloc[0] - pred_flow
                df[f"ml_flow_residual_J{i}"] = residual
                df[f"ml_flow_residual_percent_J{i}"] = (residual / (np.abs(pred_flow) + 1.0)) * 100.0
            else:
                # If for some reason actual flow is missing in input, we can't compute residual
                df[f"ml_flow_residual_J{i}"] = 0.0
                df[f"ml_flow_residual_percent_J{i}"] = 0.0
                
        # 2. Stage 2: Leak Detection
        # Ensure all required features are present, fill missing with 0 for safety during inference
        for col in self.stage2_features:
            if col not in df.columns:
                df[col] = 0.0
                
        X_stage2 = df[self.stage2_features]
        leak_prob = self.stage2_model.predict_proba(X_stage2)[0, 1]
        is_leak = int(self.stage2_model.predict(X_stage2)[0])
        
        result = {
            "leak_detected": bool(is_leak),
            "leak_probability": float(leak_prob),
            "leak_zone": "NONE",
            "leak_rate": 0.0
        }
        
        # 3 & 4. Stages 3 and 4: Zone and Rate (Only if leak detected)
        if is_leak and self.stage3_model is not None and self.stage4_model is not None:
            zone_idx = self.stage3_model.predict(X_stage2)[0]
            zone_str = self.label_encoder.inverse_transform([zone_idx])[0]
            result["leak_zone"] = zone_str
            
            rate = self.stage4_model.predict(X_stage2)[0]
            result["leak_rate"] = float(rate)
            
        return result
