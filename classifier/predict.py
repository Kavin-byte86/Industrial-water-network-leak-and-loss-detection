import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# config.py is in the same directory (classifier/)
import config

try:
    from classifier.features import add_advanced_features, advanced_feature_names
except ImportError:  # running from inside the classifier/ directory
    from features import add_advanced_features, advanced_feature_names

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
        # Operating point. The v2 classifier is tuned to a non-default threshold on
        # the validation set and persists it; falling back to 0.5 would silently
        # evaluate a different operating point than the one that was tuned.
        self.threshold = 0.5
        self._warned_missing = set()
        self._needs_advanced_features = False

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
            self.threshold = float(data.get("threshold", 0.5))
            # Detect whether this model expects the v2 advanced feature block.
            advanced = set(advanced_feature_names())
            self._needs_advanced_features = bool(advanced & set(self.stage2_features))

        # Stage 3 (Optional, might not exist if no leaks were in training)
        s3_path = os.path.join(self.models_dir, 'stage3_leak_zone.pkl')
        if os.path.exists(s3_path):
            with open(s3_path, 'rb') as f:
                data = pickle.load(f)
                self.stage3_model = data["model"]
                self.label_encoder = data["label_encoder"]
                self.stage3_features = data.get("features", self.stage2_features)

        # Stage 4 (Optional)
        s4_path = os.path.join(self.models_dir, 'stage4_leak_rate.pkl')
        if os.path.exists(s4_path):
            with open(s4_path, 'rb') as f:
                data = pickle.load(f)
                self.stage4_model = data["model"]
                self.stage4_features = data.get("features", self.stage2_features)

        print(f"Models loaded successfully (decision threshold {self.threshold:.2f}).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_stage1(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds ML expected flows and residuals for all 16 sensors, vectorised."""
        X_stage1 = df[self.num_features_s1 + self.cat_features_s1]

        new_cols = {}
        for i in range(1, 17):
            target = f"flow_J{i}"
            preds = self.stage1_models[target].predict(X_stage1)

            new_cols[f"ml_expected_flow_J{i}"] = preds
            if target in df.columns:
                residual = df[target].to_numpy() - preds
                new_cols[f"ml_flow_residual_J{i}"] = residual
                new_cols[f"ml_flow_residual_percent_J{i}"] = (
                    residual / (np.abs(preds) + 1.0)
                ) * 100.0
            else:
                # Without the actual reading there is no residual to compute.
                new_cols[f"ml_flow_residual_J{i}"] = 0.0
                new_cols[f"ml_flow_residual_percent_J{i}"] = 0.0

        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _ensure_features(self, df: pd.DataFrame, features: List[str],
                         stage: str = "") -> pd.DataFrame:
        """Returns df[features], filling any absent column with 0.0.

        Zero-filling a feature the model was trained on degrades predictions
        silently, so anything still missing at this point is reported once rather
        than swallowed.
        """
        missing = [c for c in features if c not in df.columns]
        if missing:
            key = (stage, tuple(missing))
            if key not in self._warned_missing:
                self._warned_missing.add(key)
                preview = ", ".join(missing[:5])
                more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
                print(
                    f"WARNING: {stage} is missing {len(missing)} trained feature(s); "
                    f"substituting 0.0 and predictions will be degraded: {preview}{more}"
                )
            df = pd.concat(
                [df, pd.DataFrame(0.0, index=df.index, columns=missing)], axis=1
            )
        return df[features]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the 4-stage prediction pipeline on a single network state (dictionary).

        The state dict MUST contain all raw sensor readings (flow, pressure),
        operational context (production rates, statuses), and temporal/rolling features
        as expected by the models.
        """
        return self.predict_batch(pd.DataFrame([state]))[0]

    def predict_batch(self, states: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Runs the 4-stage pipeline over many network states at once.

        Each stage costs one vectorised model call for the whole batch instead of
        one per row, which is what makes replaying a window of telemetry practical.
        Returns one result dict per input row, in input order.
        """
        if len(states) == 0:
            return []

        df = self._apply_stage1(states.reset_index(drop=True))

        # The v2 classifier is trained on lag / deviation / production-relative
        # features. Build them from the batch rather than letting them fall
        # through to the zero-fill path below.
        if self._needs_advanced_features:
            df = add_advanced_features(df)

        # Stage 2: leak detection
        X_stage2 = self._ensure_features(df, self.stage2_features, "Stage 2")
        probs = self.stage2_model.predict_proba(X_stage2)[:, 1]
        flags = probs >= self.threshold

        results = [
            {
                "leak_detected": bool(flag),
                "leak_probability": float(prob),
                "leak_zone": "NONE",
                "leak_rate": 0.0,
            }
            for prob, flag in zip(probs, flags)
        ]

        # Stages 3 and 4 only run on the rows flagged as leaking.
        leak_idx = np.flatnonzero(flags)
        if len(leak_idx) and self.stage3_model is not None and self.stage4_model is not None:
            leak_rows = df.iloc[leak_idx]

            X_stage3 = self._ensure_features(leak_rows, self.stage3_features, "Stage 3")
            zone_idx = self.stage3_model.predict(X_stage3)
            zones = self.label_encoder.inverse_transform(zone_idx)

            X_stage4 = self._ensure_features(leak_rows, self.stage4_features, "Stage 4")
            rates = self.stage4_model.predict(X_stage4)

            for pos, zone, rate in zip(leak_idx, zones, rates):
                results[pos]["leak_zone"] = str(zone)
                results[pos]["leak_rate"] = float(rate)

        return results
