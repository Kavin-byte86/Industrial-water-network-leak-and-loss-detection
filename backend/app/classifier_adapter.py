"""
Thin adapter for calling the classifier from the backend.

Gracefully handles missing model files (they're gitignored) —
returns a "models not loaded" status instead of crashing.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Add classifier directory to path so we can import predict.py
_CLASSIFIER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "classifier")
)
if _CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, _CLASSIFIER_DIR)

_detector = None
_load_error: str | None = None


def _ensure_loaded() -> bool:
    """Attempt to load the classifier models once. Returns True if ready."""
    global _detector, _load_error
    if _detector is not None:
        return True
    if _load_error is not None:
        return False  # already tried and failed

    try:
        from predict import WaterNetworkLeakDetector
        _detector = WaterNetworkLeakDetector()
        return True
    except Exception as e:
        _load_error = str(e)
        return False


def predict_from_state(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Run the 4-stage classifier pipeline on a single flat state dict
    (same shape as GET /datasink/latest).

    Returns a result dict with leak_detected, leak_probability, etc.,
    or a status dict explaining why prediction isn't available.
    """
    if not _ensure_loaded():
        return {
            "status": "models_not_loaded",
            "detail": f"Classifier models could not be loaded: {_load_error}",
            "leak_detected": None,
            "leak_probability": None,
            "leak_zone": None,
            "leak_rate": None,
        }

    try:
        result = _detector.predict(state_dict)
        result["status"] = "ok"
        return result
    except Exception as e:
        return {
            "status": "prediction_error",
            "detail": str(e),
            "leak_detected": None,
            "leak_probability": None,
            "leak_zone": None,
            "leak_rate": None,
        }
