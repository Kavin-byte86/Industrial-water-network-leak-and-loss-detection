"""
The trained 4-stage classifier, running on live telemetry.

This is the ML path. It does not reimplement any feature: it builds the exact
master-table schema the generator produced, hands it to
`classifier/feature_engineering.py` — the same function that built the training
set — and then calls the same `WaterNetworkLeakDetector` used offline. Anything
computed twice would eventually diverge; nothing here is computed twice.

    history of ticks  ->  master schema DataFrame
                      ->  build_feature_engineered_dataset()   (training code)
                      ->  WaterNetworkLeakDetector.predict_batch()
                      ->  Stage 2 leak? Stage 3 zone, Stage 4 rate

Rolling means, standard deviations, first differences and lag features all need
history, so the detector reports `warming_up` until enough ticks exist. Serving
a prediction from a half-filled window would just be a confident guess.
"""

from __future__ import annotations

import os
import sys
import threading

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CLASSIFIER_DIR = os.path.join(_REPO_ROOT, "classifier")
for _p in (_REPO_ROOT, _CLASSIFIER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.classifier_config import PRESSURE_SENSORS  # noqa: E402

# The feature pipeline needs at least a full rolling window (12 ticks = 1 hour)
# plus the longest lag (6) before its outputs mean anything.
MIN_HISTORY_TICKS = 18
# How much history to feed per call. More than this costs time for no gain,
# since the longest window is 12.
WINDOW_TICKS = 40

MODEL_VERSION = os.getenv("CLASSIFIER_VERSION", "v2")

_lock = threading.Lock()
_detector = None
_pd = None
_build_features = None
_load_error: str | None = None


def _ensure_loaded() -> bool:
    """Load pandas, the feature builder and the models once."""
    global _detector, _pd, _build_features, _load_error
    if _detector is not None:
        return True
    if _load_error is not None:
        return False
    with _lock:
        if _detector is not None:
            return True
        try:
            import pandas as pd
            from classifier.feature_engineering import build_feature_engineered_dataset
            from classifier.predict import WaterNetworkLeakDetector

            models_dir = os.path.join(_CLASSIFIER_DIR, "models", MODEL_VERSION)
            if not os.path.isdir(models_dir):
                models_dir = os.path.join(_CLASSIFIER_DIR, "models")

            _pd = pd
            _build_features = build_feature_engineered_dataset
            _detector = WaterNetworkLeakDetector(models_dir=models_dir)
            return True
        except Exception as exc:  # pragma: no cover - depends on local install
            _load_error = f"{type(exc).__name__}: {exc}"
            return False


def _shift_number(hour: int) -> int:
    """Shift as the training data encodes it: Day=1, Evening=2, Night=3."""
    if 6 <= hour < 14:
        return 1
    if 14 <= hour < 22:
        return 2
    return 3


def _snapshot_to_row(snap: dict) -> dict:
    """One tick in the generator's master-table schema."""
    timestamp = snap["timestamp"]
    date, _, time_part = timestamp.partition("T")

    row = {"date": date, "time": time_part or "00:00:00"}

    for jid, value in snap.get("flows", {}).items():
        row[f"flow_{jid}"] = value

    # The training data only has pressure at the five headers.
    pressures = snap.get("pressures", {})
    for jid in PRESSURE_SENSORS:
        if jid in pressures:
            row[f"pressure_{jid}"] = pressures[jid]

    for mid, info in snap.get("machines", {}).items():
        row[f"production_{mid}"] = info.get("production_pct", 0.0)
        row[f"machine_status_{mid}"] = info.get("state", "OFF")

    for tid, info in snap.get("taps", {}).items():
        row[f"tap_status_{tid}"] = info.get("state", "CLOSED")

    # The feature builder moves these to the end and excludes them from inputs;
    # they are never read as features.
    row["leak"] = 0
    row["leak_rate"] = 0.0
    row["leak_zone"] = "NONE"
    return row


def unavailable(reason: str, **extra) -> dict:
    return {
        "available": False,
        "reason": reason,
        "leak_detected": None,
        "leak_probability": None,
        "leak_zone": None,
        "leak_rate_lpm": None,
        "model_version": MODEL_VERSION,
        **extra,
    }


def predict_from_history(history: list) -> dict:
    """
    Run the trained pipeline over the most recent ticks.

    Returns the prediction for the latest tick, or a structured "not available"
    result explaining why — never a fabricated number.
    """
    if not _ensure_loaded():
        return unavailable(
            "models_not_loaded",
            detail=_load_error,
        )

    if len(history) < MIN_HISTORY_TICKS:
        return unavailable(
            "warming_up",
            detail=(
                f"Needs {MIN_HISTORY_TICKS} ticks of history for rolling and lag "
                f"features; have {len(history)}."
            ),
            ticks_needed=MIN_HISTORY_TICKS - len(history),
        )

    try:
        window = history[-WINDOW_TICKS:]
        master = _pd.DataFrame([_snapshot_to_row(s) for s in window])

        # Shift is numeric in the training data; the feature builder derives it
        # from the timestamp, so nothing needs to be passed in.
        engineered = _build_features(master)

        results = _detector.predict_batch(engineered)
        latest = results[-1]

        return {
            "available": True,
            "reason": None,
            "leak_detected": bool(latest["leak_detected"]),
            "leak_probability": round(float(latest["leak_probability"]), 4),
            "leak_zone": latest["leak_zone"] if latest["leak_detected"] else None,
            "leak_rate_lpm": round(float(latest["leak_rate"]), 2) if latest["leak_detected"] else 0.0,
            "threshold": _detector.threshold,
            "model_version": MODEL_VERSION,
            "features_used": len(_detector.stage2_features),
            "history_ticks": len(window),
        }
    except Exception as exc:  # pragma: no cover - surfaced rather than hidden
        return unavailable("prediction_error", detail=f"{type(exc).__name__}: {exc}")


def status() -> dict:
    """Whether the models are loaded, for diagnostics."""
    loaded = _ensure_loaded()
    return {
        "loaded": loaded,
        "model_version": MODEL_VERSION,
        "error": _load_error,
        "min_history_ticks": MIN_HISTORY_TICKS,
        "stage2_features": len(_detector.stage2_features) if loaded else None,
        "threshold": _detector.threshold if loaded else None,
    }
