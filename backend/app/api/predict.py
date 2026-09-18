"""
GET /predict/current  — the trained 4-stage classifier's verdict on the latest tick.
GET /predict/status   — whether the models are loaded, and what they expect.

The prediction is computed during the tick (see simulation/engine.py) so the
dashboard gets it in the same poll as everything else; this endpoint exposes it
on its own for tooling and debugging.
"""

from fastapi import APIRouter

from app import ml_detector
from app.dependencies import store

router = APIRouter(prefix="/predict", tags=["predict"])


@router.get("/current")
def predict_current():
    """Classifier output for the most recent tick."""
    if store.latest is None:
        return {"detail": "No simulation data yet. Call POST /simulation/step first."}
    cached = store.latest.get("ml")
    if cached is not None:
        return cached
    return ml_detector.predict_from_history(list(store.history))


@router.get("/status")
def predict_status():
    """Model loading state and feature expectations."""
    return ml_detector.status()
