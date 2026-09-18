"""
GET /predict/current — runs the classifier on the latest simulation state.

Returns leak detection results or a graceful "models not loaded" status.
"""

from fastapi import APIRouter

from app.api.datasink import _flatten_snapshot
from app.classifier_adapter import predict_from_state
from app.dependencies import store

router = APIRouter(prefix="/predict", tags=["predict"])


@router.get("/current")
def predict_current():
    """Run the classifier pipeline on the most recent tick."""
    if store.latest is None:
        return {"detail": "No simulation data yet. Call POST /simulation/step first."}
    flat = _flatten_snapshot(store.latest)
    return predict_from_state(flat)
