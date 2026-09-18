"""
GET /state/current  — §6.2
GET /state/history  — §6.3

Returns the full current synchronized network state or the last N ticks.
"""

from fastapi import APIRouter, Query

from app.dependencies import store

router = APIRouter(prefix="/state", tags=["state"])


@router.get("/current")
def get_current_state():
    """Return the full current network state snapshot."""
    if store.latest is None:
        return {"detail": "No simulation data yet. Call POST /simulation/step first."}
    return store.latest


@router.get("/history")
def get_history(limit: int = Query(100, ge=1, le=1000)):
    """Return the last N tick snapshots, oldest first."""
    ticks = store.get_history(limit)
    return {"ticks": ticks, "count": len(ticks)}
