"""
POST /control/machine — §6.4
POST /control/tap     — §6.5

Accepts control commands to change machine production rates/states and tap states.
"""

from fastapi import APIRouter, HTTPException

from app.dependencies import store
from app.network.topology import MACHINES, TAPS
from app.schemas.control import MachineControlRequest, TapControlRequest

router = APIRouter(prefix="/control", tags=["control"])


@router.post("/machine")
def control_machine(req: MachineControlRequest):
    """Update a machine's production_pct and/or state (partial update)."""
    if req.machine_id not in MACHINES:
        raise HTTPException(
            400,
            f"Unknown machine_id '{req.machine_id}'. "
            f"Valid: {', '.join(MACHINES)}",
        )
    updated = store.update_machine(
        req.machine_id,
        production_pct=req.production_pct,
        state=req.state.value if req.state else None,
    )
    return {"machine_id": req.machine_id, **updated}


@router.post("/tap")
def control_tap(req: TapControlRequest):
    """Update a tap's state."""
    if req.tap_id not in TAPS:
        raise HTTPException(
            400,
            f"Unknown tap_id '{req.tap_id}'. Valid: {', '.join(TAPS)}",
        )
    updated = store.update_tap(req.tap_id, req.state.value)
    return {"tap_id": req.tap_id, **updated}
