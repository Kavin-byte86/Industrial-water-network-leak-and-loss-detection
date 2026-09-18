"""
POST /simulation/step   — §6.6  (advance one tick, return new state)
POST /simulation/pause  — §6.7
POST /simulation/resume — §6.7
POST /simulation/reset  — §6.8
"""

from fastapi import APIRouter

from app.dependencies import engine

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/step")
def simulation_step():
    """Advance exactly one tick and return the resulting state."""
    snapshot = engine.tick()
    return snapshot


@router.post("/pause")
def simulation_pause():
    """Pause the background auto-tick loop."""
    engine.pause()
    return {"status": "paused"}


@router.post("/resume")
def simulation_resume():
    """Resume the background auto-tick loop."""
    engine.resume()
    return {"status": "resumed"}


@router.post("/reset")
def simulation_reset():
    """Reset the simulation to initial state (all OFF, clock reset)."""
    engine.reset()
    return {"status": "reset", "detail": "All machines OFF, taps CLOSED, clock reset."}
