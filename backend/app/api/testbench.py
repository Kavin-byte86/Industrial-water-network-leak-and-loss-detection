"""
Test-bench API — drives leak injection and manual sensor overrides.

This exists for the separate test-bench UI: an operator picks a node, sets a
leak rate (or forces a raw sensor value), and the main dashboard should then
detect and localise it. Nothing here is needed by the main dashboard itself.

    POST   /testbench/leak            start/update a leak at a node
    DELETE /testbench/leak/{node_id}  clear one leak
    POST   /testbench/leaks/clear     clear every leak
    GET    /testbench/leaks           list active leaks

    POST   /testbench/override        force a sensor reading
    DELETE /testbench/override/{key}  clear one override
    POST   /testbench/overrides/clear clear every override
    GET    /testbench/overrides       list active overrides

    POST   /testbench/endpoint        set a machine or tap in one call
    GET    /testbench/snapshot        everything the test UI needs, in one poll
"""

from fastapi import APIRouter, HTTPException

from app.dependencies import engine, store
from app.network.topology import (
    ALL_JUNCTION_IDS,
    ENDPOINT_TO_JUNCTION,
    MACHINES,
    TAPS,
)
from app.schemas.testbench import EndpointRequest, LeakRequest, OverrideRequest

router = APIRouter(prefix="/testbench", tags=["testbench"])

MACHINE_STATES = ("OFF", "STARTING", "RUNNING", "STOPPING", "MAINTENANCE")
TAP_STATES = ("OPEN", "CLOSED")


# ── Leaks ─────────────────────────────────────────────────────────────────
@router.post("/leak")
def set_leak(req: LeakRequest):
    """Start, update, or (with rate 0) clear a leak at one junction."""
    try:
        store.leaks.inject_leak(req.node_id, req.rate_lpm)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"leaks": store.leaks.get_active_leaks()}


@router.delete("/leak/{node_id}")
def clear_leak(node_id: str):
    store.leaks.clear_leak(node_id)
    return {"leaks": store.leaks.get_active_leaks()}


@router.post("/leaks/clear")
def clear_leaks():
    store.leaks.clear_all()
    return {"leaks": {}}


@router.get("/leaks")
def list_leaks():
    leaks = store.leaks.get_active_leaks()
    return {"leaks": leaks, "total_rate_lpm": round(store.leaks.total_rate, 2)}


# ── Sensor overrides ──────────────────────────────────────────────────────
@router.post("/override")
def set_override(req: OverrideRequest):
    """Force one sensor to a fixed reading until cleared."""
    kind, _, jid = req.sensor.partition("_")
    if kind not in ("flow", "pressure") or jid not in ALL_JUNCTION_IDS:
        raise HTTPException(
            400,
            f"Invalid sensor '{req.sensor}'. Expected flow_<J> or pressure_<J> "
            f"where <J> is one of {', '.join(ALL_JUNCTION_IDS)}.",
        )
    store.set_override(req.sensor, req.value)
    return {"overrides": dict(store.overrides)}


@router.delete("/override/{sensor}")
def clear_override(sensor: str):
    store.clear_override(sensor)
    return {"overrides": dict(store.overrides)}


@router.post("/overrides/clear")
def clear_overrides():
    store.clear_overrides()
    return {"overrides": {}}


@router.get("/overrides")
def list_overrides():
    return {"overrides": dict(store.overrides)}


# ── Endpoints (machines and taps) ─────────────────────────────────────────
@router.post("/endpoint")
def set_endpoint(req: EndpointRequest):
    """Set a machine's production/state or a tap's state by endpoint id."""
    eid = req.endpoint_id
    if eid in MACHINES:
        if req.state is not None and req.state not in MACHINE_STATES:
            raise HTTPException(
                400, f"Invalid machine state '{req.state}'. "
                     f"Valid: {', '.join(MACHINE_STATES)}")
        updated = store.update_machine(eid, req.production_pct, req.state)
        return {"endpoint_id": eid, **updated}
    if eid in TAPS:
        if req.state not in TAP_STATES:
            raise HTTPException(
                400, f"Invalid tap state '{req.state}'. "
                     f"Valid: {', '.join(TAP_STATES)}")
        updated = store.update_tap(eid, req.state)
        return {"endpoint_id": eid, **updated}
    raise HTTPException(
        400,
        f"Unknown endpoint '{eid}'. Valid: {', '.join(MACHINES + TAPS)}",
    )


# ── Combined snapshot ─────────────────────────────────────────────────────
@router.get("/snapshot")
def testbench_snapshot():
    """Everything the test UI needs in one request."""
    return {
        "tick": store.latest,
        "leaks": store.leaks.get_active_leaks(),
        "overrides": dict(store.overrides),
        "paused": engine.is_paused,
        "junctions": ALL_JUNCTION_IDS,
        "machines": MACHINES,
        "taps": TAPS,
        "endpoint_junctions": ENDPOINT_TO_JUNCTION,
    }
