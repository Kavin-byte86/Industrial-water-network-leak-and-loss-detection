"""
GET /datasink/latest          — §7.1
GET /datasink/export?since_ticks=N — §7.2

Flat-row views over the session store history, matching the ML training
dataset column naming as closely as possible.

/datasink/export reads from Postgres so historical data survives restarts.
"""

import logging

from fastapi import APIRouter, Query

from app.dependencies import store

router = APIRouter(prefix="/datasink", tags=["datasink"])

logger = logging.getLogger(__name__)


def _flatten_snapshot(snap: dict) -> dict:
    """
    Convert a nested tick snapshot into the flat column schema expected by
    the classifier pipeline (see §7.1).

    Output keys:
        timestamp, hour, day_of_week, month, shift,
        flow_J1 … flow_J16,
        pressure_J1 … pressure_J16,
        production_M1 … production_M8,
        machine_status_M1 … machine_status_M8,
        tap_status_T1 … tap_status_T3
    """
    flat: dict = {
        "timestamp": snap["timestamp"],
        "hour": snap["hour"],
        "day_of_week": snap["day_of_week"],
        "month": snap["month"],
        "shift": snap["shift"],
    }

    # Flows (J1–J16)
    for jid, val in snap.get("flows", {}).items():
        flat[f"flow_{jid}"] = val

    # Pressures (J1–J16)
    for jid, val in snap.get("pressures", {}).items():
        flat[f"pressure_{jid}"] = val

    # Machine production & state
    for mid, minfo in snap.get("machines", {}).items():
        flat[f"production_{mid}"] = minfo["production_pct"]
        flat[f"machine_status_{mid}"] = minfo["state"]

    # Tap state
    for tid, tinfo in snap.get("taps", {}).items():
        flat[f"tap_status_{tid}"] = tinfo["state"]

    # Leak labels: ground truth from injection, not the detector's opinion, so
    # exported rows are usable as supervised training data.
    injected = snap.get("injected_leaks") or {}
    flat["leak"] = 1 if injected else 0
    flat["leak_rate"] = round(sum(injected.values()), 2)
    flat["leak_zone"] = (
        f"ZONE_{max(injected, key=injected.get)}" if injected else None
    )

    return flat


@router.get("/latest")
def datasink_latest():
    """Return the most recent tick in flat ML-dataset column format."""
    if store.latest is None:
        return {"detail": "No simulation data yet."}
    return _flatten_snapshot(store.latest)


@router.get("/export")
def datasink_export(since_ticks: int = Query(100, ge=1, le=10000)):
    """Return the last N ticks as flat rows.

    Reads from Postgres (survives restarts) with an in-memory fallback
    if the DB is unavailable.
    """
    try:
        from app.db.database import get_session
        from app.db.models import TickRecord

        session = get_session()
        try:
            records = (
                session.query(TickRecord)
                .order_by(TickRecord.id.desc())
                .limit(since_ticks)
                .all()
            )
            rows = [r.to_flat_dict() for r in reversed(records)]
            return {"rows": rows, "count": len(rows), "source": "postgres"}
        finally:
            session.close()
    except Exception as exc:
        logger.warning("DB read failed, falling back to in-memory: %s", exc)
        snapshots = store.get_history(since_ticks)
        rows = [_flatten_snapshot(s) for s in snapshots]
        return {"rows": rows, "count": len(rows), "source": "memory"}

