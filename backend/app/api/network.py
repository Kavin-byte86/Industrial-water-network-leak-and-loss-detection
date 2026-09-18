"""
GET /network/topology — §6.1

Returns the static graph (nodes + edges) for frontend rendering.
No simulation state included.
"""

from fastapi import APIRouter

from app.network.topology import get_topology_serialized

router = APIRouter(prefix="/network", tags=["network"])


@router.get("/topology")
def get_topology():
    """Return the full static network topology."""
    return {"nodes": get_topology_serialized()}
