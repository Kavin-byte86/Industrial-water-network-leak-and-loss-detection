"""
Static network topology — matches §2 of BACKEND_IMPLEMENTATION_PLAN.md.

Topology tree:
    J1 (main incoming line)
     ├─ J2
     │   ├─ J5 → M1
     │   └─ J6 → M2
     ├─ J3
     │   ├─ J8  → M3
     │   ├─ J9  → M4
     │   └─ J10 → M5
     ├─ J4
     │   ├─ J11 → M6
     │   └─ J12 → M7
     └─ J7
         ├─ J13 → M8
         ├─ J14 → T1 (tap)
         ├─ J15 → T2 (tap)
         └─ J16 → T3 (tap)

Balance equations:
    J1  ≈ J2  + J3  + J4  + J7
    J2  ≈ J5  + J6
    J3  ≈ J8  + J9  + J10
    J4  ≈ J11 + J12
    J7  ≈ J13 + J14 + J15 + J16
"""

from __future__ import annotations

from typing import Literal


# ── Node type alias ────────────────────────────────────────────────────────
NodeType = Literal["junction", "machine", "tap"]


# ── Node definition ────────────────────────────────────────────────────────
class TopologyNode:
    """Represents a single node in the water-network topology."""

    __slots__ = ("id", "type", "parent_id", "children", "endpoint_id")

    def __init__(
        self,
        id: str,
        type: NodeType,
        parent_id: str | None = None,
        children: list[str] | None = None,
        endpoint_id: str | None = None,
    ):
        self.id = id
        self.type = type
        self.parent_id = parent_id
        self.children: list[str] = children or []
        # For leaf junctions, this is the machine/tap id that the junction feeds
        self.endpoint_id = endpoint_id

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.id,
            "type": self.type,
            "parent_id": self.parent_id,
            "children": self.children,
        }
        if self.endpoint_id is not None:
            d["endpoint_id"] = self.endpoint_id
        return d


# ── Full topology ──────────────────────────────────────────────────────────
# Hardcoded as specified — do NOT infer dynamically.

NODES: dict[str, TopologyNode] = {}


def _add(node: TopologyNode) -> None:
    NODES[node.id] = node


# Leaf junctions (connected directly to a machine or tap)
_add(TopologyNode("J5",  "junction", parent_id="J2", endpoint_id="M1"))
_add(TopologyNode("J6",  "junction", parent_id="J2", endpoint_id="M2"))
_add(TopologyNode("J8",  "junction", parent_id="J3", endpoint_id="M3"))
_add(TopologyNode("J9",  "junction", parent_id="J3", endpoint_id="M4"))
_add(TopologyNode("J10", "junction", parent_id="J3", endpoint_id="M5"))
_add(TopologyNode("J11", "junction", parent_id="J4", endpoint_id="M6"))
_add(TopologyNode("J12", "junction", parent_id="J4", endpoint_id="M7"))
_add(TopologyNode("J13", "junction", parent_id="J7", endpoint_id="M8"))
_add(TopologyNode("J14", "junction", parent_id="J7", endpoint_id="T1"))
_add(TopologyNode("J15", "junction", parent_id="J7", endpoint_id="T2"))
_add(TopologyNode("J16", "junction", parent_id="J7", endpoint_id="T3"))

# Intermediate junctions (children are other junctions)
_add(TopologyNode("J2", "junction", parent_id="J1", children=["J5", "J6"]))
_add(TopologyNode("J3", "junction", parent_id="J1", children=["J8", "J9", "J10"]))
_add(TopologyNode("J4", "junction", parent_id="J1", children=["J11", "J12"]))
_add(TopologyNode("J7", "junction", parent_id="J1", children=["J13", "J14", "J15", "J16"]))

# Root junction
_add(TopologyNode("J1", "junction", parent_id=None, children=["J2", "J3", "J4", "J7"]))


# ── Machine / tap catalogue (for reference only — no flow info here) ──────
MACHINES: list[str] = [f"M{i}" for i in range(1, 9)]
TAPS: list[str] = [f"T{i}" for i in range(1, 4)]

# Junction → endpoint mapping (leaf junctions only)
JUNCTION_TO_ENDPOINT: dict[str, str] = {
    node.id: node.endpoint_id
    for node in NODES.values()
    if node.endpoint_id is not None
}

# Endpoint → junction reverse mapping
ENDPOINT_TO_JUNCTION: dict[str, str] = {v: k for k, v in JUNCTION_TO_ENDPOINT.items()}

# Leaf junction IDs (junctions with no child junctions — they feed a machine or tap)
LEAF_JUNCTIONS: list[str] = [
    nid for nid, node in NODES.items() if not node.children
]

# Ordered list of all junction IDs (useful for consistent iteration)
ALL_JUNCTION_IDS: list[str] = sorted(NODES.keys(), key=lambda x: int(x[1:]))


def get_topology_serialized() -> list[dict]:
    """Return the full topology as a JSON-serializable list for the API."""
    return [NODES[jid].to_dict() for jid in ALL_JUNCTION_IDS]
