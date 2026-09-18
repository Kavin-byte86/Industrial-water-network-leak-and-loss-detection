"""
Bottom-up junction aggregation — §3.3 of BACKEND_IMPLEMENTATION_PLAN.md.

Walks the topology tree from leaves to root, summing child flows at each
junction and adding Gaussian measurement noise.

The leak extension hook is called (as a no-op) between the raw sum and the
noise application, per §8.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from app.classifier_config import FLOW_NOISE_FRACTION as NOISE_SIGMA_FRACTION
from app.network.topology import (
    ALL_JUNCTION_IDS,
    JUNCTION_TO_ENDPOINT,
    NODES,
)

if TYPE_CHECKING:
    pass


def _measurement_noise(value: float) -> float:
    """Gaussian noise with σ = NOISE_SIGMA_FRACTION × value."""
    if value == 0.0:
        return 0.0
    sigma = abs(value) * NOISE_SIGMA_FRACTION
    return random.gauss(0, sigma)


def aggregate_flows(
    endpoint_flows: dict[str, float],
    leak_rates: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute flow at every junction by walking the topology bottom-up.

    Parameters
    ----------
    endpoint_flows : dict
        Mapping of endpoint id (M1–M8, T1–T3) → flow (L/min) for this tick,
        as produced by the production model.
    leak_rates : dict, optional
        Node id → leak rate (L/min). A leak is added to its own node after that
        node's children have been summed, so it propagates upward to every
        ancestor while leaving the children reading normal demand. See
        app/leak_extension_point.py.

    Returns
    -------
    dict
        Mapping of junction id (J1–J16) → measured flow (L/min) including
        measurement noise.
    """
    leak_rates = leak_rates or {}
    junction_flows: dict[str, float] = {}

    # Pass 1 — leaf junctions: flow equals their endpoint's flow
    for jid in ALL_JUNCTION_IDS:
        node = NODES[jid]
        if not node.children:
            ep_id = JUNCTION_TO_ENDPOINT.get(jid)
            raw = endpoint_flows.get(ep_id, 0.0) if ep_id else 0.0
            # Leak water passes this meter but never reaches the endpoint, so it
            # is added before noise — the sensor cannot tell it apart from demand.
            raw += leak_rates.get(jid, 0.0)
            junction_flows[jid] = raw + _measurement_noise(raw)

    # Pass 2 — intermediate junctions (processed in reverse topological
    # order: children first, parents last). Because the topology is a
    # simple fixed tree we can sort by depth (deeper first).
    # Depth map: leaf = 2, mid = 1, root = 0.
    depth_order = sorted(
        [jid for jid in ALL_JUNCTION_IDS if NODES[jid].children],
        key=lambda jid: -_depth(jid),
    )

    for jid in depth_order:
        node = NODES[jid]
        # Children already carry their own leaks, so only this node's leak is
        # added here; the result propagates on up to the root.
        raw = sum(junction_flows.get(cid, 0.0) for cid in node.children)
        raw += leak_rates.get(jid, 0.0)
        junction_flows[jid] = raw + _measurement_noise(raw)

    return junction_flows


def _depth(jid: str, _cache: dict[str, int] = {}) -> int:
    """Return the depth of a junction (root = 0)."""
    if jid in _cache:
        return _cache[jid]
    node = NODES[jid]
    if node.parent_id is None:
        _cache[jid] = 0
    else:
        _cache[jid] = _depth(node.parent_id) + 1
    return _cache[jid]
