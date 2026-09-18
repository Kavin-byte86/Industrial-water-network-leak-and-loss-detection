"""
Leak injection — implements the interface this module previously reserved.

Physics (section 2 of the topology docs, and README section 5):

A leak at node X escapes *below* X's meter but *above* the meters of X's
children. So the extra water is measured at X and at every ancestor of X, while
X's children still read normal demand:

    leak at J3  ->  J1 up, J3 up, J8/J9/J10 unchanged
                ->  balance_J3 = flow_J3 - (J8+J9+J10) = leak_rate

For a leaf junction there are no child meters, so the same leak instead shows up
as the junction reading more than its machine or tap is actually drawing.

Aggregation applies this by adding the leak at each node *after* summing that
node's children (the children already carry their own leaks) and *before*
measurement noise, so a leak is indistinguishable from real flow to the sensor.
"""

from __future__ import annotations

from app.network.topology import ALL_JUNCTION_IDS


class LeakRegistry:
    """Active leaks, keyed by the node they occur at."""

    def __init__(self) -> None:
        self._leaks: dict[str, float] = {}

    def inject_leak(self, node_id: str, rate_lpm: float) -> None:
        """Start (or update) a leak of `rate_lpm` L/min at `node_id`."""
        if node_id not in ALL_JUNCTION_IDS:
            raise ValueError(
                f"Unknown node '{node_id}'. Valid: {', '.join(ALL_JUNCTION_IDS)}"
            )
        if rate_lpm < 0:
            raise ValueError("Leak rate must be non-negative.")
        if rate_lpm == 0:
            self._leaks.pop(node_id, None)
        else:
            self._leaks[node_id] = float(rate_lpm)

    def clear_leak(self, node_id: str) -> None:
        """Stop the leak at `node_id`, if any."""
        self._leaks.pop(node_id, None)

    def clear_all(self) -> None:
        self._leaks.clear()

    def get_active_leaks(self) -> dict[str, float]:
        """Node id → leak rate (L/min) for every active leak."""
        return dict(self._leaks)

    @property
    def total_rate(self) -> float:
        return sum(self._leaks.values())

    def rate_at(self, node_id: str) -> float:
        return self._leaks.get(node_id, 0.0)
