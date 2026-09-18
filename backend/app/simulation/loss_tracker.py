"""
Loss accounting — turns a leak rate into quantity lost and money lost.

A rate in L/min answers "how bad is it right now". What a plant manager needs
is "how much water has this cost me so far", which means integrating the rate
over the life of the event and pricing the result.

Each detected leak becomes an *event*, keyed by node, that persists across ticks
until the leak clears. While it is open the tracker accumulates:

    volume_litres += rate_lpm * minutes_per_tick
    cost          = volume_m3 * (supply tariff + effluent tariff)

Leaked process water is normally charged twice — once to buy it, once to treat
it as effluent — so both tariffs are applied by default.
"""

from __future__ import annotations

from datetime import datetime

from app.core.config import (
    CURRENCY,
    CURRENCY_SYMBOL,
    EFFLUENT_COST_PER_M3,
    TICK_SIMULATED_MINUTES,
    WATER_COST_PER_M3,
)

# What one cubic metre of lost water actually costs the plant.
TOTAL_COST_PER_M3 = WATER_COST_PER_M3 + EFFLUENT_COST_PER_M3


def cost_of(volume_litres: float) -> float:
    """Financial impact of a lost volume, in the configured currency."""
    return (volume_litres / 1000.0) * TOTAL_COST_PER_M3


def projected_daily_cost(rate_lpm: float) -> float:
    """What this leak costs per day if left unrepaired."""
    return cost_of(rate_lpm * 60.0 * 24.0)


class LossTracker:
    """Accumulates volume and cost per leak event across ticks."""

    def __init__(self) -> None:
        self._events: dict[str, dict] = {}
        self._closed: list[dict] = []

    def reset(self) -> None:
        self._events.clear()
        self._closed.clear()

    def update(self, leaks: list, timestamp: str) -> list:
        """
        Fold this tick's detected leaks into the running event log.

        Returns the leak list enriched with duration, volume lost and cost.
        Events for nodes that no longer leak are closed and archived.
        """
        minutes = TICK_SIMULATED_MINUTES
        seen = set()
        enriched = []

        for leak in leaks:
            node = leak["node"]
            seen.add(node)
            event = self._events.get(node)

            if event is None:
                event = {
                    "node": node,
                    "zone": leak["zone"],
                    "endpoint": leak.get("endpoint"),
                    "first_seen": timestamp,
                    "ticks": 0,
                    "volume_litres": 0.0,
                    "peak_rate_lpm": 0.0,
                }
                self._events[node] = event

            event["ticks"] += 1
            event["volume_litres"] += leak["rate_lpm"] * minutes
            event["peak_rate_lpm"] = max(event["peak_rate_lpm"], leak["rate_lpm"])
            event["last_seen"] = timestamp
            event["severity"] = leak["severity"]

            duration_min = event["ticks"] * minutes
            volume = event["volume_litres"]

            enriched.append({
                **leak,
                "first_seen": event["first_seen"],
                "duration_minutes": duration_min,
                "duration_hours": round(duration_min / 60.0, 2),
                "volume_lost_litres": round(volume, 1),
                "volume_lost_m3": round(volume / 1000.0, 3),
                "peak_rate_lpm": round(event["peak_rate_lpm"], 2),
                "cost_so_far": round(cost_of(volume), 2),
                "cost_per_hour": round(cost_of(leak["rate_lpm"] * 60.0), 2),
                "projected_daily_cost": round(projected_daily_cost(leak["rate_lpm"]), 2),
            })

        # Close events whose leak has stopped.
        for node in list(self._events):
            if node not in seen:
                closed = self._events.pop(node)
                closed["cost"] = round(cost_of(closed["volume_litres"]), 2)
                closed["volume_litres"] = round(closed["volume_litres"], 1)
                self._closed.append(closed)

        return enriched

    def summary(self, active: list) -> dict:
        """Plant-level loss and cost totals, for the dashboard header."""
        live_volume = sum(leak["volume_lost_litres"] for leak in active)
        live_rate = sum(leak["rate_lpm"] for leak in active)
        closed_volume = sum(event["volume_litres"] for event in self._closed)
        total_volume = live_volume + closed_volume

        return {
            "active_leaks": len(active),
            "current_loss_lpm": round(live_rate, 2),
            "current_loss_lph": round(live_rate * 60.0, 1),
            "volume_lost_litres": round(total_volume, 1),
            "volume_lost_m3": round(total_volume / 1000.0, 3),
            "cost_so_far": round(cost_of(total_volume), 2),
            "cost_per_hour": round(cost_of(live_rate * 60.0), 2),
            "projected_daily_cost": round(projected_daily_cost(live_rate), 2),
            "projected_annual_cost": round(projected_daily_cost(live_rate) * 365.0, 2),
            "resolved_events": len(self._closed),
            "currency": CURRENCY,
            "currency_symbol": CURRENCY_SYMBOL,
            "tariff_per_m3": round(TOTAL_COST_PER_M3, 2),
        }

    @property
    def closed_events(self) -> list:
        return list(self._closed)
