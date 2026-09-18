"""Pydantic schemas for simulation state responses."""

from __future__ import annotations

from pydantic import BaseModel


class MachineState(BaseModel):
    production_pct: float
    state: str
    flow_lpm: float


class TapState(BaseModel):
    state: str
    flow_lpm: float


class TickSnapshot(BaseModel):
    timestamp: str
    hour: int
    day_of_week: int
    month: int
    shift: str
    flows: dict[str, float]
    pressures: dict[str, float]
    machines: dict[str, MachineState]
    taps: dict[str, TapState]


class HistoryResponse(BaseModel):
    ticks: list[TickSnapshot]
    count: int
