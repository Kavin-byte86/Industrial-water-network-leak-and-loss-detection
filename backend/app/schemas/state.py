"""Pydantic schemas for simulation state responses.

Pydantic resolves annotations at runtime, so `from __future__ import annotations`
does not make PEP 585/604 syntax safe here. These use `typing` constructs so the
package imports on the same Python 3.8 interpreter that runs the classifier.
"""

from typing import Dict, List

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
    flows: Dict[str, float]
    pressures: Dict[str, float]
    machines: Dict[str, MachineState]
    taps: Dict[str, TapState]


class HistoryResponse(BaseModel):
    ticks: List[TickSnapshot]
    count: int
