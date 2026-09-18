"""Pydantic schemas for control endpoints."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MachineStateEnum(str, Enum):
    OFF = "OFF"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    MAINTENANCE = "MAINTENANCE"


class TapStateEnum(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class MachineControlRequest(BaseModel):
    """POST /control/machine — either field is optional (partial update)."""
    machine_id: str
    production_pct: Optional[float] = Field(None, ge=0.0, le=200.0)
    state: Optional[MachineStateEnum] = None


class TapControlRequest(BaseModel):
    """POST /control/tap."""
    tap_id: str
    state: TapStateEnum
