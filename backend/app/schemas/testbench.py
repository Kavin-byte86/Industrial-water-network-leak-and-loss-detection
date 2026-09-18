"""Pydantic schemas for the test-bench (leak injection / sensor override) API.

Uses `typing` constructs: pydantic resolves annotations at runtime, so PEP
585/604 syntax would break on Python 3.8.
"""

from typing import Optional

from pydantic import BaseModel, Field


class LeakRequest(BaseModel):
    """POST /testbench/leak — start or update a leak at one node."""
    node_id: str = Field(..., description="Junction id, e.g. 'J3'")
    rate_lpm: float = Field(..., ge=0.0, le=5000.0,
                            description="Leak rate in L/min; 0 clears the leak")


class OverrideRequest(BaseModel):
    """POST /testbench/override — force a single sensor reading."""
    sensor: str = Field(..., description="Sensor key, e.g. 'flow_J3' or 'pressure_J1'")
    value: float = Field(..., description="Value the sensor should report")


class EndpointRequest(BaseModel):
    """POST /testbench/endpoint — set a machine or tap in one call."""
    endpoint_id: str = Field(..., description="Machine (M1-M8) or tap (T1-T3) id")
    production_pct: Optional[float] = Field(None, ge=0.0, le=200.0)
    state: Optional[str] = Field(None, description="Machine or tap state")
