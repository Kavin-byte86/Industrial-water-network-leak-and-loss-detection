"""Pydantic schemas for the network topology API.

Uses `typing` constructs rather than PEP 585/604 syntax: pydantic resolves
annotations at runtime, so the modern syntax would fail on Python 3.8.
"""

from typing import List, Optional

from pydantic import BaseModel


class TopologyNodeSchema(BaseModel):
    id: str
    type: str
    parent_id: Optional[str] = None
    children: List[str] = []
    endpoint_id: Optional[str] = None


class TopologyResponse(BaseModel):
    nodes: List[TopologyNodeSchema]
