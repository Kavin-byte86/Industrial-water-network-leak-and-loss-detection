"""Pydantic schemas for the network topology API."""

from __future__ import annotations

from pydantic import BaseModel


class TopologyNodeSchema(BaseModel):
    id: str
    type: str
    parent_id: str | None = None
    children: list[str] = []
    endpoint_id: str | None = None


class TopologyResponse(BaseModel):
    nodes: list[TopologyNodeSchema]
