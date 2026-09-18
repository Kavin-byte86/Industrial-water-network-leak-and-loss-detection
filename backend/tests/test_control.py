"""
Test control endpoints — §10.6 / §10.7 of BACKEND_IMPLEMENTATION_PLAN.md.

Uses FastAPI TestClient to verify:
  - POST /control/machine and /control/tap change next tick flows
  - POST /simulation/step, /pause, /resume, /reset behave correctly
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


# ── Simulation control ────────────────────────────────────────────────────

def test_reset_clears_state():
    """POST /simulation/reset should reset everything."""
    resp = client.post("/simulation/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reset"


def test_step_returns_snapshot():
    """POST /simulation/step should return a valid tick snapshot."""
    client.post("/simulation/reset")
    resp = client.post("/simulation/step")
    assert resp.status_code == 200
    data = resp.json()
    assert "timestamp" in data
    assert "flows" in data
    assert "pressures" in data
    assert "machines" in data
    assert "taps" in data


def test_pause_and_resume():
    """POST /simulation/pause and /resume should return expected status."""
    resp = client.post("/simulation/pause")
    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"

    resp = client.post("/simulation/resume")
    assert resp.status_code == 200
    assert resp.json()["status"] == "resumed"


# ── Machine control ───────────────────────────────────────────────────────

def test_machine_control_changes_flow():
    """Setting a machine to RUNNING should produce non-zero flow on next tick."""
    client.post("/simulation/reset")

    # Set M1 to 100% RUNNING
    resp = client.post("/control/machine", json={
        "machine_id": "M1",
        "production_pct": 100.0,
        "state": "RUNNING",
    })
    assert resp.status_code == 200
    assert resp.json()["production_pct"] == 100.0

    # Step and check M1 has non-zero flow
    resp = client.post("/simulation/step")
    snap = resp.json()
    assert snap["machines"]["M1"]["flow_lpm"] > 0


def test_machine_off_produces_zero_flow():
    """A machine in OFF state should produce zero flow."""
    client.post("/simulation/reset")

    resp = client.post("/control/machine", json={
        "machine_id": "M1",
        "production_pct": 100.0,
        "state": "OFF",
    })
    assert resp.status_code == 200

    resp = client.post("/simulation/step")
    snap = resp.json()
    assert snap["machines"]["M1"]["flow_lpm"] == 0.0


def test_invalid_machine_id_rejected():
    """An unknown machine_id should return 400."""
    resp = client.post("/control/machine", json={
        "machine_id": "M99",
        "production_pct": 50.0,
    })
    assert resp.status_code == 400


def test_production_pct_validation():
    """production_pct outside 0–200 should be rejected by Pydantic."""
    resp = client.post("/control/machine", json={
        "machine_id": "M1",
        "production_pct": 250.0,
    })
    assert resp.status_code == 422  # Pydantic validation error


# ── Tap control ───────────────────────────────────────────────────────────

def test_tap_control_changes_flow():
    """Opening a tap should produce non-zero flow on next tick."""
    client.post("/simulation/reset")

    resp = client.post("/control/tap", json={
        "tap_id": "T1",
        "state": "OPEN",
    })
    assert resp.status_code == 200

    resp = client.post("/simulation/step")
    snap = resp.json()
    assert snap["taps"]["T1"]["flow_lpm"] > 0


def test_tap_closed_produces_zero():
    """A CLOSED tap should have zero flow."""
    client.post("/simulation/reset")

    resp = client.post("/simulation/step")
    snap = resp.json()
    assert snap["taps"]["T1"]["flow_lpm"] == 0.0


def test_invalid_tap_id_rejected():
    """An unknown tap_id should return 400."""
    resp = client.post("/control/tap", json={
        "tap_id": "T99",
        "state": "OPEN",
    })
    assert resp.status_code == 400


# ── State & datasink endpoints ────────────────────────────────────────────

def test_topology_returns_all_nodes():
    """GET /network/topology should return 16 junctions."""
    resp = client.get("/network/topology")
    assert resp.status_code == 200
    nodes = resp.json()["nodes"]
    assert len(nodes) == 16


def test_state_current_after_step():
    """GET /state/current should return data after a step."""
    client.post("/simulation/reset")
    client.post("/simulation/step")
    resp = client.get("/state/current")
    assert resp.status_code == 200
    assert "flows" in resp.json()


def test_datasink_latest_flat_format():
    """GET /datasink/latest should return flat column keys."""
    client.post("/simulation/reset")
    # Set at least one machine running
    client.post("/control/machine", json={
        "machine_id": "M1", "production_pct": 100.0, "state": "RUNNING",
    })
    client.post("/simulation/step")

    resp = client.get("/datasink/latest")
    assert resp.status_code == 200
    data = resp.json()
    # Check flat keys exist
    assert "flow_J1" in data
    assert "pressure_J1" in data
    assert "production_M1" in data
    assert "machine_status_M1" in data
    assert "tap_status_T1" in data
    assert "hour" in data
    assert "shift" in data
