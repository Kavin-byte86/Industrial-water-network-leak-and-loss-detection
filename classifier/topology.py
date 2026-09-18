"""
topology.py
Industrial Water Network Topology definition, sensor metadata,
graph hierarchy, and physical mass-balance relationships.
"""

import json
from typing import Dict, List, Any
import pandas as pd

# ==============================================================================
# NETWORK GRAPH TOPOLOGY
# Directed acyclic graph from Main Inlet (J1) down to terminal endpoints
# ==============================================================================
TOPOLOGY = {
    "J1": ["J2", "J3", "J4", "J7"],
    "J2": ["J5", "J6"],
    "J3": ["J8", "J9", "J10"],
    "J4": ["J11", "J12"],
    "J7": ["J13", "J14", "J15", "J16"]
}

# Terminal machine and tap mappings
TERMINAL_NODES = {
    "J5": {"type": "machine", "id": "M1", "name": "Scouring machine"},
    "J6": {"type": "machine", "id": "M2", "name": "Bleaching machine"},
    "J8": {"type": "machine", "id": "M3", "name": "Dyeing machine 1"},
    "J9": {"type": "machine", "id": "M4", "name": "Dyeing machine 2"},
    "J10": {"type": "machine", "id": "M5", "name": "Washing machine"},
    "J11": {"type": "machine", "id": "M6", "name": "Finishing machine"},
    "J12": {"type": "machine", "id": "M7", "name": "Washing/rinsing machine"},
    "J13": {"type": "machine", "id": "M8", "name": "Utility/process machine"},
    "J14": {"type": "tap", "id": "T1", "name": "Utility Washdown Tap 1"},
    "J15": {"type": "tap", "id": "T2", "name": "Equipment Cleaning Tap 2"},
    "J16": {"type": "tap", "id": "T3", "name": "Sampling & Dilution Tap 3"},
}

# Major junction headers
JUNCTION_HEADERS = ["J1", "J2", "J3", "J4", "J7"]

# All 16 flow sensor identifiers
ALL_SENSORS = [f"J{i}" for i in range(1, 17)]

# ==============================================================================
# LEAK ZONES AND HUMAN-READABLE DESCRIPTIONS
# ==============================================================================
LEAK_ZONES = {
    "ZONE_J1": {
        "description": "Main inlet distribution header (post-J1)",
        "affected_upstream_sensors": ["J1"],
        "downstream_imbalance_junction": "J1",
        "affected_pressure_sensors": ["J1", "J2", "J3", "J4", "J7"]
    },
    "ZONE_J2": {
        "description": "Branch A manifold header (between J2 and J5/J6)",
        "affected_upstream_sensors": ["J1", "J2"],
        "downstream_imbalance_junction": "J2",
        "affected_pressure_sensors": ["J1", "J2"]
    },
    "ZONE_J3": {
        "description": "Branch B manifold header (between J3 and J8/J9/J10)",
        "affected_upstream_sensors": ["J1", "J3"],
        "downstream_imbalance_junction": "J3",
        "affected_pressure_sensors": ["J1", "J3"]
    },
    "ZONE_J4": {
        "description": "Branch C manifold header (between J4 and J11/J12)",
        "affected_upstream_sensors": ["J1", "J4"],
        "downstream_imbalance_junction": "J4",
        "affected_pressure_sensors": ["J1", "J4"]
    },
    "ZONE_J5": {
        "description": "M1 Scouring machine feed line",
        "affected_upstream_sensors": ["J1", "J2", "J5"],
        "downstream_imbalance_junction": None,
        "affected_pressure_sensors": ["J1", "J2"]
    },
    "ZONE_J6": {
        "description": "M2 Bleaching machine feed line",
        "affected_upstream_sensors": ["J1", "J2", "J6"],
        "downstream_imbalance_junction": None,
        "affected_pressure_sensors": ["J1", "J2"]
    },
    "ZONE_J7": {
        "description": "Branch D utility manifold header (between J7 and J13-J16)",
        "affected_upstream_sensors": ["J1", "J7"],
        "downstream_imbalance_junction": "J7",
        "affected_pressure_sensors": ["J1", "J7"]
    },
    "ZONE_J8": {
        "description": "M3 Dyeing machine 1 feed line",
        "affected_upstream_sensors": ["J1", "J3", "J8"],
        "downstream_imbalance_junction": None,
        "affected_pressure_sensors": ["J1", "J3"]
    },
    "ZONE_J9": {
        "description": "M4 Dyeing machine 2 feed line",
        "affected_upstream_sensors": ["J1", "J3", "J9"],
        "downstream_imbalance_junction": None,
        "affected_pressure_sensors": ["J1", "J3"]
    },
    "ZONE_J10": {
        "description": "M5 Washing machine feed line",
        "affected_upstream_sensors": ["J1", "J3", "J10"],
        "downstream_imbalance_junction": None,
        "affected_pressure_sensors": ["J1", "J3"]
    },
    "ZONE_J11": {
        "description": "M6 Finishing machine feed line",
        "affected_upstream_sensors": ["J1", "J4", "J11"],
        "downstream_imbalance_junction": None,
        "affected_pressure_sensors": ["J1", "J4"]
    },
    "ZONE_J12": {
        "description": "M7 Washing/rinsing machine feed line",
        "affected_upstream_sensors": ["J1", "J4", "J12"],
        "downstream_imbalance_junction": None,
        "affected_pressure_sensors": ["J1", "J4"]
    },
    "ZONE_J13": {
        "description": "M8 Utility/process machine feed line",
        "affected_upstream_sensors": ["J1", "J7", "J13"],
        "downstream_imbalance_junction": None,
        "affected_pressure_sensors": ["J1", "J7"]
    },
    "ZONE_J14": {
        "description": "Tap 1 Utility Washdown line",
        "affected_upstream_sensors": ["J1", "J7", "J14"],
        "downstream_imbalance_junction": None,
        "affected_pressure_sensors": ["J1", "J7"]
    },
    "ZONE_J15": {
        "description": "Tap 2 Equipment Cleaning line",
        "affected_upstream_sensors": ["J1", "J7", "J15"],
        "downstream_imbalance_junction": None,
        "affected_pressure_sensors": ["J1", "J7"]
    },
    "ZONE_J16": {
        "description": "Tap 3 Sampling & Dilution line",
        "affected_upstream_sensors": ["J1", "J7", "J16"],
        "downstream_imbalance_junction": None,
        "affected_pressure_sensors": ["J1", "J7"]
    },
}

# ==============================================================================
# SENSOR METADATA BUILDER
# ==============================================================================
def get_sensor_metadata() -> pd.DataFrame:
    """Builds and returns the sensor_metadata DataFrame."""
    rows = [
        {"sensor_id": "J1",  "location": "Main inlet", "sensor_type": "flow", "parent": "source", "downstream_nodes": "J2,J3,J4,J7", "measurement_unit": "L/min"},
        {"sensor_id": "J2",  "location": "Branch A (M1-M2)", "sensor_type": "flow", "parent": "J1", "downstream_nodes": "J5,J6", "measurement_unit": "L/min"},
        {"sensor_id": "J3",  "location": "Branch B (M3-M5)", "sensor_type": "flow", "parent": "J1", "downstream_nodes": "J8,J9,J10", "measurement_unit": "L/min"},
        {"sensor_id": "J4",  "location": "Branch C (M6-M7)", "sensor_type": "flow", "parent": "J1", "downstream_nodes": "J11,J12", "measurement_unit": "L/min"},
        {"sensor_id": "J5",  "location": "Machine M1 branch", "sensor_type": "flow", "parent": "J2", "downstream_nodes": "M1", "measurement_unit": "L/min"},
        {"sensor_id": "J6",  "location": "Machine M2 branch", "sensor_type": "flow", "parent": "J2", "downstream_nodes": "M2", "measurement_unit": "L/min"},
        {"sensor_id": "J7",  "location": "Branch D (M8, T1-T3)", "sensor_type": "flow", "parent": "J1", "downstream_nodes": "J13,J14,J15,J16", "measurement_unit": "L/min"},
        {"sensor_id": "J8",  "location": "Machine M3 branch", "sensor_type": "flow", "parent": "J3", "downstream_nodes": "M3", "measurement_unit": "L/min"},
        {"sensor_id": "J9",  "location": "Machine M4 branch", "sensor_type": "flow", "parent": "J3", "downstream_nodes": "M4", "measurement_unit": "L/min"},
        {"sensor_id": "J10", "location": "Machine M5 branch", "sensor_type": "flow", "parent": "J3", "downstream_nodes": "M5", "measurement_unit": "L/min"},
        {"sensor_id": "J11", "location": "Machine M6 branch", "sensor_type": "flow", "parent": "J4", "downstream_nodes": "M6", "measurement_unit": "L/min"},
        {"sensor_id": "J12", "location": "Machine M7 branch", "sensor_type": "flow", "parent": "J4", "downstream_nodes": "M7", "measurement_unit": "L/min"},
        {"sensor_id": "J13", "location": "Machine M8 branch", "sensor_type": "flow", "parent": "J7", "downstream_nodes": "M8", "measurement_unit": "L/min"},
        {"sensor_id": "J14", "location": "Tap 1", "sensor_type": "flow", "parent": "J7", "downstream_nodes": "T1", "measurement_unit": "L/min"},
        {"sensor_id": "J15", "location": "Tap 2", "sensor_type": "flow", "parent": "J7", "downstream_nodes": "T2", "measurement_unit": "L/min"},
        {"sensor_id": "J16", "location": "Tap 3", "sensor_type": "flow", "parent": "J7", "downstream_nodes": "T3", "measurement_unit": "L/min"},
    ]
    return pd.DataFrame(rows)

def export_network_topology_json() -> str:
    """Returns JSON string of the network topology graph."""
    return json.dumps(TOPOLOGY, indent=2)
