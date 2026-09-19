"""
Test: insert a tick record, read it back, confirm values match exactly.

Uses the real Neon database configured in DATABASE_URL.
"""

import pytest

from app.db.database import get_session, create_tables, engine as db_engine, Base
from app.db.models import TickRecord


@pytest.fixture(scope="module", autouse=True)
def ensure_tables():
    """Make sure tick_records exists before tests run."""
    create_tables()
    yield


@pytest.fixture()
def session():
    """Provide a fresh session and clean up test rows afterwards."""
    s = get_session()
    yield s
    # Clean up any rows inserted during the test
    s.rollback()
    s.close()


# A representative flat tick matching the datasink schema.
SAMPLE_TICK = {
    "timestamp": "2026-01-01T00:05:00",
    "hour": 0,
    "day_of_week": 3,
    "month": 1,
    "shift": "C",
    # Flows
    "flow_J1": 987.654,
    "flow_J2": 234.567,
    "flow_J3": 345.678,
    "flow_J4": 123.456,
    "flow_J5": 111.111,
    "flow_J6": 122.222,
    "flow_J7": 199.999,
    "flow_J8": 100.100,
    "flow_J9": 110.110,
    "flow_J10": 120.120,
    "flow_J11": 60.060,
    "flow_J12": 55.055,
    "flow_J13": 45.045,
    "flow_J14": 20.020,
    "flow_J15": 18.018,
    "flow_J16": 15.015,
    # Pressures
    "pressure_J1": 5.950,
    "pressure_J2": 5.800,
    "pressure_J3": 5.700,
    "pressure_J4": 5.850,
    "pressure_J5": 5.600,
    "pressure_J6": 5.620,
    "pressure_J7": 5.780,
    "pressure_J8": 5.500,
    "pressure_J9": 5.520,
    "pressure_J10": 5.540,
    "pressure_J11": 5.650,
    "pressure_J12": 5.670,
    "pressure_J13": 5.580,
    "pressure_J14": 5.450,
    "pressure_J15": 5.460,
    "pressure_J16": 5.470,
    # Machine production
    "production_M1": 100.0,
    "production_M2": 85.5,
    "production_M3": 110.0,
    "production_M4": 95.0,
    "production_M5": 120.0,
    "production_M6": 70.0,
    "production_M7": 90.0,
    "production_M8": 50.0,
    # Machine states
    "machine_status_M1": "RUNNING",
    "machine_status_M2": "RUNNING",
    "machine_status_M3": "STARTING",
    "machine_status_M4": "RUNNING",
    "machine_status_M5": "RUNNING",
    "machine_status_M6": "MAINTENANCE",
    "machine_status_M7": "RUNNING",
    "machine_status_M8": "OFF",
    # Tap states
    "tap_status_T1": "OPEN",
    "tap_status_T2": "CLOSED",
    "tap_status_T3": "OPEN",
    # Leak labels
    "leak": 0,
    "leak_rate": 0.0,
    "leak_zone": None,
}


def test_insert_and_read_back(session):
    """Insert a tick, read it back, confirm every field matches exactly."""
    # Insert
    record = TickRecord(**SAMPLE_TICK)
    session.add(record)
    session.commit()
    record_id = record.id
    assert record_id is not None

    # Read back
    fetched = session.get(TickRecord, record_id)
    assert fetched is not None

    # Verify every field
    for key, expected in SAMPLE_TICK.items():
        actual = getattr(fetched, key)
        if isinstance(expected, float):
            assert abs(actual - expected) < 1e-6, (
                f"{key}: expected {expected}, got {actual}"
            )
        else:
            assert actual == expected, (
                f"{key}: expected {expected!r}, got {actual!r}"
            )

    # Also verify to_flat_dict round-trips correctly
    flat = fetched.to_flat_dict()
    for key, expected in SAMPLE_TICK.items():
        actual = flat[key]
        if isinstance(expected, float):
            assert abs(actual - expected) < 1e-6, (
                f"to_flat_dict {key}: expected {expected}, got {actual}"
            )
        else:
            assert actual == expected, (
                f"to_flat_dict {key}: expected {expected!r}, got {actual!r}"
            )

    # Cleanup: delete the test row
    session.delete(fetched)
    session.commit()
