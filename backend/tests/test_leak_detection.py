"""
Tests for leak injection, propagation and mass-balance detection.

Tolerances are generous because every meter adds ~1% Gaussian noise, so an
injected rate is never recovered exactly.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.dependencies import engine, store  # noqa: E402
from app.simulation.detection import classify_severity  # noqa: E402



# Determinism
# -----------
# The simulator draws measurement noise from the global `random` module. Left
# unseeded, every tolerance in this file becomes a probabilistic bet: bounds at
# 2-3 sigma fail a few percent of runs, and assertions like "a clean network
# never false-positives" fail whenever noise crosses a 3-sigma threshold, which
# it does by design. Seeding per test makes each run reproducible, so a failure
# means the behaviour changed rather than the dice landed badly. The tolerances
# below stay generous so they still describe the physics rather than one draw.

@pytest.fixture(autouse=True)
def clean_sim():
    random.seed(20260919)
    """Every test starts from a full plant with no leaks."""
    store.reset()
    engine.clock.reset()
    for i in range(1, 9):
        store.update_machine(f"M{i}", production_pct=100.0, state="RUNNING")
    yield
    store.reset()


def test_no_leak_is_not_detected():
    snap = engine.tick()
    assert snap["detection"]["leak_detected"] is False
    assert snap["detection"]["leak_node"] is None


def test_leak_at_internal_junction_is_localised():
    store.leaks.inject_leak("J3", 100.0)
    snap = engine.tick()
    det = snap["detection"]

    assert det["leak_detected"] is True
    assert det["leak_node"] == "J3"
    assert det["leak_zone"] == "ZONE_J3"
    # Measured: mean 101.5, sd 10.1 -> 40 is ~4 sigma.
    assert det["leak_rate_lpm"] == pytest.approx(100.0, abs=40.0)


def test_leak_at_leaf_junction_is_localised_to_the_leaf():
    """A leaf leak shows as the feed reading more than its machine draws."""
    store.leaks.inject_leak("J8", 80.0)
    det = engine.tick()["detection"]

    assert det["leak_node"] == "J8"
    assert det["endpoint"] == "M3"
    assert det["leak_rate_lpm"] == pytest.approx(80.0, abs=25.0)


def test_leak_propagates_upstream_but_not_downstream():
    """J1 and J3 rise by the leak; J3's children keep reading normal demand.

    Tolerance is ~4 sigma. Measured tick-to-tick spread on J1 is about 18 L/min
    (every meter in the tree contributes independent noise), so a tighter bound
    such as 40 fails on roughly 2% of runs for no real reason.
    """
    before = engine.tick()["flows"]
    store.leaks.inject_leak("J3", 200.0)
    after = engine.tick()["flows"]

    # J1's delta is the noisiest quantity in the suite: it combines two ticks
    # of the main meter at ~880 L/min plus every child meter beneath it.
    # Measured mean 194.8, sd 34.8 -> 140 is ~4 sigma. J3 is much tighter
    # (sd 15.6) because less of the tree feeds it.
    assert after["J1"] - before["J1"] == pytest.approx(200.0, abs=140.0)
    assert after["J3"] - before["J3"] == pytest.approx(200.0, abs=65.0)

    # The children should not move at all; measured spread of their summed
    # delta is sd ~10 L/min, so 45 is ~4 sigma. What matters is that this stays
    # far below the 200 L/min the parent moved.
    downstream = sum(after[j] for j in ("J8", "J9", "J10"))
    downstream_before = sum(before[j] for j in ("J8", "J9", "J10"))
    assert downstream == pytest.approx(downstream_before, abs=45.0)


def test_root_balance_absorbs_a_downstream_leak():
    """Only the leaking node shows a residual, which is what localises it.

    The root bound is ~4 sigma. J1's residual combines five 2%-noise meters at
    ~880 L/min, giving a spread near 19 L/min around a mean of ~0 — so the old
    40 L/min bound was barely 2 sigma and failed about 1.7% of runs even though
    the balance itself is exact.
    """
    store.leaks.inject_leak("J3", 150.0)
    det = engine.tick()["detection"]

    assert det["residuals"]["J1"] == pytest.approx(0.0, abs=80.0)
    assert det["residuals"]["J3"] > 100.0


def test_clearing_a_leak_stops_detection():
    store.leaks.inject_leak("J7", 120.0)
    assert engine.tick()["detection"]["leak_detected"] is True

    store.leaks.clear_all()
    assert engine.tick()["detection"]["leak_detected"] is False


def test_severity_bands():
    assert classify_severity(20.0) == "Small"
    assert classify_severity(100.0) == "Medium"
    assert classify_severity(300.0) == "Large"


def test_detected_severity_matches_injected_size():
    store.leaks.inject_leak("J4", 300.0)
    det = engine.tick()["detection"]
    assert det["severity"] == "Large"


def test_small_leak_below_noise_floor_is_not_reported():
    """Honesty check: a 1 L/min leak on a ~880 L/min main is not detectable.

    Asserts J1 specifically rather than "nothing anywhere fired" — the latter
    also fails on an unrelated false positive elsewhere in the tree, which is a
    different claim.
    """
    store.leaks.inject_leak("J1", 1.0)
    det = engine.tick()["detection"]
    assert "J1" not in [leak["node"] for leak in det["leaks"]]


def test_flow_override_forces_the_reading():
    store.set_override("flow_J5", 999.0)
    snap = engine.tick()
    assert snap["flows"]["J5"] == pytest.approx(999.0)

    store.clear_override("flow_J5")
    assert engine.tick()["flows"]["J5"] != pytest.approx(999.0)


def test_pressure_override_forces_the_reading():
    store.set_override("pressure_J1", 2.5)
    assert engine.tick()["pressures"]["J1"] == pytest.approx(2.5)


def test_override_creating_an_imbalance_is_detected():
    """Faking a high reading on a feed line looks exactly like a leak."""
    engine.tick()
    store.set_override("flow_J6", 400.0)
    det = engine.tick()["detection"]

    assert det["leak_detected"] is True
    assert det["leak_node"] == "J6"


def test_reset_clears_leaks_and_overrides():
    store.leaks.inject_leak("J2", 50.0)
    store.set_override("flow_J2", 500.0)
    engine.reset()

    assert store.leaks.get_active_leaks() == {}
    assert store.overrides == {}


def test_datasink_rows_carry_ground_truth_labels():
    from app.api.datasink import _flatten_snapshot

    store.leaks.inject_leak("J9", 90.0)
    flat = _flatten_snapshot(engine.tick())

    assert flat["leak"] == 1
    assert flat["leak_zone"] == "ZONE_J9"
    assert flat["leak_rate"] == pytest.approx(90.0)


def test_datasink_rows_are_unlabelled_when_clean():
    from app.api.datasink import _flatten_snapshot

    flat = _flatten_snapshot(engine.tick())
    assert flat["leak"] == 0
    assert flat["leak_zone"] is None


# ── Simultaneous leaks ────────────────────────────────────────────────────
# Residuals are independent per junction, so concurrent leaks must each be
# reported rather than collapsed into the single largest finding.

def test_two_leaks_in_different_branches_are_both_reported():
    store.leaks.inject_leak("J3", 200.0)
    store.leaks.inject_leak("J11", 120.0)
    det = engine.tick()["detection"]

    assert det["leak_count"] == 2
    found = {leak["node"]: leak["rate_lpm"] for leak in det["leaks"]}
    assert set(found) == {"J3", "J11"}
    assert found["J3"] == pytest.approx(200.0, abs=35.0)
    assert found["J11"] == pytest.approx(120.0, abs=25.0)


def test_nested_leaks_do_not_contaminate_each_other():
    """A leak at a parent and at its own child are separable."""
    store.leaks.inject_leak("J3", 180.0)
    store.leaks.inject_leak("J8", 90.0)
    det = engine.tick()["detection"]

    found = {leak["node"]: leak["rate_lpm"] for leak in det["leaks"]}
    assert set(found) == {"J3", "J8"}
    # J3's residual recovers J3's rate alone, not 180 + 90.
    # Measured: J3 mean 178.9 sd 13.6, J8 mean 90.0 sd 4.5 -> 4 sigma bounds.
    assert found["J3"] == pytest.approx(180.0, abs=55.0)
    assert found["J8"] == pytest.approx(90.0, abs=20.0)


def test_primary_is_the_largest_leak_not_the_deepest():
    """The headline must not hide a bigger loss behind a smaller, deeper one."""
    store.leaks.inject_leak("J3", 250.0)    # shallower, bigger
    store.leaks.inject_leak("J11", 60.0)    # deeper, smaller
    det = engine.tick()["detection"]

    assert det["leak_node"] == "J3"
    assert det["leak_rate_lpm"] > 150.0


def test_leaks_are_sorted_largest_first():
    store.leaks.inject_leak("J14", 40.0)
    store.leaks.inject_leak("J3", 220.0)
    store.leaks.inject_leak("J12", 110.0)
    rates = [leak["rate_lpm"] for leak in engine.tick()["detection"]["leaks"]]

    assert rates == sorted(rates, reverse=True)
    assert len(rates) == 3


def test_total_loss_sums_every_leak():
    store.leaks.inject_leak("J3", 150.0)
    store.leaks.inject_leak("J11", 100.0)
    det = engine.tick()["detection"]

    assert det["total_loss_lpm"] == pytest.approx(250.0, abs=45.0)
    assert det["total_loss_lpm"] == pytest.approx(
        sum(leak["rate_lpm"] for leak in det["leaks"]), abs=0.05
    )


def test_each_leak_carries_its_own_severity():
    store.leaks.inject_leak("J3", 300.0)    # Large
    store.leaks.inject_leak("J14", 30.0)    # Small
    by_node = {l["node"]: l for l in engine.tick()["detection"]["leaks"]}

    assert by_node["J3"]["severity"] == "Large"
    assert by_node["J14"]["severity"] == "Small"


def test_clearing_one_of_two_leaks_leaves_the_other():
    store.leaks.inject_leak("J3", 200.0)
    store.leaks.inject_leak("J11", 120.0)
    assert engine.tick()["detection"]["leak_count"] == 2

    store.leaks.clear_leak("J3")
    det = engine.tick()["detection"]
    assert det["leak_count"] == 1
    assert det["leaks"][0]["node"] == "J11"


def test_no_leaks_reports_empty_list_and_zero_total():
    det = engine.tick()["detection"]
    assert det["leaks"] == []
    assert det["leak_count"] == 0
    assert det["total_loss_lpm"] == 0.0


def test_datasink_labels_use_the_largest_of_several_leaks():
    from app.api.datasink import _flatten_snapshot

    store.leaks.inject_leak("J3", 200.0)
    store.leaks.inject_leak("J11", 120.0)
    flat = _flatten_snapshot(engine.tick())

    assert flat["leak"] == 1
    assert flat["leak_zone"] == "ZONE_J3"          # largest injected
    assert flat["leak_rate"] == pytest.approx(320.0)  # total escaping water
