"""
Tests for the four questions the problem statement asks beyond "is there a leak":
where, how severe, how much water, and what it costs — plus the key challenge of
telling production ramps apart from genuine losses.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import TICK_SIMULATED_MINUTES  # noqa: E402
from app.dependencies import engine, store  # noqa: E402
from app.simulation.expected import reconcile  # noqa: E402
from app.simulation.loss_tracker import TOTAL_COST_PER_M3, cost_of  # noqa: E402



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
    random.seed(20260920)
    store.reset()
    engine.clock.reset()
    engine.losses.reset()
    for i in range(1, 9):
        store.update_machine(f"M{i}", production_pct=100.0, state="RUNNING")
    yield
    store.reset()
    engine.losses.reset()


def run(ticks):
    snap = None
    for _ in range(ticks):
        snap = engine.tick()
    return snap


# ── Key challenge: production vs loss ─────────────────────────────────────

def test_production_ramp_alone_raises_no_unexplained_flow():
    """Doubling production doubles measured flow without implying a loss."""
    at_100 = run(3)["production"]
    for i in range(1, 9):
        store.update_machine(f"M{i}", production_pct=200.0, state="RUNNING")
    at_200 = run(3)["production"]

    # Flow rises steeply — though not exactly 2x, because each machine's curve
    # has a constant base_flow term that does not scale with production.
    # Measured ratio: mean 1.81, sd 0.06, so 1.55 is ~4 sigma below the mean.
    # (1.65 sat inside the noise and failed ~0.7% of runs.)
    assert at_200["measured_total_lpm"] > 1.55 * at_100["measured_total_lpm"]
    # ...and expected demand tracks it, so nothing is unaccounted for.
    # ~4 sigma: at 200% load the main runs ~1600 L/min and every meter carries
    # 2% noise, giving the unexplained figure a spread near 35 L/min. The mean
    # is ~6, so the estimator is unbiased; the bound just has to respect noise.
    assert abs(at_200["unexplained_lpm"]) < 140.0
    # Measured: mean 99.1%, sd 1.39, min 94.5 -> 93 is ~4 sigma below the mean.
    # (95 sat inside the noise and failed ~0.7% of runs.)
    assert at_200["explained_pct"] > 93.0


def test_leak_shows_up_as_unexplained_flow():
    """A leak, unlike a production ramp, lands entirely in `unexplained`.

    The bound is ~3.5 sigma. This compares two independent ticks, so their
    noise adds: measured spread of the difference is ~34 L/min around a mean of
    202 (the injected 200). The old 60 L/min bound was under 2 sigma and failed
    ~6% of runs while the estimator itself was unbiased.
    """
    before = run(3)["production"]
    store.leaks.inject_leak("J3", 200.0)
    after = run(3)["production"]

    assert after["unexplained_lpm"] - before["unexplained_lpm"] == pytest.approx(
        200.0, abs=120.0
    )
    assert after["explained_pct"] < 95.0


def test_unexplained_flow_isolates_the_leak_during_a_production_ramp():
    """The hard case: production doubles *while* a leak is running."""
    store.leaks.inject_leak("J3", 200.0)
    run(3)
    for i in range(1, 9):
        store.update_machine(f"M{i}", production_pct=200.0, state="RUNNING")
    ramped = run(3)["production"]

    # Unexplained still reflects the leak, not the extra production.
    # Measured mean 196.6, sd 44.2 at 200% load -> 180 is ~4 sigma.
    assert ramped["unexplained_lpm"] == pytest.approx(200.0, abs=180.0)


def test_machines_off_expect_no_water():
    for i in range(1, 9):
        store.update_machine(f"M{i}", production_pct=0.0, state="OFF")
    prod = run(2)["production"]
    # Standby draw only: eight machines idling at a few tenths of a L/min.
    assert prod["expected_total_lpm"] < 5.0
    assert prod["machines_running"] == 0


def test_reconcile_is_pure_and_matches_production():
    machines = {f"M{i}": {"production_pct": 100.0, "state": "RUNNING"} for i in range(1, 9)}
    taps = {f"T{i}": {"state": "CLOSED"} for i in range(1, 4)}
    # Expected total is the sum of the per-machine demand curves, ~880 L/min.
    out = reconcile({"J1": 880.0}, machines, taps)
    assert 850.0 < out["expected_total_lpm"] < 910.0
    assert abs(out["unexplained_lpm"]) < 35.0


# ── Quantity of water lost ────────────────────────────────────────────────

def test_volume_lost_integrates_rate_over_time():
    store.leaks.inject_leak("J3", 200.0)
    snap = run(12)                      # 12 ticks x 5 min = 1 simulated hour
    leak = snap["detection"]["leaks"][0]

    assert leak["duration_hours"] == pytest.approx(1.0, abs=0.01)
    # 200 L/min for 60 min = 12,000 L
    assert leak["volume_lost_litres"] == pytest.approx(12000.0, abs=900.0)
    assert leak["volume_lost_m3"] == pytest.approx(12.0, abs=0.9)


def test_volume_keeps_accumulating_while_the_leak_runs():
    store.leaks.inject_leak("J3", 150.0)
    first = run(6)["detection"]["leaks"][0]["volume_lost_litres"]
    second = run(6)["detection"]["leaks"][0]["volume_lost_litres"]
    assert second > first * 1.8


def test_each_leak_accounts_for_its_own_volume():
    store.leaks.inject_leak("J3", 200.0)
    store.leaks.inject_leak("J11", 100.0)
    leaks = {l["node"]: l for l in run(12)["detection"]["leaks"]}

    assert leaks["J3"]["volume_lost_litres"] == pytest.approx(12000.0, abs=900.0)
    assert leaks["J11"]["volume_lost_litres"] == pytest.approx(6000.0, abs=700.0)


# ── Financial impact ──────────────────────────────────────────────────────

def test_cost_follows_the_configured_tariff():
    assert cost_of(1000.0) == pytest.approx(TOTAL_COST_PER_M3)
    assert cost_of(0.0) == 0.0


def test_leak_cost_matches_its_volume():
    store.leaks.inject_leak("J3", 200.0)
    leak = run(12)["detection"]["leaks"][0]
    assert leak["cost_so_far"] == pytest.approx(
        cost_of(leak["volume_lost_litres"]), abs=0.05
    )


def test_projected_costs_scale_from_the_current_rate():
    store.leaks.inject_leak("J3", 200.0)
    leak = run(3)["detection"]["leaks"][0]
    # 200 L/min -> 288 m3/day
    assert leak["projected_daily_cost"] == pytest.approx(
        cost_of(leak["rate_lpm"] * 60 * 24), rel=0.01
    )
    assert leak["cost_per_hour"] > 0


def test_plant_summary_totals_every_leak():
    store.leaks.inject_leak("J3", 200.0)
    store.leaks.inject_leak("J11", 100.0)
    snap = run(12)
    summary = snap["losses"]

    assert summary["active_leaks"] == 2
    assert summary["current_loss_lpm"] == pytest.approx(300.0, abs=45.0)
    assert summary["cost_so_far"] == pytest.approx(
        cost_of(summary["volume_lost_litres"]), abs=0.05
    )
    assert summary["currency"] == "INR"


def test_clean_network_costs_nothing():
    summary = run(6)["losses"]
    assert summary["active_leaks"] == 0
    assert summary["current_loss_lpm"] == 0.0
    assert summary["cost_so_far"] == 0.0


def test_resolved_leak_keeps_its_volume_in_the_total():
    """Repairing a leak stops the bleeding but does not un-spend the water."""
    store.leaks.inject_leak("J3", 200.0)
    run(12)
    store.leaks.clear_all()
    snap = run(3)

    assert snap["losses"]["active_leaks"] == 0
    assert snap["losses"]["current_loss_lpm"] == 0.0
    assert snap["losses"]["volume_lost_litres"] > 10000.0
    assert snap["losses"]["resolved_events"] == 1


# ── Pressure as corroborating evidence ────────────────────────────────────

def test_pressure_drop_corroborates_a_large_leak():
    """A large leak pulls the header pressure down, corroborating the flow finding.

    Pressure carries 0.08 bar of noise against a 0.05 bar evidence threshold, so
    a single tick corroborates ~98% of the time. The physics claim is that the
    drop is there on average; asserting one tick's flag would fail ~2% of runs.
    """
    run(14)                                  # build a pressure baseline
    store.leaks.inject_leak("J3", 300.0)

    drops = []
    corroborated = 0
    for _ in range(5):
        leak = run(1)["detection"]["leaks"][0]
        assert leak["pressure"] is not None
        drops.append(leak["pressure"]["drop_bar"])
        corroborated += bool(leak["pressure"]["corroborates"])

    assert sum(drops) / len(drops) > 0.05, f"mean drop {sum(drops)/len(drops):.3f} bar"
    assert corroborated >= 3, f"only {corroborated}/5 ticks corroborated"


# ── Historical usage patterns ─────────────────────────────────────────────

def test_baselines_are_learned_from_history():
    """Each sensor builds a baseline from its own recent history.

    `abnormal` is a 3-sigma flag, so on clean data it fires on ~1% of ticks by
    construction. The test asserts the baseline is learned and sane, not that a
    threshold never trips — that would be asserting noise away.
    """
    snap = run(14)
    baselines = snap["detection"]["baselines"]

    assert "J1" in baselines
    assert baselines["J1"]["baseline_lpm"] > 0
    assert baselines["J1"]["sd_lpm"] >= 0
    # Measured: z has mean ~0, sd ~1.1 on a clean network.
    assert abs(baselines["J1"]["z_score"]) < 6.0


def test_baseline_needs_enough_history_first():
    snap = run(2)
    assert snap["detection"]["baselines"] == {}
