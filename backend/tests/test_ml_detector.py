"""
Tests for the trained 4-stage classifier running on live telemetry.

These assert the ML path specifically — that the real models load, that they
fire on real leaks and stay quiet on clean data, and that a missing model or a
cold history degrades to a structured "unavailable" rather than a made-up number.
"""

from __future__ import annotations

import os
import sys
import warnings

import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import ml_detector  # noqa: E402
from app.dependencies import engine, store  # noqa: E402

MODELS_PRESENT = ml_detector.status()["loaded"]
needs_models = pytest.mark.skipif(
    not MODELS_PRESENT,
    reason="trained model files are gitignored; run classifier training first",
)


@pytest.fixture(autouse=True)
def clean_sim():
    store.reset()
    engine.clock.reset()
    engine.losses.reset()
    for i in range(1, 9):
        store.update_machine(f"M{i}", production_pct=100.0, state="RUNNING")
    yield
    store.reset()
    engine.losses.reset()


def warm(ticks=22):
    for _ in range(ticks):
        engine.tick()


def predict():
    return ml_detector.predict_from_history(list(store.history))


# ── Availability contract ─────────────────────────────────────────────────

def test_cold_history_reports_warming_up_not_a_guess():
    engine.tick()
    result = predict()
    assert result["available"] is False
    assert result["reason"] == "warming_up"
    assert result["leak_detected"] is None       # never fabricated


def test_unavailable_results_keep_a_stable_shape():
    result = ml_detector.unavailable("models_not_loaded", detail="x")
    for key in ("available", "reason", "leak_detected", "leak_probability",
                "leak_zone", "leak_rate_lpm", "model_version"):
        assert key in result


# ── The model itself ──────────────────────────────────────────────────────

@needs_models
def test_model_loads_with_all_four_stages():
    status = ml_detector.status()
    assert status["loaded"] is True
    assert status["stage2_features"] > 100
    assert 0.0 < status["threshold"] <= 1.0


@needs_models
def test_clean_network_is_not_flagged():
    warm()
    result = predict()
    assert result["available"] is True
    assert result["leak_detected"] is False
    assert result["leak_probability"] < 0.5


@needs_models
def test_model_detects_an_injected_leak():
    warm()
    store.leaks.inject_leak("J3", 250.0)
    warm(14)
    result = predict()

    assert result["leak_detected"] is True
    assert result["leak_probability"] > 0.5


@needs_models
def test_model_localises_the_leak_zone():
    warm()
    store.leaks.inject_leak("J3", 250.0)
    warm(14)
    assert predict()["leak_zone"] == "ZONE_J3"


@needs_models
def test_model_estimates_a_plausible_rate():
    warm()
    store.leaks.inject_leak("J3", 250.0)
    warm(14)
    rate = predict()["leak_rate_lpm"]
    assert rate == pytest.approx(250.0, abs=120.0)


@needs_models
def test_probability_rises_when_a_leak_starts():
    warm()
    before = predict()["leak_probability"]
    store.leaks.inject_leak("J2", 300.0)
    warm(14)
    after = predict()["leak_probability"]
    assert after > before


@needs_models
def test_prediction_is_published_on_every_tick():
    warm()
    snap = engine.tick()
    assert "ml" in snap
    assert snap["ml"]["available"] is True


@needs_models
def test_production_ramp_does_not_fool_the_model():
    """The key challenge, checked against the model rather than the physics."""
    warm()
    for i in range(1, 9):
        store.update_machine(f"M{i}", production_pct=200.0, state="RUNNING")
    warm(16)
    result = predict()

    assert result["leak_detected"] is False, (
        f"model fired on a pure production ramp (P={result['leak_probability']})"
    )
