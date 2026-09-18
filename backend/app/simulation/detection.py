"""
Mass-balance leak detection.

This is the deterministic, physics-based detector: it finds leaks from
conservation of mass rather than from the ML classifier.

    internal junction:  residual(J) = flow(J) - sum(flow(child) for children)
    leaf junction:      residual(J) = flow(J) - demand(machine or tap it feeds)

A leak at node X inflates X and its ancestors equally, so the ancestors' child
sums absorb the extra and only X shows a positive residual.

This holds independently for every leak, including nested ones: with leaks at
both J3 and its child J8, flow_J3 carries both but the J8 child-sum absorbs J8's
share, so residual(J3) recovers J3's rate alone. Concurrent leaks are therefore
reported as separate findings rather than collapsed into one.

Why not the ML classifier: the trained models expect the textile generator's
feature set (rolling statistics, network balances, per-machine flow curves) which
this simulator does not reproduce — see README section 9. This detector operates
on the same residual principle the models were built around, but computes it
exactly instead of estimating it.
"""

from __future__ import annotations

from app.core.config import (
    BASELINE_MIN_TICKS,
    BASELINE_SIGMAS,
    BASELINE_WINDOW_TICKS,
)
# Must be the noise the simulator actually injects. Reading a stale 1% constant
# here while the meters ran at 2% halved every threshold and produced a 21%
# false-alarm rate.
from app.classifier_config import FLOW_NOISE_FRACTION as NOISE_SIGMA_FRACTION
from app.network.topology import ALL_JUNCTION_IDS, JUNCTION_TO_ENDPOINT, NODES

# A leak downstream of a header pulls that header's pressure down, because the
# extra flow raises friction loss. Pressure never localises a leak on its own,
# but a coincident drop corroborates a flow-side finding.
PRESSURE_DROP_EVIDENCE_BAR = 0.05

# A residual must clear measurement noise before it counts as a leak. Noise is
# proportional to flow, and a parent's residual combines its own noise with its
# children's, so the threshold scales with flow and is floored for small pipes.
NOISE_SIGMAS = 3.0
MIN_ABSOLUTE_THRESHOLD_LPM = 4.0

# Severity bands match LEAK_CONFIG in the classifier's config.py.
SEVERITY_BANDS = (
    (50.0, "Small"),
    (150.0, "Medium"),
    (float("inf"), "Large"),
)


def classify_severity(rate_lpm: float) -> str:
    for upper, label in SEVERITY_BANDS:
        if rate_lpm < upper:
            return label
    return "Large"


def _threshold_for(jid: str, flows: dict, endpoint_flows: dict) -> float:
    """Noise-aware detection threshold for one junction, in L/min."""
    node = NODES[jid]
    # Every meter contributing to this residual adds independent noise.
    contributors = [abs(flows.get(jid, 0.0))]
    if node.children:
        contributors += [abs(flows.get(c, 0.0)) for c in node.children]
    else:
        ep = JUNCTION_TO_ENDPOINT.get(jid)
        contributors.append(abs(endpoint_flows.get(ep, 0.0)) if ep else 0.0)

    # Independent noise sources add in quadrature.
    combined_sigma = (
        sum((v * NOISE_SIGMA_FRACTION) ** 2 for v in contributors) ** 0.5
    )
    return max(MIN_ABSOLUTE_THRESHOLD_LPM, NOISE_SIGMAS * combined_sigma)


def _baseline_analysis(flows: dict, history: list) -> dict:
    """Compare each junction against its own recent history.

    Mass balance catches water that goes missing between meters. It cannot
    catch a machine that is simply using more than it used to — that shows up
    only against the sensor's own past. Both are "abnormal water usage", so
    both are reported, kept separate because the remedies differ.
    """
    result: dict = {}
    if len(history) < BASELINE_MIN_TICKS:
        return result

    window = history[-BASELINE_WINDOW_TICKS:]
    for jid in ALL_JUNCTION_IDS:
        samples = [
            snap["flows"][jid] for snap in window
            if "flows" in snap and jid in snap["flows"]
        ]
        if len(samples) < BASELINE_MIN_TICKS:
            continue
        mean = sum(samples) / len(samples)
        variance = sum((v - mean) ** 2 for v in samples) / len(samples)
        sd = variance ** 0.5
        current = flows.get(jid, 0.0)
        # A flat baseline would flag ordinary noise, so floor the spread.
        spread = max(sd, abs(mean) * NOISE_SIGMA_FRACTION, 1.0)
        z = (current - mean) / spread
        result[jid] = {
            "baseline_lpm": round(mean, 2),
            "sd_lpm": round(sd, 2),
            "current_lpm": round(current, 2),
            "deviation_lpm": round(current - mean, 2),
            "z_score": round(z, 2),
            "abnormal": bool(z > BASELINE_SIGMAS),
        }
    return result


def _pressure_evidence(jid: str, pressures: dict, history: list) -> dict | None:
    """Pressure change at a junction versus its recent baseline."""
    if not pressures or jid not in pressures or len(history) < BASELINE_MIN_TICKS:
        return None
    samples = [
        snap["pressures"][jid] for snap in history[-BASELINE_WINDOW_TICKS:]
        if "pressures" in snap and jid in snap["pressures"]
    ]
    if len(samples) < BASELINE_MIN_TICKS:
        return None
    baseline = sum(samples) / len(samples)
    drop = baseline - pressures[jid]
    return {
        "baseline_bar": round(baseline, 3),
        "current_bar": round(pressures[jid], 3),
        "drop_bar": round(drop, 3),
        "corroborates": bool(drop > PRESSURE_DROP_EVIDENCE_BAR),
    }


def detect(
    flows: dict,
    endpoint_flows: dict,
    pressures: dict | None = None,
    history: list | None = None,
) -> dict:
    """
    Run mass-balance detection over one tick's measured flows.

    Parameters
    ----------
    flows : dict
        Junction id → measured flow (L/min), including noise and any override.
    endpoint_flows : dict
        Endpoint id (M1-M8, T1-T3) → the demand that endpoint actually drew.
    pressures : dict, optional
        Junction id → pressure (bar), used as corroborating evidence.
    history : list, optional
        Recent tick snapshots, used to build each sensor's own baseline.

    Returns
    -------
    dict
        Detection result, always with the same shape so the UI can render it
        unconditionally.
    """
    residuals: dict[str, float] = {}
    thresholds: dict[str, float] = {}

    for jid in ALL_JUNCTION_IDS:
        node = NODES[jid]
        measured = flows.get(jid, 0.0)
        if node.children:
            expected = sum(flows.get(c, 0.0) for c in node.children)
        else:
            ep = JUNCTION_TO_ENDPOINT.get(jid)
            expected = endpoint_flows.get(ep, 0.0) if ep else 0.0
        residuals[jid] = measured - expected
        thresholds[jid] = _threshold_for(jid, flows, endpoint_flows)

    # Every junction clearing its own noise threshold is a distinct leak: the
    # residuals are independent, so N candidates mean N leaks, not one.
    candidates = [
        jid for jid in ALL_JUNCTION_IDS if residuals[jid] > thresholds[jid]
    ]

    history = history or []
    pressures = pressures or {}
    baselines = _baseline_analysis(flows, history)

    leaks = [
        {
            "node": jid,
            "zone": f"ZONE_{jid}",
            "endpoint": JUNCTION_TO_ENDPOINT.get(jid),
            "rate_lpm": round(residuals[jid], 2),
            "severity": classify_severity(residuals[jid]),
            # How far the residual clears the noise floor, capped at 1.
            "confidence": round(
                min(1.0, residuals[jid] / (thresholds[jid] * 2.0)), 3
            ),
            "pressure": _pressure_evidence(jid, pressures, history),
        }
        for jid in candidates
    ]
    # Largest first: the biggest loss is the one to send a crew to.
    leaks.sort(key=lambda leak: leak["rate_lpm"], reverse=True)

    # Junctions drawing far more than their own history, which mass balance
    # alone would not flag (e.g. a machine quietly wasting water).
    abnormal = [
        {"node": jid, **info}
        for jid, info in baselines.items()
        if info["abnormal"] and jid not in candidates
    ]
    abnormal.sort(key=lambda item: item["deviation_lpm"], reverse=True)

    base = {
        "baselines": baselines,
        "abnormal_consumption": abnormal,
        "residuals": {k: round(v, 2) for k, v in residuals.items()},
        "thresholds": {k: round(v, 2) for k, v in thresholds.items()},
        "candidates": candidates,
        "leaks": leaks,
        "leak_count": len(leaks),
        "total_loss_lpm": round(sum(leak["rate_lpm"] for leak in leaks), 2),
    }

    if not leaks:
        return {
            **base,
            "leak_detected": False,
            "leak_node": None,
            "leak_zone": None,
            "leak_rate_lpm": 0.0,
            "severity": None,
            "confidence": 0.0,
            "endpoint": None,
        }

    # The top-level fields describe the primary (largest) leak and are kept so
    # existing single-leak consumers keep working.
    primary = leaks[0]
    return {
        **base,
        "leak_detected": True,
        "leak_node": primary["node"],
        "leak_zone": primary["zone"],
        "leak_rate_lpm": primary["rate_lpm"],
        "severity": primary["severity"],
        "confidence": primary["confidence"],
        "endpoint": primary["endpoint"],
    }
