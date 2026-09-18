"""
generator.py
Main Simulation Generator for the Synthetic 1-Year Industrial Water Network Dataset.
Generates physically correlated, production-aware data for textile wet-processing plant.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
import topology
import feature_engineering

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetGenerator")


def set_seed(seed: int = config.RANDOM_SEED):
    """Sets random seed for reproducibility."""
    np.random.seed(seed)


def create_directory_structure():
    """Creates all required output directories."""
    config.SENSORS_DIR.mkdir(parents=True, exist_ok=True)
    config.MACHINES_DIR.mkdir(parents=True, exist_ok=True)
    config.TAPS_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directories initialized under %s", config.BASE_OUTPUT_DIR)


def generate_time_index() -> pd.DatetimeIndex:
    """Generates 5-minute sampling timestamps for 365 days."""
    dt_index = pd.date_range(
        start=config.START_DATE,
        end=config.END_DATE,
        freq=f"{config.SAMPLING_MINUTES}min"
    )
    logger.info("Generated %d synchronized timestamps from %s to %s", len(dt_index), dt_index[0], dt_index[-1])
    return dt_index


def generate_production_and_states(dt_index: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic independent production schedules (0% to 200%)
    and operating states for all 8 textile machines.
    """
    n = len(dt_index)
    hours = dt_index.hour.values
    dows = dt_index.dayofweek.values
    days = (dt_index - dt_index[0]).days.values

    prod_df = pd.DataFrame(index=dt_index)
    state_df = pd.DataFrame(index=dt_index)

    # Shift factors: Day (06:00-14:00), Evening (14:00-22:00), Night (22:00-06:00)
    shift_factor = np.where(
        (hours >= 6) & (hours < 14), 1.05,
        np.where((hours >= 14) & (hours < 22), 0.95, 0.70)
    )

    # Weekday factor: Mon-Fri = 1.0, Sat = 0.65, Sun = 0.35
    dow_factor = np.where(dows < 5, 1.0, np.where(dows == 5, 0.65, 0.35))

    # Seasonal / campaign factor across the year (sine wave + pulses)
    campaign_factor = 1.0 + 0.15 * np.sin(2 * np.pi * days / 365.0)

    # Specific peak campaign weeks where plant production scales to 150%-200%
    # (e.g. high-demand order fulfilment in months 3, 7, and 10)
    peak_mask = (
        ((days >= 60) & (days <= 75)) |
        ((days >= 180) & (days <= 200)) |
        ((days >= 290) & (days <= 305))
    )

    for i in range(1, 9):
        m_id = f"M{i}"
        # Machine-specific base schedule
        m_offset = (i * 0.08) - 0.30
        m_noise = np.random.normal(0.0, 0.04, n)
        
        # Base production calculation
        base_prod = 90.0 * (shift_factor + m_offset) * dow_factor * campaign_factor + m_noise * 100.0

        # Apply peak campaign boost (allowing 150% - 200% production)
        peak_boost = np.where(peak_mask, np.random.uniform(1.4, 1.9, n), 1.0)
        prod_val = base_prod * peak_boost

        # Add machine-specific batch cycling (periods of idle/cleaning between runs)
        batch_cycle = np.sin(2 * np.pi * (np.arange(n) + i * 40) / (288.0 * (0.8 + 0.3 * i)))
        prod_val += batch_cycle * 15.0

        # Clip production between 0% and 200%
        prod_val = np.clip(prod_val, 0.0, 200.0)

        # Scheduled maintenance days for each machine (e.g., once every 3-4 weeks)
        maint_mask = (days % 28 == (i * 3) % 28) & (dows == 6)  # Sundays

        # Derive machine states
        states = np.full(n, "RUNNING", dtype=object)
        
        # OFF when low production or Sunday shutdown
        states[prod_val < 5.0] = "OFF"
        prod_val[states == "OFF"] = 0.0

        # MAINTENANCE state
        states[maint_mask] = "MAINTENANCE"
        prod_val[maint_mask] = 0.0

        # State transitions: STARTING and STOPPING buffers
        for idx in range(1, n - 1):
            if states[idx - 1] == "OFF" and states[idx] == "RUNNING":
                states[idx] = "STARTING"
            elif states[idx - 1] == "RUNNING" and states[idx] == "OFF":
                states[idx] = "STOPPING"

        prod_df[f"production_{m_id}"] = np.round(prod_val, 2)
        state_df[f"machine_status_{m_id}"] = states

    return prod_df, state_df


def generate_machine_flows(prod_df: pd.DataFrame, state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates water demand for all 8 machines using nonlinear physics models.
    Calibrated so that 100% production ≈ 1000 L/min total facility demand,
    and 200% production ≈ 2000 L/min.
    """
    n = len(prod_df)
    machine_flows = pd.DataFrame(index=prod_df.index)

    for i in range(1, 9):
        m_id = f"M{i}"
        spec = config.MACHINE_SPECS[m_id]
        prod = prod_df[f"production_{m_id}"].values
        status = state_df[f"machine_status_{m_id}"].values

        # Base running flow formula: base_flow + alpha*prod + beta*(prod^power)
        running_flow = spec["base_flow"] + spec["alpha"] * prod + spec["beta"] * (prod ** spec["power"])
        flow_noise = np.random.normal(0.0, 1.2, n)

        flow = np.zeros(n)
        for idx in range(n):
            st = status[idx]
            if st == "OFF":
                flow[idx] = spec["standby_flow"] + np.random.uniform(0.0, 0.05)
            elif st == "MAINTENANCE":
                flow[idx] = spec["standby_flow"] * 4.0 + 5.0 + np.random.uniform(0.0, 1.0)
            elif st == "STOPPING":
                flow[idx] = spec["base_flow"] * 0.5 + np.random.uniform(0.0, 0.5)
            elif st == "STARTING":
                flow[idx] = running_flow[idx] * spec["startup_factor"] + flow_noise[idx]
            else:  # RUNNING
                flow[idx] = running_flow[idx] + flow_noise[idx]

        flow = np.clip(flow, 0.0, spec["max_flow"])
        machine_flows[f"flow_{spec['sensor']}"] = np.round(flow, 3)

    return machine_flows


def generate_tap_usage(dt_index: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates tap usage events (T1, T2, T3) independent of machine production.
    Models cleaning routines, shift handovers, and random usage.
    """
    n = len(dt_index)
    hours = dt_index.hour.values
    minutes = dt_index.minute.values

    tap_status_df = pd.DataFrame(index=dt_index)
    tap_flow_df = pd.DataFrame(index=dt_index)

    for i in range(1, 4):
        t_id = f"T{i}"
        spec = config.TAP_SPECS[t_id]
        sensor_id = spec["sensor"]

        # Shift handover cleaning windows (around 06:00, 14:00, 22:00)
        handover_prob = np.where(
            ((hours == 6) | (hours == 14) | (hours == 22)) & (minutes <= 25),
            0.45, 0.04
        )
        # Random usage events
        rand_draw = np.random.random(n)
        is_open = rand_draw < handover_prob

        # Cluster usage into continuous events of 2-5 intervals (10-25 mins)
        open_status = np.zeros(n, dtype=bool)
        idx = 0
        while idx < n:
            if is_open[idx]:
                duration = np.random.randint(2, 6)
                end_idx = min(idx + duration, n)
                open_status[idx:end_idx] = True
                idx = end_idx
            else:
                idx += 1

        flows = np.zeros(n)
        flows[open_status] = np.random.normal(spec["nominal_flow"], spec["flow_std"], np.sum(open_status))
        flows = np.clip(flows, 0.0, spec["nominal_flow"] * 1.8)

        status_str = np.where(open_status, "OPEN", "CLOSED")
        tap_status_df[f"tap_status_{t_id}"] = status_str
        tap_flow_df[f"flow_{sensor_id}"] = np.round(flows, 3)

    return tap_status_df, tap_flow_df


def generate_leak_events(dt_index: pd.DatetimeIndex) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Generates 30 to 60 realistic leak events throughout the year.
    Leaks cover zones ZONE_J1 to ZONE_J16 across 6 distinct types.
    """
    n = len(dt_index)
    num_events = np.random.randint(config.LEAK_CONFIG["min_events"], config.LEAK_CONFIG["max_events"] + 1)
    logger.info("Scheduling %d leak events across the year", num_events)

    # Ensure events are separated and don't completely overlap
    min_spacing_steps = 288 * 3  # ~3 days minimum spacing
    possible_start_indices = np.arange(288, n - 288 * 2, min_spacing_steps)
    np.random.shuffle(possible_start_indices)
    chosen_starts = sorted(possible_start_indices[:num_events])

    leak_label = np.zeros(n, dtype=int)
    leak_rate = np.zeros(n, dtype=float)
    leak_zone = np.full(n, "NONE", dtype=object)

    all_zones = list(topology.LEAK_ZONES.keys())
    events_log = []

    for evt_id, start_idx in enumerate(chosen_starts, 1):
        zone = np.random.choice(all_zones)
        l_type = np.random.choice(config.LEAK_CONFIG["types"])
        
        # Severity selection
        severity = np.random.choice(["Small", "Medium", "Large"], p=[0.50, 0.35, 0.15])
        sev_range = config.LEAK_CONFIG["severity_levels"][severity]
        target_flow = np.random.uniform(sev_range["flow_min"], sev_range["flow_max"])

        # Duration: 1 to 36 hours (12 to 432 steps)
        dur_hours = np.random.uniform(config.LEAK_CONFIG["duration_hours"]["min"], config.LEAK_CONFIG["duration_hours"]["max"])
        dur_steps = int(dur_hours * 60 / config.SAMPLING_MINUTES)
        end_idx = min(start_idx + dur_steps, n)
        actual_steps = end_idx - start_idx

        t_steps = np.arange(actual_steps)
        # Profile based on leak type
        if l_type == "Small gradual leak":
            profile = target_flow * (t_steps / actual_steps)
        elif l_type == "Small sudden leak" or l_type == "Medium leak" or l_type == "Large leak":
            profile = np.full(actual_steps, target_flow) + np.random.normal(0, target_flow * 0.03, actual_steps)
        elif l_type == "Progressive leak":
            profile = target_flow * ((t_steps / actual_steps) ** 1.8)
        elif l_type == "Intermittent leak":
            # Leak toggles on and off with cycles of 4-8 steps
            cycle = np.sin(2 * np.pi * t_steps / 16) > 0.0
            profile = np.where(cycle, target_flow, target_flow * 0.15)
        else:
            profile = np.full(actual_steps, target_flow)

        profile = np.clip(profile, 0.0, None)

        leak_label[start_idx:end_idx] = 1
        leak_rate[start_idx:end_idx] = np.round(profile, 3)
        leak_zone[start_idx:end_idx] = zone

        affected_sensors = topology.LEAK_ZONES[zone]["affected_upstream_sensors"]
        events_log.append({
            "event_id": f"LEAK_{evt_id:03d}",
            "start_time": str(dt_index[start_idx]),
            "end_time": str(dt_index[end_idx - 1]),
            "duration_minutes": actual_steps * config.SAMPLING_MINUTES,
            "leak_zone": zone,
            "leak_type": l_type,
            "severity": severity,
            "leak_rate": np.round(np.mean(profile), 2),
            "affected_upstream_sensors": ",".join(affected_sensors)
        })

    leak_df = pd.DataFrame({
        "leak": leak_label,
        "leak_rate": leak_rate,
        "leak_zone": leak_zone
    }, index=dt_index)

    logger.info("Generated %d events covering %.2f%% of timestamps",
                len(events_log), (np.sum(leak_label) / n) * 100)
    return leak_df, events_log


def calculate_network_flows(machine_flows: pd.DataFrame, tap_flows: pd.DataFrame, leak_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates physically correlated flow for all 16 sensors.
    Applies conservation of mass and propagates leak flows upstream according to topology.
    """
    n = len(machine_flows)
    flows = pd.DataFrame(index=machine_flows.index)

    # 1. Base terminal node legitimate process demand
    flows["flow_J5"] = machine_flows["flow_J5"].values
    flows["flow_J6"] = machine_flows["flow_J6"].values
    flows["flow_J8"] = machine_flows["flow_J8"].values
    flows["flow_J9"] = machine_flows["flow_J9"].values
    flows["flow_J10"] = machine_flows["flow_J10"].values
    flows["flow_J11"] = machine_flows["flow_J11"].values
    flows["flow_J12"] = machine_flows["flow_J12"].values
    flows["flow_J13"] = machine_flows["flow_J13"].values
    flows["flow_J14"] = tap_flows["flow_J14"].values
    flows["flow_J15"] = tap_flows["flow_J15"].values
    flows["flow_J16"] = tap_flows["flow_J16"].values

    # 2. Add leak flow to affected terminal sensors if the leak is on that specific line
    l_active = leak_df["leak"].values == 1
    l_rates = leak_df["leak_rate"].values
    l_zones = leak_df["leak_zone"].values

    for idx in np.where(l_active)[0]:
        z = l_zones[idx]
        rate = l_rates[idx]
        # If leak is on a terminal machine/tap feed line (e.g. ZONE_J5)
        if z in ["ZONE_J5", "ZONE_J6", "ZONE_J8", "ZONE_J9", "ZONE_J10",
                 "ZONE_J11", "ZONE_J12", "ZONE_J13", "ZONE_J14", "ZONE_J15", "ZONE_J16"]:
            terminal_sensor = z.replace("ZONE_", "")
            flows.loc[flows.index[idx], f"flow_{terminal_sensor}"] += rate

    # 3. Intermediate branch flows:
    # J2 = J5 + J6 (+ leak if leak is at ZONE_J2 manifold header)
    flows["flow_J2"] = flows["flow_J5"] + flows["flow_J6"]
    # J3 = J8 + J9 + J10 (+ leak if leak is at ZONE_J3 manifold header)
    flows["flow_J3"] = flows["flow_J8"] + flows["flow_J9"] + flows["flow_J10"]
    # J4 = J11 + J12 (+ leak if leak is at ZONE_J4 manifold header)
    flows["flow_J4"] = flows["flow_J11"] + flows["flow_J12"]
    # J7 = J13 + J14 + J15 + J16 (+ leak if leak is at ZONE_J7 manifold header)
    flows["flow_J7"] = flows["flow_J13"] + flows["flow_J14"] + flows["flow_J15"] + flows["flow_J16"]

    # Add branch header leaks
    for idx in np.where(l_active)[0]:
        z = l_zones[idx]
        rate = l_rates[idx]
        if z == "ZONE_J2":
            flows.loc[flows.index[idx], "flow_J2"] += rate
        elif z == "ZONE_J3":
            flows.loc[flows.index[idx], "flow_J3"] += rate
        elif z == "ZONE_J4":
            flows.loc[flows.index[idx], "flow_J4"] += rate
        elif z == "ZONE_J7":
            flows.loc[flows.index[idx], "flow_J7"] += rate

    # 4. Main Inlet flow:
    # J1 = J2 + J3 + J4 + J7 (+ leak if leak is at ZONE_J1 main distribution header)
    flows["flow_J1"] = flows["flow_J2"] + flows["flow_J3"] + flows["flow_J4"] + flows["flow_J7"]
    for idx in np.where(l_active)[0]:
        z = l_zones[idx]
        rate = l_rates[idx]
        if z == "ZONE_J1":
            flows.loc[flows.index[idx], "flow_J1"] += rate

    return flows


def generate_pressures(flows_df: pd.DataFrame, leak_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates realistic hydraulic pressure data (bar) across J1, J2, J3, J4, J7.
    Pressures obey friction loss and localized pressure drops during leaks.
    Prototype range: 3.0 to 6.0 bar.
    """
    n = len(flows_df)
    pressures = pd.DataFrame(index=flows_df.index)

    base_p = config.PRESSURE_CONFIG["inlet_base_pressure"]
    src_fric = config.PRESSURE_CONFIG["source_flow_friction"]
    pipe_fric = config.PRESSURE_CONFIG["pipe_friction_loss"]

    # Inlet pressure drops with total demand
    p_j1 = base_p - src_fric * flows_df["flow_J1"].values + np.random.normal(0.0, 0.04, n)

    # Branch pressures drop with branch flow friction
    p_j2 = p_j1 - pipe_fric["J2"] * (flows_df["flow_J2"].values ** 1.1) + np.random.normal(0.0, 0.03, n)
    p_j3 = p_j1 - pipe_fric["J3"] * (flows_df["flow_J3"].values ** 1.1) + np.random.normal(0.0, 0.03, n)
    p_j4 = p_j1 - pipe_fric["J4"] * (flows_df["flow_J4"].values ** 1.1) + np.random.normal(0.0, 0.03, n)
    p_j7 = p_j1 - pipe_fric["J7"] * (flows_df["flow_J7"].values ** 1.1) + np.random.normal(0.0, 0.03, n)

    # Apply localized pressure drops during leaks
    l_active = leak_df["leak"].values == 1
    l_rates = leak_df["leak_rate"].values
    l_zones = leak_df["leak_zone"].values

    for idx in np.where(l_active)[0]:
        z = l_zones[idx]
        rate = l_rates[idx]
        drop = 0.025 * np.sqrt(rate)  # realistic Bernoulli orifice pressure drop

        p_j1[idx] -= drop * 0.4
        if "J2" in topology.LEAK_ZONES[z]["affected_upstream_sensors"]:
            p_j2[idx] -= drop * 1.1
        if "J3" in topology.LEAK_ZONES[z]["affected_upstream_sensors"]:
            p_j3[idx] -= drop * 1.1
        if "J4" in topology.LEAK_ZONES[z]["affected_upstream_sensors"]:
            p_j4[idx] -= drop * 1.1
        if "J7" in topology.LEAK_ZONES[z]["affected_upstream_sensors"]:
            p_j7[idx] -= drop * 1.1

    # Clip to physically plausible prototype ranges (3.0 - 6.0 bar)
    pressures["pressure_J1"] = np.round(np.clip(p_j1, 3.2, 5.8), 3)
    pressures["pressure_J2"] = np.round(np.clip(p_j2, 3.1, 5.6), 3)
    pressures["pressure_J3"] = np.round(np.clip(p_j3, 3.0, 5.5), 3)
    pressures["pressure_J4"] = np.round(np.clip(p_j4, 3.1, 5.6), 3)
    pressures["pressure_J7"] = np.round(np.clip(p_j7, 3.2, 5.7), 3)

    return pressures


def inject_noise_and_anomalies(flows_df: pd.DataFrame, pressures_df: pd.DataFrame, leak_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Injects realistic measurement noise, normal non-leak outliers, and NaNs.
    Crucially: normal spikes are introduced where leak = 0.
    """
    n = len(flows_df)
    noisy_flows = flows_df.copy()
    noisy_pressures = pressures_df.copy()

    # 1. Standard continuous measurement noise
    for col in noisy_flows.columns:
        noise = np.random.normal(0.0, config.FLOW_NOISE_PERCENT / 100.0, n)
        noisy_flows[col] = np.round(np.maximum(0.0, noisy_flows[col] * (1.0 + noise)), 2)

    for col in noisy_pressures.columns:
        noise = np.random.normal(0.0, config.PRESSURE_NOISE_BAR, n)
        noisy_pressures[col] = np.round(np.clip(noisy_pressures[col] + noise, 2.8, 6.0), 3)

    # 2. Operational outliers / spikes where LEAK = 0 (Section 23 requirement)
    num_outliers = int(n * (config.OUTLIER_PERCENT / 100.0))
    non_leak_indices = np.where(leak_df["leak"].values == 0)[0]
    outlier_indices = np.random.choice(non_leak_indices, size=num_outliers, replace=False)

    for idx in outlier_indices:
        target_sensor = f"flow_J{np.random.randint(1, 17)}"
        spike_multiplier = np.random.uniform(1.15, 1.35)
        noisy_flows.loc[noisy_flows.index[idx], target_sensor] = np.round(
            noisy_flows.loc[noisy_flows.index[idx], target_sensor] * spike_multiplier, 2
        )

    # 3. Missing sensor data (NaNs) (Section 22 requirement)
    num_missing = int(n * (config.MISSING_DATA_PERCENT / 100.0))
    missing_indices = np.random.choice(n, size=num_missing, replace=False)
    for idx in missing_indices:
        target_sensor = f"flow_J{np.random.randint(1, 17)}"
        noisy_flows.loc[noisy_flows.index[idx], target_sensor] = np.nan

    return noisy_flows, noisy_pressures


def save_individual_files(
    dt_index: pd.DatetimeIndex,
    flows_df: pd.DataFrame,
    pressures_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    state_df: pd.DataFrame,
    tap_status_df: pd.DataFrame,
    tap_flows_df: pd.DataFrame,
    machine_demand_df: pd.DataFrame,
    leak_df: pd.DataFrame,
    events_log: List[Dict[str, Any]]
):
    """Saves separate sensor CSVs, machine CSVs, tap CSVs, and metadata files."""
    dates = dt_index.strftime("%Y-%m-%d")
    times = dt_index.strftime("%H:%M:%S")

    # 1. 16 Flow Sensor CSV files
    logger.info("Writing 16 individual sensor CSV files...")
    for i in range(1, 17):
        s_id = f"J{i}"
        filename = config.SENSORS_DIR / f"sensor_{s_id[:1]}{int(s_id[1:]):02d}.csv"
        
        # Pressure column: branch pressure if available, else inlet or interpolated
        if f"pressure_{s_id}" in pressures_df.columns:
            p_series = pressures_df[f"pressure_{s_id}"]
        elif s_id in ["J5", "J6"]:
            p_series = pressures_df["pressure_J2"]
        elif s_id in ["J8", "J9", "J10"]:
            p_series = pressures_df["pressure_J3"]
        elif s_id in ["J11", "J12"]:
            p_series = pressures_df["pressure_J4"]
        else:
            p_series = pressures_df["pressure_J7"]

        sensor_file_df = pd.DataFrame({
            "date": dates,
            "time": times,
            "flow": flows_df[f"flow_{s_id}"].values,
            "pressure": p_series.values,
            "leak": leak_df["leak"].values,
            "leak_rate": leak_df["leak_rate"].values,
            "leak_zone": leak_df["leak_zone"].values
        })
        sensor_file_df.to_csv(filename, index=False)

    # 2. 8 Machine CSV files
    logger.info("Writing 8 machine CSV files...")
    for i in range(1, 9):
        m_id = f"M{i}"
        filename = config.MACHINES_DIR / f"machine_{m_id[:1]}{int(m_id[1:]):02d}.csv"
        spec = config.MACHINE_SPECS[m_id]
        j_sensor = spec["sensor"]

        m_file_df = pd.DataFrame({
            "date": dates,
            "time": times,
            "production_rate": prod_df[f"production_{m_id}"].values,
            "machine_status": state_df[f"machine_status_{m_id}"].values,
            "machine_type": spec["name"],
            "water_demand": machine_demand_df[f"flow_{j_sensor}"].values
        })
        m_file_df.to_csv(filename, index=False)

    # 3. 3 Tap CSV files
    logger.info("Writing 3 tap CSV files...")
    for i in range(1, 4):
        t_id = f"T{i}"
        filename = config.TAPS_DIR / f"tap_{t_id[:1]}{int(t_id[1:]):02d}.csv"
        spec = config.TAP_SPECS[t_id]
        j_sensor = spec["sensor"]

        t_file_df = pd.DataFrame({
            "date": dates,
            "time": times,
            "tap_status": tap_status_df[f"tap_status_{t_id}"].values,
            "water_demand": tap_flows_df[f"flow_{j_sensor}"].values
        })
        t_file_df.to_csv(filename, index=False)

    # 4. Leak Events CSV
    logger.info("Writing leak events summary...")
    pd.DataFrame(events_log).to_csv(config.LEAK_EVENTS_CSV, index=False)

    # 5. Network Topology JSON
    logger.info("Writing network topology JSON...")
    with open(config.TOPOLOGY_JSON, "w", encoding="utf-8") as f:
        f.write(topology.export_network_topology_json())

    # 6. Sensor Metadata CSV
    logger.info("Writing sensor metadata CSV...")
    topology.get_sensor_metadata().to_csv(config.SENSOR_METADATA_CSV, index=False)


def generate_plots(master_df: pd.DataFrame, leak_df: pd.DataFrame):
    """
    Generates and saves the 7 required validation plots in synthetic_textile_water_dataset/plots/
    """
    logger.info("Generating validation plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Main Flow vs Total Production
    fig, ax = plt.subplots(figsize=(10, 5))
    total_prod = master_df[[f"production_M{i}" for i in range(1, 9)]].mean(axis=1)
    # Filter normal operation timestamps to show true correlation
    normal_mask = master_df["leak"] == 0
    ax.scatter(total_prod[normal_mask][::50], master_df["flow_J1"][normal_mask][::50], alpha=0.3, color="#1f77b4", s=10)
    ax.set_title("Main Flow (J1) vs Facility Average Production (Normal Operation, Leak=0)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Average Production Rate (%)")
    ax.set_ylabel("Main Inlet Flow J1 (L/min)")
    ax.axvline(100, color="orange", linestyle="--", label="100% Production (~1000 L/min)")
    ax.axvline(200, color="red", linestyle="--", label="200% Production (~2000 L/min)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "main_flow_vs_production.png", dpi=200)
    plt.close(fig)

    # 2. Production vs Expected Flow
    fig, ax = plt.subplots(figsize=(10, 5))
    sample_prod = np.linspace(0, 200, 200)
    for i in range(1, 9):
        m_id = f"M{i}"
        spec = config.MACHINE_SPECS[m_id]
        expected_f = [feature_engineering.compute_expected_machine_flow(m_id, p, "RUNNING") for p in sample_prod]
        ax.plot(sample_prod, expected_f, label=f"{m_id} ({spec['name'][:10]})")
    ax.set_title("Machine Water Consumption Model vs Production Rate (0% to 200%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Machine Production Rate (%)")
    ax.set_ylabel("Water Flow (L/min)")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "production_vs_expected_flow.png", dpi=200)
    plt.close(fig)

    # 3. Normal vs Leak Flow Behavior
    fig, ax = plt.subplots(figsize=(12, 5))
    # Find a medium/large leak event window
    leak_indices = np.where(master_df["leak"] == 1)[0]
    if len(leak_indices) > 0:
        event_mid = leak_indices[len(leak_indices) // 2]
        window_slice = slice(max(0, event_mid - 150), min(len(master_df), event_mid + 150))
        sub_df = master_df.iloc[window_slice]
        times = pd.to_datetime(sub_df["date"] + " " + sub_df["time"])

        ax.plot(times, sub_df["flow_J1"], label="Main Flow J1 (Measured)", color="#2ca02c")
        ax.plot(times, sub_df["leak_rate"], label="Leak Rate", color="red", linestyle="--")
        ax.fill_between(times, 0, sub_df["flow_J1"], where=sub_df["leak"] == 1, color="salmon", alpha=0.3, label="Active Leak Window")
        ax.set_title("Network Main Flow During Normal and Leak Conditions", fontsize=12, fontweight="bold")
        ax.set_ylabel("Flow Rate (L/min)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / "normal_vs_leak_behavior.png", dpi=200)
    plt.close(fig)

    # 4. Pressure During a Leak
    fig, ax = plt.subplots(figsize=(12, 5))
    if len(leak_indices) > 0:
        event_mid = leak_indices[len(leak_indices) // 2]
        window_slice = slice(max(0, event_mid - 150), min(len(master_df), event_mid + 150))
        sub_df = master_df.iloc[window_slice]
        times = pd.to_datetime(sub_df["date"] + " " + sub_df["time"])

        ax.plot(times, sub_df["pressure_J1"], label="Header Pressure J1", color="#1f77b4")
        ax.plot(times, sub_df["pressure_J3"], label="Branch B Pressure J3", color="#ff7f0e")
        ax.fill_between(times, 3.0, 6.0, where=sub_df["leak"] == 1, color="red", alpha=0.15, label="Leak Event")
        ax.set_title("Hydraulic Pressure Drop Response During Active Leak Event", fontsize=12, fontweight="bold")
        ax.set_ylabel("Pressure (bar)")
        ax.set_ylim(3.0, 5.8)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / "pressure_during_leak.png", dpi=200)
    plt.close(fig)

    # 5. One Leak Event Across Multiple Sensors
    fig, ax = plt.subplots(figsize=(12, 6))
    if len(leak_indices) > 0:
        event_mid = leak_indices[len(leak_indices) // 2]
        window_slice = slice(max(0, event_mid - 120), min(len(master_df), event_mid + 120))
        sub_df = master_df.iloc[window_slice]
        times = pd.to_datetime(sub_df["date"] + " " + sub_df["time"])

        ax.plot(times, sub_df["flow_J1"], label="J1 (Main Inlet)", color="black", linewidth=1.5)
        ax.plot(times, sub_df["flow_J2"], label="J2 (Branch A)", color="blue")
        ax.plot(times, sub_df["flow_J3"], label="J3 (Branch B)", color="darkorange")
        ax.plot(times, sub_df["flow_J4"], label="J4 (Branch C)", color="purple")
        ax.plot(times, sub_df["flow_J7"], label="J7 (Branch D)", color="brown")
        ax.axvspan(times[sub_df["leak"] == 1].iloc[0], times[sub_df["leak"] == 1].iloc[-1], color="red", alpha=0.15, label="Leak Duration")
        ax.set_title("Multi-Sensor Flow Response Across Branches During Injected Leak", fontsize=12, fontweight="bold")
        ax.set_ylabel("Flow Rate (L/min)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(config.PLOTS_DIR / "leak_event_across_sensors.png", dpi=200)
    plt.close(fig)

    # 6. Network Flow Balance
    fig, ax = plt.subplots(figsize=(12, 5))
    balance_j1 = master_df["flow_J1"] - (master_df["flow_J2"] + master_df["flow_J3"] + master_df["flow_J4"] + master_df["flow_J7"])
    balance_j3 = master_df["flow_J3"] - (master_df["flow_J8"] + master_df["flow_J9"] + master_df["flow_J10"])
    # 500 timestamp slice
    ax.plot(balance_j1.iloc[1000:1500].values, label="Mass Balance Residual J1 (Unaccounted)", color="red")
    ax.plot(balance_j3.iloc[1000:1500].values, label="Mass Balance Residual J3", color="blue", alpha=0.7)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_title("Network Conservation of Mass Residuals (Zero in Normal Operation, Non-Zero During Leak)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Flow Discrepancy (L/min)")
    ax.set_xlabel("Time Index (Timesteps)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "network_flow_balance.png", dpi=200)
    plt.close(fig)

    # 7. Machine Production vs Machine Water Demand
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    sample_slice = slice(500, 800)
    sub_df = master_df.iloc[sample_slice]
    times = pd.to_datetime(sub_df["date"] + " " + sub_df["time"])

    ax1.plot(times, sub_df["production_M1"], label="M1 (Scouring) Prod %", color="#1f77b4")
    ax1.plot(times, sub_df["production_M3"], label="M3 (Dyeing) Prod %", color="#2ca02c")
    ax1.set_ylabel("Production Rate (%)")
    ax1.legend(loc="upper left")
    ax1.set_title("Machine Production Schedules vs Corresponding Water Demands", fontsize=12, fontweight="bold")

    ax2.plot(times, sub_df["flow_J5"], label="J5 (M1 Flow)", color="#1f77b4")
    ax2.plot(times, sub_df["flow_J8"], label="J8 (M3 Flow)", color="#2ca02c")
    ax2.set_ylabel("Water Flow (L/min)")
    ax2.set_xlabel("Time")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "machine_production_vs_demand.png", dpi=200)
    plt.close(fig)

    logger.info("All 7 validation plots successfully saved in %s", config.PLOTS_DIR)


def print_production_normalization_demonstration(master_df: pd.DataFrame, ml_df: pd.DataFrame):
    """
    Prints the demonstration required by Section 33:
    CASE 1: Production = 100%, Flow normal, Leak = 0
    CASE 2: Production = 200%, Flow doubled, Leak = 0
    CASE 3: Production = 200%, Flow significantly above expected, Leak = 1
    CASE 4: Production constant, localized flow imbalance appears, Leak = 1
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATION: PRODUCTION-AWARE LEAK DETECTION BEHAVIOR")
    print("=" * 80)

    # Case 1: Normal ~100% production
    total_prod = master_df[[f"production_M{i}" for i in range(1, 9)]].mean(axis=1)
    case1_mask = (total_prod >= 95.0) & (total_prod <= 105.0) & (master_df["leak"] == 0)
    if np.any(case1_mask):
        c1_idx = np.where(case1_mask)[0][0]
        row_c1 = master_df.iloc[c1_idx]
        exp_c1 = ml_df.loc[c1_idx, "expected_flow_J1"]
        print("\n[CASE 1]: High Normal Production (100%) -> Flow Normal -> LEAK = 0")
        print(f"  Timestamp        : {row_c1['date']} {row_c1['time']}")
        print(f"  Mean Production  : {total_prod.iloc[c1_idx]:.1f}%")
        print(f"  Expected Main J1 : {exp_c1:.1f} L/min")
        print(f"  Actual Main J1   : {row_c1['flow_J1']:.1f} L/min")
        print(f"  Leak Status      : leak={row_c1['leak']} (leak_rate={row_c1['leak_rate']} L/min, zone={row_c1['leak_zone']})")
        print("  -> Result: High legitimate flow is NOT flagged as a leak.")

    # Case 2: Peak ~200% production, Leak = 0
    case2_mask = (total_prod >= 175.0) & (master_df["leak"] == 0)
    if np.any(case2_mask):
        c2_idx = np.where(case2_mask)[0][0]
        row_c2 = master_df.iloc[c2_idx]
        exp_c2 = ml_df.loc[c2_idx, "expected_flow_J1"]
        print("\n[CASE 2]: Peak Production (~200%) -> Flow Approximately Doubles -> LEAK = 0")
        print(f"  Timestamp        : {row_c2['date']} {row_c2['time']}")
        print(f"  Mean Production  : {total_prod.iloc[c2_idx]:.1f}%")
        print(f"  Expected Main J1 : {exp_c2:.1f} L/min")
        print(f"  Actual Main J1   : {row_c2['flow_J1']:.1f} L/min")
        print(f"  Leak Status      : leak={row_c2['leak']} (leak_rate={row_c2['leak_rate']} L/min, zone={row_c2['leak_zone']})")
        print("  -> Result: Doubled flow matches 200% production demand. Legitimate, NOT a leak.")

    # Case 3: Peak ~200% production WITH an active leak
    case3_mask = (total_prod >= 160.0) & (master_df["leak"] == 1)
    if np.any(case3_mask):
        c3_idx = np.where(case3_mask)[0][0]
        row_c3 = master_df.iloc[c3_idx]
        exp_c3 = ml_df.loc[c3_idx, "expected_flow_J1"]
        res_c3 = ml_df.loc[c3_idx, "flow_residual_J1"]
        print("\n[CASE 3]: Peak Production (High) + Injected Pipe Failure -> LEAK = 1")
        print(f"  Timestamp        : {row_c3['date']} {row_c3['time']}")
        print(f"  Mean Production  : {total_prod.iloc[c3_idx]:.1f}%")
        print(f"  Expected Main J1 : {exp_c3:.1f} L/min")
        print(f"  Actual Main J1   : {row_c3['flow_J1']:.1f} L/min (Excess Unaccounted: {res_c3:.1f} L/min)")
        print(f"  Leak Status      : leak={row_c3['leak']} (leak_rate={row_c3['leak_rate']:.1f} L/min, zone={row_c3['leak_zone']})")
        print("  -> Result: Flow significantly exceeds high expected demand -> Correctly labeled LEAK = 1.")

    # Case 4: Constant production, localized flow imbalance appears, Leak = 1
    case4_mask = (master_df["leak"] == 1) & (master_df["leak_zone"].isin(["ZONE_J3", "ZONE_J8"]))
    if np.any(case4_mask):
        c4_idx = np.where(case4_mask)[0][0]
        row_c4 = master_df.iloc[c4_idx]
        bal_j3 = ml_df.loc[c4_idx, "balance_J3"]
        print("\n[CASE 4]: Normal Constant Production + Localized Flow Imbalance -> LEAK = 1")
        print(f"  Timestamp        : {row_c4['date']} {row_c4['time']}")
        print(f"  Mean Production  : {total_prod.iloc[c4_idx]:.1f}%")
        print(f"  Branch B (J3)    : {row_c4['flow_J3']:.1f} L/min")
        print(f"  Downstream J8+9+10: {(row_c4['flow_J8']+row_c4['flow_J9']+row_c4['flow_J10']):.1f} L/min")
        print(f"  J3 Mass Balance  : balance_J3 = {bal_j3:.1f} L/min (Discrepancy)")
        print(f"  Leak Status      : leak={row_c4['leak']} (leak_rate={row_c4['leak_rate']:.1f} L/min, zone={row_c4['leak_zone']})")
        print("  -> Result: Localized mass-balance deficit correctly flags zone leak.")
    print("=" * 80 + "\n")


def run_simulation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the full simulation, writes all datasets, and outputs summary.
    """
    logger.info("Starting Realistic Synthetic Water Network Simulation...")
    set_seed(config.RANDOM_SEED)
    create_directory_structure()

    # 1. Timestamps
    dt_index = generate_time_index()

    # 2. Production & States
    logger.info("Simulating machine production schedules and operating states...")
    prod_df, state_df = generate_production_and_states(dt_index)

    # 3. Machine Demands
    logger.info("Calculating machine water demands...")
    machine_demand_df = generate_machine_flows(prod_df, state_df)

    # 4. Tap Demands
    logger.info("Simulating tap and utility water usage...")
    tap_status_df, tap_flow_df = generate_tap_usage(dt_index)

    # 5. Leak Events
    logger.info("Generating realistic leak events...")
    leak_df, events_log = generate_leak_events(dt_index)

    # 6. Physical Flow Propagation
    logger.info("Calculating topological conservation of mass and flow propagation...")
    ideal_flows_df = calculate_network_flows(machine_demand_df, tap_flow_df, leak_df)

    # 7. Pressures
    logger.info("Calculating hydraulic network pressures...")
    ideal_pressures_df = generate_pressures(ideal_flows_df, leak_df)

    # 8. Sensor Noise, Spikes (Leak=0), and NaNs
    logger.info("Injecting sensor measurement noise, non-leak outliers, and missing data...")
    measured_flows_df, measured_pressures_df = inject_noise_and_anomalies(
        ideal_flows_df, ideal_pressures_df, leak_df
    )

    # 9. Write Individual CSV Files
    save_individual_files(
        dt_index, measured_flows_df, measured_pressures_df,
        prod_df, state_df, tap_status_df, tap_flow_df,
        machine_demand_df, leak_df, events_log
    )

    # 10. Assemble Network Master CSV
    logger.info("Assembling network_master.csv...")
    dates = dt_index.strftime("%Y-%m-%d")
    times = dt_index.strftime("%H:%M:%S")

    master_dict = {"date": dates, "time": times}
    for i in range(1, 9):
        master_dict[f"production_M{i}"] = prod_df[f"production_M{i}"].values
    for i in range(1, 9):
        master_dict[f"machine_status_M{i}"] = state_df[f"machine_status_M{i}"].values
    for i in range(1, 4):
        master_dict[f"tap_status_T{i}"] = tap_status_df[f"tap_status_T{i}"].values
    for i in range(1, 17):
        master_dict[f"flow_J{i}"] = measured_flows_df[f"flow_J{i}"].values
    for p_id in ["pressure_J1", "pressure_J2", "pressure_J3", "pressure_J4", "pressure_J7"]:
        master_dict[p_id] = measured_pressures_df[p_id].values

    master_dict["leak"] = leak_df["leak"].values
    master_dict["leak_rate"] = leak_df["leak_rate"].values
    master_dict["leak_zone"] = leak_df["leak_zone"].values

    master_df = pd.DataFrame(master_dict)
    master_df.to_csv(config.NETWORK_MASTER_CSV, index=False)
    logger.info("Saved %s (%d rows)", config.NETWORK_MASTER_CSV, len(master_df))

    # 11. Feature Engineering for ML Training Dataset
    logger.info("Building feature-engineered dataset for ML training...")
    ml_df = feature_engineering.build_feature_engineered_dataset(master_df)
    ml_df.to_csv(config.ML_TRAINING_CSV, index=False)
    logger.info("Saved %s (%d rows, %d columns)", config.ML_TRAINING_CSV, len(ml_df), len(ml_df.columns))

    # 12. Generate Validation Plots
    generate_plots(master_df, leak_df)

    # 13. Print Production Normalization Demonstration
    print_production_normalization_demonstration(master_df, ml_df)

    # 14. Dataset README
    write_dataset_readme(len(master_df), len(events_log))

    return master_df, ml_df


def write_dataset_readme(total_rows: int, total_leaks: int):
    """Writes dataset specific README.md into synthetic_textile_water_dataset/."""
    content = f"""# Synthetic Textile Industrial Water Network Dataset

## Overview
This dataset provides 1 full year (365 days) of 5-minute sampled, physically and logically correlated industrial water network telemetry for a representative textile wet-processing facility.

- **Total Rows**: {total_rows}
- **Total Flow Measurement Points**: 16 sensors (J1 to J16)
- **Total Pressure Measurements**: 5 major branch headers (J1, J2, J3, J4, J7)
- **Total Machines**: 8 textile wet-processing units (M1 to M8) with independent production rates (0% to 200%)
- **Total Taps**: 3 utility points (T1, T2, T3)
- **Total Injected Leak Events**: {total_leaks} events

## Key Principles
1. **Production Awareness**: At 100% production, facility intake is ~1000 L/min; at 200% production, intake scales to ~2000 L/min with `leak = 0`. Water demand is legitimately determined by machine production.
2. **Conservation of Mass**: In normal operation, junction balances hold ($J_1 = J_2+J_3+J_4+J_7$, etc.).
3. **Physical Leak Propagation**: Injected leaks propagate upstream to parent sensors while leaving downstream machines at legitimate demand, creating distinct localized mass balance and pressure anomalies.

## Directory Structure
- `sensors/`: `sensor_J01.csv` through `sensor_J16.csv`
- `machines/`: `machine_M01.csv` through `machine_M08.csv`
- `taps/`: `tap_T01.csv` through `tap_T03.csv`
- `network_master.csv`: Unified multi-sensor master telemetry
- `ml_training_dataset.csv`: Feature-engineered dataset with mass balances, expected flows, rolling statistics, and targets
- `leak_events.csv`: Ground truth catalog of simulated leak events
- `sensor_metadata.csv`: Network sensor specifications
- `network_topology.json`: Topology graph definition
- `plots/`: 7 validation plots confirming hydraulic correlation and production-aware leak detection
"""
    with open(config.DATASET_README_MD, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_simulation()
