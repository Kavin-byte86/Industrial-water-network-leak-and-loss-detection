import os
import glob
import pandas as pd
import numpy as np

def parse_epanet_topology(inp_file_path):
    """
    Parses the EPANET .inp file to extract the [PIPES] topology.
    Returns a list of dicts: {'link': id, 'from': node1, 'to': node2}
    """
    links = []
    in_pipes_section = False
    with open(inp_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('[PIPES]'):
                in_pipes_section = True
                continue
            if in_pipes_section and line.startswith('['):
                break
            if in_pipes_section and line and not line.startswith(';'):
                parts = line.split()
                if len(parts) >= 3:
                    links.append({
                        'link': parts[0],
                        'from': parts[1],
                        'to': parts[2]
                    })
    return links

def build_network_balances(df, topology):
    """
    Constructs mass balance features for each node.
    Balance at Node = Sum(Inflows) - Sum(Outflows) - Demand
    """
    nodes = set()
    for t in topology:
        nodes.add(t['from'])
        nodes.add(t['to'])
        
    for node in nodes:
        inflow_cols = [f"flow_{t['link']}" for t in topology if t['to'] == node]
        outflow_cols = [f"flow_{t['link']}" for t in topology if t['from'] == node]
        
        # Only compute if all columns exist
        all_cols_exist = all(c in df.columns for c in inflow_cols + outflow_cols)
        demand_col = f"demand_{node}"
        if all_cols_exist:
            balance = pd.Series(0.0, index=df.index)
            for c in inflow_cols:
                balance += df[c]
            for c in outflow_cols:
                balance -= df[c]
            if demand_col in df.columns:
                balance -= df[demand_col]
            df[f"balance_{node}"] = balance
            
    return df

def load_scenario(scenario_dir, scenario_id):
    """
    Loads all data for a single LeakDB scenario into a DataFrame.
    """
    print(f"Loading {scenario_id}...")
    
    # 1. Labels (Master time index)
    labels_path = os.path.join(scenario_dir, "Labels.csv")
    df_labels = pd.read_csv(labels_path, parse_dates=["Timestamp"])
    df_labels = df_labels.rename(columns={"Label": "leak"}).set_index("Timestamp")
    
    dfs_to_concat = [df_labels]
    
    # 2. Flows (Links)
    flows_dir = os.path.join(scenario_dir, "Flows")
    for f in glob.glob(os.path.join(flows_dir, "*.csv")):
        link_id = os.path.basename(f).replace("Link_", "").replace(".csv", "")
        df_flow = pd.read_csv(f, parse_dates=["Timestamp"]).set_index("Timestamp")
        df_flow.columns = [f"flow_{link_id}"]
        dfs_to_concat.append(df_flow)
        
    # 3. Pressures (Nodes)
    press_dir = os.path.join(scenario_dir, "Pressures")
    for f in glob.glob(os.path.join(press_dir, "*.csv")):
        node_id = os.path.basename(f).replace("Node_", "").replace(".csv", "")
        df_press = pd.read_csv(f, parse_dates=["Timestamp"]).set_index("Timestamp")
        df_press.columns = [f"pressure_{node_id}"]
        dfs_to_concat.append(df_press)
        
    # 4. Demands (Nodes)
    demands_dir = os.path.join(scenario_dir, "Demands")
    for f in glob.glob(os.path.join(demands_dir, "*.csv")):
        node_id = os.path.basename(f).replace("Node_", "").replace(".csv", "")
        df_dem = pd.read_csv(f, parse_dates=["Timestamp"]).set_index("Timestamp")
        df_dem.columns = [f"demand_{node_id}"]
        dfs_to_concat.append(df_dem)
        
    df_merged = pd.concat(dfs_to_concat, axis=1)
        
    # 5. Leak Info (Actual leak demand if it exists)
    leaks_dir = os.path.join(scenario_dir, "Leaks")
    leak_rate = pd.Series(0.0, index=df_merged.index)
    
    # If there are leaks in this scenario
    if os.path.exists(leaks_dir):
        for f in glob.glob(os.path.join(leaks_dir, "*_demand.csv")):
            df_leak = pd.read_csv(f, parse_dates=["Timestamp"]).set_index("Timestamp")
            # LeakDB uses "Description" or "Value" randomly, so use positional 0
            leak_rate += df_leak.iloc[:, 0].reindex(df_merged.index).fillna(0.0)
            
    df_merged["leak_rate"] = leak_rate
    
    # --- 5-MINUTE RESAMPLING ---
    df_merged = df_merged.sort_index()
    df_resampled = df_merged.resample('5T').asfreq()
    
    df_resampled["leak"] = df_resampled["leak"].ffill()
    
    for col in df_resampled.columns:
        if col != "leak":
            df_resampled[col] = df_resampled[col].interpolate(method='linear')
            
    df_merged = df_resampled
    # ---------------------------
    
    df_merged["scenario_id"] = scenario_id
    
    # Reset index to keep Timestamp as a column
    df_merged = df_merged.reset_index()
    return df_merged

def generate_event_summary(df):
    """
    Generates a summary of all leak events in the loaded dataset.
    """
    events = []
    
    for sid in df["scenario_id"].unique():
        sdf = df[df["scenario_id"] == sid].copy()
        
        is_leak = sdf["leak"] > 0
        leak_starts = sdf.index[is_leak & ~is_leak.shift(1).fillna(False)]
        leak_ends = sdf.index[~is_leak & is_leak.shift(1).fillna(False)]
        
        # If leak goes to the end of the scenario
        if len(leak_starts) > len(leak_ends):
            leak_ends = leak_ends.append(pd.Index([sdf.index[-1]]))
            
        for start, end in zip(leak_starts, leak_ends):
            max_rate = sdf.loc[start:end, "leak_rate"].max()
            events.append({
                "scenario_id": sid,
                "start_time": sdf.loc[start, "Timestamp"],
                "end_time": sdf.loc[end, "Timestamp"],
                "duration_hours": (sdf.loc[end, "Timestamp"] - sdf.loc[start, "Timestamp"]).total_seconds() / 3600.0,
                "max_leak_rate_CMH": max_rate
            })
            
        if len(leak_starts) == 0:
            events.append({
                "scenario_id": sid,
                "start_time": None,
                "end_time": None,
                "duration_hours": 0.0,
                "max_leak_rate_CMH": 0.0
            })
            
    summary_df = pd.DataFrame(events)
    out_path = os.path.join(os.path.dirname(__file__), "leakdb_event_summary.csv")
    summary_df.to_csv(out_path, index=False)
    
    print("\n--- LeakDB Event Summary ---")
    print(f"Total Scenarios: {len(df['scenario_id'].unique())}")
    print(f"Normal Scenarios: {len(summary_df[summary_df['start_time'].isnull()])}")
    print(f"Leak Scenarios: {len(summary_df[summary_df['start_time'].notnull()])}")
    print(f"Total Leak Events: {len(summary_df[summary_df['start_time'].notnull()])}")
    
    return summary_df

def load_all_scenarios(base_dir):
    """
    Loads and compiles the entire LeakDB network benchmark.
    """
    all_dfs = []
    scenarios = [d for d in os.listdir(base_dir) if d.startswith("Scenario-")]
    
    topology = None
    for s in scenarios:
        s_dir = os.path.join(base_dir, s)
        df = load_scenario(s_dir, s)
        
        # Parse topology from the first .inp file found
        if topology is None:
            inp_files = glob.glob(os.path.join(s_dir, "*.inp"))
            if inp_files:
                topology = parse_epanet_topology(inp_files[0])
                
        if topology:
            df = build_network_balances(df, topology)
            
        all_dfs.append(df)
        
    master_df = pd.concat(all_dfs, ignore_index=True)
    generate_event_summary(master_df)
    
    return master_df

if __name__ == "__main__":
    # Test loader
    base_dir = os.path.join(os.path.dirname(__file__), "LeakDB", "CCWI-WDSA2018", "Benchmarks", "Hanoi_CMH")
    if not os.path.exists(base_dir):
        print(f"Error: Dataset not found at {base_dir}")
        sys.exit(1)
        
    df = load_all_scenarios(base_dir)
    print(f"\nMaster DataFrame Shape: {df.shape}")
