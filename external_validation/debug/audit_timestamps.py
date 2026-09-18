import os
import glob
import pandas as pd
import numpy as np

def audit_timestamps(scenario_dir, scenario_id):
    print(f"\n--- Timestamp Audit for {scenario_id} ---")
    
    labels_path = os.path.join(scenario_dir, "Labels.csv")
    df_labels = pd.read_csv(labels_path, parse_dates=["Timestamp"]).set_index("Timestamp")
    
    leaks_dir = os.path.join(scenario_dir, "Leaks")
    
    if not os.path.exists(leaks_dir):
        print("No leaks directory. Normal scenario.")
        return
        
    leak_files = glob.glob(os.path.join(leaks_dir, "*_demand.csv"))
    
    if not leak_files:
        print("No leak demand files.")
        return
        
    # Read the first leak demand file
    df_leak = pd.read_csv(leak_files[0], parse_dates=["Timestamp"]).set_index("Timestamp")
    # LeakDB sometimes uses "Description" as the column name
    col_name = df_leak.columns[0]
    
    # Identify leak start and end from Labels
    is_leak = df_labels["Label"] > 0
    leak_starts = df_labels.index[is_leak & ~is_leak.shift(1).fillna(False)]
    leak_ends = df_labels.index[~is_leak & is_leak.shift(1).fillna(False)]
    
    if len(leak_starts) == 0:
        print("No leak in labels.")
        return
        
    for i, (start) in enumerate(leak_starts):
        end = leak_ends[i] if i < len(leak_ends) else df_labels.index[-1]
        
        # Check alignment +/- 2 time steps
        print(f"\nLeak Event {i+1}")
        print("--- Start Alignment ---")
        window_start = start - pd.Timedelta(hours=1)
        window_end = start + pd.Timedelta(hours=1)
        
        merged = df_labels.loc[window_start:window_end].copy()
        merged["Leak_Rate"] = df_leak.loc[window_start:window_end, col_name]
        print(merged.to_string())
        
        print("\n--- End Alignment ---")
        window_start = end - pd.Timedelta(hours=1)
        window_end = end + pd.Timedelta(hours=1)
        
        merged = df_labels.loc[window_start:window_end].copy()
        merged["Leak_Rate"] = df_leak.loc[window_start:window_end, col_name]
        print(merged.to_string())

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "LeakDB", "CCWI-WDSA2018", "Benchmarks", "Hanoi_CMH")
    
    # Audit test scenarios (3 and 6)
    audit_timestamps(os.path.join(base_dir, "Scenario-3"), "Scenario-3")
    audit_timestamps(os.path.join(base_dir, "Scenario-6"), "Scenario-6")
