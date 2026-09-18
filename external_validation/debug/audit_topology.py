import os
import pandas as pd
import numpy as np

def audit_topology():
    print("--- Auditing Topology for Hanoi_CMH ---")
    
    # We load the merged df from results
    train_df = pd.read_pickle(os.path.join(os.path.dirname(__file__), "..", "results", "train_df.pkl"))
    
    # Let's pick Node 10
    # Inflow: Link 9
    # Outflow: Link 10, Link 13
    
    print("\nNode 10 Verification:")
    sample = train_df.sample(5, random_state=42)
    
    for idx, row in sample.iterrows():
        t = row["Timestamp"]
        f9 = row["flow_9"]
        f10 = row["flow_10"]
        f13 = row["flow_13"]
        d10 = row["demand_10"]
        b10_generated = row["balance_10"]
        
        b10_manual = f9 - f10 - f13 - d10
        
        print(f"[{t}] F9: {f9:.2f} | F10: {f10:.2f} | F13: {f13:.2f} | D10: {d10:.2f}")
        print(f"    Manual Balance: {b10_manual:.4f}")
        print(f"    Generated Bal:  {b10_generated:.4f}")
        if not np.isclose(b10_manual, b10_generated, atol=1e-3):
            print("    [!] MISMATCH!")
        else:
            print("    [OK] MATCH")
            
    print("\nNode 23 Verification:")
    # Inflow: Link 23
    # Outflow: Link 24, Link 29
    # (Based on EPANET INP file review)
    for idx, row in sample.iterrows():
        t = row["Timestamp"]
        f23 = row["flow_23"]
        f24 = row["flow_24"]
        f29 = row["flow_29"]
        d23 = row["demand_23"]
        b23_generated = row["balance_23"]
        
        b23_manual = f23 - f24 - f29 - d23
        
        print(f"[{t}] F23: {f23:.2f} | F24: {f24:.2f} | F29: {f29:.2f} | D23: {d23:.2f}")
        print(f"    Manual Balance: {b23_manual:.4f}")
        print(f"    Generated Bal:  {b23_generated:.4f}")
        if not np.isclose(b23_manual, b23_generated, atol=1e-3):
            print("    [!] MISMATCH!")
        else:
            print("    [OK] MATCH")

if __name__ == "__main__":
    audit_topology()
