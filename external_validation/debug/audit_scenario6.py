import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR = os.path.join(os.path.dirname(__file__), "scenario6_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

def audit_scenario6():
    print("Auditing Scenario 6...")
    
    # Load test data and models
    test_df = pd.read_pickle(os.path.join(RESULTS_DIR, "test_df.pkl"))
    with open(os.path.join(MODELS_DIR, "leakdb_classifiers.pkl"), "rb") as f:
        models = pickle.load(f)
        
    s6_df = test_df[test_df["scenario_id"] == "Scenario-6"].copy()
    if s6_df.empty:
        print("Scenario 6 not found in test_df")
        return
        
    s6_df["Timestamp"] = pd.to_datetime(s6_df["Timestamp"])
    s6_df.set_index("Timestamp", inplace=True)
    
    # Generate predictions from D_ResidualOnly
    model_name = "D_ResidualOnly"
    data = models[model_name]
    model = data["model"]
    feats = data["features"]
    
    s6_df["prob"] = model.predict_proba(s6_df[feats])[:, 1]
    
    # Identify leak event
    is_leak = s6_df["leak"] > 0
    leak_starts = s6_df.index[is_leak & ~is_leak.shift(1).fillna(False)]
    leak_ends = s6_df.index[~is_leak & is_leak.shift(1).fillna(False)]
    
    if len(leak_starts) == 0:
        print("No leak event found in Scenario 6")
        return
        
    start_time = leak_starts[0]
    end_time = leak_ends[0] if len(leak_ends) > 0 else s6_df.index[-1]
    
    # Window to plot (7 days before, 7 days after)
    plot_start = start_time - pd.Timedelta(days=7)
    plot_end = end_time + pd.Timedelta(days=7)
    
    plot_df = s6_df.loc[plot_start:plot_end]
    
    print(f"Leak Start: {start_time}")
    print(f"Leak End: {end_time}")
    print(f"Max Leak Rate: {plot_df['leak_rate'].max()}")
    
    fig, axes = plt.subplots(6, 1, figsize=(15, 20), sharex=True)
    
    # Identify a node and link near the leak. 
    # Let's plot the total residual sum, and top residual feature from importance
    top_residual_col = "residual_flow_27"
    top_flow_col = "flow_27"
    top_exp_col = "expected_flow_27"
    top_press_col = "pressure_6"
    
    axes[0].plot(plot_df.index, plot_df[top_flow_col], label="Actual Flow 27", color='blue')
    axes[0].plot(plot_df.index, plot_df[top_exp_col], label="Expected Flow 27", color='orange', alpha=0.7)
    axes[0].set_ylabel("Flow (CMH)")
    axes[0].legend()
    axes[0].set_title(f"Flows for Link 27")
    
    axes[1].plot(plot_df.index, plot_df[top_residual_col], label="Residual Flow 27", color='red')
    axes[1].set_ylabel("Residual (CMH)")
    axes[1].legend()
    
    axes[2].plot(plot_df.index, plot_df[top_press_col], label="Pressure Node 6", color='green')
    axes[2].set_ylabel("Pressure (m)")
    axes[2].legend()
    
    axes[3].plot(plot_df.index, plot_df["leak_rate"], label="Actual Leak Rate", color='black')
    axes[3].set_ylabel("Leak Rate (CMH)")
    axes[3].legend()
    
    axes[4].plot(plot_df.index, plot_df["leak"], label="Leak Label (Ground Truth)", color='purple')
    axes[4].set_ylabel("Label (0/1)")
    axes[4].set_ylim([-0.1, 1.1])
    axes[4].legend()
    
    axes[5].plot(plot_df.index, plot_df["prob"], label=f"Predicted Probability ({model_name})", color='magenta')
    axes[5].axhline(y=0.5, color='r', linestyle='--', label="Threshold (0.5)")
    axes[5].set_ylabel("Probability")
    axes[5].set_ylim([-0.1, 1.1])
    axes[5].legend()
    
    for ax in axes:
        ax.axvline(x=start_time, color='r', linestyle='--', alpha=0.5, label='Leak Start' if ax==axes[0] else "")
        ax.axvline(x=end_time, color='b', linestyle='--', alpha=0.5, label='Leak End' if ax==axes[0] else "")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scenario6_audit.png"))
    plt.close()
    
    print(f"Generated Scenario 6 audit plots in {OUT_DIR}")
    
if __name__ == "__main__":
    audit_scenario6()
