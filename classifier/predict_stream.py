import os
import sys
import time
import pandas as pd
from datetime import datetime

# config.py is in the same directory (classifier/)
import config
from predict import WaterNetworkLeakDetector

def stream_simulation(start_idx: int = 80000, num_steps: int = 100, delay_sec: float = 0.5):
    """
    Simulates a live data stream by reading sequentially from the test portion of the dataset.
    Feeds each row into the prediction API in real-time.
    """
    print("Initializing Water Network ML Inference Engine...")
    detector = WaterNetworkLeakDetector()
    
    dataset_path = config.ML_TRAINING_CSV
    if not os.path.exists(dataset_path):
        print(f"Error: Could not find dataset at {dataset_path}")
        return
        
    print(f"Loading stream data from {dataset_path} (Starting at index {start_idx})...")
    # Load only the chunk we need for memory efficiency
    try:
        df_stream = pd.read_csv(dataset_path, skiprows=range(1, start_idx), nrows=num_steps)
    except Exception as e:
        print(f"Failed to load dataset slice: {e}")
        return
        
    print(f"--- STREAM SIMULATION STARTED ({num_steps} iterations) ---")
    
    for idx, row in df_stream.iterrows():
        # Convert row to dict to simulate JSON payload from SCADA system
        state = row.to_dict()
        
        # We simulate that the SCADA system provides actual features (including rolling stats).
        # In a purely raw stream, the state machine here would maintain a deque of length 12
        # and calculate rolling means in memory.
        
        timestamp_str = f"{state['date']} {state['time']}"
        
        t0 = time.time()
        result = detector.predict(state)
        latency_ms = (time.time() - t0) * 1000.0
        
        # Format output
        actual_leak = int(state.get('leak', 0))
        actual_zone = state.get('leak_zone', 'NONE')
        
        status_color = "\033[91m" if result['leak_detected'] else "\033[92m"
        reset_color = "\033[0m"
        
        print(f"[{timestamp_str}] Flow J1: {state['flow_J1']:7.2f} L/min | "
              f"Pred: {status_color}{result['leak_detected']}{reset_color} "
              f"(Prob: {result['leak_probability']:.2f}) | "
              f"Zone: {result['leak_zone']:<10} | "
              f"Rate: {result['leak_rate']:<5.1f} | "
              f"Actual: {actual_leak} ({actual_zone}) | "
              f"Latency: {latency_ms:.1f}ms")
              
        time.sleep(delay_sec)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run simulated SCADA data stream through ML model.")
    parser.add_argument("--start", type=int, default=100000, help="Row index to start streaming from (test set is ~73500 onwards)")
    parser.add_argument("--steps", type=int, default=100, help="Number of timestamps to simulate")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between rows in seconds")
    
    args = parser.parse_args()
    stream_simulation(start_idx=args.start, num_steps=args.steps, delay_sec=args.delay)
