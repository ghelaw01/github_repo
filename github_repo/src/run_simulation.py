# src/run_simulation.py

import yaml
import numpy as np
import pandas as pd
from components.drift_detection import detect_drift

def run_simulation(config):
    """
    Runs the main fairness drift simulation.

    Args:
        config (dict): The simulation configuration.
    """
    print("Starting simulation...")
    
    # Load reference data (simulated)
    reference_data = np.random.rand(config["simulation"]["n_samples"], len(config["features"]))
    
    # Simulate time steps
    for t in range(config["simulation"]["time_steps"]):
        print(f"Running time step {t+1}...")
        
        # Generate new data with potential drift
        new_data = reference_data.copy()
        if np.random.rand() < config["simulation"]["drift_probability"]:
            drift_feature_index = np.random.randint(0, len(config["features"]))
            new_data[:, drift_feature_index] += np.random.normal(0, config["simulation"]["drift_magnitude"], config["simulation"]["n_samples"])
            print(f"Drift introduced in feature: {config["features"][drift_feature_index]}")
        
        # Detect drift
        for i, feature in enumerate(config["features"]):
            drift_results = detect_drift(reference_data, new_data, i)
            if drift_results["drift_detected"]:
                print(f"Drift detected in {feature} with p-value: {drift_results["p_value"]:.4f}")

    print("Simulation finished.")

if __name__ == "__main__":
    with open("../configs/simulation_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_simulation(config)
