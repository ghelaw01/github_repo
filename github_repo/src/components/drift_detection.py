# src/components/drift_detection.py

import numpy as np
from scipy.stats import ks_2samp

def detect_drift(reference_data, new_data, feature_index, threshold=0.05):
    """
    Detects drift in a specific feature using the Kolmogorov-Smirnov test.

    Args:
        reference_data (np.ndarray): The reference dataset.
        new_data (np.ndarray): The new dataset to check for drift.
        feature_index (int): The index of the feature to check.
        threshold (float): The p-value threshold for drift detection.

    Returns:
        dict: A dictionary containing drift status and the p-value.
    """
    ks_statistic, p_value = ks_2samp(reference_data[:, feature_index], new_data[:, feature_index])
    
    drift_detected = p_value < threshold
    
    return {
        "drift_detected": drift_detected,
        "p_value": p_value,
        "ks_statistic": ks_statistic
    }
