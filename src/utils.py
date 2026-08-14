"""
Utility functions for random seeding, metric computation, latency benchmarking, and formatting.
"""

import random
import time
import numpy as np
import torch

def set_seed(seed=42):
    """Fix random seed across python, numpy, and torch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred):
    """
    Compute Mean Absolute Error (MAE), Root Mean Square Error (RMSE), 
    and Coefficient of Determination (R^2).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum(errors ** 2)
    
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def measure_execution_time(func, *args, **kwargs):
    """Measure function execution wall-clock time in milliseconds."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000.0
    return result, elapsed_ms
