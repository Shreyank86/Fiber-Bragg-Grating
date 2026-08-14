"""
Phase 9 Script: 1000-Sample Bootstrap Analysis for 95% Confidence Intervals.
"""

import os
import json
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RESULTS_JSON_PATH, SAVED_MODELS_DIR, RANDOM_SEED
from src.evaluate import run_bootstrap_analysis

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("==================================================================")
    print("PHASE 9: BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS (N=1000)")
    print("==================================================================")
    
    pred_path = os.path.join(SAVED_MODELS_DIR, "pinn_predictions.npz")
    if not os.path.exists(pred_path):
        print("Predictions file not found! Run scripts/run_baselines.py first.")
        return

    data = np.load(pred_path)
    y_test = data["y_test"]
    delta_pred = data["delta_pred"]

    bootstrap_results = run_bootstrap_analysis(y_test, delta_pred, n_bootstrap=1000, seed=RANDOM_SEED)

    print("\nBootstrap Analysis Summary (1000 Resamples):")
    print(f"  MAE Mean:  {bootstrap_results['MAE']['mean']:.4f} pm | 95% CI: [{bootstrap_results['MAE']['ci95'][0]:.4f}, {bootstrap_results['MAE']['ci95'][1]:.4f}]")
    print(f"  RMSE Mean: {bootstrap_results['RMSE']['mean']:.4f} pm | 95% CI: [{bootstrap_results['RMSE']['ci95'][0]:.4f}, {bootstrap_results['RMSE']['ci95'][1]:.4f}]")

    # Clean out raw samples array for concise JSON serialization
    json_save_data = {
        "MAE": bootstrap_results["MAE"],
        "RMSE": bootstrap_results["RMSE"]
    }

    # Save to JSON
    output_data = {}
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            output_data = json.load(open(RESULTS_JSON_PATH, "r"))
        except Exception:
            output_data = {}
            
    output_data["Phase9_Bootstrap"] = json_save_data
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nSaved bootstrap results to: {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    main()
