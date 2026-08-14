"""
Phase 8 Script: 5-Fold Cross-Validation for PINN evaluating Mean, Std, and 95% Confidence Intervals.
"""

import os
import json
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RESULTS_JSON_PATH, RANDOM_SEED
from src.data import get_train_test_data
from src.evaluate import run_cross_validation

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("==================================================================")
    print("PHASE 8: 5-FOLD CROSS-VALIDATION ANALYSIS")
    print("==================================================================")
    
    data_dict = get_train_test_data()
    X_sc = np.vstack([data_dict["X_train_pinn"], data_dict["X_test_pinn"]])
    y = np.vstack([data_dict["y_train"], data_dict["y_test"]])

    cv_results = run_cross_validation(X_sc, y, n_splits=5, epochs=300, seed=RANDOM_SEED)

    s = cv_results["summary"]
    print("\nCross-Validation Summary:")
    print(f"  MAE:  {s['MAE']['mean']:.4f} ± {s['MAE']['std']:.4f} (95% CI: [{s['MAE']['ci95'][0]:.4f}, {s['MAE']['ci95'][1]:.4f}])")
    print(f"  RMSE: {s['RMSE']['mean']:.4f} ± {s['RMSE']['std']:.4f} (95% CI: [{s['RMSE']['ci95'][0]:.4f}, {s['RMSE']['ci95'][1]:.4f}])")
    print(f"  R²:   {s['R2']['mean']:.4f} ± {s['R2']['std']:.4f} (95% CI: [{s['R2']['ci95'][0]:.4f}, {s['R2']['ci95'][1]:.4f}])")

    # Save to JSON
    output_data = {}
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            output_data = json.load(open(RESULTS_JSON_PATH, "r"))
        except Exception:
            output_data = {}
            
    output_data["Phase8_CrossValidation"] = cv_results
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nSaved cross-validation results to: {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    main()
