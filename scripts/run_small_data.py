"""
Phase 6 Script: Small Data Analysis across 20%, 40%, 60%, 80%, 100% Training Data Subsamples for ALL models.
Fast, clean evaluation across Linear Regression, Random Forest, SVR, MLP, Gaussian Process, and Proposed PINN.
"""

import os
import json
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RESULTS_JSON_PATH, RANDOM_SEED
from src.data import get_train_test_data, get_subsampled_data
from src.models import ClassicalMLBaselines
from src.pinn import FBG_PINN_Trainer
from src.utils import set_seed, compute_metrics

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("==================================================================")
    print("PHASE 6: SMALL DATA ANALYSIS ACROSS ALL MODELS (20%-100% SUBSAMPLES)")
    print("==================================================================")
    
    set_seed(RANDOM_SEED)
    data_dict = get_train_test_data()
    X_tr_ml = data_dict["X_train_ml"]
    X_te_ml = data_dict["X_test_ml"]
    X_tr_pinn = data_dict["X_train_pinn"]
    X_te_pinn = data_dict["X_test_pinn"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]

    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    model_names = ["Linear Regression", "Random Forest", "SVR", "MLP", "Gaussian Process", "Proposed PINN"]

    small_data_results = {}

    for frac in fractions:
        frac_key = f"{int(frac*100)}%"
        print(f"\nEvaluating Subsample Fraction: {frac_key}...")

        # Subsample data deterministically
        X_tr_ml_sub, y_tr_sub = get_subsampled_data(X_tr_ml, y_train, fraction=frac, seed=RANDOM_SEED)
        X_tr_pinn_sub, _ = get_subsampled_data(X_tr_pinn, y_train, fraction=frac, seed=RANDOM_SEED)

        frac_summary = {}

        # Fit classical models
        baselines = ClassicalMLBaselines(seed=RANDOM_SEED)
        c_res = baselines.fit_and_evaluate(X_tr_ml_sub, y_tr_sub, X_te_ml, y_test)
        for m_name, r in c_res.items():
            frac_summary[m_name] = r["metrics"]

        # Fit PINN
        pinn_trainer = FBG_PINN_Trainer(physics_weight=1.0, seed=RANDOM_SEED)
        pinn_trainer.train(X_tr_pinn_sub, y_tr_sub, epochs=250, verbose=False)
        pinn_m, _, _, _ = pinn_trainer.evaluate(X_te_pinn, y_test)
        frac_summary["Proposed PINN"] = pinn_m

        small_data_results[frac_key] = frac_summary
        print(f"  -> Proposed PINN: MAE={pinn_m['MAE']:.4f} pm | RMSE={pinn_m['RMSE']:.4f} pm | R2={pinn_m['R2']:.4f}")

    # Save to JSON
    output_data = {}
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            output_data = json.load(open(RESULTS_JSON_PATH, "r"))
        except Exception:
            output_data = {}
            
    output_data["Phase6_SmallData"] = small_data_results
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nSaved multi-model small data results to: {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    main()
