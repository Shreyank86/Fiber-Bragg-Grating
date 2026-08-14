"""
Phase 5 Script: Noise Robustness Analysis under 0%, 1%, 3%, 5%, 10% Gaussian Noise Injection across ALL models.
Evaluates Linear Regression, Random Forest, SVR, MLP, Gaussian Process, and Proposed PINN.
"""

import os
import json
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RESULTS_JSON_PATH, RANDOM_SEED
from src.data import get_train_test_data, add_gaussian_noise
from src.models import ClassicalMLBaselines
from src.pinn import FBG_PINN_Trainer
from src.utils import set_seed, compute_metrics

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("==================================================================")
    print("PHASE 5: NOISE ROBUSTNESS EVALUATION ACROSS ALL MODELS (0%-10% GAUSSIAN NOISE)")
    print("==================================================================")
    
    set_seed(RANDOM_SEED)
    data_dict = get_train_test_data()
    X_tr_ml = data_dict["X_train_ml"]
    X_te_ml = data_dict["X_test_ml"]
    X_tr_pinn = data_dict["X_train_pinn"]
    X_te_pinn = data_dict["X_test_pinn"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]

    noise_levels = [0.0, 1.0, 3.0, 5.0, 10.0]
    
    # Fit base classical models once on clean training data
    print("\nFitting base classical ML models...")
    baselines = ClassicalMLBaselines(seed=RANDOM_SEED)
    baselines.fit_and_evaluate(X_tr_ml, y_train, X_te_ml, y_test)
    
    # Fit PINN on clean training data
    print("Fitting PINN model...")
    pinn_trainer = FBG_PINN_Trainer(physics_weight=1.0, seed=RANDOM_SEED)
    pinn_trainer.train(X_tr_pinn, y_train, epochs=400, verbose=False)

    noise_results = {}

    for noise in noise_levels:
        lvl_key = f"{noise}%"
        print(f"\nEvaluating Noise Level: {lvl_key}...")
        y_test_noisy = add_gaussian_noise(y_test, noise_percentage=noise, seed=RANDOM_SEED)
        y_noisy_flat = y_test_noisy.flatten()

        lvl_metrics = {}

        # Evaluate Classical Models under noise
        for name, model in baselines.fitted_models.items():
            y_pred = model.predict(X_te_ml)
            m = compute_metrics(y_noisy_flat, y_pred)
            lvl_metrics[name] = m

        # Evaluate PINN under noise
        _, _, delta_pred, _ = pinn_trainer.predict(X_te_pinn)
        pinn_m = compute_metrics(y_noisy_flat, delta_pred)
        lvl_metrics["Proposed PINN"] = pinn_m

        noise_results[lvl_key] = lvl_metrics
        print(f"  -> Proposed PINN: MAE={pinn_m['MAE']:.4f} pm | RMSE={pinn_m['RMSE']:.4f} pm | R2={pinn_m['R2']:.4f}")

    # Save to JSON
    output_data = {}
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            output_data = json.load(open(RESULTS_JSON_PATH, "r"))
        except Exception:
            output_data = {}
            
    output_data["Phase5_NoiseRobustness"] = noise_results
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nSaved multi-model noise robustness results to: {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    main()
