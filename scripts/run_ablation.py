"""
Phase 4 Script: Physics Loss Weight (λ) Ablation Study.
Compares standard NN (No Physics Loss, λ=0) vs Physics-Informed NN (λ=0.1, 1.0, 10.0).
"""

import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RESULTS_JSON_PATH, RANDOM_SEED
from src.data import get_train_test_data
from src.pinn import FBG_PINN_Trainer
from src.utils import set_seed

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("==================================================================")
    print("PHASE 4: PHYSICS LOSS WEIGHT (lambda) ABLATION STUDY")
    print("==================================================================")
    
    set_seed(RANDOM_SEED)
    data_dict = get_train_test_data()
    X_tr = data_dict["X_train_pinn"]
    X_te = data_dict["X_test_pinn"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]

    lambdas = [0.0, 0.1, 1.0, 10.0]
    ablation_results = {}

    for lam in lambdas:
        label = "NN (No Physics)" if lam == 0.0 else f"PINN (λ={lam})"
        print(f"\nTraining {label}...")
        trainer = FBG_PINN_Trainer(physics_weight=lam, seed=RANDOM_SEED)
        train_time = trainer.train(X_tr, y_train, epochs=400, verbose=False)
        metrics, _, _, _ = trainer.evaluate(X_te, y_test)
        metrics["Train_Time_Sec"] = float(train_time)
        metrics["Loss_History"] = trainer.loss_history
        
        ablation_results[f"lambda_{lam}"] = {
            "label": label,
            "lambda": lam,
            "metrics": {k: v for k, v in metrics.items() if k != "Loss_History"}
        }
        print(f"  -> {label:20s}: MAE={metrics['MAE']:.4f} pm | RMSE={metrics['RMSE']:.4f} pm | R2={metrics['R2']:.4f}")

    # Save results
    output_data = {}
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            output_data = json.load(open(RESULTS_JSON_PATH, "r"))
        except Exception:
            output_data = {}
            
    output_data["Phase4_Ablation"] = ablation_results
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nSaved ablation results to: {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    main()
