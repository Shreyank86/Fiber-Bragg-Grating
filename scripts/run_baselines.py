"""
Phase 3 Script: Machine Learning Benchmarking across Linear Regression, Random Forest, SVR, MLP, Gaussian Process, and PINN.
Saves all trained models to saved_models/ and records JSON metrics in outputs/results.json.
"""

import os
import json
import sys
import numpy as np

# Add parent directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RESULTS_JSON_PATH, SAVED_MODELS_DIR, RANDOM_SEED
from src.data import get_train_test_data
from src.models import ClassicalMLBaselines
from src.pinn import FBG_PINN_Trainer
from src.utils import set_seed

def main():
    print("==================================================================")
    print("PHASE 3: MACHINE LEARNING BENCHMARKING (MODELS & METRICS)")
    print("==================================================================")
    
    set_seed(RANDOM_SEED)
    data_dict = get_train_test_data()
    
    X_tr_ml = data_dict["X_train_ml"]
    X_te_ml = data_dict["X_test_ml"]
    X_tr_pinn = data_dict["X_train_pinn"]
    X_te_pinn = data_dict["X_test_pinn"]
    y_train = data_dict["y_train"]
    y_test = data_dict["y_test"]
    
    # 1. Train Classical Baseline Models
    print("\n[1/2] Training Classical Baseline Models (LR, RF, SVR, MLP, GP)...")
    baselines = ClassicalMLBaselines(seed=RANDOM_SEED)
    baseline_results = baselines.fit_and_evaluate(X_tr_ml, y_train, X_te_ml, y_test)
    
    all_results = {}
    for name, res in baseline_results.items():
        all_results[name] = res["metrics"]
        print(f"  -> {name:20s}: MAE={res['metrics']['MAE']:.4f} pm | RMSE={res['metrics']['RMSE']:.4f} pm | R2={res['metrics']['R2']:.4f} | TrainTime={res['metrics']['Train_Time_Sec']:.3f}s")
        
    # 2. Train Proposed Physics-Informed Neural Network (PINN)
    print("\n[2/2] Training Proposed Measurement-Constrained PINN...")
    pinn_trainer = FBG_PINN_Trainer(physics_weight=1.0, seed=RANDOM_SEED)
    train_time_sec = pinn_trainer.train(X_tr_pinn, y_train, epochs=500, verbose=False)
    pinn_metrics, strain_pred, temp_pred, delta_pred = pinn_trainer.evaluate(X_te_pinn, y_test)
    pinn_metrics["Train_Time_Sec"] = float(train_time_sec)
    
    # Save PINN model
    saved_pinn_path = pinn_trainer.save(SAVED_MODELS_DIR)
    print(f"  -> Saved PINN Model to: {saved_pinn_path}")
    
    # Save PINN prediction outputs
    np.savez(
        os.path.join(SAVED_MODELS_DIR, "pinn_predictions.npz"),
        strain_pred=strain_pred,
        temp_pred=temp_pred,
        delta_pred=delta_pred,
        y_test=y_test
    )
    
    all_results["Proposed PINN"] = pinn_metrics
    print(f"  -> {'Proposed PINN':20s}: MAE={pinn_metrics['MAE']:.4f} pm | RMSE={pinn_metrics['RMSE']:.4f} pm | R2={pinn_metrics['R2']:.4f} | TrainTime={pinn_metrics['Train_Time_Sec']:.3f}s")
    
    # Print Final Comparison Table in Terminal
    print("\n==========================================================================================")
    print(f"{'Model':22s} | {'MAE (pm)':10s} | {'RMSE (pm)':10s} | {'R²':8s} | {'Train (s)':10s} | {'Infer (s)':10s}")
    print("==========================================================================================")
    for model_name, m in all_results.items():
        print(f"{model_name:22s} | {m['MAE']:10.4f} | {m['RMSE']:10.4f} | {m['R2']:8.4f} | {m['Train_Time_Sec']:10.4f} | {m['Infer_Time_Sec']:10.4f}")
    print("==========================================================================================")

    # Save metrics to results.json
    output_data = {}
    if os.path.exists(RESULTS_JSON_PATH):
        try:
            output_data = json.load(open(RESULTS_JSON_PATH, "r"))
        except Exception:
            output_data = {}
            
    output_data["Phase3_Baselines"] = all_results
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"\nSaved benchmark results to: {RESULTS_JSON_PATH}")

if __name__ == "__main__":
    main()
