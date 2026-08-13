import numpy as np
import pandas as pd
from src.data_loader import load_fbg_dataset, get_scaled_train_test_split
from src.models.mlp import MLPModel
from src.models.pinn import PINNModel

def run_small_data_experiment(data_fractions=[0.2, 0.4, 0.6, 0.8, 1.0]):
    """Evaluates MLP vs PINN under limited training dataset fractions."""
    print("Executing Small Data Regime Experiment...")
    X, y = load_fbg_dataset("both")
    X_train_s, X_test_s, y_train, y_test, _ = get_scaled_train_test_split(X, y, scale=True)
    
    results = []
    for frac in data_fractions:
        n_samples = int(len(X_train_s) * frac)
        X_sub = X_train_s[:n_samples]
        y_sub = y_train[:n_samples]
        
        mlp = MLPModel()
        mlp.fit(X_sub, y_sub)
        mlp_m = mlp.evaluate(X_test_s, y_test)
        
        pinn = PINNModel()
        pinn.fit(X_sub, y_sub)
        pinn_m = pinn.evaluate(X_test_s, y_test)
        
        results.append({
            "Data_Fraction": f"{int(frac*100)}%",
            "Samples": n_samples,
            "MLP_R2": mlp_m["R2"],
            "PINN_R2": pinn_m["R2"],
            "MLP_RMSE": mlp_m["RMSE"],
            "PINN_RMSE": pinn_m["RMSE"]
        })
        
    df_small = pd.DataFrame(results)
    print("\n=== SMALL DATA REGIME COMPARISON ===")
    print(df_small.to_string(index=False))
    return df_small

if __name__ == "__main__":
    run_small_data_experiment()
