import numpy as np
import pandas as pd
from src.data_loader import load_fbg_dataset, get_scaled_train_test_split
from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.svr import SVRModel
from src.models.mlp import MLPModel
from src.models.gpr import GPRModel
from src.models.pinn import PINNModel

def run_small_data_experiment(data_fractions=[0.2, 0.4, 0.6, 0.8, 1.0]):
    """Evaluates ALL 6 models under limited training dataset fractions."""
    print("Executing Comprehensive Small Data Regime Experiment across ALL 6 models...")
    X, y = load_fbg_dataset("both")
    X_train_s, X_test_s, y_train, y_test, _ = get_scaled_train_test_split(X, y, scale=True)
    
    results = []
    for frac in data_fractions:
        n_samples = int(len(X_train_s) * frac)
        X_sub = X_train_s[:n_samples]
        y_sub = y_train[:n_samples]
        print(f"  Training on {int(frac*100)}% data ({n_samples} samples)...")
        
        models = {
            "LR": LinearRegressionModel(),
            "RF": RandomForestModel(),
            "SVR": SVRModel(),
            "MLP": MLPModel(),
            "GPR": GPRModel(),
            "PINN": PINNModel(epochs=100, batch_size=128)
        }
        
        row = {
            "Data_Fraction": f"{int(frac*100)}%",
            "Samples": n_samples
        }
        
        for name, model in models.items():
            model.fit(X_sub, y_sub)
            m = model.evaluate(X_test_s, y_test)
            row[f"{name}_RMSE"] = m["RMSE"]
            row[f"{name}_R2"] = m["R2"]
            
        results.append(row)
        
    df_small = pd.DataFrame(results)
    print("\n=== COMPREHENSIVE SMALL DATA REGIME COMPARISON ===")
    print(df_small.to_string(index=False))
    return df_small

if __name__ == "__main__":
    run_small_data_experiment()
