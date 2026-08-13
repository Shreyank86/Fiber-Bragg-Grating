import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_fbg_dataset, get_scaled_train_test_split
from src.evaluation import print_metrics_summary
from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.svr import SVRModel
from src.models.mlp import MLPModel
from src.models.gpr import GPRModel
from src.models.pinn import PINNModel

def run_benchmark():
    """Runs standard 6-model benchmarking suite."""
    print("Loading FBG Combined Temperature + Strain Dataset...")
    X, y = load_fbg_dataset("both")
    X_train_s, X_test_s, y_train, y_test, _ = get_scaled_train_test_split(X, y, scale=True)
    
    models = {
        "Linear Regression": LinearRegressionModel(),
        "Random Forest": RandomForestModel(),
        "SVR (Member 1)": SVRModel(),
        "MLP (Member 2)": MLPModel(),
        "GPR (Member 3)": GPRModel(),
        "PINN (Proposed)": PINNModel()
    }
    
    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_s, y_train)
        metrics = model.evaluate(X_test_s, y_test)
        print_metrics_summary(name, metrics)
        metrics["Model"] = name
        results.append(metrics)
        
    df_results = pd.DataFrame(results)[["Model", "MAE", "RMSE", "R2", "Train_Time_Sec", "Infer_Time_Ms"]]
    print("\n" + "=" * 60)
    print("FINAL MASTER BENCHMARK COMPARISON")
    print("=" * 60)
    print(df_results.to_string(index=False))
    return df_results

if __name__ == "__main__":
    run_benchmark()
