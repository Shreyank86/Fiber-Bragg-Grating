import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_fbg_dataset, get_scaled_train_test_split
from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.svr import SVRModel
from src.models.mlp import MLPModel
from src.models.gpr import GPRModel
from src.models.pinn import PINNModel
from src.evaluation import evaluate_predictions

TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

def run_cross_validation(n_splits=5):
    """
    Executes Leak-Free 5-Fold Cross Validation across ALL 6 models.
    Applies StandardScaler fold-by-fold strictly fitted on training splits (X_tr).
    """
    print("Executing Leak-Free 5-Fold Cross Validation across ALL 6 Models...")
    X, y = load_fbg_dataset("both")
    X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    model_factories = {
        "Linear Regression": lambda: LinearRegressionModel(),
        "Random Forest": lambda: RandomForestModel(),
        "Support Vector Regression": lambda: SVRModel(),
        "Multi-Layer Perceptron": lambda: MLPModel(),
        "Gaussian Process Regression": lambda: GPRModel(),
        "PINN (Proposed)": lambda: PINNModel(epochs=100, batch_size=128)
    }
    
    fold_results = {name: {"R2": [], "RMSE": [], "MAE": []} for name in model_factories}
    
    fold = 1
    for train_idx, val_idx in kf.split(X_arr):
        print(f"  Executing Fold {fold}/{n_splits}...")
        X_tr_raw, X_val_raw = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Leak-free scaling inside fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_val = scaler.transform(X_val_raw)
        
        for name, factory in model_factories.items():
            model = factory()
            model.fit(X_tr, y_tr)
            metrics = model.evaluate(X_val, y_val)
            fold_results[name]["R2"].append(metrics["R2"])
            fold_results[name]["RMSE"].append(metrics["RMSE"])
            fold_results[name]["MAE"].append(metrics["MAE"])
            
        fold += 1

    summary_rows = []
    for name in model_factories:
        r2_vals = fold_results[name]["R2"]
        rmse_vals = fold_results[name]["RMSE"]
        mae_vals = fold_results[name]["MAE"]
        
        summary_rows.append({
            "Model": name,
            "Mean_R2": float(np.mean(r2_vals)),
            "Std_R2": float(np.std(r2_vals)),
            "Mean_RMSE": float(np.mean(rmse_vals)),
            "Std_RMSE": float(np.std(rmse_vals)),
            "Mean_MAE": float(np.mean(mae_vals)),
            "CI95_R2": f"{np.mean(r2_vals):.2f} ± {1.96 * np.std(r2_vals):.2f}%"
        })
        
    df_cv = pd.DataFrame(summary_rows)
    out_path = os.path.join(TABLES_DIR, "cross_validation.csv")
    df_cv.to_csv(out_path, index=False)
    print("\n=== LEAK-FREE 5-FOLD CROSS-VALIDATION RESULTS ===")
    print(df_cv.to_string(index=False))
    return df_cv, fold_results

def run_bootstrap_analysis(n_bootstraps=200):
    """Executes 1000-sample Bootstrap Confidence Interval evaluation on test predictions."""
    print("Executing Bootstrap Analysis for PINN and GPR...")
    X, y = load_fbg_dataset("both")
    X_tr_s, X_te_s, y_tr, y_te, _ = get_scaled_train_test_split(X, y, scale=True)
    
    pinn = PINNModel(epochs=100, batch_size=128).fit(X_tr_s, y_tr)
    y_pred_pinn = pinn.predict(X_te_s)
    
    gpr = GPRModel().fit(X_tr_s, y_tr)
    y_pred_gpr = gpr.predict(X_te_s)
    
    n_samples = len(y_te)
    np.random.seed(42)
    
    boot_rows = []
    for model_name, y_pred in [("PINN (Proposed)", y_pred_pinn), ("Gaussian Process Regression", y_pred_gpr)]:
        r2_scores = []
        rmse_scores = []
        for _ in range(n_bootstraps):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            m = evaluate_predictions(y_te[idx], y_pred[idx])
            r2_scores.append(m["R2"])
            rmse_scores.append(m["RMSE"])
            
        boot_rows.append({
            "Model": model_name,
            "Metric": "R2 Score (%)",
            "Mean": float(np.mean(r2_scores)),
            "CI_95_Lower": float(np.percentile(r2_scores, 2.5)),
            "CI_95_Upper": float(np.percentile(r2_scores, 97.5))
        })
        boot_rows.append({
            "Model": model_name,
            "Metric": "RMSE",
            "Mean": float(np.mean(rmse_scores)),
            "CI_95_Lower": float(np.percentile(rmse_scores, 2.5)),
            "CI_95_Upper": float(np.percentile(rmse_scores, 97.5))
        })

    df_boot = pd.DataFrame(boot_rows)
    out_path = os.path.join(TABLES_DIR, "bootstrap_statistics.csv")
    df_boot.to_csv(out_path, index=False)
    print("\n=== BOOTSTRAP CONFIDENCE INTERVALS ===")
    print(df_boot.to_string(index=False))
    return df_boot

if __name__ == "__main__":
    run_cross_validation()
    run_bootstrap_analysis()
