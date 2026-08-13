import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.data_loader import load_fbg_dataset, get_scaled_train_test_split
from src.models.gpr import GPRModel
from src.models.pinn import PINNModel
from src.evaluation import evaluate_predictions

TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

def run_cross_validation(n_splits=5):
    """Executes 5-Fold Cross Validation for GPR and PINN models."""
    print("Executing 5-Fold Cross Validation...")
    X, y = load_fbg_dataset("both")
    X_arr = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    gpr_r2 = []
    gpr_rmse = []
    pinn_r2 = []
    pinn_rmse = []
    
    fold = 1
    for train_idx, val_idx in kf.split(X_arr):
        print(f"  Fold {fold}/{n_splits}...")
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # GPR
        gpr = GPRModel().fit(X_tr, y_tr)
        m_gpr = gpr.evaluate(X_val, y_val)
        gpr_r2.append(m_gpr["R2"])
        gpr_rmse.append(m_gpr["RMSE"])
        
        # PINN
        pinn = PINNModel(epochs=100, batch_size=128).fit(X_tr, y_tr)
        m_pinn = pinn.evaluate(X_val, y_val)
        pinn_r2.append(m_pinn["R2"])
        pinn_rmse.append(m_pinn["RMSE"])
        
        fold += 1

    df_cv = pd.DataFrame([
        {
            "Model": "GPR (Member 3)",
            "Mean_R2": float(np.mean(gpr_r2)),
            "Std_R2": float(np.std(gpr_r2)),
            "Mean_RMSE": float(np.mean(gpr_rmse)),
            "Std_RMSE": float(np.std(gpr_rmse)),
            "CI95_R2": f"{np.mean(gpr_r2):.2f} ± {1.96 * np.std(gpr_r2):.2f}%"
        },
        {
            "Model": "PINN (Proposed)",
            "Mean_R2": float(np.mean(pinn_r2)),
            "Std_R2": float(np.std(pinn_r2)),
            "Mean_RMSE": float(np.mean(pinn_rmse)),
            "Std_RMSE": float(np.std(pinn_rmse)),
            "CI95_R2": f"{np.mean(pinn_r2):.2f} ± {1.96 * np.std(pinn_r2):.2f}%"
        }
    ])
    
    out_path = os.path.join(TABLES_DIR, "cross_validation.csv")
    df_cv.to_csv(out_path, index=False)
    print("\n=== CROSS-VALIDATION RESULTS ===")
    print(df_cv.to_string(index=False))
    return df_cv

def run_bootstrap_analysis(n_bootstraps=200):
    """Executes 1000-sample Bootstrap Confidence Interval evaluation on test predictions."""
    print("Executing Bootstrap Analysis...")
    X, y = load_fbg_dataset("both")
    X_tr_s, X_te_s, y_tr, y_te, _ = get_scaled_train_test_split(X, y, scale=True)
    
    pinn = PINNModel(epochs=100, batch_size=128).fit(X_tr_s, y_tr)
    y_pred_pinn = pinn.predict(X_te_s)
    
    n_samples = len(y_te)
    r2_scores = []
    rmse_scores = []
    
    np.random.seed(42)
    for _ in range(n_bootstraps):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        m = evaluate_predictions(y_te[idx], y_pred_pinn[idx])
        r2_scores.append(m["R2"])
        rmse_scores.append(m["RMSE"])
        
    df_boot = pd.DataFrame([{
        "Model": "PINN (Proposed)",
        "Metric": "R2 Score (%)",
        "Mean": float(np.mean(r2_scores)),
        "CI_95_Lower": float(np.percentile(r2_scores, 2.5)),
        "CI_95_Upper": float(np.percentile(r2_scores, 97.5))
    }, {
        "Model": "PINN (Proposed)",
        "Metric": "RMSE",
        "Mean": float(np.mean(rmse_scores)),
        "CI_95_Lower": float(np.percentile(rmse_scores, 2.5)),
        "CI_95_Upper": float(np.percentile(rmse_scores, 97.5))
    }])
    
    out_path = os.path.join(TABLES_DIR, "bootstrap_statistics.csv")
    df_boot.to_csv(out_path, index=False)
    print("\n=== BOOTSTRAP CONFIDENCE INTERVALS ===")
    print(df_boot.to_string(index=False))
    return df_boot

if __name__ == "__main__":
    run_cross_validation()
    run_bootstrap_analysis()
