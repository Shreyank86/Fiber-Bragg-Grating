import os
import pandas as pd
from src.data_loader import load_fbg_dataset, get_scaled_train_test_split
from src.models.pinn import PINNModel

TABLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

def run_hyperparameter_sensitivity_study():
    """Evaluates PINN sensitivity over Learning Rate, Batch Size, and Physics Weight Lambda."""
    print("Executing Hyperparameter Sensitivity Study for PINN...")
    X, y = load_fbg_dataset("both")
    X_tr_s, X_te_s, y_tr, y_te, _ = get_scaled_train_test_split(X, y, scale=True)
    
    results = []
    
    # 1. Learning Rate Variation
    lrs = [0.001, 0.005, 0.01]
    for lr in lrs:
        pinn = PINNModel(lr=lr, epochs=100, batch_size=128)
        pinn.fit(X_tr_s, y_tr)
        m = pinn.evaluate(X_te_s, y_te)
        results.append({
            "Parameter_Type": "Learning_Rate",
            "Parameter_Value": str(lr),
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "R2": m["R2"]
        })
        
    # 2. Batch Size Variation
    batch_sizes = [32, 64, 128]
    for bs in batch_sizes:
        pinn = PINNModel(batch_size=bs, epochs=100, lr=0.005)
        pinn.fit(X_tr_s, y_tr)
        m = pinn.evaluate(X_te_s, y_te)
        results.append({
            "Parameter_Type": "Batch_Size",
            "Parameter_Value": str(bs),
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "R2": m["R2"]
        })

    # 3. Physics Loss Weight Variation
    lambdas = [0.01, 0.1, 1.0, 10.0]
    for l_val in lambdas:
        pinn = PINNModel(lambda_phys=l_val, epochs=100, batch_size=128)
        pinn.fit(X_tr_s, y_tr)
        m = pinn.evaluate(X_te_s, y_te)
        results.append({
            "Parameter_Type": "Physics_Lambda",
            "Parameter_Value": str(l_val),
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "R2": m["R2"]
        })

    df_sens = pd.DataFrame(results)
    out_path = os.path.join(TABLES_DIR, "hyperparameter_sensitivity.csv")
    df_sens.to_csv(out_path, index=False)
    print("\n=== HYPERPARAMETER SENSITIVITY RESULTS ===")
    print(df_sens.to_string(index=False))
    return df_sens

if __name__ == "__main__":
    run_hyperparameter_sensitivity_study()
