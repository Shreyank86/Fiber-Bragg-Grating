import pandas as pd
from src.data_loader import load_fbg_dataset, get_scaled_train_test_split
from src.models.pinn import PINNModel

def run_ablation_study(lambdas=[0.0, 0.01, 0.1, 1.0, 10.0]):
    """Evaluates impact of physics loss weight lambda in PINN."""
    print("Executing Physics Loss Weight Lambda Ablation Study...")
    X, y = load_fbg_dataset("both")
    X_train_s, X_test_s, y_train, y_test, _ = get_scaled_train_test_split(X, y, scale=True)
    
    results = []
    for l_val in lambdas:
        pinn = PINNModel(lambda_phys=l_val, epochs=100, batch_size=128)
        pinn.fit(X_train_s, y_train)
        m = pinn.evaluate(X_test_s, y_test)
        
        results.append({
            "Lambda_Phys": l_val,
            "MAE": m["MAE"],
            "RMSE": m["RMSE"],
            "R2": m["R2"]
        })
        
    df_ablation = pd.DataFrame(results)
    print("\n=== PHYSICS LOSS WEIGHT ABLATION STUDY ===")
    print(df_ablation.to_string(index=False))
    return df_ablation

if __name__ == "__main__":
    run_ablation_study()
