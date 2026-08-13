import numpy as np
import pandas as pd
from src.data_loader import load_fbg_dataset, get_scaled_train_test_split
from src.models.mlp import MLPModel
from src.models.pinn import PINNModel

def run_noise_robustness_experiment(noise_levels=[0.01, 0.03, 0.05, 0.10]):
    """Evaluates MLP vs PINN under varying Gaussian noise levels."""
    print("Executing Noise Robustness Experiment...")
    X, y = load_fbg_dataset("both")
    X_train_s, X_test_s, y_train, y_test, _ = get_scaled_train_test_split(X, y, scale=True)
    
    mlp = MLPModel()
    mlp.fit(X_train_s, y_train)
    
    pinn = PINNModel()
    pinn.fit(X_train_s, y_train)
    
    results = []
    for noise in noise_levels:
        noise_matrix = np.random.normal(0, noise, X_test_s.shape)
        X_test_noisy = X_test_s + noise_matrix
        
        mlp_metrics = mlp.evaluate(X_test_noisy, y_test)
        pinn_metrics = pinn.evaluate(X_test_noisy, y_test)
        
        results.append({
            "Noise_Level": f"{int(noise*100)}%",
            "MLP_RMSE": mlp_metrics["RMSE"],
            "PINN_RMSE": pinn_metrics["RMSE"],
            "MLP_R2": mlp_metrics["R2"],
            "PINN_R2": pinn_metrics["R2"]
        })
        
    df_noise = pd.DataFrame(results)
    print("\n=== NOISE ROBUSTNESS COMPARISON ===")
    print(df_noise.to_string(index=False))
    return df_noise

if __name__ == "__main__":
    run_noise_robustness_experiment()
