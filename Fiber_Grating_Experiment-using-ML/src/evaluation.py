import time
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred, train_time=0.0, infer_time=0.0):
    """
    Computes standard evaluation metrics across target dimensions.
    Returns dictionary with MAE, MSE, RMSE, R2, training_time, inference_time.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2 * 100.0,  # Percentage
        "Train_Time_Sec": train_time,
        "Infer_Time_Ms": infer_time
    }

def print_metrics_summary(model_name, metrics):
    """Prints clean formatted metrics summary."""
    print(f"=== {model_name} Performance Metrics ===")
    print(f"  MAE           : {metrics['MAE']:.4f}")
    print(f"  RMSE          : {metrics['RMSE']:.4f}")
    print(f"  R² Score      : {metrics['R2']:.3f}%")
    print(f"  Training Time : {metrics['Train_Time_Sec']:.4f} s")
    print(f"  Inference Time: {metrics['Infer_Time_Ms']:.2f} ms")
    print("=" * 40)
