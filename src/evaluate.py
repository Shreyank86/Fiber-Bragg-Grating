"""
Evaluation module containing routines for Cross-Validation, Bootstrap Analysis, Noise Robustness, 
Small-Data Regime Analysis, and Hyperparameter Sensitivity Sweeps.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import RANDOM_SEED, K_EPSILON, K_TEMP
from .utils import compute_metrics, set_seed
from .pinn import FBG_PINN_Trainer
from .data import add_gaussian_noise, get_subsampled_data

def run_cross_validation(X, y, n_splits=5, epochs=300, seed=RANDOM_SEED):
    """
    Executes 5-Fold Cross Validation for PINN, returning fold metrics, means, stds, and 95% CIs.
    """
    set_seed(seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    mae_list, rmse_list, r2_list = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        trainer = FBG_PINN_Trainer(seed=seed + fold)
        trainer.train(X_tr, y_tr, epochs=epochs, verbose=False)
        metrics, _, _, _ = trainer.evaluate(X_te, y_te)

        mae_list.append(metrics["MAE"])
        rmse_list.append(metrics["RMSE"])
        r2_list.append(metrics["R2"])

    def calc_ci95(arr):
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        return mean, std, [ci_lower, ci_upper]

    mae_mean, mae_std, mae_ci = calc_ci95(mae_list)
    rmse_mean, rmse_std, rmse_ci = calc_ci95(rmse_list)
    r2_mean, r2_std, r2_ci = calc_ci95(r2_list)

    return {
        "folds": {
            "MAE": mae_list,
            "RMSE": rmse_list,
            "R2": r2_list
        },
        "summary": {
            "MAE": {"mean": mae_mean, "std": mae_std, "ci95": mae_ci},
            "RMSE": {"mean": rmse_mean, "std": rmse_std, "ci95": rmse_ci},
            "R2": {"mean": r2_mean, "std": r2_std, "ci95": r2_ci}
        }
    }

def run_bootstrap_analysis(y_true, y_pred, n_bootstrap=1000, seed=RANDOM_SEED):
    """
    Computes 1000 non-parametric bootstrap resamples to establish 95% Confidence Intervals for MAE and RMSE.
    """
    rng = np.random.default_rng(seed)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    n_samples = len(y_true)

    mae_samples, rmse_samples = [], []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]
        
        errors = sample_true - sample_pred
        mae_samples.append(float(np.mean(np.abs(errors))))
        rmse_samples.append(float(np.sqrt(np.mean(errors ** 2))))

    def ci95(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "MAE": {
            "mean": float(np.mean(mae_samples)),
            "std": float(np.std(mae_samples)),
            "ci95": ci95(mae_samples)
        },
        "RMSE": {
            "mean": float(np.mean(rmse_samples)),
            "std": float(np.std(rmse_samples)),
            "ci95": ci95(rmse_samples)
        },
        "mae_samples": mae_samples,
        "rmse_samples": rmse_samples
    }

def run_noise_robustness(X_train, y_train, X_test, y_test, noise_levels=[0.0, 1.0, 3.0, 5.0, 10.0], seed=RANDOM_SEED):
    """
    Evaluates impact of Gaussian measurement noise injection (0% to 10%) on model performance.
    """
    results = {}
    
    for noise in noise_levels:
        y_test_noisy = add_gaussian_noise(y_test, noise_percentage=noise, seed=seed)
        
        trainer = FBG_PINN_Trainer(seed=seed)
        trainer.train(X_train, y_train, epochs=300, verbose=False)
        metrics, _, _, _ = trainer.evaluate(X_test, y_test_noisy)
        
        results[f"{noise}%"] = metrics
        
    return results

def run_small_data_analysis(X_train, y_train, X_test, y_test, fractions=[0.2, 0.4, 0.6, 0.8, 1.0], seed=RANDOM_SEED):
    """
    Evaluates model performance across training data subsample fractions (20% to 100%).
    """
    results = {}
    
    for frac in fractions:
        X_sub, y_sub = get_subsampled_data(X_train, y_train, fraction=frac, seed=seed)
        
        trainer = FBG_PINN_Trainer(seed=seed)
        trainer.train(X_sub, y_sub, epochs=300, verbose=False)
        metrics, _, _, _ = trainer.evaluate(X_test, y_test)
        
        results[f"{int(frac*100)}%"] = metrics
        
    return results
