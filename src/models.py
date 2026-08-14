"""
Classical Machine Learning Baseline Models (LR, RF, SVR, MLP, Gaussian Process).
"""

import time
import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from .config import SAVED_MODELS_DIR, RANDOM_SEED
from .utils import compute_metrics

class ClassicalMLBaselines:
    """Wrapper class for training, evaluating, saving, and inferring classical ML models."""
    
    def __init__(self, seed=RANDOM_SEED):
        self.seed = seed
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=self.seed, n_jobs=-1),
            "SVR": SVR(kernel='rbf', C=10.0, epsilon=0.1),
            "MLP": MLPRegressor(hidden_layer_sizes=(64, 64, 32), max_iter=500, random_state=self.seed, early_stopping=True),
            "Gaussian Process": GaussianProcessRegressor(
                kernel=C(1.0) * RBF(length_scale=1.0),
                alpha=1e-2,
                n_restarts_optimizer=2,
                random_state=self.seed
            )
        }
        self.fitted_models = {}

    def fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train all classical models and measure MAE, RMSE, R2, Train Time, and Inference Time."""
        y_train_flat = y_train.flatten()
        y_test_flat = y_test.flatten()
        results = {}

        for name, model in self.models.items():
            # Measure Training Time
            t0 = time.perf_counter()
            if name == "Gaussian Process" and len(X_train) > 1000:
                rng = np.random.default_rng(self.seed)
                idx = rng.choice(len(X_train), size=1000, replace=False)
                model.fit(X_train[idx], y_train_flat[idx])
            else:
                model.fit(X_train, y_train_flat)
            t_train_sec = time.perf_counter() - t0

            # Measure Inference Time
            t0 = time.perf_counter()
            if name == "Gaussian Process":
                y_pred, std_pred = model.predict(X_test, return_std=True)
            else:
                y_pred = model.predict(X_test)
                std_pred = None
            t_infer_sec = time.perf_counter() - t0

            metrics = compute_metrics(y_test_flat, y_pred)
            metrics["Train_Time_Sec"] = float(t_train_sec)
            metrics["Infer_Time_Sec"] = float(t_infer_sec)
            if std_pred is not None:
                metrics["Uncertainty_Mean_Std"] = float(np.mean(std_pred))

            results[name] = {
                "metrics": metrics,
                "y_pred": y_pred,
                "std_pred": std_pred
            }
            self.fitted_models[name] = model

            # Save model artifact to saved_models directory with compression
            model_filename = name.lower().replace(" ", "_") + ".pkl"
            save_path = os.path.join(SAVED_MODELS_DIR, model_filename)
            joblib.dump(model, save_path, compress=3)

        return results
