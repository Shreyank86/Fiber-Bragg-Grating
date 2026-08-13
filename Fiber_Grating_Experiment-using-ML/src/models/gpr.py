import time
import os
import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from src.evaluation import evaluate_predictions

class GPRModel:
    """
    Gaussian Process Regression (GPR) model for FBG Strain-Temperature Decoupling.
    Implements standard model contract, uncertainty quantification, sample subsampling for efficiency,
    and joblib persistence.
    """
    def __init__(self, random_state=42, max_train_samples=2000, **kwargs):
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        gpr = GaussianProcessRegressor(
            kernel=kernel,
            random_state=random_state,
            normalize_y=True,
            **kwargs
        )
        self.model = MultiOutputRegressor(gpr)
        self.max_train_samples = max_train_samples
        self.train_time = 0.0
        self.infer_time = 0.0

    def fit(self, X_train, y_train, train_groups=None):
        """Trains GPR model with random subsampling if sample count exceeds max_train_samples."""
        start_time = time.time()
        
        n_samples = len(X_train)
        if n_samples > self.max_train_samples:
            idx = np.random.choice(n_samples, self.max_train_samples, replace=False)
            X_fit = X_train[idx] if isinstance(X_train, np.ndarray) else X_train.iloc[idx]
            y_fit = y_train[idx]
        else:
            X_fit = X_train
            y_fit = y_train

        self.model.fit(X_fit, y_fit)
        self.train_time = time.time() - start_time
        return self

    def predict(self, X_test):
        """Predicts target outputs and records inference latency in ms."""
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        self.infer_time = (time.time() - start_time) * 1000.0  # ms
        return y_pred

    def predict_with_uncertainty(self, X_test):
        """
        Predicts target outputs (y_pred) along with 1-sigma standard deviation (y_std)
        and 95% confidence intervals (lower_bound, upper_bound).
        Returns: y_pred, y_std, lower_bound, upper_bound
        """
        start_time = time.time()
        means = []
        stds = []
        for est in self.model.estimators_:
            m, s = est.predict(X_test, return_std=True)
            means.append(m)
            stds.append(s)
        
        y_pred = np.column_stack(means)
        y_std = np.column_stack(stds)
        lower_bound = y_pred - 1.96 * y_std
        upper_bound = y_pred + 1.96 * y_std
        
        self.infer_time = (time.time() - start_time) * 1000.0
        return y_pred, y_std, lower_bound, upper_bound

    def evaluate(self, X_test, y_test):
        """Evaluates predictions and returns metrics dictionary."""
        y_pred = self.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred, self.train_time, self.infer_time)
        
        if hasattr(self.model, "estimators_") and len(self.model.estimators_) > 0:
            _, y_std, _, _ = self.predict_with_uncertainty(X_test[:100])
            metrics["Avg_Uncertainty_Interval_Width"] = float(np.mean(2 * 1.96 * y_std))
        return metrics

    def save_model(self, filepath="saved_models/gpr_model.joblib"):
        """Saves GPR model to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"GPR Model saved to: {filepath}")

    def load_model(self, filepath="saved_models/gpr_model.joblib"):
        """Loads GPR model from file."""
        self.model = joblib.load(filepath)
        print(f"GPR Model loaded from: {filepath}")
        return self

