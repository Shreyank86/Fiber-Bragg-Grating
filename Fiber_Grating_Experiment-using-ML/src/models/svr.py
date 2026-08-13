import time
import os
import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from src.evaluation import evaluate_predictions

class SVRModel:
    """
    Support Vector Regression (SVR) model implementation.
    Uses RBF kernel wrapped in MultiOutputRegressor to simultaneously
    predict Temperature and Strain.
    """
    def __init__(self, kernel='rbf', C=50.0, epsilon=0.01, gamma='scale', cache_size=1000, max_train_samples=4000, **kwargs):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.cache_size = cache_size
        self.max_train_samples = max_train_samples
        self.kwargs = kwargs
        
        base_svr = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
            cache_size=self.cache_size,
            **kwargs
        )
        self.model = MultiOutputRegressor(base_svr)
        self.train_time = 0.0
        self.infer_time = 0.0

    def fit(self, X_train, y_train, train_groups=None):
        start_time = time.time()
        
        if len(X_train) > self.max_train_samples:
            idx = np.random.choice(len(X_train), self.max_train_samples, replace=False)
            X_fit = X_train[idx] if isinstance(X_train, np.ndarray) else X_train.iloc[idx]
            y_fit = y_train[idx]
        else:
            X_fit = X_train
            y_fit = y_train

        self.model.fit(X_fit, y_fit)
        self.train_time = time.time() - start_time
        return self

    def predict(self, X_test):
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        self.infer_time = (time.time() - start_time) * 1000.0
        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return evaluate_predictions(y_test, y_pred, self.train_time, self.infer_time)

    def save_model(self, filepath="saved_models/svr_model.joblib"):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model successfully saved to: {filepath}")

    def load_model(self, filepath="saved_models/svr_model.joblib"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        self.model = joblib.load(filepath)
        print(f"Model successfully loaded from: {filepath}")
        return self
