import time
import os
import joblib
from sklearn.neural_network import MLPRegressor
from src.evaluation import evaluate_predictions

class MLPModel:
    """
    Multi-Layer Perceptron (MLP) pure data-driven model for Member 2.
    Uses 2 hidden layers (64, 32) with ReLU activation and Adam optimizer.
    """
    def __init__(self, hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42, **kwargs):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        self.train_time = 0.0
        self.infer_time = 0.0

    def fit(self, X_train, y_train, train_groups=None):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
        return self

    def predict(self, X_test):
        start_time = time.time()
        y_pred = self.model.predict(X_test)
        self.infer_time = (time.time() - start_time) * 1000.0  # ms
        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return evaluate_predictions(y_test, y_pred, self.train_time, self.infer_time)

    def save_model(self, filepath="saved_models/mlp_model.joblib"):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"MLP Model saved to: {filepath}")

    def load_model(self, filepath="saved_models/mlp_model.joblib"):
        self.model = joblib.load(filepath)
        print(f"MLP Model loaded from: {filepath}")
        return self

