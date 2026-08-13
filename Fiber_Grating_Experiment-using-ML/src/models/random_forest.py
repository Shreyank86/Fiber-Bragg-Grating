import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from src.evaluation import evaluate_predictions

class RandomForestModel:
    """Wrapper for Random Forest baseline."""
    def __init__(self, n_estimators=300, max_depth=None, random_state=42, **kwargs):
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, **kwargs)
        self.model = MultiOutputRegressor(rf)
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
