---
name: siddharth-mlp-guide
description: Role-specific workspace skill for Siddharth (Member 2) working on Multi-Layer Perceptron (MLP) model implementation, feature scaling, model saving, and evaluation.
---

# Siddharth (Member 2) — MLP Implementation Skill Guide

This skill provides step-by-step guidance for **Siddharth (Member 2)** working on the **Multi-Layer Perceptron (MLP)** pure data-driven neural network module for FBG strain-temperature decoupling.

---

## 1. Role Summary & Assigned Files
- **Assigned Developer**: Siddharth (Member 2)
- **Model File to Edit**: [src/models/mlp.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/mlp.py)
- **Output Model File to Save**: `saved_models/mlp_model.joblib`

---

## 2. Technical Instructions & Implementation Steps

### Step 1: Open [src/models/mlp.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/mlp.py)
Implement the `MLPModel` class using `sklearn.neural_network.MLPRegressor`:

```python
import time
import os
import joblib
from sklearn.neural_network import MLPRegressor
from src.evaluation import evaluate_predictions

class MLPModel:
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
```

---

## 3. How to Test Your Implementation
Run this Python script to verify your MLP model:

```python
from src.data_loader import load_fbg_dataset, get_scaled_train_test_split
from src.models.mlp import MLPModel

# Load data and apply leak-free feature scaling
X, y = load_fbg_dataset("both")
X_tr, X_te, y_tr, y_te, _ = get_scaled_train_test_split(X, y, scale=True)

# Train and evaluate MLP
mlp = MLPModel()
mlp.fit(X_tr, y_tr)
metrics = mlp.evaluate(X_te, y_te)
print("Siddharth's MLP Metrics:", metrics)

# Save extracted model binary file
mlp.save_model()
```

---

## 4. Deliverables Checklist
- [ ] `MLPModel` class implemented in `src/models/mlp.py`.
- [ ] Leak-free scaling applied via `get_scaled_train_test_split`.
- [ ] Model trained and saved to `saved_models/mlp_model.joblib`.
- [ ] Metrics (`MAE`, `RMSE`, `R2`, `Train_Time_Sec`, `Infer_Time_Ms`) evaluated cleanly.
