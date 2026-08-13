---
name: member1-svr-guide
description: Role-specific workspace skill for Member 1 (Project Lead) working on SVR model implementation, scaling, serialization, and benchmarking.
---

# Member 1 (Project Lead) — SVR Implementation Skill Guide

This skill provides step-by-step guidance for **Member 1 (Project Lead)** working on the **Support Vector Regression (SVR)** module for FBG strain-temperature decoupling.

---

## 1. Role Summary & File Location
- **Assigned Module**: Support Vector Regression (SVR)
- **Model File**: [src/models/svr.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/svr.py)
- **Extracted Saved Model File**: `saved_models/svr_model.joblib`
- **Extracted Scaler File**: `saved_models/scaler.joblib`

---

## 2. Technical Requirements
1. **Model Architecture**: RBF kernel Support Vector Regression wrapped in `sklearn.multioutput.MultiOutputRegressor`.
2. **Leak-free Feature Scaling**: Use `StandardScaler` fitted ONLY on training data via `src.data_loader.get_scaled_train_test_split`.
3. **Model Serialization**:
   - `save_model("saved_models/svr_model.joblib")` using `joblib.dump`.
   - `load_model("saved_models/svr_model.joblib")` using `joblib.load`.
4. **Standard Contract**:
   - `fit(X_train, y_train)`: Fits model and records training time in seconds.
   - `predict(X_test)`: Generates predictions ($\hat{T}, \hat{\epsilon}$) and records inference latency in milliseconds.
   - `evaluate(X_test, y_test)`: Returns dictionary with `MAE`, `MSE`, `RMSE`, `R2`, `Train_Time_Sec`, `Infer_Time_Ms`.

---

## 3. Quick Verification Command
```python
from src.data_loader import load_fbg_dataset, get_scaled_train_test_split
from src.models.svr import SVRModel

X, y = load_fbg_dataset("both")
X_tr, X_te, y_tr, y_te, _ = get_scaled_train_test_split(X, y, scale=True)
model = SVRModel()
model.fit(X_tr, y_tr)
print(model.evaluate(X_te, y_te))
model.save_model()
```
