---
name: shreyank-gpr-pinn-guide
description: Role-specific workspace skill for Shreyank (Member 3) working on Gaussian Process Regression (GPR) and Measurement-Constrained Physics-Informed Neural Network (PINN).
---

# Shreyank (Member 3) — GPR & PINN Implementation Skill Guide

This skill provides step-by-step guidance for **Shreyank (Member 3)** working on **Gaussian Process Regression (GPR)** and the **Physics-Informed Neural Network (PINN)** module for FBG strain-temperature decoupling.

---

## 1. Role Summary & Assigned Files
- **Assigned Developer**: Shreyank (Member 3)
- **GPR Model File to Edit**: [src/models/gpr.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/gpr.py)
- **PINN Model File to Edit**: [src/models/pinn.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/pinn.py)
- **Output Saved Files**:
  - `saved_models/gpr_model.joblib`
  - `saved_models/pinn_model.pth`

---

## 2. Technical Instructions for GPR (`src/models/gpr.py`)

### Step 1: Implement Gaussian Process Regression
Use `sklearn.gaussian_process.GaussianProcessRegressor` with `RBF() + WhiteKernel()`:

```python
import time
import os
import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from src.evaluation import evaluate_predictions

class GPRModel:
    def __init__(self, random_state=42, max_train_samples=2000, **kwargs):
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=random_state, normalize_y=True, **kwargs)
        self.model = MultiOutputRegressor(gpr)
        self.max_train_samples = max_train_samples
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
        self.infer_time = (time.time() - start_time) * 1000.0  # ms
        return y_pred

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return evaluate_predictions(y_test, y_pred, self.train_time, self.infer_time)

    def save_model(self, filepath="saved_models/gpr_model.joblib"):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"GPR Model saved to: {filepath}")

    def load_model(self, filepath="saved_models/gpr_model.joblib"):
        self.model = joblib.load(filepath)
        print(f"GPR Model loaded from: {filepath}")
        return self
```

---

## 3. Technical Instructions for PINN (`src/models/pinn.py`)

### Physics Loss Formulation:
- Sensitivity Constants: $k_T = 0.01015\,\text{nm/}^\circ\text{C}, k_S = 0.00121\,\text{nm/}\mu\epsilon$.
- Recombination: $\Delta \lambda_B = k_T \cdot (\hat{T} - 20) + k_S \cdot (\hat{\epsilon} - 0)$.
- Physics Loss: $\mathcal{L}_{\text{physics}} = \text{MSE}(\Delta \lambda_B, k_T (\hat{T} - 20) + k_S \cdot \hat{\epsilon})$.
- Total Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{physics}}$.

```python
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.evaluation import evaluate_predictions
from src.physics import K_T, K_S

class PINNModule(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(PINNModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class PINNModel:
    def __init__(self, lambda_phys=1.0, epochs=200, lr=0.005, batch_size=64, random_state=42):
        torch.manual_seed(random_state)
        self.lambda_phys = lambda_phys
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        self.train_time = 0.0
        self.infer_time = 0.0

    def fit(self, X_train, y_train, train_groups=None):
        start_time = time.time()
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        input_dim = X_train.shape[1]
        
        self.model = PINNModule(input_dim=input_dim, output_dim=2)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        shift_col_idx = input_dim - 1
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                
                loss_data = criterion(preds, batch_y)
                
                delta_t = preds[:, 0] - 20.0
                delta_s = preds[:, 1] - 0.0
                wave_phys = K_T * delta_t + K_S * delta_s
                obs_shift = batch_x[:, shift_col_idx]
                
                loss_phys = criterion(wave_phys, obs_shift)
                loss_total = loss_data + self.lambda_phys * loss_phys
                
                loss_total.backward()
                optimizer.step()
                
        self.train_time = time.time() - start_time
        return self

    def predict(self, X_test):
        start_time = time.time()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32)
            preds = self.model(X_t).numpy()
        self.infer_time = (time.time() - start_time) * 1000.0
        return preds

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return evaluate_predictions(y_test, y_pred, self.train_time, self.infer_time)

    def save_model(self, filepath="saved_models/pinn_model.pth"):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"PINN Model weights saved to: {filepath}")

    def load_model(self, filepath="saved_models/pinn_model.pth", input_dim=7):
        self.model = PINNModule(input_dim=input_dim, output_dim=2)
        self.model.load_state_dict(torch.load(filepath))
        print(f"PINN Model weights loaded from: {filepath}")
        return self
```

---

## 4. Deliverables Checklist
- [ ] `GPRModel` implemented in `src/models/gpr.py` & saved to `saved_models/gpr_model.joblib`.
- [ ] `PINNModel` implemented in `src/models/pinn.py` & state dict saved to `saved_models/pinn_model.pth`.
- [ ] Run `python run_experiments.py` to generate PINN vs ML benchmarks, noise robustness plots, and data efficiency curves!
