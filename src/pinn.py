"""
Physics-Informed Neural Network (PINN) architecture and physics-constrained training module implemented in PyTorch.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import K_EPSILON, K_TEMP, SAVED_MODELS_DIR, RANDOM_SEED, DEFAULT_LEARNING_RATE
from .utils import compute_metrics, set_seed

class MeasurementConstrainedPINN(nn.Module):
    """
    PyTorch Deep Neural Network with two separate output heads:
    Head 1: Strain Prediction (microstrain)
    Head 2: Temperature Prediction (degree C)
    """

    def __init__(self, input_dim=2):
        super(MeasurementConstrainedPINN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.strain_head = nn.Linear(32, 1)
        self.temp_head = nn.Linear(32, 1)

    def forward(self, x):
        features = self.backbone(x)
        strain_pred = self.strain_head(features)
        temp_pred = self.temp_head(features)
        return strain_pred, temp_pred

def physics_loss_fn(y_true, strain_pred, temp_pred, k_e=K_EPSILON, k_t=K_TEMP):
    """
    Computes physics loss enforcing analytical Bragg wavelength shift equation:
    delta_pred = k_e * strain_pred + k_t * temp_pred
    """
    delta_pred = k_e * strain_pred + k_t * temp_pred
    return torch.mean((y_true - delta_pred) ** 2)

class FBG_PINN_Trainer:
    """Trainer class for PyTorch Measurement-Constrained Physics-Informed Neural Network."""

    def __init__(self, physics_weight=1.0, learning_rate=DEFAULT_LEARNING_RATE, seed=RANDOM_SEED):
        set_seed(seed)
        self.physics_weight = physics_weight
        self.learning_rate = learning_rate
        self.model = MeasurementConstrainedPINN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_history = []
        self.val_loss_history = []

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=500, verbose=False):
        """PyTorch training loop with physics loss regularization."""
        self.model.train()
        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32)

        if X_val is not None and y_val is not None:
            X_v = torch.tensor(X_val, dtype=torch.float32)
            y_v = torch.tensor(y_val, dtype=torch.float32)
        else:
            X_v, y_v = None, None

        t0 = time.perf_counter()

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()
            strain_out, temp_out = self.model(X_tr)
            delta_pred = K_EPSILON * strain_out + K_TEMP * temp_out

            data_loss = torch.mean((y_tr - delta_pred) ** 2)
            phys_loss = physics_loss_fn(y_tr, strain_out, temp_out)
            total_loss = data_loss + self.physics_weight * phys_loss

            total_loss.backward()
            self.optimizer.step()

            self.loss_history.append(float(total_loss.item()))

            if X_v is not None and y_v is not None:
                self.model.eval()
                with torch.no_grad():
                    s_v, t_v = self.model(X_v)
                    val_delta = K_EPSILON * s_v + K_TEMP * t_v
                    v_loss = torch.mean((y_v - val_delta) ** 2)
                    self.val_loss_history.append(float(v_loss.item()))
                self.model.train()

            if verbose and (epoch % 50 == 0 or epoch == epochs):
                print(f"Epoch {epoch:4d}/{epochs} | Total Loss: {total_loss.item():.4f}")

        train_time_sec = time.perf_counter() - t0
        return train_time_sec

    def predict(self, X_test):
        """Predicts strain, temperature, and wavelength shift on test data."""
        self.model.eval()
        t0 = time.perf_counter()
        X_te = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            strain_pred, temp_pred = self.model(X_te)
        infer_time_sec = time.perf_counter() - t0

        strain_np = strain_pred.numpy()
        temp_np = temp_pred.numpy()
        delta_pred_np = (K_EPSILON * strain_np + K_TEMP * temp_np)

        return strain_np, temp_np, delta_pred_np, infer_time_sec

    def evaluate(self, X_test, y_test):
        """Computes test set metrics."""
        strain_np, temp_np, delta_pred_np, infer_time_sec = self.predict(X_test)
        metrics = compute_metrics(y_test, delta_pred_np)
        metrics["Infer_Time_Sec"] = float(infer_time_sec)
        return metrics, strain_np, temp_np, delta_pred_np

    def save(self, model_dir=SAVED_MODELS_DIR):
        """Saves PyTorch PINN model state dict."""
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, "pinn_model.pt")
        torch.save(self.model.state_dict(), save_path)
        return save_path
