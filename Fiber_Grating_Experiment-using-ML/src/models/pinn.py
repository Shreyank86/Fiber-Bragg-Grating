import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.evaluation import evaluate_predictions
from src.physics import K_T, K_S

class PINNModule(nn.Module):
    """
    Multi-Layer Perceptron neural network for physical forward and inverse mapping.
    Architecture: input_dim -> 64 (Tanh) -> 32 (Tanh) -> 2 (Temperature, Strain).
    """
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
    """
    Measurement-Constrained Physics-Informed Neural Network (PINN) Wrapper.
    Embeds physical sensitivity equation Delta_Lambda_B = k_T * (T - 20) + k_S * Strain into network loss.
    Supports CUDA GPU acceleration automatically if available.
    """
    def __init__(self, lambda_phys=1.0, epochs=200, lr=0.005, batch_size=64, random_state=42):
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
        self.lambda_phys = lambda_phys
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_time = 0.0
        self.infer_time = 0.0
        print(f"PINN Initialized using device: {self.device}")

    def fit(self, X_train, y_train, train_groups=None):
        """Trains PINN model using Adam optimizer with dual Data + Physics loss on target device."""
        start_time = time.time()
        
        X_arr = X_train if isinstance(X_train, torch.Tensor) else torch.tensor(X_train, dtype=torch.float32)
        y_arr = y_train if isinstance(y_train, torch.Tensor) else torch.tensor(y_train, dtype=torch.float32)
        
        input_dim = X_arr.shape[1]
        shift_col_idx = input_dim - 1  # Wavelength shift column (last column)
        
        self.model = PINNModule(input_dim=input_dim, output_dim=2).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(X_arr, y_arr)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                preds = self.model(batch_x)
                
                # 1. Data-driven Loss
                loss_data = criterion(preds, batch_y)
                
                # 2. Physics-informed Loss (Delta_Lambda_B = k_T * (T - 20) + k_S * Strain)
                delta_t = preds[:, 0] - 20.0
                delta_s = preds[:, 1] - 0.0
                wave_phys = K_T * delta_t + K_S * delta_s
                obs_shift = batch_x[:, shift_col_idx]
                
                loss_phys = criterion(wave_phys, obs_shift)
                
                # 3. Combined Loss
                loss_total = loss_data + self.lambda_phys * loss_phys
                
                loss_total.backward()
                optimizer.step()
                
        self.train_time = time.time() - start_time
        return self

    def predict(self, X_test):
        """Predicts target output parameters (Temperature, Strain) using trained PINN."""
        start_time = time.time()
        if self.model is None:
            raise RuntimeError("PINN Model has not been trained yet. Call fit() or load_model().")
            
        self.model.eval()
        with torch.no_grad():
            X_arr = X_test if isinstance(X_test, torch.Tensor) else torch.tensor(X_test, dtype=torch.float32)
            X_t = X_arr.to(self.device)
            preds = self.model(X_t).cpu().numpy()
            
        self.infer_time = (time.time() - start_time) * 1000.0  # ms
        return preds

    def evaluate(self, X_test, y_test):
        """Evaluates predictions and returns metrics dictionary."""
        y_pred = self.predict(X_test)
        return evaluate_predictions(y_test, y_pred, self.train_time, self.infer_time)

    def save_model(self, filepath="saved_models/pinn_model.pth"):
        """Saves model weights to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"PINN Model weights saved to: {filepath}")

    def load_model(self, filepath="saved_models/pinn_model.pth", input_dim=7):
        """Loads model weights from file."""
        self.model = PINNModule(input_dim=input_dim, output_dim=2).to(self.device)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        print(f"PINN Model weights loaded from: {filepath}")
        return self

