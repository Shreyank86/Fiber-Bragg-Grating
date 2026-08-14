# Table 2: Model Architecture & Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| **Input Layer Neurons** | 2 (`[Time, delta_lambda_pm]`) |
| **Hidden Layers** | 3 Dense Layers (64, 64, 32 neurons) |
| **Activation Function** | ReLU |
| **Output Heads** | 2 Heads (`strain_pred`, `temp_pred`) |
| **Physics Loss Weight (\lambda)** | 1.0 |
| **Optimizer** | Adam ($lr = 10^{-3}$) |
| **Batch Size** | Full batch / 64 |
| **Training Epochs** | 500 epochs |
