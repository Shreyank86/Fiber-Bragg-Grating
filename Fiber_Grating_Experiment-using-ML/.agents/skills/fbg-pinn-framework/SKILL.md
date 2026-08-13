---
name: fbg-pinn-framework
description: Master technical guidelines, physics formulation, repository architecture, and team role assignments for FBG Sensor ML + PINN Inverse Measurement project.
---

# FBG Sensor ML + PINN Framework — Team Guide & AI Assistant Skill

This document serves as the **master skill and project context** for AI coding assistants (Google Antigravity) and team members working on the **Fiber Bragg Grating (FBG) Sensor Strain-Temperature Decoupling as an Inverse Measurement Problem**.

---

## 1. Project Overview & Physical Problem

### Physics Context
A **Fiber Bragg Grating (FBG)** optical sensor reflects light at a Bragg wavelength ($\lambda_B = 2 \cdot n_{\text{eff}} \cdot \Lambda$). Both **Temperature change ($\Delta T$)** and **Strain ($\Delta \epsilon$)** cause a red-shift in the reflected wavelength:
$$\Delta \lambda_B = \lambda_B \cdot (\alpha + \xi) \cdot \Delta T + \lambda_B \cdot (1 - p_e) \cdot \Delta \epsilon = k_T \cdot \Delta T + k_S \cdot \Delta \epsilon$$
Where:
- $k_T \approx 0.01015\,\text{nm/}^\circ\text{C}$ (Temperature sensitivity constant)
- $k_S \approx 0.00121\,\text{nm/}\mu\epsilon$ (Strain sensitivity constant)

### Scientific Positioning
This work solves an **Inverse Measurement Problem** (recovering unknown strain $\Delta \epsilon$ and temperature $\Delta T$ from observed optical wavelength shift $\Delta \lambda_B$) rather than a forward PDE simulation.

---

## 2. Dataset & Feature Engineering

### Experimental Datasets (`Dataset/`)
1. `TEMP_EXPERIMENT-1.csv`: Temperature-only heating ($20^\circ\text{C} \to 80^\circ\text{C}$).
2. `TEMP EXPERIMENT-2.csv`: Strain-only loading ($0 \to 1000\,\mu\epsilon$).
3. `TEMP AND STRAIN EXPERIMENT-3.csv`: Simultaneous Temperature + Strain loading (9,064 samples, ~5 Hz).

### Input Features ($X$) — 50-Sample Sliding Window (~10 seconds)
Extracted automatically via `src/data_loader.py`:
1. `Wavelength`: Raw Bragg wavelength ($\sim 1523.6 - 1524.6\,\text{nm}$).
2. `mean`: 50-sample rolling mean.
3. `std`: 50-sample rolling standard deviation.
4. `slope`: Rate of wavelength change ($\text{diff}(5) / (dt \times 5)$).
5. `skew`: 50-sample rolling skewness.
6. `kurtosis`: 50-sample rolling kurtosis.
7. `shift`: Wavelength shift from initial baseline ($\Delta \lambda_B = \lambda_B - \lambda_{B,0}$).

### Target Outputs ($y$)
Multi-output stacked array: `[y_temp, y_strain]` ($^\circ\text{C}$ and $\mu\epsilon$).

---

## 3. Team Member Assignments & Deliverables

When a developer or AI assistant starts a task, specify who is working to load the exact role context:

---

### 👤 Member 1 (RAKSHITH): Support Vector Regression (SVR)
- **Assigned Script**: [src/models/svr.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/svr.py)
- **Model**: `Support Vector Regression` (`MultiOutputRegressor(SVR())`)
- **Deliverables**:
  1. Implement RBF kernel SVR with leak-free `StandardScaler`.
  2. Perform hyperparameter tuning over $C \in [1, 10, 100]$, $\epsilon \in [0.01, 0.1]$, $\gamma \in ['scale', 'auto']$.
  3. Track MAE, RMSE, $R^2$, Training Time (sec), and Inference Latency (ms).

---

### 👤 Member 2 (Siddharth): Multi-Layer Perceptron (MLP)
- **Assigned Script**: [src/models/mlp.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/mlp.py)
- **Model**: `Multi-Layer Perceptron (Pure Data-Driven Neural Network)`
- **Deliverables**:
  1. Implement CPU-friendly MLP architecture (e.g., 2 hidden layers: 64 and 32 neurons with ReLU/Tanh activation).
  2. Ensure feature scaling using `StandardScaler` (fit on train set, transform on test set).
  3. Ensure reproducible random seed initialization (`random_state=42`).
  4. Track MAE, RMSE, $R^2$, Training Time (sec), and Inference Latency (ms).
  5. Provide pure data-driven baseline comparison against PINN.

---

### 👤 Member 3 (Shreyank): Gaussian Process (GPR) & PINN
- **Assigned Scripts**: 
  - [src/models/gpr.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/gpr.py) (Gaussian Process Regression)
  - [src/models/pinn.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/pinn.py) (Measurement-Constrained PINN)
- **Deliverables**:
  1. **GPR**: Implement `GaussianProcessRegressor(kernel=RBF() + WhiteKernel())`. Extract predictions and $95\%$ confidence bounds ($\pm 1.96\sigma$).
  2. **PINN**: Implement PyTorch Measurement-Constrained PINN embedding physical loss:
     $$\mathcal{L}_{\text{physics}} = \frac{1}{N} \sum_{i=1}^N \left| \Delta \lambda_{B, i} - (k_T \cdot \hat{\Delta T}_i + k_S \cdot \hat{\Delta \epsilon}_i) \right|^2$$
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{physics}}$$
  3. Prove PINN superiority under noise (1%–10% Gaussian noise) and small data regimes (20%–100% data).

---

## 4. Standard Model Class Contract

All model classes inside `src/models/` MUST implement the following unified Python interface:

```python
class ModelWrapper:
    def __init__(self, **kwargs):
        pass

    def fit(self, X_train, y_train, train_groups=None):
        """Train model and record training time (sec)."""
        pass

    def predict(self, X_test):
        """Return predictions y_pred and record inference latency (ms)."""
        pass

    def evaluate(self, X_test, y_test):
        """Returns dict containing MAE, RMSE, R2, Train_Time_Sec, Infer_Time_Ms."""
        pass
```

---

## 5. Summary of Project Architecture

```
Fiber_Grating_Experiment-using-ML/
├── .agents/skills/fbg-pinn-framework/SKILL.md  # [SKILL CONTEXT]
├── Dataset/                                     # Raw experimental CSVs
├── src/                                         # Shared pipeline
│   ├── data_loader.py                           # Dataset loading & sliding window features
│   ├── physics.py                               # FBG sensitivity constants (k_T, k_S)
│   ├── evaluation.py                            # Metric calculation & printing
│   ├── models/
│   │   ├── linear_regression.py                 # Baseline LR
│   │   ├── random_forest.py                     # Baseline RF
│   │   ├── svr.py                               # [Member 1 Assignment] SVR
│   │   ├── mlp.py                               # [Siddharth Assignment] MLP
│   │   ├── gpr.py                               # [Shreyank Assignment] GPR
│   │   └── pinn.py                              # [Shreyank Assignment] PINN
│   └── experiments/
│       ├── benchmark.py                         # Master comparison runner
│       ├── noise_robustness.py                  # 1%-10% Gaussian noise evaluation
│       ├── small_data.py                        # 20%-100% data fraction analysis
│       └── ablation.py                          # Physics loss lambda study
├── run_experiments.py                           # Master script executing all experiments
├── requirements.txt
└── README.md
```
