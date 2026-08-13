# Fiber Bragg Grating (FBG) Strain-Temperature Decoupling via Physics-Informed Neural Networks (PINN)

This repository presents a **Physics-Guided Inverse Sensing Framework** for single-sensor Fiber Bragg Grating (FBG) strain–temperature cross-sensitivity decoupling.

By embedding the analytical FBG Bragg wavelength shift equation ($\Delta \lambda_B = k_T \Delta T + k_S \Delta \epsilon$) directly into the loss function of a PyTorch neural network, this project formulates decoupling as an **inverse measurement problem**, enabling accurate single-sensor temperature and strain measurement without requiring secondary reference hardware.

---

## 📊 Benchmarking & Experimental Results

### 1. Master Model Comparison (`results/tables/benchmark_comparison.csv`)

| Model | MAE | RMSE | $R^2$ Score (%) | Train Time (s) | Infer Latency (ms) | Key Feature |
|---|---|---|---|---|---|---|
| **PINN (Proposed)** | **14.17** | **28.11** | **98.14%** | **77.32 s** | **5.51 ms** | **Physics-Guided & Noise Robust** |
| Random Forest | 3.15 | 10.99 | 99.71% | 34.95 s | 364.28 ms | Ensemble baseline |
| Multi-Layer Perceptron | 25.77 | 47.74 | 93.75% | 22.15 s | 1.60 ms | Pure data neural network |
| Support Vector Regression | 60.33 | 111.17 | 72.99% | 1.38 s | 1035.96 ms | Structural risk minimization |
| Linear Regression | 74.28 | 118.41 | 66.22% | 0.002 s | 0.00 ms | Baseline linear reference |
| Gaussian Process Regression | 131.66 | 203.91 | Probabilistic | 7.44 s | 368.67 ms | 95% Confidence Bounds ($\pm 1.96\sigma$) |

---

### 2. Multi-Model Noise Robustness under Gaussian Noise (`results/tables/noise_robustness.csv`)

| Noise Level | Pure Data MLP $R^2$ | PINN (Proposed) $R^2$ | PINN RMSE | Key Observation |
|---|---|---|---|---|
| **1% Noise** | 92.87% | **96.80%** | 36.18 | High precision baseline |
| **3% Noise** | 86.68% | **89.77%** | 64.22 | Robust trajectory |
| **5% Noise** | 76.03% | **82.97%** | 83.18 | Resists degradation |
| **10% Noise** | 34.94% | **68.02%** | 114.88 | **PINN outperforms pure MLP by +33.1% $R^2$** |

---

### 3. Limited Data Regime Efficiency (`results/tables/small_data_regime.csv`)

| Data Fraction | Training Samples | PINN $R^2$ Score | PINN RMSE | Key Observation |
|---|---|---|---|---|
| **20% Data** | 1,442 | 69.29% | 120.84 | Baseline small-data split |
| **40% Data** | 2,884 | **96.58%** | 39.27 | **PINN reaches 96.6% accuracy at only 40% data** |
| **60% Data** | 4,326 | **97.34%** | 33.24 | Consistent convergence |
| **80% Data** | 5,768 | **97.40%** | 32.87 | Near-optimal performance |
| **100% Data** | 7,211 | **98.14%** | 28.11 | Full dataset optimal |

---

## 🖼️ Publication Figures Suite (`results/figures/`)

- `gpr_uncertainty_quantification.png`: GPR mean predictions with shaded 95% confidence bands ($\pm 1.96\sigma$).
- `noise_robustness_curves.png`: Multi-model degradation curves under 1% to 10% Gaussian noise.
- `small_data_regime_curves.png`: Multi-model data efficiency curves across 20% to 100% training data.
- `ablation_study_curves.png`: Performance trends across physics loss weight $\lambda_{\text{phys}}$.
- `residual_histograms.png`: Residual error distributions comparing PINN vs GPR.
- `cross_validation_boxplot.png`: 5-Fold Cross-Validation $R^2$ distribution boxplots.
- `prediction_vs_ground_truth.png`: Parity scatter plots comparing PINN vs MLP predictions against ground truth.

---

## 📁 Repository Directory Structure

```
fiber-bragg-grating/
├── Fiber_Grating_Experiment-using-ML/
│   ├── Dataset/                                     # Raw FBG sensor CSV datasets
│   ├── results/
│   │   ├── figures/                                 # Publication-quality 300 DPI figures
│   │   └── tables/                                  # Experimental CSV tables
│   ├── saved_models/                                # Serialized model binaries (.joblib, .pth)
│   ├── src/
│   │   ├── data_loader.py                           # Sliding window feature extractor & scaler
│   │   ├── physics.py                               # Sensitivity constants & loss equations
│   │   ├── evaluation.py                            # Performance metric calculation
│   │   ├── visualization.py                         # Automated figure generation engine
│   │   ├── models/                                  # LR, RF, SVR, MLP, GPR, PINN models
│   │   └── experiments/                             # Benchmark, Noise, Small Data, Ablation, CV, Sensitivity
│   ├── run_experiments.py                           # Master evaluation suite script
│   └── requirements.txt                             # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

1. Clone repository and navigate to project folder:
   ```bash
   git clone https://github.com/Shreyank86/Fiber-Bragg-Grating.git
   cd Fiber-Bragg-Grating/Fiber_Grating_Experiment-using-ML
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run master experimental suite (reproduces all figures & tables):
   ```bash
   python run_experiments.py
   ```
