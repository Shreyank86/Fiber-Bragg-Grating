# Measurement-Constrained PINN for FBG Sensor Strain–Temperature Decoupling

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository presents a **Physics-Guided Inverse Sensing Framework** and **Measurement-Constrained Physics-Informed Neural Network (MC-PINN)** for single-sensor Fiber Bragg Grating (FBG) strain–temperature cross-sensitivity decoupling.

By embedding the analytical FBG Bragg wavelength shift equation ($\Delta \lambda_B = k_\varepsilon \cdot \varepsilon + k_T \cdot \Delta T$) directly into the neural network loss function, this project formulates strain-temperature separation as an **inverse measurement problem**, enabling accurate, physically consistent strain and temperature prediction without auxiliary sensing hardware.

> **Strict Non-Destructive Guarantee**: All original raw datasets and notebooks (such as `Internship_PINN.ipynb` and `TEMP_STRAIN_CSV.csv`) remain 100% untouched to ensure historical reproducibility.

---

## 📊 Benchmarking & Experimental Results

### 1. Master Model Comparison (`outputs/tables/table3_baseline_comparison.md`)

| Model | MAE (pm) | RMSE (pm) | $R^2$ Score | Training Time (s) | Inference Latency (s) | Key Feature |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposed MC-PINN** | **0.3897** | **0.4993** | **1.0000** | **5.068 s** | **0.0012 s** | **Physics-Guided & Noise Robust** |
| **Random Forest** | 1.8645 | 6.7924 | 0.9995 | 0.1938 s | 0.0310 s | Ensemble baseline |
| **Multi-Layer Perceptron (MLP)** | 20.9321 | 44.1278 | 0.9780 | 7.2624 s | 0.0031 s | Pure data neural network |
| **Support Vector Regression (SVR)** | 22.9181 | 47.2421 | 0.9748 | 1.4724 s | 0.6307 s | Structural risk minimization |
| **Linear Regression** | 147.3706 | 173.6049 | 0.6591 | 0.0008 s | 0.0001 s | Baseline linear reference |
| **Gaussian Process Regression** | 691.0143 | 751.1688 | -5.3814 | 0.5376 s | 0.0579 s | Probabilistic ($\pm 1.96\sigma$) |

---

### 2. Multi-Model Noise Robustness under Gaussian Noise (`outputs/tables/table6_noise_robustness.md`)

| Noise Level (%) | Proposed PINN MAE (pm) | Proposed PINN RMSE (pm) | Proposed PINN $R^2$ | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **0.0% (Clean)** | 0.4441 | 0.5846 | 1.0000 | Perfect Physical Alignment |
| **1.0% Noise** | 2.4182 | 3.0586 | 0.9999 | High Precision Baseline |
| **3.0% Noise** | 7.1115 | 8.9995 | 0.9991 | Robust Trajectory |
| **5.0% Noise** | 11.8343 | 14.9702 | 0.9975 | Resists Degradation |
| **10.0% Noise** | 23.6499 | 29.9102 | 0.9899 | **Degrades Gracefully** |

---

### 3. Limited Data Regime Efficiency (`outputs/tables/table7_small_data_experiment.md`)

| Training Data (%) | Proposed PINN MAE (pm) | Proposed PINN RMSE (pm) | Proposed PINN $R^2$ | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| **20% Data** | 1.2030 | 1.5169 | 1.0000 | **High accuracy at only 20% training data** |
| **40% Data** | 0.9866 | 1.2398 | 1.0000 | Consistent small-data convergence |
| **60% Data** | 1.5769 | 2.2070 | 0.9999 | Stable out-of-sample performance |
| **80% Data** | 1.5267 | 2.1319 | 0.9999 | Near-optimal performance |
| **100% Data** | 0.4441 | 0.5846 | 1.0000 | Full dataset optimal |

---

## 🖼️ Publication Figures Suite (`outputs/figures/`)

- `training_loss.png`: Training loss vs epoch (*Labeled: Proposed Measurement-Constrained PINN*).
- `validation_loss.png`: Validation loss vs epoch (*Labeled: Proposed Measurement-Constrained PINN*).
- `prediction_vs_gt.png`: **Parity Scatter Plot ($y=x$)** comparing Actual $\Delta\lambda$ vs Predicted $\Delta\lambda$ with ideal reference line and sorted time series panel.
- `residual_histogram.png`: Residual error distributions comparing PINN predictions.
- `rmse_comparison.png`, `mae_comparison.png`, `r2_comparison.png`: Bar charts with **exact numerical value callouts on top of every bar**.
- `noise_robustness.png`: Multi-model degradation curves under 0% to 10% Gaussian noise.
- `small_data_experiment.png`: Multi-model data efficiency curves across 20% to 100% training data.
- `lambda_ablation.png`: Performance sensitivity trends across physics loss weight $\lambda$.
- `cv_boxplot.png`: 5-Fold Cross-Validation metric distributions.
- `bootstrap_ci.png`: **Gaussian Normal Distribution Bell Curves (KDE)** with shaded 95% Confidence Interval bands.

---

## 📁 Repository Directory Structure

```
FBG-Sensor-PINN/
├── Internship_PINN.ipynb         # Original Notebook (UNTOUCHED)
├── STRAIN_CSV.csv                # Original Dataset (UNTOUCHED)
├── TEMP_CSV.csv                  # Original Dataset (UNTOUCHED)
├── TEMP_STRAIN_CSV.csv           # Original Dataset (UNTOUCHED)
├── Week1_combined_final.csv      # Original Dataset (UNTOUCHED)
│
├── saved_models/                          # Saved Trained Model Files (.pkl, .pt, .npz)
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── svr.pkl
│   ├── mlp.pkl
│   ├── gaussian_process.pkl
│   ├── pinn_model.pt
│   └── pinn_predictions.npz
│
├── docs/                                  # Repository Documentation
│   ├── repository_review.md               # Phase 1: 8-10 Page Complete Repository Guide
│   ├── literature_review.md               # Phase 1: Literature Review & Gap Analysis
│   └── explainability_uncertainty.md      # Phase 12 & 13: Physics Regularization & Uncertainty
│
├── src/                                   # Modular Source Code
│   ├── config.py                          # Physical Constants & Configuration Parameters
│   ├── data.py                            # Data Loading, Splitting, Scaling & Noise Injection
│   ├── models.py                          # Classical Machine Learning Baselines (LR, RF, SVR, MLP, GP)
│   ├── pinn.py                            # Measurement-Constrained PINN Architecture & Loss
│   ├── evaluate.py                        # Evaluation Pipelines (CV, Bootstrap, Noise, Small Data)
│   └── utils.py                           # Random Seed Management & Metrics Calculation
│
├── scripts/                               # Automated Execution Scripts
│   ├── run_baselines.py                   # Phase 3: Benchmarking All ML Baselines & PINN
│   ├── run_ablation.py                    # Phase 4: Physics Weight (λ) Ablation Study
│   ├── run_noise_robustness.py            # Phase 5: 1%-10% Gaussian Noise Injection Study
│   ├── run_small_data.py                  # Phase 6: 20%-100% Low-Data Regime Evaluation
│   ├── run_cross_validation.py            # Phase 8: 5-Fold Cross Validation with 95% CIs
│   ├── run_bootstrap_analysis.py          # Phase 9: 1,000-Sample Bootstrap Confidence Intervals
│   ├── generate_figures.py                # Phase 14: Automated Publication Figure Generator
│   └── generate_tables.py                 # Phase 15: Automated LaTeX & Markdown Table Generator
│
├── paper/                                 # Refined Manuscript Sections
│   ├── positioning_and_contributions.md   # Phase 2: Inverse Sensing Framing & Reframed Contributions
│   ├── methodology.md                     # Technical Methodology Specification
│   ├── results_and_discussion.md          # Phase 10, 11 & 16: Results & Extended Discussion
│   └── conclusion_and_future_work.md      # Phase 17 & 18: Conclusion & 9 Future Directions
│
└── outputs/                               # Generated Outputs
    ├── figures/                           # 12 Publication-Quality PNG Figures
    ├── tables/                            # 9 LaTeX (.tex) and Markdown (.md) Tables
    └── results.json                       # Comprehensive Benchmark Metrics JSON File
```

---

## ⚡ Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Benchmarking & Train All Models
Train Linear Regression, Random Forest, SVR, MLP, Gaussian Process, and PINN:
```bash
python scripts/run_baselines.py
```
This automatically saves all trained models into `saved_models/` and logs benchmark metrics into `outputs/results.json`.

### 3. Run Experiments (Ablation, Noise, Small Data, CV, Bootstrap)
```bash
python scripts/run_ablation.py
python scripts/run_noise_robustness.py
python scripts/run_small_data.py
python scripts/run_cross_validation.py
python scripts/run_bootstrap_analysis.py
```

### 4. Generate Publication Figures & LaTeX Tables
```bash
python scripts/generate_figures.py
python scripts/generate_tables.py
```
Outputs will be saved in `outputs/figures/` and `outputs/tables/`.

---

## 📜 Citation & License

This project is licensed under the MIT License.
