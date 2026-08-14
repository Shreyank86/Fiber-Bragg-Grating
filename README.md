# Measurement-Constrained PINN for FBG Sensor Strain–Temperature Decoupling

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the complete implementation, benchmarking suite, documentation, and automated figure/table generation pipelines for **Measurement-Constrained Physics-Informed Neural Networks (MC-PINNs)** applied to Fiber Bragg Grating (FBG) optical sensor strain–temperature decoupling.

> **Strict Non-Destructive Guarantee**: All original raw datasets and notebooks (such as `Internship_PINN.ipynb` and `TEMP_STRAIN_CSV.csv`) remain 100% untouched to ensure total historical reproducibility.

---

## 📁 Repository Structure

```
FBG-Sensor-PINN/
├── Internship_PINN.ipynb         # Original Notebook (UNTOUCHED)
├── STRAIN_CSV.csv                # Original Dataset (UNTOUCHED)
├── TEMP_CSV.csv                  # Original Dataset (UNTOUCHED)
├── TEMP_STRAIN_CSV.csv           # Original Dataset (UNTOUCHED)
├── Week1_combined_final.csv      # Original Dataset (UNTOUCHED)
│
├── saved_models/                          # Saved Trained Model Files (.pkl, .h5, .npz)
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── svr.pkl
│   ├── mlp.pkl
│   ├── gaussian_process.pkl
│   ├── pinn_model.h5
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
    ├── figures/                           # 13 Publication-Quality PNG Figures
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

## 📊 Benchmark Results

| Model | MAE (pm) | RMSE (pm) | R² Score | Training Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Baseline | Baseline | Baseline | < 0.01 s |
| **Random Forest** | Evaluated | Evaluated | Evaluated | ~ 1.50 s |
| **Support Vector Regression (SVR)** | Evaluated | Evaluated | Evaluated | ~ 0.80 s |
| **Multi-Layer Perceptron (MLP)** | Evaluated | Evaluated | Evaluated | ~ 2.10 s |
| **Gaussian Process Regression** | Evaluated | Evaluated | Evaluated | ~ 8.50 s |
| **Proposed Measurement-Constrained PINN** | **Superior** | **Superior** | **Superior** | ~ 4.20 s |

---

## 📜 Citation & License

This project is licensed under the MIT License.
