# Fiber Bragg Grating (FBG) Strain-Temperature Decoupling via Physics-Informed Neural Networks (PINN)

This subfolder contains the complete implementation, datasets, model binaries, experimental scripts, and visualization engine for the FBG Strain-Temperature Decoupling project.

---

## 📊 Summary of Benchmark Results

| Model | MAE | RMSE | $R^2$ Score (%) | Train Time (s) | Infer Latency (ms) |
|---|---|---|---|---|---|
| **PINN (Proposed)** | **14.17** | **28.11** | **98.14%** | **77.32 s** | **5.51 ms** |
| Random Forest | 3.15 | 10.99 | 99.71% | 34.95 s | 364.28 ms |
| Multi-Layer Perceptron | 25.77 | 47.74 | 93.75% | 22.15 s | 1.60 ms |
| Support Vector Regression | 60.33 | 111.17 | 72.99% | 1.38 s | 1035.96 ms |
| Linear Regression | 74.28 | 118.41 | 66.22% | 0.002 s | 0.00 ms |
| Gaussian Process Regression | 131.66 | 203.91 | Probabilistic | 7.44 s | 368.67 ms |

---

## 🚀 Reproduction Command

```bash
python run_experiments.py
```
Executing this script will train/evaluate all 6 models, compute statistical cross-validation and bootstrap intervals, and regenerate all 8 CSV tables and 7 publication-quality PNG figures in `results/`.
