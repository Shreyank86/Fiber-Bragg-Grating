# Fiber Bragg Grating (FBG) Strain-Temperature Decoupling via ML + PINN

This repository implements a **Physics-Guided Inverse Sensing Framework** to solve single-sensor FBG strain and temperature cross-sensitivity decoupling.

## 👥 Team Work Division

- **Member 1 (Project Lead / User)**: Support Vector Regression ([src/models/svr.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/svr.py))
- **Member 2 (Siddharth)**: Multi-Layer Perceptron ([src/models/mlp.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/mlp.py))
- **Member 3 (Shreyank)**: Gaussian Process Regression ([src/models/gpr.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/gpr.py)) & PINN ([src/models/pinn.py](file:///c:/Users/raksh/OneDrive/Desktop/FBG/Fiber_Grating_Experiment-using-ML/src/models/pinn.py))

---

## 📁 Repository Structure

```
Fiber_Grating_Experiment-using-ML/
├── .agents/skills/fbg-pinn-framework/SKILL.md  # Custom workspace skill context
├── Dataset/                                     # Raw FBG sensor CSV datasets
├── src/
│   ├── data_loader.py                           # Sliding window feature extractor & leak-free scaler
│   ├── physics.py                               # Physical forward equations & sensitivity constants
│   ├── evaluation.py                            # Metric evaluators (MAE, RMSE, R2, Latency)
│   ├── models/                                  # Model implementations (LR, RF, SVR, MLP, GPR, PINN)
│   └── experiments/                             # Experiment runners (Benchmark, Noise, Small Data, Ablation)
├── run_experiments.py                           # Master experiment suite runner
├── requirements.txt                             # Dependencies
└── README.md
```

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run master evaluation suite:
   ```bash
   python run_experiments.py
   ```
