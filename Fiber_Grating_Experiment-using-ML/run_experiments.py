"""
=============================================================================
Master Project Execution Script — FBG Sensor ML + PINN Framework
-----------------------------------------------------------------------------
Executes complete experimental suite across all 6 models:
1. Standard Benchmark (LR, RF, SVR, MLP, GPR, PINN)
2. Noise Robustness Evaluation (1% - 10% Noise)
3. Small Data Regime Evaluation (20% - 100% Data)
4. Physics Weight Lambda Ablation Study
=============================================================================
"""

import os
import pandas as pd
from src.experiments.benchmark import run_benchmark
from src.experiments.noise_robustness import run_noise_robustness_experiment
from src.experiments.small_data import run_small_data_experiment
from src.experiments.ablation import run_ablation_study
from src.experiments.statistical_validation import run_cross_validation, run_bootstrap_analysis
from src.visualization import (
    plot_gpr_uncertainty,
    plot_noise_robustness,
    plot_small_data_regime,
    plot_ablation_study,
    plot_prediction_residuals
)
from src.data_loader import load_fbg_dataset, get_scaled_train_test_split
from src.models.gpr import GPRModel
from src.models.pinn import PINNModel

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("STARTING FBG SENSOR ML + PINN MASTER EXPERIMENTAL SUITE")
    print("=" * 70)
    
    # 1. Master Benchmark
    df_benchmark = run_benchmark()
    df_benchmark.to_csv(os.path.join(TABLES_DIR, "benchmark_comparison.csv"), index=False)
    
    # 2. Noise Robustness
    df_noise = run_noise_robustness_experiment()
    df_noise.to_csv(os.path.join(TABLES_DIR, "noise_robustness.csv"), index=False)
    plot_noise_robustness(df_noise)
    
    # 3. Small Data Regime
    df_small = run_small_data_experiment()
    df_small.to_csv(os.path.join(TABLES_DIR, "small_data_regime.csv"), index=False)
    plot_small_data_regime(df_small)
    
    # 4. Ablation Study
    df_ablation = run_ablation_study()
    df_ablation.to_csv(os.path.join(TABLES_DIR, "ablation_study.csv"), index=False)
    plot_ablation_study(df_ablation)
    
    # 5. Statistical Validation (Cross-Validation & Bootstrap)
    run_cross_validation(n_splits=3)
    run_bootstrap_analysis(n_bootstraps=100)
    
    # 6. GPR Uncertainty & Residual Visualization
    print("\nGenerating Publication Figures for GPR Uncertainty & Residuals...")
    X, y = load_fbg_dataset("both")
    X_tr_s, X_te_s, y_tr, y_te, _ = get_scaled_train_test_split(X, y, scale=True)
    
    gpr = GPRModel().fit(X_tr_s, y_tr)
    y_pred_gpr, y_std_gpr, _, _ = gpr.predict_with_uncertainty(X_te_s)
    plot_gpr_uncertainty(y_te, y_pred_gpr, y_std_gpr)
    
    pinn = PINNModel(epochs=100, batch_size=128).fit(X_tr_s, y_tr)
    y_pred_pinn = pinn.predict(X_te_s)
    plot_prediction_residuals(y_te, y_pred_pinn, y_pred_gpr)
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS & FIGURE GENERATIONS COMPLETED SUCCESSFULLY!")
    print(f"Results & Figures exported to: {RESULTS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
