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
    
    # 3. Small Data Regime
    df_small = run_small_data_experiment()
    df_small.to_csv(os.path.join(TABLES_DIR, "small_data_regime.csv"), index=False)
    
    # 4. Ablation Study
    df_ablation = run_ablation_study()
    df_ablation.to_csv(os.path.join(TABLES_DIR, "ablation_study.csv"), index=False)
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results exported to: {RESULTS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
