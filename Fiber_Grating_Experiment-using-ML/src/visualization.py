import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_gpr_uncertainty(y_test, y_pred_gpr, y_std_gpr, sample_range=100, save_path=None):
    """Plots GPR prediction vs Ground Truth with shaded 95% Confidence Bounds (+/- 1.96 * sigma)."""
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "gpr_uncertainty_quantification.png")
        
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    idx = np.arange(sample_range)
    
    # Temperature Target
    axes[0].plot(idx, y_test[:sample_range, 0], 'k--', label='Ground Truth (Temperature °C)', linewidth=1.8)
    axes[0].plot(idx, y_pred_gpr[:sample_range, 0], 'b-', label='GPR Mean Prediction', linewidth=1.5)
    axes[0].fill_between(
        idx,
        y_pred_gpr[:sample_range, 0] - 1.96 * y_std_gpr[:sample_range, 0],
        y_pred_gpr[:sample_range, 0] + 1.96 * y_std_gpr[:sample_range, 0],
        color='blue', alpha=0.25, label='95% Confidence Interval (±1.96σ)'
    )
    axes[0].set_ylabel("Temperature (°C)", fontsize=11)
    axes[0].set_title("Gaussian Process Regression (GPR) — Temperature Uncertainty Quantification", fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Strain Target
    axes[1].plot(idx, y_test[:sample_range, 1], 'k--', label='Ground Truth (Strain µε)', linewidth=1.8)
    axes[1].plot(idx, y_pred_gpr[:sample_range, 1], 'r-', label='GPR Mean Prediction', linewidth=1.5)
    axes[1].fill_between(
        idx,
        y_pred_gpr[:sample_range, 1] - 1.96 * y_std_gpr[:sample_range, 1],
        y_pred_gpr[:sample_range, 1] + 1.96 * y_std_gpr[:sample_range, 1],
        color='red', alpha=0.25, label='95% Confidence Interval (±1.96σ)'
    )
    axes[1].set_xlabel("Sample Index", fontsize=11)
    axes[1].set_ylabel("Strain (µε)", fontsize=11)
    axes[1].set_title("Gaussian Process Regression (GPR) — Strain Uncertainty Quantification", fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")

def plot_noise_robustness(df_noise, save_path=None):
    """Plots Noise Robustness comparison (RMSE and R2 vs Gaussian Noise Level) across ALL 6 models."""
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "noise_robustness_curves.png")
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    noise_labels = df_noise["Noise_Level"].values
    
    models = ["LR", "RF", "SVR", "MLP", "GPR", "PINN"]
    colors = ['gray', 'green', 'purple', 'crimson', 'blue', 'teal']
    markers = ['v', '^', 'D', 'o', 'x', 's']

    for i, m in enumerate(models):
        if f"{m}_RMSE" in df_noise.columns:
            lw = 2.5 if m == "PINN" else 1.5
            axes[0].plot(noise_labels, df_noise[f"{m}_RMSE"], label=m, color=colors[i], marker=markers[i], linewidth=lw)
            axes[1].plot(noise_labels, df_noise[f"{m}_R2"], label=m, color=colors[i], marker=markers[i], linewidth=lw)
            
    axes[0].set_title("Noise Degradation: RMSE vs Noise %", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Gaussian Measurement Noise Level", fontsize=11)
    axes[0].set_ylabel("Root Mean Squared Error (RMSE)", fontsize=11)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].set_title("Noise Robustness: R² Score (%) vs Noise %", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Gaussian Measurement Noise Level", fontsize=11)
    axes[1].set_ylabel("R² Score (%)", fontsize=11)
    axes[1].legend(loc='lower left')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")

def plot_small_data_regime(df_small, save_path=None):
    """Plots Small Data Regime performance (RMSE and R2 vs Training Data %) across ALL 6 models."""
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "small_data_regime_curves.png")
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    frac_labels = df_small["Data_Fraction"].values
    
    models = ["LR", "RF", "SVR", "MLP", "GPR", "PINN"]
    colors = ['gray', 'green', 'purple', 'orange', 'blue', 'darkgreen']
    markers = ['v', '^', 'D', 'o', 'x', 's']

    for i, m in enumerate(models):
        if f"{m}_RMSE" in df_small.columns:
            lw = 2.5 if m == "PINN" else 1.5
            axes[0].plot(frac_labels, df_small[f"{m}_RMSE"], label=m, color=colors[i], marker=markers[i], linewidth=lw)
            axes[1].plot(frac_labels, df_small[f"{m}_R2"], label=m, color=colors[i], marker=markers[i], linewidth=lw)

    axes[0].set_title("Data Efficiency: RMSE vs Training Data %", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Percentage of Training Data Used", fontsize=11)
    axes[0].set_ylabel("Root Mean Squared Error (RMSE)", fontsize=11)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].set_title("Data Efficiency: R² Score (%) vs Training Data %", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Percentage of Training Data Used", fontsize=11)
    axes[1].set_ylabel("R² Score (%)", fontsize=11)
    axes[1].legend(loc='lower right')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")

def plot_ablation_study(df_ablation, save_path=None):
    """Plots Physics Weight Lambda Ablation curves."""
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "ablation_study_curves.png")
        
    fig, ax1 = plt.subplots(figsize=(8, 5))
    lambdas = df_ablation["Lambda_Phys"].astype(str).values
    
    color1 = 'tab:blue'
    ax1.set_xlabel("Physics Loss Weight (λ_phys)", fontsize=11)
    ax1.set_ylabel("Root Mean Squared Error (RMSE)", color=color1, fontsize=11)
    ax1.plot(lambdas, df_ablation["RMSE"], color=color1, marker='o', linewidth=2.5, label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel("R² Score (%)", color=color2, fontsize=11)
    ax2.plot(lambdas, df_ablation["R2"], color=color2, marker='s', linestyle='--', linewidth=2, label='R² Score')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title("Physics Loss Weight (λ_phys) Ablation Study", fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")

def plot_prediction_residuals(y_test, y_pred_pinn, y_pred_gpr, save_path=None):
    """Plots Residual Error Histograms for PINN and GPR."""
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "residual_histograms.png")
        
    res_pinn_temp = y_test[:, 0] - y_pred_pinn[:, 0]
    res_pinn_strain = y_test[:, 1] - y_pred_pinn[:, 1]
    
    res_gpr_temp = y_test[:, 0] - y_pred_gpr[:, 0]
    res_gpr_strain = y_test[:, 1] - y_pred_gpr[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    sns.histplot(res_pinn_temp, ax=axes[0, 0], kde=True, color='teal')
    axes[0, 0].set_title("PINN Temperature Residuals (°C)", fontweight='bold')
    
    sns.histplot(res_pinn_strain, ax=axes[0, 1], kde=True, color='teal')
    axes[0, 1].set_title("PINN Strain Residuals (µε)", fontweight='bold')
    
    sns.histplot(res_gpr_temp, ax=axes[1, 0], kde=True, color='navy')
    axes[1, 0].set_title("GPR Temperature Residuals (°C)", fontweight='bold')
    
    sns.histplot(res_gpr_strain, ax=axes[1, 1], kde=True, color='navy')
    axes[1, 1].set_title("GPR Strain Residuals (µε)", fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")

def plot_cross_validation_boxplot(fold_results, save_path=None):
    """Plots 5-Fold Cross Validation R2 Boxplot across all models."""
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "cross_validation_boxplot.png")
        
    data_list = []
    for model_name, metrics in fold_results.items():
        for r2 in metrics["R2"]:
            data_list.append({"Model": model_name, "R2_Score": r2})
            
    df_box = pd.DataFrame(data_list)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Model", y="R2_Score", data=df_box, hue="Model", palette="Set2", legend=False)
    sns.stripplot(x="Model", y="R2_Score", data=df_box, color='black', alpha=0.6, jitter=True)
    
    plt.title("5-Fold Cross Validation R² Score Distribution Across Models", fontsize=12, fontweight='bold')
    plt.ylabel("R² Score (%)", fontsize=11)
    plt.xlabel("Machine Learning & PINN Models", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")

def plot_parity_plots(y_test, y_pred_pinn, y_pred_mlp, save_path=None):
    """Plots Parity Scatter Plots (Predicted vs Ground Truth) for Temperature and Strain."""
    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "prediction_vs_ground_truth.png")
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Temperature Parity
    axes[0].scatter(y_test[:, 0], y_pred_mlp[:, 0], alpha=0.5, label='Multi-Layer Perceptron', color='orange', s=20)
    axes[0].scatter(y_test[:, 0], y_pred_pinn[:, 0], alpha=0.5, label='PINN (Proposed)', color='teal', s=20)
    axes[0].plot([20, 80], [20, 80], 'k--', label='Ideal 1:1 Line')
    axes[0].set_title("Temperature Parity Plot (°C)", fontweight='bold')
    axes[0].set_xlabel("Ground Truth (°C)")
    axes[0].set_ylabel("Predicted (°C)")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Strain Parity
    axes[1].scatter(y_test[:, 1], y_pred_mlp[:, 1], alpha=0.5, label='Multi-Layer Perceptron', color='orange', s=20)
    axes[1].scatter(y_test[:, 1], y_pred_pinn[:, 1], alpha=0.5, label='PINN (Proposed)', color='teal', s=20)
    axes[1].plot([0, 1000], [0, 1000], 'k--', label='Ideal 1:1 Line')
    axes[1].set_title("Strain Parity Plot (µε)", fontweight='bold')
    axes[1].set_xlabel("Ground Truth (µε)")
    axes[1].set_ylabel("Predicted (µε)")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure: {save_path}")
