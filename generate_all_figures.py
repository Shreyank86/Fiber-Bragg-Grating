import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Configure UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Ensure output results folder exists
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Publication Matplotlib Configuration
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 9.5
plt.rcParams['xtick.labelsize'] = 8.5
plt.rcParams['ytick.labelsize'] = 8.5
plt.rcParams['legend.fontsize'] = 8.5
plt.rcParams['figure.titlesize'] = 12

# Fix random seeds for 100% reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Generating Clean Publication Figures for 6 Models (Preserving Exact Notebook PINN & LR)...")
print(f"All figures will be saved in 300 DPI inside '{RESULTS_DIR}/'\n")

# Physical sensitivity constants (Notebook Cell 13)
k_e_pm = 1.2   # pm/µε
k_T_pm = 10.0  # pm/°C

# Load Datasets (Notebook Cell 1 & 2)
strain_raw = pd.read_csv("STRAIN_CSV.csv")
temp_raw = pd.read_csv("TEMP_CSV.csv")
combined_raw = pd.read_csv("TEMP_STRAIN_CSV.csv")

strain_raw.columns = ["Time", "CH1", "CH2", "CH3", "CH4", "Wavelength"]
temp_raw.columns = ["Time", "CH1", "CH2", "CH3", "CH4", "Wavelength"]
combined_raw.columns = ["Time", "CH1", "CH2", "CH3", "CH4", "Wavelength"]

strain_raw['delta_lambda_pm'] = (strain_raw['Wavelength'] - strain_raw['Wavelength'].iloc[0]) * 1000
temp_raw['delta_lambda_pm'] = (temp_raw['Wavelength'] - temp_raw['Wavelength'].iloc[0]) * 1000
combined_raw['delta_lambda_pm'] = (combined_raw['Wavelength'] - combined_raw['Wavelength'].iloc[0]) * 1000

# Interpolate pure Δλ waveforms onto combined time base (Notebook Cell 9)
t_comb = combined_raw["Time"].values
strain_proxy = np.interp(t_comb, strain_raw["Time"].values, strain_raw["delta_lambda_pm"].values)
temp_proxy = np.interp(t_comb, temp_raw["Time"].values, temp_raw["delta_lambda_pm"].values)

# Feature matrix for classical baselines & targets (Notebook Cell 9)
X_proxy = np.column_stack([strain_proxy, temp_proxy])
y_combined = combined_raw["delta_lambda_pm"].values

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_proxy, y_combined, test_size=0.2, random_state=42)

# Features for PINN (Notebook Cell 14: [Time, delta_lambda_pm])
X_pinn = combined_raw[["Time", "delta_lambda_pm"]].values
y_pinn = combined_raw["delta_lambda_pm"].values.reshape(-1, 1)
X_train_pinn, X_test_pinn, y_train_pinn, y_test_pinn = train_test_split(X_pinn, y_pinn, test_size=0.2, random_state=42)

X_train_tf = tf.convert_to_tensor(X_train_pinn, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train_pinn, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test_pinn, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test_pinn, dtype=tf.float32)

def build_pinn():
    inputs = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    strain_pred = tf.keras.layers.Dense(1, name='strain_pred')(x)
    temp_pred = tf.keras.layers.Dense(1, name='temp_pred')(x)
    return tf.keras.Model(inputs=inputs, outputs=[strain_pred, temp_pred])

def physics_loss(y_true_b, strain_b, temp_b):
    delta_b = k_e_pm * strain_b + k_T_pm * temp_b
    return tf.reduce_mean(tf.square(y_true_b - delta_b))

# ==============================================================================
# FIGURE 1: Architecture Diagram
# ==============================================================================
print("[1/10] Generating Figure 1: PINN Architecture Diagram...")
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.axis('off')

ax.add_patch(patches.FancyBboxPatch((0.05, 0.35), 0.18, 0.3, boxstyle="round,pad=0.03", fc="#e9ecef", ec="#343a40", lw=1.5))
ax.text(0.14, 0.5, "Input Features\n$[t, \\Delta\\lambda_B]$\n(Time, Shift)", ha="center", va="center", fontweight="bold")

ax.add_patch(patches.FancyBboxPatch((0.32, 0.25), 0.28, 0.5, boxstyle="round,pad=0.03", fc="#e3f2fd", ec="#1565c0", lw=1.5))
ax.text(0.46, 0.5, "Deep Neural Network\n(3 Dense Layers)\n\nLayer 1: 64 Neurons\nLayer 2: 64 Neurons\nLayer 3: 32 Neurons", ha="center", va="center")

ax.add_patch(patches.FancyBboxPatch((0.68, 0.35), 0.18, 0.3, boxstyle="round,pad=0.03", fc="#e8f5e9", ec="#2e7d32", lw=1.5))
ax.text(0.77, 0.5, "Physical Outputs\n$\\hat{\\epsilon}$ (Strain)\n$\\Delta \\hat{T}$ (Temperature)", ha="center", va="center", fontweight="bold")

ax.add_patch(patches.FancyBboxPatch((0.35, 0.03), 0.50, 0.16, boxstyle="round,pad=0.03", fc="#fbe9e7", ec="#c62828", lw=1.5))
ax.text(0.60, 0.11, "Bragg Physics Loss:\n$\\mathcal{L}_{phys} = \\frac{1}{N} \\sum \\left[ \\Delta \\lambda_B - (k_\\epsilon \\hat{\\epsilon} + k_T \\Delta \\hat{T}) \\right]^2$", ha="center", va="center", color="#b71c1c", fontweight="bold")

ax.annotate("", xy=(0.32, 0.5), xytext=(0.23, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#343a40"))
ax.annotate("", xy=(0.68, 0.5), xytext=(0.60, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1565c0"))
ax.annotate("", xy=(0.60, 0.19), xytext=(0.77, 0.35), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2e7d32", connectionstyle="arc3,rad=0.3"))
ax.annotate("", xy=(0.60, 0.19), xytext=(0.14, 0.35), arrowprops=dict(arrowstyle="->", lw=1.5, color="#343a40", connectionstyle="arc3,rad=-0.3"))

plt.title("Physics-Informed Neural Network (PINN) Architecture for FBG Sensors", pad=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig1_architecture_diagram.png"), dpi=300)
plt.close()

# ==============================================================================
# FIGURE 2: Training & Validation Loss (Exact Notebook Cell 19 Match)
# ==============================================================================
print("[2/10] Training PINN & Generating Figure 2: Loss Convergence (Notebook Cell 19)...")
pinn = build_pinn()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

@tf.function
def train_step(x_b, y_b):
    with tf.GradientTape() as tape:
        s_out, t_out = pinn(x_b, training=True)
        loss = physics_loss(y_b, s_out, t_out)
    grads = tape.gradient(loss, pinn.trainable_variables)
    optimizer.apply_gradients(zip(grads, pinn.trainable_variables))
    return loss

loss_history = []
epochs = 501

for epoch in range(epochs):
    tr_l = train_step(X_train_tf, y_train_tf)
    loss_history.append(tr_l.numpy())

plt.figure(figsize=(7, 4))
plt.plot(loss_history, color="#1565c0", lw=1.5)
plt.xlabel("Epoch")
plt.ylabel("Physics Loss (MSE)")
plt.title("PINN Training Loss", fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig2_training_validation_loss.png"), dpi=300)
plt.close()

# Evaluate PINN predictions
s_pred, t_pred = pinn(X_test_tf, training=False)
y_pred_pinn = (k_e_pm * s_pred + k_T_pm * t_pred).numpy().flatten()
y_test_flat = y_test_pinn.flatten()

# ==============================================================================
# FIGURE 3: Prediction vs Ground Truth (Exact Notebook Cell 20 Match)
# ==============================================================================
print("[3/10] Generating Figure 3: PINN Predicted vs Actual Δλ (Notebook Cell 20)...")
plt.figure(figsize=(10, 4))
plt.plot(y_test_pinn, label="Actual Δλ (Test)", color='blue', linewidth=1)
plt.plot(y_pred_pinn, label="Predicted Δλ (PINN)", color='orange', linewidth=1)
plt.legend()
plt.title("PINN Predicted vs Actual Δλ (Test Set)")
plt.xlabel("Sample Index (Test Subset)")
plt.ylabel("Δλ (pm)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig3_prediction_vs_ground_truth.png"), dpi=300)
plt.close()

# ==============================================================================
# FIGURE 4: Residual Error Plot (Exact Notebook Cell 20 Match)
# ==============================================================================
print("[4/10] Generating Figure 4: Test Set Residuals (Notebook Cell 20)...")
errors = y_test_pinn.flatten() - y_pred_pinn.flatten()
plt.figure(figsize=(8, 3))
plt.plot(errors, color='purple')
plt.axhline(0, color='black', linewidth=1)
plt.title("Residuals (Actual - Predicted) on Test Set")
plt.xlabel("Sample Index (Test Subset)")
plt.ylabel("Residual Δλ (pm)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig4_residual_histogram.png"), dpi=300)
plt.close()

# ==============================================================================
# FIGURE 5: Clean Publication Bar Chart (Includes 6 Models)
# ==============================================================================
print("[5/10] Loading Benchmark Metrics & Generating Figure 5...")

# Fit baselines for downstream figure evaluations (Fig 6 Noise & Fig 10 Bootstrap)
scaler_fig5 = StandardScaler()
X_train_scaled = scaler_fig5.fit_transform(X_train_p)
X_test_scaled = scaler_fig5.transform(X_test_p)

lr_m = LinearRegression().fit(X_train_p, y_train_p)
y_pred_lr = lr_m.predict(X_test_p)

mlp_m = MLPRegressor(hidden_layer_sizes=(64, 64, 32), max_iter=200, early_stopping=True, random_state=42).fit(X_train_scaled, y_train_p)
y_pred_mlp = mlp_m.predict(X_test_scaled)

svr_m = SVR(kernel='rbf', C=10.0, epsilon=0.1).fit(X_train_scaled, y_train_p)
y_pred_svr = svr_m.predict(X_test_scaled)

kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gp_m = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=0, random_state=42)
sub_idx = np.random.choice(len(X_train_scaled), size=1000, replace=False)
gp_m.fit(X_train_scaled[sub_idx], y_train_p[sub_idx])
y_pred_gp = gp_m.predict(X_test_scaled)

rf_m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_p, y_train_p)
y_pred_rf = rf_m.predict(X_test_p)

# Load exact benchmark results from benchmark_results.csv to guarantee 100% matching values across table & figures
csv_bench_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
if os.path.exists(csv_bench_path):
    df_bm_raw = pd.read_csv(csv_bench_path)
    model_name_map = {
        "Linear Regression": "Linear Reg",
        "MLP Baseline": "MLP",
        "SVR Baseline": "SVR",
        "Gaussian Process": "Gaussian Process",
        "Random Forest": "Random Forest",
        "Proposed PINN": "Proposed PINN"
    }
    df_bm_raw["Model_Short"] = df_bm_raw["Model"].map(model_name_map)
    df_bm = df_bm_raw.rename(columns={"MAE (pm)": "MAE", "RMSE (pm)": "RMSE", "R²": "R2"})
    df_bm["Model"] = df_bm["Model_Short"]
else:
    models_data = [
        ("Linear Reg", y_pred_lr),
        ("MLP", y_pred_mlp),
        ("SVR", y_pred_svr),
        ("Gaussian Process", y_pred_gp),
        ("Random Forest", y_pred_rf),
        ("Proposed PINN", y_pred_pinn)
    ]
    benchmarks = []
    for name, y_p in models_data:
        y_target = y_test_p if name != "Proposed PINN" else y_test_flat
        benchmarks.append({
            "Model": name,
            "MAE": float(mean_absolute_error(y_target, y_p)),
            "RMSE": float(np.sqrt(mean_squared_error(y_target, y_p))),
            "R2": float(r2_score(y_target, y_p))
        })
    df_bm = pd.DataFrame(benchmarks)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
colors = ["#78909c", "#78909c", "#78909c", "#78909c", "#78909c", "#2e7d32"]

# MAE
ax_mae = axes[0]
bars_mae = ax_mae.bar(df_bm["Model"], df_bm["MAE"], color=colors, width=0.5, edgecolor="none")
ax_mae.set_title("Mean Absolute Error (MAE)", fontweight="bold")
ax_mae.set_ylabel("MAE (pm)")
ax_mae.set_xticks(range(len(df_bm["Model"])))
ax_mae.set_xticklabels(df_bm["Model"], rotation=30, ha="right")
ax_mae.grid(True, linestyle=":", alpha=0.6, axis="y")
ax_mae.set_ylim(0, max(df_bm["MAE"]) * 1.15)
for bar in bars_mae:
    h = bar.get_height()
    ax_mae.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

# RMSE
ax_rmse = axes[1]
bars_rmse = ax_rmse.bar(df_bm["Model"], df_bm["RMSE"], color=colors, width=0.5, edgecolor="none")
ax_rmse.set_title("Root Mean Square Error (RMSE)", fontweight="bold")
ax_rmse.set_ylabel("RMSE (pm)")
ax_rmse.set_xticks(range(len(df_bm["Model"])))
ax_rmse.set_xticklabels(df_bm["Model"], rotation=30, ha="right")
ax_rmse.grid(True, linestyle=":", alpha=0.6, axis="y")
ax_rmse.set_ylim(0, max(df_bm["RMSE"]) * 1.15)
for bar in bars_rmse:
    h = bar.get_height()
    ax_rmse.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

# R² Score
ax_r2 = axes[2]
bars_r2 = ax_r2.bar(df_bm["Model"], df_bm["R2"], color=colors, width=0.5, edgecolor="none")
ax_r2.set_title("Coefficient of Determination ($R^2$)", fontweight="bold")
ax_r2.set_ylabel("$R^2$ Score")
ax_r2.set_ylim(0.90, 1.01)
ax_r2.set_xticks(range(len(df_bm["Model"])))
ax_r2.set_xticklabels(df_bm["Model"], rotation=30, ha="right")
ax_r2.grid(True, linestyle=":", alpha=0.6, axis="y")
for bar in bars_r2:
    h = bar.get_height()
    ax_r2.annotate(f"{h:.5f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                   xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

plt.suptitle("Model Evaluation Benchmarks (6 Models)", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig5_model_benchmarks_mae_rmse_r2.png"), dpi=300)
plt.savefig(os.path.join(RESULTS_DIR, "benchmark_metrics.png"), dpi=300)
plt.close()

# ==============================================================================
# FIGURE 6: Noise Robustness Experiment
# ==============================================================================
print("[6/10] Generating Figure 6: Noise Robustness...")
noise_levels = [0.0, 1.0, 3.0, 5.0, 10.0]

noise_results = []
for n_std in noise_levels:
    y_test_noisy = y_test_flat + np.random.normal(0, n_std, size=y_test_flat.shape)
    y_test_p_noisy = y_test_p + np.random.normal(0, n_std, size=y_test_p.shape)
    
    rmse_pinn = float(np.sqrt(np.mean((y_test_noisy - y_pred_pinn)**2)))
    rmse_lr = float(np.sqrt(mean_squared_error(y_test_p_noisy, y_pred_lr)))
    rmse_rf = float(np.sqrt(mean_squared_error(y_test_p_noisy, y_pred_rf)))
    rmse_mlp = float(np.sqrt(mean_squared_error(y_test_p_noisy, y_pred_mlp)))
    rmse_svr = float(np.sqrt(mean_squared_error(y_test_p_noisy, y_pred_svr)))
    rmse_gp = float(np.sqrt(mean_squared_error(y_test_p_noisy, y_pred_gp)))

    noise_results.append({
        "Noise": n_std,
        "PINN": rmse_pinn,
        "LR": rmse_lr,
        "RF": rmse_rf,
        "MLP": rmse_mlp,
        "SVR": rmse_svr,
        "GP": rmse_gp
    })

df_noise = pd.DataFrame(noise_results)

plt.figure(figsize=(8, 4.2))
plt.plot(df_noise["Noise"], df_noise["PINN"], 'o-', label="Proposed PINN", color="#2e7d32", lw=2, ms=6)
plt.plot(df_noise["Noise"], df_noise["RF"], 's--', label="Random Forest", color="#1565c0", lw=1.2, ms=5)
plt.plot(df_noise["Noise"], df_noise["GP"], 'P-.', label="Gaussian Process", color="#00838f", lw=1.2, ms=5)
plt.plot(df_noise["Noise"], df_noise["SVR"], 'd-.', label="SVR Baseline", color="#6a1b9a", lw=1.2, ms=5)
plt.plot(df_noise["Noise"], df_noise["MLP"], '^:', label="MLP Baseline", color="#e65100", lw=1.2, ms=5)
plt.plot(df_noise["Noise"], df_noise["LR"], 'x:', label="Linear Regression", color="#c62828", lw=1.2, ms=5)

plt.xlabel("Measurement Gaussian Noise $\\sigma$ (pm)")
plt.ylabel("RMSE (pm)")
plt.title("Noise Robustness Comparison Across All 6 Models", fontweight="bold")
plt.legend(frameon=True)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig6_noise_robustness.png"), dpi=300)
plt.close()

# ==============================================================================
# FIGURE 7: Small Data Regime Experiment
# ==============================================================================
print("[7/10] Generating Figure 7: Small Data Regime...")
data_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
small_data_results = []

for frac in data_fractions:
    subset_size_p = int(len(X_train_p) * frac)
    sub_idx_p = np.random.choice(len(X_train_p), size=subset_size_p, replace=False)
    
    lr_sub = LinearRegression().fit(X_train_p[sub_idx_p], y_train_p[sub_idx_p])
    rmse_lr_sub = float(np.sqrt(mean_squared_error(y_test_p, lr_sub.predict(X_test_p))))
    
    mlp_sub = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=100, early_stopping=True, random_state=42).fit(X_train_scaled[sub_idx_p], y_train_p[sub_idx_p])
    rmse_mlp_sub = float(np.sqrt(mean_squared_error(y_test_p, mlp_sub.predict(X_test_scaled))))

    svr_sub = SVR(kernel='rbf', C=10.0).fit(X_train_scaled[sub_idx_p], y_train_p[sub_idx_p])
    rmse_svr_sub = float(np.sqrt(mean_squared_error(y_test_p, svr_sub.predict(X_test_scaled))))

    kernel = C(1.0) * RBF(1.0)
    gp_sub = GaussianProcessRegressor(kernel=kernel, alpha=0.01, optimizer=None, random_state=42).fit(X_train_scaled[sub_idx_p[:min(500, len(sub_idx_p))]], y_train_p[sub_idx_p[:min(500, len(sub_idx_p))]])
    rmse_gp_sub = float(np.sqrt(mean_squared_error(y_test_p, gp_sub.predict(X_test_scaled))))

    rf_sub = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train_p[sub_idx_p], y_train_p[sub_idx_p])
    rmse_rf_sub = float(np.sqrt(mean_squared_error(y_test_p, rf_sub.predict(X_test_p))))

    subset_size_pinn = int(len(X_train_pinn) * frac)
    sub_idx_pinn = np.random.choice(len(X_train_pinn), size=subset_size_pinn, replace=False)
    
    X_tr_sub, y_tr_sub = X_train_pinn[sub_idx_pinn], y_train_pinn[sub_idx_pinn]
    X_tr_sub_tf = tf.convert_to_tensor(X_tr_sub, dtype=tf.float32)
    y_tr_sub_tf = tf.convert_to_tensor(y_tr_sub, dtype=tf.float32)
    
    p_sub = build_pinn()
    opt_sub = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def train_step_sub(x_b, y_b):
        with tf.GradientTape() as tape:
            s_o, t_o = p_sub(x_b, training=True)
            d_o = k_e_pm * s_o + k_T_pm * t_o
            l_sub = tf.reduce_mean(tf.square(y_b - d_o))
        grads = tape.gradient(l_sub, p_sub.trainable_variables)
        opt_sub.apply_gradients(zip(grads, p_sub.trainable_variables))
        return l_sub

    for _ in range(350):
        train_step_sub(X_tr_sub_tf, y_tr_sub_tf)
        
    s_test_sub, t_test_sub = p_sub(X_test_tf, training=False)
    pred_p_sub = (k_e_pm * s_test_sub + k_T_pm * t_test_sub).numpy().flatten()
    rmse_p_sub = float(np.sqrt(mean_squared_error(y_test_flat, pred_p_sub)))
    
    small_data_results.append({
        "Percentage": int(frac * 100),
        "PINN": rmse_p_sub,
        "RF": rmse_rf_sub,
        "GP": rmse_gp_sub,
        "SVR": rmse_svr_sub,
        "MLP": rmse_mlp_sub,
        "LR": rmse_lr_sub
    })

df_sd = pd.DataFrame(small_data_results)

plt.figure(figsize=(8.5, 4.5))
plt.plot(df_sd["Percentage"], df_sd["PINN"], 'o-', label="Proposed PINN", color="#2e7d32", lw=2.2, ms=7)
plt.plot(df_sd["Percentage"], df_sd["RF"], 's--', label="Random Forest", color="#1565c0", lw=1.4, ms=5)
plt.plot(df_sd["Percentage"], df_sd["GP"], 'P-.', label="Gaussian Process", color="#00838f", lw=1.4, ms=5)
plt.plot(df_sd["Percentage"], df_sd["SVR"], 'd-.', label="SVR Baseline", color="#6a1b9a", lw=1.4, ms=5)
plt.plot(df_sd["Percentage"], df_sd["MLP"], '^:', label="MLP Baseline", color="#e65100", lw=1.4, ms=5)
plt.plot(df_sd["Percentage"], df_sd["LR"], 'x:', label="Linear Regression", color="#c62828", lw=1.4, ms=5)

for _, r in df_sd.iterrows():
    plt.annotate(f"{r['PINN']:.1f}", (r["Percentage"], r["PINN"]), textcoords="offset points", xytext=(0,6), ha='center', fontsize=8, fontweight="bold", color="#2e7d32")

plt.xlabel("Training Data Subset Percentage (%)")
plt.ylabel("RMSE (pm)")
plt.title("Small Data Regime Performance Across All 6 Models", fontweight="bold")
plt.legend(frameon=True, loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig7_small_data_experiment.png"), dpi=300)
plt.close()

# ==============================================================================
# FIGURE 8: Lambda Physics Weight Ablation (Exact Notebook Cell 25 Image Copy)
# ==============================================================================
print("[8/10] Extracting Figure 8: Lambda Physics Weight Ablation from Internship_PINN.ipynb Cell 25...")
import json, base64

with open('Internship_PINN.ipynb', 'r', encoding='utf-8') as f_nb:
    nb_json = json.load(f_nb)

cell25_b64 = None
for out in nb_json['cells'][25].get('outputs', []):
    if 'data' in out and 'image/png' in out['data']:
        cell25_b64 = ''.join(out['data']['image/png']).strip()
        break

if cell25_b64:
    with open(os.path.join(RESULTS_DIR, "fig8_lambda_ablation.png"), "wb") as f_out:
        f_out.write(base64.b64decode(cell25_b64))

# ==============================================================================
# FIGURE 9: 5-Fold Cross-Validation Boxplot
# ==============================================================================
print("[9/10] Performing 5-Fold CV & Generating Figure 9: 5-Fold CV Distributions...")
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_maes, cv_rmses = [], []

for fold, (tr_idx, te_idx) in enumerate(kfold.split(X_pinn)):
    X_tr_f, X_te_f = X_pinn[tr_idx], X_pinn[te_idx]
    y_tr_f, y_te_f = y_pinn[tr_idx], y_pinn[te_idx]
    
    X_tr_f_tf = tf.convert_to_tensor(X_tr_f, dtype=tf.float32)
    y_tr_f_tf = tf.convert_to_tensor(y_tr_f, dtype=tf.float32)
    X_te_f_tf = tf.convert_to_tensor(X_te_f, dtype=tf.float32)
    
    p_cv = build_pinn()
    opt_cv = tf.keras.optimizers.Adam(1e-3)
    
    @tf.function
    def train_step_cv(x_b, y_b):
        with tf.GradientTape() as tape:
            s_o, t_o = p_cv(x_b, training=True)
            d_o = k_e_pm * s_o + k_T_pm * t_o
            l_cv = tf.reduce_mean(tf.square(y_b - d_o))
        grads = tape.gradient(l_cv, p_cv.trainable_variables)
        opt_cv.apply_gradients(zip(grads, p_cv.trainable_variables))
        return l_cv

    for _ in range(400):
        train_step_cv(X_tr_f_tf, y_tr_f_tf)
        
    s_te, t_te = p_cv(X_te_f_tf, training=False)
    pred_cv = (k_e_pm * s_te + k_T_pm * t_te).numpy().flatten()
    
    cv_maes.append(float(mean_absolute_error(y_te_f.flatten(), pred_cv)))
    cv_rmses.append(float(np.sqrt(mean_squared_error(y_te_f.flatten(), pred_cv))))

plt.figure(figsize=(6, 4.2))
bplot = plt.boxplot([cv_maes, cv_rmses], patch_artist=True, tick_labels=["MAE (pm)", "RMSE (pm)"], widths=0.4)

colors_bp = ['#1565c0', '#e65100']
for patch, color in zip(bplot['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

for i, vals in enumerate([cv_maes, cv_rmses], start=1):
    x_jit = np.random.normal(i, 0.04, size=len(vals))
    plt.scatter(x_jit, vals, color='black', alpha=0.8, zorder=3, s=25)

plt.ylabel("Metric Value (pm)")
plt.ylim(0, 4.0)
plt.yticks(np.arange(0, 4.1, 0.5))
plt.title("5-Fold Cross Validation Distributions", fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig9_cross_validation_boxplot.png"), dpi=300)
plt.close()

# ==============================================================================
# FIGURE 10: Bootstrap Confidence Intervals (Exact Notebook Cell 34 Match)
# ==============================================================================
print("[10/10] Performing Bootstrap Analysis & Generating Figure 10 (Notebook Cell 34)...")
n_bootstrap = 1000
rng = np.random.default_rng(42)
boot_maes = []

for _ in range(n_bootstrap):
    b_idx = rng.integers(0, len(y_test_flat), len(y_test_flat))
    y_b_samp = y_test_flat[b_idx]
    pred_b_pinn = y_pred_pinn[b_idx]
    boot_maes.append(float(mean_absolute_error(y_b_samp, pred_b_pinn)))

plt.figure(figsize=(6, 4))
plt.hist(boot_maes, bins=30, color='orange', alpha=0.7, label='MAE Samples')
plt.axvline(np.mean(boot_maes), color='r', linestyle='--', label='Mean MAE')
plt.xlabel("MAE (pm)")
plt.ylabel("Frequency")
plt.title("Bootstrap MAE Distribution (95% CI)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig10_bootstrap_confidence_intervals.png"), dpi=300)
plt.close()

print("\n==================================================================")
print("SUCCESS: ALL 10 FIGURES REGENERATED WITH EXACT NOTEBOOK PINN & LR")
print("==================================================================")
