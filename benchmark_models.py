import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

sys.stdout.reconfigure(encoding='utf-8')

# Ensure output results folder exists
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fix random seeds for 100% exact reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Starting Benchmarking (Exact Notebook PINN & LR from Internship_PINN.ipynb + Baselines)...")

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

# Feature matrix for classical models & targets (Notebook Cell 9)
X_proxy = np.column_stack([strain_proxy, temp_proxy])
y_combined = combined_raw["delta_lambda_pm"].values

# Train-test split for classical models (80/20, seed=42)
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_proxy, y_combined, test_size=0.2, random_state=42)

# Features for PINN (Notebook Cell 14: [Time, delta_lambda_pm])
X_pinn = combined_raw[["Time", "delta_lambda_pm"]].values
y_pinn = combined_raw["delta_lambda_pm"].values.reshape(-1, 1)
X_train_pinn, X_test_pinn, y_train_pinn, y_test_pinn = train_test_split(X_pinn, y_pinn, test_size=0.2, random_state=42)

# StandardScaler for distance-sensitive baselines (SVR, MLP, GP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_p)
X_test_scaled = scaler.transform(X_test_p)

results = []

def get_metrics(name, y_t,
 y_p, t_tr, t_inf, s_pred=None, t_pred=None):
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    
    if s_pred is not None and t_pred is not None:
        phys_reconstr = k_e_pm * s_pred + k_T_pm * t_pred
        phys_res = float(np.mean(np.abs(y_t.flatten() - phys_reconstr.flatten())))
    else:
        phys_res = float(mae)
        
    return {
        "Model": name,
        "MAE (pm)": round(float(mae), 4),
        "RMSE (pm)": round(float(rmse), 4),
        "R²": round(float(r2), 5),
        "Physics Residual (pm)": round(float(phys_res), 4),
        "Training Time (s)": round(float(t_tr), 4),
        "Inference Time (s)": round(float(t_inf), 4)
    }

# --- Model 1: Linear Regression (Exact Notebook Cell 9) ---
print("\n[1/6] Training Notebook Linear Regression Baseline (Cell 9)...")
t0 = time.time()
lr_model = LinearRegression()
lr_model.fit(X_train_p, y_train_p)
t_train_lr = time.time() - t0

t0 = time.time()
y_pred_lr = lr_model.predict(X_test_p)
t_infer_lr = time.time() - t0
results.append(get_metrics("Linear Regression", y_test_p, y_pred_lr, t_train_lr, t_infer_lr))

# --- Model 2: Multi-Layer Perceptron Baseline ---
print("[2/6] Training Multi-Layer Perceptron (MLP Baseline)...")
t0 = time.time()
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 64, 32), max_iter=200, early_stopping=True, random_state=42)
mlp_model.fit(X_train_scaled, y_train_p)
t_train_mlp = time.time() - t0

t0 = time.time()
y_pred_mlp = mlp_model.predict(X_test_scaled)
t_infer_mlp = time.time() - t0
results.append(get_metrics("MLP Baseline", y_test_p, y_pred_mlp, t_train_mlp, t_infer_mlp))

# --- Model 3: Support Vector Regression Baseline ---
print("[3/6] Training Support Vector Regression (SVR Baseline)...")
t0 = time.time()
svr_model = SVR(kernel='rbf', C=10.0, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train_p)
t_train_svr = time.time() - t0

t0 = time.time()
y_pred_svr = svr_model.predict(X_test_scaled)
t_infer_svr = time.time() - t0
results.append(get_metrics("SVR Baseline", y_test_p, y_pred_svr, t_train_svr, t_infer_svr))

# --- Model 4: Gaussian Process Regression Baseline ---
print("[4/6] Training Gaussian Process Regression Baseline (StandardScaler Normalization)...")
t0 = time.time()
kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gp_model = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=3, random_state=42)
sub_idx = np.random.choice(len(X_train_scaled), size=1000, replace=False)
gp_model.fit(X_train_scaled[sub_idx], y_train_p[sub_idx])
t_train_gp = time.time() - t0

t0 = time.time()
y_pred_gp = gp_model.predict(X_test_scaled)
t_infer_gp = time.time() - t0
results.append(get_metrics("Gaussian Process", y_test_p, y_pred_gp, t_train_gp, t_infer_gp))

# --- Model 5: Random Forest Regression Baseline ---
print("[5/6] Training Random Forest Regression Baseline...")
t0 = time.time()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_p, y_train_p)
t_train_rf = time.time() - t0

t0 = time.time()
y_pred_rf = rf_model.predict(X_test_p)
t_infer_rf = time.time() - t0
results.append(get_metrics("Random Forest", y_test_p, y_pred_rf, t_train_rf, t_infer_rf))

# --- Model 6: EXACT PRESERVED PINN (Notebook Cell 13-15) ---
print("[6/6] Training Preserved PINN (Exact Notebook Architecture & Physics Loss)...")
X_train_tf = tf.convert_to_tensor(X_train_pinn, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(y_train_pinn, dtype=tf.float32)
X_test_tf = tf.convert_to_tensor(X_test_pinn, dtype=tf.float32)
y_test_tf = tf.convert_to_tensor(y_test_pinn, dtype=tf.float32)

t0 = time.time()

# Exact Notebook Cell 15 Architecture
inputs = tf.keras.Input(shape=(2,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
strain_pred = tf.keras.layers.Dense(1, name='strain_pred')(x)
temp_pred = tf.keras.layers.Dense(1, name='temp_pred')(x)
pinn_model = tf.keras.Model(inputs=inputs, outputs=[strain_pred, temp_pred])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

def physics_loss(y_true_b, strain_b, temp_b):
    delta_b = k_e_pm * strain_b + k_T_pm * temp_b
    return tf.reduce_mean(tf.square(y_true_b - delta_b))

@tf.function
def train_step(x_b, y_b):
    with tf.GradientTape() as tape:
        s_out, t_out = pinn_model(x_b, training=True)
        loss = physics_loss(y_b, s_out, t_out)
    grads = tape.gradient(loss, pinn_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, pinn_model.trainable_variables))
    return loss

epochs = 501
for epoch in range(epochs):
    train_loss = train_step(X_train_tf, y_train_tf)

t_train_pinn = time.time() - t0

t0 = time.time()
s_pred, t_pred = pinn_model(X_test_tf, training=False)
y_pred_pinn = (k_e_pm * s_pred + k_T_pm * t_pred).numpy().flatten()
t_infer_pinn = time.time() - t0

results.append(get_metrics("Proposed PINN", y_test_pinn.flatten(), y_pred_pinn, t_train_pinn, t_infer_pinn, s_pred.numpy(), t_pred.numpy()))

# Save DataFrame Results
df_results = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
df_results.to_csv(csv_path, index=False)

# Save Summary Markdown
md_path = os.path.join(RESULTS_DIR, "benchmark_summary.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Benchmark Comparison Summary\n\n")
    f.write("## Performance Metrics Across All 6 Models\n\n")
    f.write(df_results.to_markdown(index=False))
    f.write("\n\n## Visualizations\n\n")
    f.write("![Metrics](fig5_model_benchmarks_mae_rmse_r2.png)\n\n")
    f.write("![Predictions](fig3_prediction_vs_ground_truth.png)\n")

print("\n" + "="*85)
print("BENCHMARK TABLE INCLUDING GAUSSIAN PROCESS & PHYSICS RESIDUAL (6 MODELS)")
print("="*85)
print(df_results.to_string(index=False))

# Save LaTeX Table
format_dict = {
    "MAE (pm)": "{:.4f}".format,
    "RMSE (pm)": "{:.4f}".format,
    "R²": "{:.5f}".format,
    "Physics Residual (pm)": "{:.4f}".format,
    "Training Time (s)": "{:.3f}".format,
    "Inference Time (s)": "{:.3f}".format
}
latex_table = df_results.to_latex(index=False, formatters=format_dict)
tex_path = os.path.join(RESULTS_DIR, "benchmark_table.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex_table)
print(f"\n✅ LaTeX table saved to '{tex_path}'")
