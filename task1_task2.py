import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Constants
k_e_pm = 1.2
k_T_pm = 10.0
CYCLE_DURATION = 26.0 # 25.81s average detected, 26s safe threshold

os.makedirs("results/robustness", exist_ok=True)

# 1. Load Data
def load_data():
    strain_raw = pd.read_csv("STRAIN_CSV.csv")
    temp_raw = pd.read_csv("TEMP_CSV.csv")
    combined_raw = pd.read_csv("TEMP_STRAIN_CSV.csv")

    for df in [strain_raw, temp_raw, combined_raw]:
        df.columns = ["Time", "CH1", "CH2", "CH3", "CH4", "Wavelength"]
        df['delta_lambda_pm'] = (df['Wavelength'] - df['Wavelength'].iloc[0]) * 1000

    # Interpolate proxies
    t_comb = combined_raw["Time"].values
    strain_proxy = np.interp(t_comb, strain_raw["Time"].values, strain_raw["delta_lambda_pm"].values)
    temp_proxy = np.interp(t_comb, temp_raw["Time"].values, temp_raw["delta_lambda_pm"].values)

    X_proxy = np.column_stack([strain_proxy, temp_proxy]) # For classical models
    X_pinn = combined_raw[["Time", "delta_lambda_pm"]].values # For PINN
    y = combined_raw["delta_lambda_pm"].values

    # Assign Cycle ID based on time
    combined_raw['Cycle_ID'] = (combined_raw['Time'] // CYCLE_DURATION).astype(int)
    groups = combined_raw['Cycle_ID'].values

    return X_proxy, X_pinn, y, groups

def train_eval_models(X_train_proxy, X_test_proxy, X_train_pinn, X_test_pinn, y_train, y_test):
    results = {}
    
    # Scale classical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proxy)
    X_test_scaled = scaler.transform(X_test_proxy)

    # 1. Linear Regression
    lr = LinearRegression().fit(X_train_proxy, y_train)
    y_pred_lr = lr.predict(X_test_proxy)
    results['Linear Regression'] = (mean_absolute_error(y_test, y_pred_lr), np.sqrt(mean_squared_error(y_test, y_pred_lr)), r2_score(y_test, y_pred_lr))

    # 2. MLP
    mlp = MLPRegressor(hidden_layer_sizes=(64, 64, 32), max_iter=200, early_stopping=True, random_state=42).fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_test_scaled)
    results['MLP'] = (mean_absolute_error(y_test, y_pred_mlp), np.sqrt(mean_squared_error(y_test, y_pred_mlp)), r2_score(y_test, y_pred_mlp))

    # 3. SVR
    svr = SVR(kernel='rbf', C=10.0, epsilon=0.1).fit(X_train_scaled, y_train)
    y_pred_svr = svr.predict(X_test_scaled)
    results['SVR'] = (mean_absolute_error(y_test, y_pred_svr), np.sqrt(mean_squared_error(y_test, y_pred_svr)), r2_score(y_test, y_pred_svr))

    # 4. GPR (Subsampled for speed as in original)
    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=1, random_state=42)
    sub_idx = np.random.choice(len(X_train_scaled), size=min(1000, len(X_train_scaled)), replace=False)
    gpr.fit(X_train_scaled[sub_idx], y_train[sub_idx])
    y_pred_gpr = gpr.predict(X_test_scaled)
    results['GPR'] = (mean_absolute_error(y_test, y_pred_gpr), np.sqrt(mean_squared_error(y_test, y_pred_gpr)), r2_score(y_test, y_pred_gpr))

    # 5. Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train_proxy, y_train)
    y_pred_rf = rf.predict(X_test_proxy)
    results['Random Forest'] = (mean_absolute_error(y_test, y_pred_rf), np.sqrt(mean_squared_error(y_test, y_pred_rf)), r2_score(y_test, y_pred_rf))

    # 6. PINN
    X_train_tf = tf.convert_to_tensor(X_train_pinn, dtype=tf.float32)
    y_train_tf = tf.convert_to_tensor(y_train.reshape(-1, 1), dtype=tf.float32)
    X_test_tf = tf.convert_to_tensor(X_test_pinn, dtype=tf.float32)

    inputs = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    strain_pred = tf.keras.layers.Dense(1, name='strain_pred')(x)
    temp_pred = tf.keras.layers.Dense(1, name='temp_pred')(x)
    pinn_model = tf.keras.Model(inputs=inputs, outputs=[strain_pred, temp_pred])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    @tf.function
    def train_step(x_b, y_b):
        with tf.GradientTape() as tape:
            s_out, t_out = pinn_model(x_b, training=True)
            delta_b = k_e_pm * s_out + k_T_pm * t_out
            loss = tf.reduce_mean(tf.square(y_b - delta_b))
        grads = tape.gradient(loss, pinn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, pinn_model.trainable_variables))
        return loss

    epochs = 200 # slightly reduced for evaluation speed, original was 500
    for epoch in range(epochs):
        train_step(X_train_tf, y_train_tf)

    s_pred, t_pred = pinn_model(X_test_tf, training=False)
    y_pred_pinn = (k_e_pm * s_pred + k_T_pm * t_pred).numpy().flatten()
    results['PINN'] = (mean_absolute_error(y_test, y_pred_pinn), np.sqrt(mean_squared_error(y_test, y_pred_pinn)), r2_score(y_test, y_pred_pinn))

    return results

def run_task1():
    print("--- TASK 1: Cycle-wise Train/Test Split ---")
    X_proxy, X_pinn, y, groups = load_data()
    
    # 80/20 Group split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_proxy, y, groups))
    
    train_groups = np.unique(groups[train_idx])
    test_groups = np.unique(groups[test_idx])
    print(f"Training cycles ({len(train_groups)}): {train_groups}")
    print(f"Testing cycles ({len(test_groups)}): {test_groups}")
    
    res = train_eval_models(X_proxy[train_idx], X_proxy[test_idx], 
                            X_pinn[train_idx], X_pinn[test_idx], 
                            y[train_idx], y[test_idx])
                            
    df = pd.DataFrame.from_dict(res, orient='index', columns=['MAE (pm)', 'RMSE (pm)', 'R²'])
    print(df.to_string())
    df.to_csv("results/robustness/Task1_Cycle_Split_Results.csv")

def run_task2():
    print("\n--- TASK 2: Cycle-wise K-Fold Validation ---")
    X_proxy, X_pinn, y, groups = load_data()
    
    gkf = GroupKFold(n_splits=5)
    
    fold_results = {model: {'MAE': [], 'RMSE': [], 'R2': []} for model in ['Linear Regression', 'MLP', 'SVR', 'GPR', 'Random Forest', 'PINN']}
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_proxy, y, groups)):
        print(f"Running Fold {fold+1}/5...")
        res = train_eval_models(X_proxy[train_idx], X_proxy[test_idx], 
                                X_pinn[train_idx], X_pinn[test_idx], 
                                y[train_idx], y[test_idx])
                                
        for model, metrics in res.items():
            fold_results[model]['MAE'].append(metrics[0])
            fold_results[model]['RMSE'].append(metrics[1])
            fold_results[model]['R2'].append(metrics[2])
            
    final_res = {}
    for model, metrics in fold_results.items():
        final_res[model] = {
            'MAE (pm)': f"{np.mean(metrics['MAE']):.4f} ± {np.std(metrics['MAE']):.4f}",
            'RMSE (pm)': f"{np.mean(metrics['RMSE']):.4f} ± {np.std(metrics['RMSE']):.4f}",
            'R²': f"{np.mean(metrics['R2']):.4f} ± {np.std(metrics['R2']):.4f}"
        }
        
    df = pd.DataFrame.from_dict(final_res, orient='index')
    print(df.to_string())
    df.to_csv("results/robustness/Task2_GroupKFold_Results.csv")

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    run_task1()
    run_task2()
