import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants
k_e_pm = 1.2
k_T_pm = 10.0
CYCLE_DURATION = 26.0

os.makedirs("results/robustness", exist_ok=True)

def load_data():
    strain_raw = pd.read_csv("STRAIN_CSV.csv")
    temp_raw = pd.read_csv("TEMP_CSV.csv")
    combined_raw = pd.read_csv("TEMP_STRAIN_CSV.csv")

    for df in [strain_raw, temp_raw, combined_raw]:
        df.columns = ["Time", "CH1", "CH2", "CH3", "CH4", "Wavelength"]
        df['delta_lambda_pm'] = (df['Wavelength'] - df['Wavelength'].iloc[0]) * 1000

    combined_raw['Cycle_ID'] = (combined_raw['Time'] // CYCLE_DURATION).astype(int)
    groups = combined_raw['Cycle_ID'].values
    
    return combined_raw, groups

def build_and_train_pinn(X_train, y_train, X_test, y_test, input_dim=2, physics_lambda=1.0):
    inputs = tf.keras.Input(shape=(input_dim,))
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
            data_loss = tf.reduce_mean(tf.square(y_b - delta_b))
            
            if physics_lambda > 0:
                physics_loss = tf.reduce_mean(tf.square(y_b - delta_b)) # For this simplified setup, physics loss and data loss are basically identical constraints if no other governing equations are present. In the original notebook, it's just the delta_b formula. We'll multiply by lambda.
                loss = data_loss * physics_lambda
            else:
                # If lambda = 0, purely data driven means predicting target without strict constraint? 
                # Wait, the notebook's PINN has no direct output for y, only strain/temp.
                # If lambda=0, the physics loss is 0. But how does it learn without data loss?
                # Actually, in the notebook, `tf.reduce_mean(tf.square(y_b - delta_b))` IS the loss. 
                # If the loss is scaled by lambda, lambda=0 means loss=0, no learning.
                # In standard PINN: Loss = Data_Loss + lambda * Physics_Loss.
                # The notebook formulation seems to be ONLY `reduce_mean(tf.square(y_true_b - delta_b))`.
                # If that is the only loss, lambda is just a learning rate scaler.
                # But let's assume Data_Loss = MSE(y_true, y_pred_network), Physics_Loss = MSE(y_pred_network, physical_formula).
                # The architecture gives s_pred and t_pred. There is no direct delta_b output from the network!
                # So if lambda=0, it's impossible for this specific architecture to train because there's no data loss on s_pred and t_pred (labels don't exist).
                # To be a true data-driven baseline (lambda=0), we must add a dense layer to predict delta_lambda directly!
                pass # Handled below
            
        grads = tape.gradient(loss, pinn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, pinn_model.trainable_variables))
        return loss

    # To handle lambda properly:
    # A standard PINN: pred_y = NN(x). Loss = MSE(pred_y, true_y) + lambda * MSE(pred_y, physical_formula(pred_s, pred_t))
    # Let's adjust the training loop to match a proper PINN ablation:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # Direct data prediction
    delta_lambda_pred = tf.keras.layers.Dense(1, name='delta_lambda_pred')(x)
    
    # Physics intermediate predictions
    strain_pred = tf.keras.layers.Dense(1, name='strain_pred')(x)
    temp_pred = tf.keras.layers.Dense(1, name='temp_pred')(x)
    
    pinn_model = tf.keras.Model(inputs=inputs, outputs=[delta_lambda_pred, strain_pred, temp_pred])
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def train_step_ablation(x_b, y_b):
        with tf.GradientTape() as tape:
            d_out, s_out, t_out = pinn_model(x_b, training=True)
            data_loss = tf.reduce_mean(tf.square(y_b - d_out))
            
            phys_constr = k_e_pm * s_out + k_T_pm * t_out
            physics_loss = tf.reduce_mean(tf.square(d_out - phys_constr))
            
            loss = data_loss + physics_lambda * physics_loss
            
        grads = tape.gradient(loss, pinn_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, pinn_model.trainable_variables))
        return loss

    epochs = 501
    for epoch in range(epochs):
        train_step_ablation(X_train, y_train)

    d_out, s_out, t_out = pinn_model(X_test, training=False)
    y_pred = d_out.numpy().flatten()
    
    return mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)

def run_task3():
    print("--- TASK 3: Time-Input Ablation ---")
    combined_raw, groups = load_data()
    
    # Model B features (delta_lambda + time)
    X_B = combined_raw[["Time", "delta_lambda_pm"]].values
    # Model A features (delta_lambda only)
    X_A = combined_raw[["delta_lambda_pm"]].values
    
    y = combined_raw["delta_lambda_pm"].values.reshape(-1, 1)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_B, y, groups))
    
    # Train Model A
    X_train_A = tf.convert_to_tensor(X_A[train_idx], dtype=tf.float32)
    X_test_A = tf.convert_to_tensor(X_A[test_idx], dtype=tf.float32)
    y_train = tf.convert_to_tensor(y[train_idx], dtype=tf.float32)
    y_test = y[test_idx].flatten()
    
    mae_A, rmse_A, r2_A = build_and_train_pinn(X_train_A, y_train, X_test_A, y_test, input_dim=1)
    
    # Train Model B
    X_train_B = tf.convert_to_tensor(X_B[train_idx], dtype=tf.float32)
    X_test_B = tf.convert_to_tensor(X_B[test_idx], dtype=tf.float32)
    
    mae_B, rmse_B, r2_B = build_and_train_pinn(X_train_B, y_train, X_test_B, y_test, input_dim=2)
    
    res = {
        "ΔλB only": [mae_A, rmse_A, r2_A],
        "ΔλB + time": [mae_B, rmse_B, r2_B]
    }
    
    df = pd.DataFrame.from_dict(res, orient='index', columns=['MAE (pm)', 'RMSE (pm)', 'R²'])
    print(df.to_string())
    df.to_csv("results/robustness/Task3_TimeInput_Ablation.csv")

def run_task4():
    print("\n--- TASK 4: Complete Physics-Loss Ablation ---")
    combined_raw, groups = load_data()
    X = combined_raw[["Time", "delta_lambda_pm"]].values
    y = combined_raw["delta_lambda_pm"].values.reshape(-1, 1)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train = tf.convert_to_tensor(X[train_idx], dtype=tf.float32)
    X_test = tf.convert_to_tensor(X[test_idx], dtype=tf.float32)
    y_train = tf.convert_to_tensor(y[train_idx], dtype=tf.float32)
    y_test = y[test_idx].flatten()
    
    lambdas = [0, 0.01, 0.1, 1, 10, 100]
    res = {}
    
    for l in lambdas:
        print(f"Training with lambda = {l}...")
        mae, rmse, r2 = build_and_train_pinn(X_train, y_train, X_test, y_test, input_dim=2, physics_lambda=l)
        res[l] = [mae, rmse, r2]
        
    df = pd.DataFrame.from_dict(res, orient='index', columns=['MAE (pm)', 'RMSE (pm)', 'R²'])
    df.index.name = 'λ'
    print(df.to_string())
    df.to_csv("results/robustness/Task4_PhysicsLoss_Ablation.csv")
    
    # Plot
    plt.figure(figsize=(10, 5))
    valid_lambdas = [l for l in lambdas if l > 0]
    rmses = [res[l][1] for l in valid_lambdas]
    maes = [res[l][0] for l in valid_lambdas]
    
    plt.plot(valid_lambdas, rmses, marker='o', label='RMSE (pm)')
    plt.plot(valid_lambdas, maes, marker='x', label='MAE (pm)')
    plt.xscale('log')
    plt.xlabel('Physics Loss Weight (λ)')
    plt.ylabel('Error (pm)')
    plt.title('Effect of Physics Loss Constraint on Model Error')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/robustness/Task4_Lambda_vs_Error.png")

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    run_task3()
    run_task4()
