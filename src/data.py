"""
Data processing module for loading, preprocessing, splitting, scaling, noise injection, and small-data subsampling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import DATA_COMBINED_PATH, RANDOM_SEED, TEST_SIZE

def load_fbg_dataset(file_path=DATA_COMBINED_PATH):
    """
    Loads FBG sensor dataset, ensuring column names and derived delta_lambda_pm are formatted.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    df = pd.read_csv(file_path)
    
    if 'Unnamed: 5' in df.columns:
        df = df.rename(columns={'Unnamed: 5': 'Wavelength'})
        
    if 'Wavelength' in df.columns and 'delta_lambda_pm' not in df.columns:
        lambda_0 = df['Wavelength'].iloc[0]
        df['delta_lambda_pm'] = (df['Wavelength'] - lambda_0) * 1000.0
        
    return df

def get_train_test_data(test_size=TEST_SIZE, seed=RANDOM_SEED):
    """
    Extracts features for PINN (Time, delta_lambda_pm) and Classical ML (Time), along with target y = delta_lambda_pm.
    """
    df = load_fbg_dataset()
    
    # PINN input uses [Time, delta_lambda_pm], Classical ML uses [Time]
    X_pinn = df[['Time', 'delta_lambda_pm']].values
    X_ml = df[['Time']].values
    y = df['delta_lambda_pm'].values.reshape(-1, 1)
    
    # Standard train/test split
    X_train_pinn, X_test_pinn, y_train, y_test = train_test_split(
        X_pinn, y, test_size=test_size, random_state=seed, shuffle=True
    )
    
    X_train_ml, X_test_ml, _, _ = train_test_split(
        X_ml, y, test_size=test_size, random_state=seed, shuffle=True
    )
    
    # Scalers for ML models
    scaler_ml = StandardScaler()
    X_train_ml_sc = scaler_ml.fit_transform(X_train_ml)
    X_test_ml_sc = scaler_ml.transform(X_test_ml)
    
    return {
        "X_train_pinn": X_train_pinn,
        "X_test_pinn": X_test_pinn,
        "X_train_ml": X_train_ml_sc,
        "X_test_ml": X_test_ml_sc,
        "y_train": y_train,
        "y_test": y_test,
        "full_df": df
    }

def add_gaussian_noise(y_data, noise_percentage, seed=RANDOM_SEED):
    """
    Injects zero-mean Gaussian noise into target wavelength shift measurements.
    """
    if noise_percentage <= 0:
        return y_data.copy()
        
    rng = np.random.default_rng(seed)
    std_dev = np.std(y_data) * (noise_percentage / 100.0)
    noise = rng.normal(0, std_dev, size=y_data.shape)
    return y_data + noise

def get_subsampled_data(X_train, y_train, fraction, seed=RANDOM_SEED):
    """
    Subsamples training set for small-data regime experiments.
    """
    if fraction >= 1.0:
        return X_train, y_train
        
    n_samples = int(len(X_train) * fraction)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X_train), size=n_samples, replace=False)
    
    return X_train[indices], y_train[indices]
