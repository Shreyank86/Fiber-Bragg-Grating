import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Dataset")

def compute_sliding_features(x, windows=50, dt=0.5):
    """
    Computes 5 rolling time-series features over a sliding window of 50 samples:
    mean, std, slope, skew, kurtosis.
    """
    df = pd.DataFrame({'x': x})
    df['mean'] = df['x'].rolling(windows).mean()
    df['std'] = df['x'].rolling(windows).std()
    df['slope'] = df['x'].diff(periods=5) / (dt * 5)
    df['skew'] = df['x'].rolling(windows).skew()
    df['kurtosis'] = df['x'].rolling(windows).kurt()
    return df.dropna().reset_index(drop=True)

def load_fbg_dataset(experiment_type="both", windows=50, dt=0.5):
    """
    Loads raw CSV and returns feature dataframe X and targets y.
    
    experiment_type: "temp", "strain", or "both"
    """
    if experiment_type == "temp":
        file_path = os.path.join(DATASET_DIR, "TEMP_EXPERIMENT-1.csv")
        df = pd.read_csv(file_path)
        features = compute_sliding_features(df['Wavelength'], windows=windows, dt=dt)
        features['shift'] = features['x'] - features['x'].iloc[0]
        X = features.rename(columns={'x': 'Wavelength'})
        y = np.linspace(20, 80, len(X))
        return X, y

    elif experiment_type == "strain":
        file_path = os.path.join(DATASET_DIR, "TEMP EXPERIMENT-2.csv")
        df = pd.read_csv(file_path)
        features = compute_sliding_features(df['Wavelength'], windows=windows, dt=dt)
        features['shift'] = features['x'] - features['x'].iloc[0]
        X = features.rename(columns={'x': 'Wavelength'})
        y = np.linspace(0, 1000, len(X))
        return X, y

    elif experiment_type == "both":
        file_path = os.path.join(DATASET_DIR, "TEMP AND STRAIN EXPERIMENT-3.csv")
        df = pd.read_csv(file_path)
        features = compute_sliding_features(df['Wavelength'], windows=windows, dt=dt)
        features['shift'] = features['x'] - features['x'].iloc[0]
        X = features.rename(columns={'x': 'Wavelength'})
        
        y_temp = np.linspace(20, 80, len(X))
        y_strain = np.linspace(0, 1000, len(X))
        y = np.column_stack([y_temp, y_strain])
        return X, y

    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

def get_scaled_train_test_split(X, y, test_size=0.2, random_state=42, scale=True):
    """
    Splits data into train/test sets and applies StandardScaler in a leak-free manner.
    Scales features strictly on X_train.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    return X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train, \
           X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test, \
           y_train, y_test, None
