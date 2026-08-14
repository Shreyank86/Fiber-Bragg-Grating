# Physics-Informed Neural Network (PINN) for FBG Sensor Strain–Temperature Decoupling: Comprehensive Repository Review

**Document Version:** 1.0  
**Target Domain:** Fiber Bragg Grating (FBG) Optical Sensing & Scientific Machine Learning (SciML)  
**Author:** AI Research Team  

---

## 1. Executive Summary & Overview

Fiber Bragg Grating (FBG) optical sensors are standard optical components used extensively in structural health monitoring (SHM), aerospace engineering, civil infrastructure, and smart energy systems. An FBG sensor reflects light at a specific wavelength, termed the Bragg wavelength ($\lambda_B$). However, the measured shift in Bragg wavelength ($\Delta\lambda_B$) is simultaneously sensitive to both mechanical strain ($\varepsilon$) and ambient temperature changes ($\Delta T$). 

Mathematically, the analytical sensing response of a single FBG sensor is governed by:
$$\Delta\lambda_B = k_\varepsilon \cdot \varepsilon + k_T \cdot \Delta T$$
where:
- $k_\varepsilon$ is the strain sensitivity coefficient ($\text{pm}/\mu\varepsilon$)
- $k_T$ is the temperature sensitivity coefficient ($\text{pm}/^\circ\text{C}$)

Decoupling strain ($\varepsilon$) and temperature ($\Delta T$) using only a **single FBG sensor** is an ill-posed **inverse measurement problem** because a single scalar measurement ($\Delta\lambda$) must be mapped to two independent unknown state variables ($\varepsilon, \Delta T$). Conventional optical sensing systems resolve this ambiguity by deploying auxiliary hardware (e.g., dual-FBG arrays or supplementary thermocouples).

This repository implements a **Physics-Informed Neural Network (PINN)** framework—more precisely, a **Measurement-Constrained Physics-Guided Inverse Sensing Framework**—that directly embeds the analytical FBG Bragg wavelength shift equation into the loss function of a neural network. This allows accurate, physically consistent strain and temperature prediction using a single optical sensor.

---

## 2. Overall Repository Structure

The existing workspace consists of the following primary components:

```
FBG-Sensor-PINN/
│
├── Internship_PINN.ipynb         # Primary Jupyter notebook containing exploratory data analysis,
│                                 # preprocessing, PINN architecture, training loop, K-fold CV,
│                                 # bootstrap confidence intervals, and physics weight (λ) ablation.
├── STRAIN_CSV.csv                # Experimental dataset for pure strain variation (temperature constant).
├── TEMP_CSV.csv                  # Experimental dataset for pure temperature variation (strain constant).
├── TEMP_STRAIN_CSV.csv           # Combined experimental dataset with simultaneous strain and temperature variations.
├── Week1_strain_final.csv        # Processed strain dataset with column standardization.
├── Week1_temp_final.csv          # Processed temperature dataset with column standardization.
├── Week1_combined_final.csv      # Processed combined dataset with calculated wavelength shifts (Δλ).
├── classical_baseline_results.txt# Text summary of baseline linear sensitivity results.
├── data_dictionary.csv           # CSV formatted data dictionary of raw dataset features.
├── data_dictionary.md            # Markdown formatted data dictionary describing data columns.
└── frontend/                     # Web interface directory containing UI components.
```

All existing repository files remain **100% untouched** to preserve full historical reproducibility.

---

## 3. Dataset Description & Physics Parameters

### 3.1 Raw Datasets

The repository contains three primary raw CSV datasets recorded from an FBG optical interrogator system:

| Dataset File | Sample Count | Primary Physical Condition | Notes |
| :--- | :--- | :--- | :--- |
| `STRAIN_CSV.csv` | ~5,000 samples | Pure mechanical strain applied ($\Delta T \approx 0$) | Used to extract baseline strain sensitivity $k_\varepsilon$ |
| `TEMP_CSV.csv` | ~4,000 samples | Pure temperature change applied ($\varepsilon \approx 0$) | Used to extract baseline temperature sensitivity $k_T$ |
| `TEMP_STRAIN_CSV.csv` | 9,063 samples | Combined strain and temperature loading | Used for PINN model training, validation, and benchmarking |

### 3.2 Feature Specification

For each raw dataset, the optical interrogator outputs 6 columns:

1. **`Time`**: Elapsed experimental timestamp in seconds ($\text{sec}$).
2. **`CH1`**: Optical channel flag 1 (binary indicator).
3. **`CH2`**: Optical channel flag 2 (binary indicator).
4. **`CH3`**: Optical channel flag 3 (binary indicator).
5. **`CH4`**: Optical channel flag 4 (binary indicator).
6. **`Wavelength`**: Raw Bragg wavelength $\lambda_B(t)$ measured in nanometers ($\text{nm}$).

### 3.3 Derived Physical Quantities

The target physics feature is the wavelength shift $\Delta\lambda_B(t)$ measured in picometers ($\text{pm}$), calculated relative to the initial baseline Bragg wavelength $\lambda_0$:
$$\Delta\lambda_B(t) = (\lambda_B(t) - \lambda_0) \times 10^3 \quad (\text{pm})$$

In the preprocessed dataset `Week1_combined_final.csv`:
- Initial reference wavelength $\lambda_0 = 1524.22429 \text{ nm}$
- Mean relative wavelength shift $\Delta\lambda_{\text{pm}} = -694.21 \text{ pm}$
- Wavelength shift range $\Delta\lambda_{\text{pm}} \in [-923.88 \text{ pm}, +288.70 \text{ pm}]$

### 3.4 Physical Sensitivity Coefficients

Linear regression on individual strain and temperature calibration datasets yields the empirical optical sensitivity coefficients:
- **Strain Sensitivity Coefficient ($k_\varepsilon$)**: $-0.0009 \text{ pm}/\mu\varepsilon$
- **Temperature Sensitivity Coefficient ($k_T$)**: $+0.0001 \text{ pm}/^\circ\text{C}$

---

## 4. Data Preprocessing Pipeline

The data preprocessing workflow follows a structured sequence:

```
[Raw Interrogator CSV] 
       ↓
1. Column Standardisation ("Time", "CH1"-"CH4", "Wavelength")
       ↓
2. Baseline Reference Extraction (λ₀ at t = 0)
       ↓
3. Relative Shift Calculation (Δλ = (λ - λ₀) * 1000 pm)
       ↓
4. Input Feature Extraction (X = [Time, Δλ])
       ↓
5. Target Definition (y_true = Δλ)
       ↓
6. Train-Test Splitting (80% Train, 20% Test, seed=42)
```

---

## 5. Neural Network Architecture & Physics Loss

### 5.1 Deep Neural Network (DNN) Architecture

The core predictor model is a fully connected Multi-Layer Perceptron (MLP) built using TensorFlow/Keras:

```
Input Layer: [Time (s), Δλ (pm)]  (Dimension: 2)
       ↓
Dense Layer 1: 64 neurons, ReLU activation
       ↓
Dense Layer 2: 64 neurons, ReLU activation
       ↓
Dense Layer 3: 32 neurons, ReLU activation
       ↓
Split Output Heads:
   ├── Head 1: Strain Prediction (ε_pred) -> 1 neuron (Linear)
   └── Head 2: Temperature Prediction (T_pred) -> 1 neuron (Linear)
```

Total trainable parameters: $2 \times 64 + 64 + (64 \times 64 + 64) + (64 \times 32 + 32) + (32 \times 1 + 1) + (32 \times 1 + 1) = 6,562$ parameters.

### 5.2 Physics Loss Implementation

Rather than enforcing partial differential equations (PDEs), the network loss embeds the **analytical FBG optical measurement law**:

$$\hat{\Delta\lambda} = k_\varepsilon \cdot \hat{\varepsilon} + k_T \cdot \hat{T}$$

$$\mathcal{L}_{\text{phys}}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \left( \Delta\lambda_{\text{true}, i} - (k_\varepsilon \cdot \hat{\varepsilon}_i + k_T \cdot \hat{T}_i) \right)^2$$

### 5.3 Total Loss Function & Optimization

When incorporating both data fidelity and physics constraint regularization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \cdot \mathcal{L}_{\text{phys}}$$

where $\lambda \ge 0$ is the physics loss weighting hyperparameter.

- **Optimizer**: Adam ($\text{learning rate} = 10^{-3}$)
- **Epochs**: 500 epochs
- **Batching**: Full-batch tensor conversion (`tf.convert_to_tensor`)

---

## 6. Evaluation Metrics & Statistical Analysis

To rigorously assess performance, three quantitative accuracy metrics are used:

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |\Delta\lambda_i - \hat{\Delta\lambda}_i|$$

2. **Root Mean Square Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (\Delta\lambda_i - \hat{\Delta\lambda}_i)^2}$$

3. **Coefficient of Determination ($R^2$)**:
   $$R^2 = 1 - \frac{\sum_{i=1}^N (\Delta\lambda_i - \hat{\Delta\lambda}_i)^2}{\sum_{i=1}^N (\Delta\lambda_i - \bar{\Delta\lambda})^2}$$

In addition:
- **5-Fold Cross-Validation**: Evaluates stability and out-of-fold generalization error.
- **Bootstrap Resampling ($N=1000$)**: Computes non-parametric 95% confidence intervals for MAE and RMSE.

---

## 7. Analysis of Existing Experiments in `Internship_PINN.ipynb`

1. **Exploratory Data Analysis**: Standardized dataset headers and created combined CSV file (`Week1_combined_final.csv`).
2. **Classical Sensitivity Fitting**: Simple linear regression fitted $k_\varepsilon = -0.0009 \text{ pm}/\mu\varepsilon$ and $k_T = 0.0001 \text{ pm}/^\circ\text{C}$, but achieved negative $R^2 = -1.122$, demonstrating that purely linear baseline models fail on combined strain-temperature dynamics.
3. **PINN Model Training**: 500 epochs of Adam optimization on 80/20 train/test split.
4. **Validation & Sorting**: Sorted predictions by timestamp to inspect dynamic strain ($\mu\varepsilon$) and temperature ($^\circ\text{C}$) trajectories.
5. **Bootstrap Confidence Intervals**: Calculated 1000 bootstrap resamples on test set residuals.
6. **Physics Weight ($\lambda$) Ablation**: Evaluated $\lambda \in [0.0, 0.1, 1.0, 10.0]$ to verify that introducing physics constraints stabilizes model optimization.

---

## 8. Limitations & Gaps of Existing Work

1. **Missing Machine Learning Baselines**: Lack of comparison against Random Forest, Support Vector Regression (SVR), Multi-layer Perceptron (MLP), and Gaussian Process Regression (GPR).
2. **Missing Robustness Studies**: No prior evaluation under noise injection (1%-10% Gaussian noise) or small training data regimes (20%-80%).
3. **Conceptual Framing**: Described as "Engineering PINN" rather than "Measurement-Constrained Inverse Sensing Framework".
4. **Code Modularization**: Monolithic Jupyter notebook format rather than structured, reusable Python modules.

All 19 phases in `Things to do.docx` address these gaps directly.
