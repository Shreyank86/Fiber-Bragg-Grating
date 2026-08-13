import numpy as np

# Standard FBG Physical Sensitivity Coefficients
# Delta_Lambda_B = k_T * Delta_T + k_S * Delta_Strain
K_T = 0.01015   # nm / deg C
K_S = 0.00121   # nm / microstrain

def calculate_wavelength_shift(delta_temp, delta_strain, k_t=K_T, k_s=K_S):
    """
    Physical Forward Equation:
    Recombines temperature shift (deg C) and strain (microstrain) into predicted Bragg wavelength shift (nm).
    """
    return k_t * delta_temp + k_s * delta_strain

def compute_physical_residual(wavelength_shift_observed, pred_temp, pred_strain, k_t=K_T, k_s=K_S, initial_temp=20.0, initial_strain=0.0):
    """
    Computes physical residual error (nm):
    residual = observed_shift - (k_t * (pred_temp - initial_temp) + k_s * (pred_strain - initial_strain))
    """
    delta_temp = pred_temp - initial_temp
    delta_strain = pred_strain - initial_strain
    predicted_shift = calculate_wavelength_shift(delta_temp, delta_strain, k_t=k_t, k_s=k_s)
    return wavelength_shift_observed - predicted_shift
