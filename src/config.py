"""
Configuration module defining physical constants, paths, default hyperparameters, and random seeds.
"""

import os

# Base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# File Paths (READ ONLY - Existing files remain untouched)
DATA_COMBINED_PATH = os.path.join(BASE_DIR, "Week1_combined_final.csv")
DATA_RAW_TEMP_STRAIN_PATH = os.path.join(BASE_DIR, "TEMP_STRAIN_CSV.csv")
DATA_RAW_STRAIN_PATH = os.path.join(BASE_DIR, "STRAIN_CSV.csv")
DATA_RAW_TEMP_PATH = os.path.join(BASE_DIR, "TEMP_CSV.csv")

# Output Directories
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
RESULTS_JSON_PATH = os.path.join(OUTPUT_DIR, "results.json")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Ensure Output Directories Exist
for directory in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, SAVED_MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Physical FBG Sensitivity Parameters
K_EPSILON = 1.2    # Strain sensitivity coefficient (pm / microstrain: 0.0012 nm/µε * 1000)
K_TEMP = 10.0      # Temperature sensitivity coefficient (pm / degree C: 0.01 nm/°C * 1000)
LAMBDA_0 = 1524.22429 # Base Bragg wavelength (nm)

# Default Training Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 500
DEFAULT_BATCH_SIZE = 64
DEFAULT_PHYSICS_WEIGHT = 1.0
