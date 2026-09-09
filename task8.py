import pandas as pd
import os

os.makedirs("results/robustness", exist_ok=True)

def run_task8():
    print("--- TASK 8: Final Consistency Check Master Sheet ---")
    
    # Template for the master consistency sheet
    columns = [
        "Result",
        "Dataset",
        "Train/Test protocol",
        "Noise",
        "λ",
        "Input",
        "Purpose"
    ]
    
    # Pre-fill with the examples given in TASK 1.md
    data = [
        ["0.51 pm RMSE", "TEMP_STRAIN_CSV", "Random 80/20", "None", "1.0", "ΔλB + time", "Original baseline report"],
        ["0.61 pm RMSE", "TEMP_STRAIN_CSV", "Random 80/20", "None", "1.0", "ΔλB + time", "Original baseline report"],
        ["1.12 pm RMSE", "TEMP_STRAIN_CSV", "Random 80/20", "None", "1.0", "ΔλB + time", "Data Efficiency 100%"],
        ["1.99 pm MAE", "TEMP_STRAIN_CSV", "Cycle-wise 80/20", "None", "1.0", "ΔλB + time", "Task 1 Output"],
        ["3.05 pm RMSE", "TEMP_STRAIN_CSV", "Cycle-wise 80/20", "None", "1.0", "ΔλB + time", "Task 1 Output"]
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    output_path = "results/robustness/Task8_Master_Consistency_Sheet.csv"
    df.to_csv(output_path, index=False)
    print(f"Created template at {output_path}")

if __name__ == "__main__":
    run_task8()
