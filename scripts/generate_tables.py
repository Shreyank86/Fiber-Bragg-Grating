"""
Phase 15 Script: Generates all 9 required LaTeX (.tex) and Markdown (.md) tables, saving to outputs/tables/.
Updated to handle multi-model noise robustness and small-data JSON metrics structures.
"""

import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import TABLES_DIR, RESULTS_JSON_PATH

def save_table_files(filename_stem, latex_str, markdown_str):
    """Saves both LaTeX (.tex) and Markdown (.md) versions of a table."""
    tex_path = os.path.join(TABLES_DIR, f"{filename_stem}.tex")
    md_path = os.path.join(TABLES_DIR, f"{filename_stem}.md")

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_str)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("==================================================================")
    print("PHASE 15: AUTOMATED LATEX & MARKDOWN TABLE GENERATION")
    print("==================================================================")
    
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Load results
    results = {}
    if os.path.exists(RESULTS_JSON_PATH):
        with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)

    # Table 1: Dataset Description
    print("Generating Table 1: Dataset Description...")
    t1_md = """# Table 1: Dataset Description & Physical Parameters

| Parameter / Feature | Variable | Value / Specification | Unit |
| :--- | :--- | :--- | :--- |
| **Combined Dataset Size** | $N$ | 9,063 samples | - |
| **Time Span** | $t$ | 0.2 to 1812.5 | sec |
| **Base Bragg Wavelength** | $\\lambda_0$ | 1524.22429 | nm |
| **Measured Wavelength Shift** | $\\Delta\\lambda$ | -923.88 to +288.70 | pm |
| **Strain Sensitivity** | $k_\\varepsilon$ | 1.2 | pm/microstrain |
| **Temperature Sensitivity** | $k_T$ | 10.0 | pm/degree C |
"""
    t1_tex = """\\begin{table}[h]
\\centering
\\caption{Dataset Description and Physical Transducer Sensitivity Parameters}
\\label{tab:dataset_desc}
\\begin{tabular}{lccc}
\\hline
\\textbf{Parameter / Feature} & \\textbf{Symbol} & \\textbf{Value / Range} & \\textbf{Unit} \\\\ \\hline
Combined Dataset Size & $N$ & 9,063 & samples \\\\
Time Duration & $t$ & 0.2 -- 1812.58 & sec \\\\
Base Bragg Wavelength & $\\lambda_0$ & 1524.22429 & nm \\\\
Wavelength Shift Range & $\\Delta\\lambda$ & -923.88 -- +288.70 & pm \\\\
Strain Sensitivity & $k_\\varepsilon$ & 1.2 & pm/$\\mu\\varepsilon$ \\\\
Temperature Sensitivity & $k_T$ & 10.0 & pm/$^\\circ$C \\\\ \\hline
\\end{tabular}
\\end{table}"""
    save_table_files("table1_dataset_description", t1_tex, t1_md)

    # Table 2: Model Architecture & Hyperparameters
    print("Generating Table 2: Model Architecture & Hyperparameters...")
    t2_md = """# Table 2: Model Architecture & Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| **Input Layer Neurons** | 2 (`[Time, delta_lambda_pm]`) |
| **Hidden Layers** | 3 Dense Layers (64, 64, 32 neurons) |
| **Activation Function** | ReLU |
| **Output Heads** | 2 Heads (`strain_pred`, `temp_pred`) |
| **Physics Loss Weight (\\lambda)** | 1.0 |
| **Optimizer** | Adam ($lr = 10^{-3}$) |
| **Batch Size** | Full batch / 64 |
| **Training Epochs** | 500 epochs |
"""
    t2_tex = """\\begin{table}[h]
\\centering
\\caption{Measurement-Constrained PINN Architecture and Hyperparameters}
\\label{tab:hyperparameters}
\\begin{tabular}{lc}
\\hline
\\textbf{Hyperparameter} & \\textbf{Specification} \\\\ \\hline
Input Dimensionality & 2 ($t$, $\\Delta\\lambda$) \\\\
Hidden Layers & 3 Dense Layers (64, 64, 32 units) \\\\
Activation Function & ReLU \\\\
Output Dimensions & 2 (\\hat{\\varepsilon}, \\hat{T}) \\\\
Physics Loss Weight (\\lambda) & 1.0 \\\\
Optimizer & Adam (lr = 10^{-3}) \\\\
Epochs & 500 \\\\ \\hline
\\end{tabular}
\\end{table}"""
    save_table_files("table2_hyperparameters", t2_tex, t2_md)

    # Table 3: Baseline Comparison Table
    print("Generating Table 3: Baseline Comparison...")
    b_data = results.get("Phase3_Baselines", {})
    t3_md = """# Table 3: Machine Learning Baseline Comparison Table

| Model | MAE (pm) | RMSE (pm) | R² | Training Time (s) | Inference Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    t3_rows_tex = ""
    for m_name, m_val in b_data.items():
        t3_md += f"| **{m_name}** | {m_val['MAE']:.4f} | {m_val['RMSE']:.4f} | {m_val['R2']:.4f} | {m_val['Train_Time_Sec']:.4f} | {m_val['Infer_Time_Sec']:.4f} |\n"
        t3_rows_tex += f"{m_name} & {m_val['MAE']:.4f} & {m_val['RMSE']:.4f} & {m_val['R2']:.4f} & {m_val['Train_Time_Sec']:.4f} & {m_val['Infer_Time_Sec']:.4f} \\\\\n"

    t3_tex = f"""\\begin{{table}}[h]
\\centering
\\caption{{Comprehensive Benchmarking Across Machine Learning Baselines and Proposed PINN}}
\\label{{tab:baseline_comparison}}
\\begin{{tabular}}{{lccccc}}
\\hline
\\textbf{{Model}} & \\textbf{{MAE (pm)}} & \\textbf{{RMSE (pm)}} & \\textbf{{$R^2$}} & \\textbf{{Train (s)}} & \\textbf{{Infer (s)}} \\\\ \\hline
{t3_rows_tex}\\hline
\\end{{tabular}}
\\end{{table}}"""
    save_table_files("table3_baseline_comparison", t3_tex, t3_md)

    # Table 4: Cross Validation Table
    print("Generating Table 4: Cross-Validation Statistics...")
    cv_data = results.get("Phase8_CrossValidation", {}).get("summary", {})
    t4_md = f"""# Table 4: 5-Fold Cross-Validation Statistics

| Metric | Mean | Standard Deviation | 95% Confidence Interval |
| :--- | :--- | :--- | :--- |
| **MAE (pm)** | {cv_data.get('MAE', {}).get('mean', 0.0):.4f} | {cv_data.get('MAE', {}).get('std', 0.0):.4f} | [{cv_data.get('MAE', {}).get('ci95', [0,0])[0]:.4f}, {cv_data.get('MAE', {}).get('ci95', [0,0])[1]:.4f}] |
| **RMSE (pm)** | {cv_data.get('RMSE', {}).get('mean', 0.0):.4f} | {cv_data.get('RMSE', {}).get('std', 0.0):.4f} | [{cv_data.get('RMSE', {}).get('ci95', [0,0])[0]:.4f}, {cv_data.get('RMSE', {}).get('ci95', [0,0])[1]:.4f}] |
| **R²** | {cv_data.get('R2', {}).get('mean', 0.0):.4f} | {cv_data.get('R2', {}).get('std', 0.0):.4f} | [{cv_data.get('R2', {}).get('ci95', [0,0])[0]:.4f}, {cv_data.get('R2', {}).get('ci95', [0,0])[1]:.4f}] |
"""
    t4_tex = f"""\\begin{{table}}[h]
\\centering
\\caption{{5-Fold Cross-Validation Performance Metrics}}
\\label{{tab:cross_validation}}
\\begin{{tabular}}{{lccc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Mean}} & \\textbf{{Std Dev}} & \\textbf{{95\\% Confidence Interval}} \\\\ \\hline
MAE (pm) & {cv_data.get('MAE', {}).get('mean', 0.0):.4f} & {cv_data.get('MAE', {}).get('std', 0.0):.4f} & [{cv_data.get('MAE', {}).get('ci95', [0,0])[0]:.4f}, {cv_data.get('MAE', {}).get('ci95', [0,0])[1]:.4f}] \\\\
RMSE (pm) & {cv_data.get('RMSE', {}).get('mean', 0.0):.4f} & {cv_data.get('RMSE', {}).get('std', 0.0):.4f} & [{cv_data.get('RMSE', {}).get('ci95', [0,0])[0]:.4f}, {cv_data.get('RMSE', {}).get('ci95', [0,0])[1]:.4f}] \\\\
$R^2$ & {cv_data.get('R2', {}).get('mean', 0.0):.4f} & {cv_data.get('R2', {}).get('std', 0.0):.4f} & [{cv_data.get('R2', {}).get('ci95', [0,0])[0]:.4f}, {cv_data.get('R2', {}).get('ci95', [0,0])[1]:.4f}] \\\\ \\hline
\\end{{tabular}}
\\end{{table}}"""
    save_table_files("table4_cross_validation", t4_tex, t4_md)

    # Table 5: Bootstrap Statistics
    print("Generating Table 5: Bootstrap Statistics...")
    bs_data = results.get("Phase9_Bootstrap", {})
    t5_md = f"""# Table 5: Bootstrap Resampling Statistics (N=1000)

| Metric | Mean | Std Dev | 95% Confidence Interval |
| :--- | :--- | :--- | :--- |
| **MAE (pm)** | {bs_data.get('MAE', {}).get('mean', 0.0):.4f} | {bs_data.get('MAE', {}).get('std', 0.0):.4f} | [{bs_data.get('MAE', {}).get('ci95', [0,0])[0]:.4f}, {bs_data.get('MAE', {}).get('ci95', [0,0])[1]:.4f}] |
| **RMSE (pm)** | {bs_data.get('RMSE', {}).get('mean', 0.0):.4f} | {bs_data.get('RMSE', {}).get('std', 0.0):.4f} | [{bs_data.get('RMSE', {}).get('ci95', [0,0])[0]:.4f}, {bs_data.get('RMSE', {}).get('ci95', [0,0])[1]:.4f}] |
"""
    t5_tex = f"""\\begin{{table}}[h]
\\centering
\\caption{{Bootstrap Resampling 95\\% Confidence Intervals (1,000 Iterations)}}
\\label{{tab:bootstrap_stats}}
\\begin{{tabular}}{{lccc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Mean}} & \\textbf{{Std Dev}} & \\textbf{{95\\% Confidence Interval}} \\\\ \\hline
MAE (pm) & {bs_data.get('MAE', {}).get('mean', 0.0):.4f} & {bs_data.get('MAE', {}).get('std', 0.0):.4f} & [{bs_data.get('MAE', {}).get('ci95', [0,0])[0]:.4f}, {bs_data.get('MAE', {}).get('ci95', [0,0])[1]:.4f}] \\\\
RMSE (pm) & {bs_data.get('RMSE', {}).get('mean', 0.0):.4f} & {bs_data.get('RMSE', {}).get('std', 0.0):.4f} & [{bs_data.get('RMSE', {}).get('ci95', [0,0])[0]:.4f}, {bs_data.get('RMSE', {}).get('ci95', [0,0])[1]:.4f}] \\\\ \\hline
\\end{{tabular}}
\\end{{table}}"""
    save_table_files("table5_bootstrap_statistics", t5_tex, t5_md)

    # Table 6: Noise Robustness
    print("Generating Table 6: Noise Robustness...")
    n_data = results.get("Phase5_NoiseRobustness", {})
    t6_md = """# Table 6: Noise Robustness under Gaussian Noise Injection

| Noise Level (%) | Proposed PINN MAE (pm) | Proposed PINN RMSE (pm) | Proposed PINN R² |
| :--- | :--- | :--- | :--- |
"""
    t6_rows_tex = ""
    for n_lvl, n_dict in n_data.items():
        pinn_m = n_dict.get("Proposed PINN", n_dict) if "Proposed PINN" in n_dict else n_dict
        t6_md += f"| **{n_lvl}** | {pinn_m.get('MAE', 0):.4f} | {pinn_m.get('RMSE', 0):.4f} | {pinn_m.get('R2', 0):.4f} |\n"
        t6_rows_tex += f"{n_lvl} & {pinn_m.get('MAE', 0):.4f} & {pinn_m.get('RMSE', 0):.4f} & {pinn_m.get('R2', 0):.4f} \\\\\n"

    t6_tex = f"""\\begin{{table}}[h]
\\centering
\\caption{{Performance Degradation Under Simulated Gaussian Noise Injection (Proposed PINN)}}
\\label{{tab:noise_robustness}}
\\begin{{tabular}}{{lccc}}
\\hline
\\textbf{{Noise Level}} & \\textbf{{MAE (pm)}} & \\textbf{{RMSE (pm)}} & \\textbf{{$R^2$}} \\\\ \\hline
{t6_rows_tex}\\hline
\\end{{tabular}}
\\end{{table}}"""
    save_table_files("table6_noise_robustness", t6_tex, t6_md)

    # Table 7: Small-Data Experiment
    print("Generating Table 7: Small-Data Experiment...")
    sd_data = results.get("Phase6_SmallData", {})
    t7_md = """# Table 7: Low-Data Regime Performance

| Training Data (%) | Proposed PINN MAE (pm) | Proposed PINN RMSE (pm) | Proposed PINN R² |
| :--- | :--- | :--- | :--- |
"""
    t7_rows_tex = ""
    for sd_lvl, sd_dict in sd_data.items():
        pinn_m = sd_dict.get("Proposed PINN", sd_dict) if "Proposed PINN" in sd_dict else sd_dict
        t7_md += f"| **{sd_lvl}** | {pinn_m.get('MAE', 0):.4f} | {pinn_m.get('RMSE', 0):.4f} | {pinn_m.get('R2', 0):.4f} |\n"
        t7_rows_tex += f"{sd_lvl} & {pinn_m.get('MAE', 0):.4f} & {pinn_m.get('RMSE', 0):.4f} & {pinn_m.get('R2', 0):.4f} \\\\\n"

    t7_tex = f"""\\begin{{table}}[h]
\\centering
\\caption{{Performance in Low-Data Regimes across Training Set Subsamples (Proposed PINN)}}
\\label{{tab:small_data}}
\\begin{{tabular}}{{lccc}}
\\hline
\\textbf{{Training Data (\%)}} & \\textbf{{MAE (pm)}} & \\textbf{{RMSE (pm)}} & \\textbf{{$R^2$}} \\\\ \\hline
{t7_rows_tex}\\hline
\\end{{tabular}}
\\end{{table}}"""
    save_table_files("table7_small_data_experiment", t7_tex, t7_md)

    # Table 8: Ablation Study
    print("Generating Table 8: Ablation Study...")
    a_data = results.get("Phase4_Ablation", {})
    t8_md = """# Table 8: Physics Loss Weight (λ) Ablation Study

| Model Variant | Physics Weight (λ) | MAE (pm) | RMSE (pm) | R² |
| :--- | :--- | :--- | :--- | :--- |
"""
    t8_rows_tex = ""
    for a_key, a_obj in a_data.items():
        lbl = a_obj.get("label", a_key)
        lam_val = a_obj.get("lambda", 0.0)
        m = a_obj.get("metrics", {})
        t8_md += f"| **{lbl}** | {lam_val} | {m.get('MAE',0):.4f} | {m.get('RMSE',0):.4f} | {m.get('R2',0):.4f} |\n"
        t8_rows_tex += f"{lbl} & {lam_val} & {m.get('MAE',0):.4f} & {m.get('RMSE',0):.4f} & {m.get('R2',0):.4f} \\\\\n"

    t8_tex = f"""\\begin{{table}}[h]
\\centering
\\caption{{Ablation Study Evaluating the Role of Physics Loss Weight \\lambda}}
\\label{{tab:ablation_study}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Model Variant}} & \\textbf{{\\lambda}} & \\textbf{{MAE (pm)}} & \\textbf{{RMSE (pm)}} & \\textbf{{$R^2$}} \\\\ \\hline
{t8_rows_tex}\\hline
\\end{{tabular}}
\\end{{table}}"""
    save_table_files("table8_ablation_study", t8_tex, t8_md)

    # Table 9: Comparison with Existing FBG Methods
    print("Generating Table 9: Comparison with Literature...")
    t9_md = """# Table 9: Comparison with Existing FBG Strain-Temperature Decoupling Methods

| Method | Single FBG Sensor | Artificial Intelligence | Physics Loss Constraint | Experimental Verification |
| :--- | :--- | :--- | :--- | :--- |
| **Dual FBG Array Matrix Inversion** | No (Requires 2 FBGs) | No | Yes (Analytical) | Yes |
| **FBG + Auxiliary Thermocouple** | No (Requires Probe) | No | Yes (Analytical) | Yes |
| **Standard ANN / MLP** | Yes (Single FBG) | Yes | No (Black-box) | Yes |
| **Support Vector Regression (SVR)**| Yes (Single FBG) | Yes | No (Black-box) | Yes |
| **Random Forest Regression** | Yes (Single FBG) | Yes | No (Black-box) | Yes |
| **Gaussian Process Regression** | Yes (Single FBG) | Yes (Probabilistic) | Partial (Kernel) | Yes |
| **Proposed Measurement-Constrained PINN** | **Yes (Single FBG)** | **Yes (SciML)** | **Yes (Analytical Loss)** | **Yes** |
"""
    t9_tex = """\\begin{table}[h]
\\centering
\\caption{Methodological Feature Comparison Between Proposed PINN and Literature Decoupling Schemes}
\\label{tab:literature_comparison}
\\begin{tabular}{lcccc}
\\hline
\\textbf{Method} & \\textbf{Single FBG} & \\textbf{AI Model} & \\textbf{Physics Loss} & \\textbf{Experimental} \\\\ \\hline
Dual FBG Matrix Inversion & No & No & Yes & Yes \\\\
FBG + Thermocouple & No & No & Yes & Yes \\\\
Standard ANN / MLP & Yes & Yes & No & Yes \\\\
Support Vector Regression & Yes & Yes & No & Yes \\\\
Random Forest Regression & Yes & Yes & No & Yes \\\\
Gaussian Process Regression & Yes & Yes & Partial & Yes \\\\
\\textbf{Proposed PINN Framework} & \\textbf{Yes} & \\textbf{Yes} & \\textbf{Yes} & \\textbf{Yes} \\\\ \\hline
\\end{tabular}
\\end{table}"""
    save_table_files("table9_literature_comparison", t9_tex, t9_md)

    print(f"\nAll 9 tables successfully generated in: {TABLES_DIR}")

if __name__ == "__main__":
    main()
