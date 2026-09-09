**TASK 1 — DO FIRST: Cycle-wise Train/Test Split**

**Objective**

The current sample-level random train/test split may allow samples from the **same experimental cycle** to appear in both training and testing. This can lead to data leakage, especially because the FBG measurements are time-series/correlated data.

We need to demonstrate that the models generalize to **unseen experimental cycles**.

**What you need to do**

1. Identify the **cycle/experiment ID** for every sample in the dataset.
   - Do not randomly split individual samples.
   - Samples belonging to the same experimental cycle must remain together.
2. Create a **grouped/cycle-wise split**.

For example:

- 70–80% of cycles → Training
- Remaining 20–30% of cycles → Testing

**Important:** No samples from a test cycle should appear in training.

1. Use exactly the **same cycle-wise train/test split for all six models**:
   1. Linear Regression
   2. MLP
   3. SVR
   4. GPR
   5. Random Forest
   6. PINN
2. Keep the following identical wherever applicable:

- Dataset
- Input variables
- Preprocessing
- Normalization
- Training/test cycles
- Random seed
- Evaluation metrics

1. Calculate:

- MAE (pm)
- RMSE (pm)
- R²

**Required output**

Create a table:

| **Model**         | **MAE (pm)** | **RMSE (pm)** | **R²** |
| ----------------- | ------------ | ------------- | ------ |
| Linear Regression |              |               |        |
| MLP               |              |               |        |
| SVR               |              |               |        |
| GPR               |              |               |        |
| Random Forest     |              |               |        |
| PINN              |              |               |        |

**Also provide**

A figure showing the cycle allocation, for example:

**Training cycles → \[Cycle 1, 2, 3, ...\]**

**Testing cycles → \[Cycle ..., ...\]**

**Critical requirement**

Do **not** simply change the train/test split and report the result without documenting which cycles were used.

Save:

- cycle IDs used for training
- cycle IDs used for testing
- random seed
- resulting metrics
- trained model/results

**TASK 2 — Cycle-wise K-Fold Validation**

**Objective**

Perform robust validation where the **entire experimental cycle is treated as one group**.

Do NOT use ordinary random K-fold where samples from the same cycle can occur in both training and validation folds.

**What you need to do**

Use preferably:

**5-fold Group/Cycle-wise Cross-Validation**

For each fold:

**Fold 1:** 4 groups training + 1 group validation  
**Fold 2:** another group validation  
...  
**Fold 5:** remaining group validation

The exact number of folds can be adjusted depending on the number of experimental cycles.

**For each model**

Run:

1. Linear Regression
2. MLP
3. SVR
4. GPR
5. Random Forest
6. PINN

**For every fold calculate**

- MAE
- RMSE
- R²

Then calculate:

**Mean ± Standard Deviation**

For example:

| **Model**         | **MAE (pm)** | **RMSE (pm)** | **R²**    |
| ----------------- | ------------ | ------------- | --------- |
| Linear Regression | mean ± SD    | mean ± SD     | mean ± SD |
| MLP               | mean ± SD    | mean ± SD     | mean ± SD |
| SVR               | mean ± SD    | mean ± SD     | mean ± SD |
| GPR               | mean ± SD    | mean ± SD     | mean ± SD |
| Random Forest     | mean ± SD    | mean ± SD     | mean ± SD |
| PINN              | mean ± SD    | mean ± SD     | mean ± SD |

**Important**

The same cycle/group assignment must be used for comparing all six models.

**TASK 3 — Time-Input Ablation**

**Objective**

We need to establish whether **time is actually necessary as a model input**.

The paper should compare:

**Model A — ΔλB only**

Input:

FBG wavelength shift ΔλB

**Model B — ΔλB + time**

Inputs:

FBG wavelength shift ΔλB + time

**Keep everything else identical**

Use the same:

- Architecture
- Number of layers
- Number of neurons
- Activation function
- Optimizer
- Learning rate
- Epochs
- Batch size
- Training data
- Test data
- Random seed
- Physics-loss formulation

**Required results**

| **Input configuration** | **MAE (pm)** | **RMSE (pm)** | **R²** |
| ----------------------- | ------------ | ------------- | ------ |
| ΔλB only                |              |               |        |
| ΔλB + time              |              |               |        |

**Interpretation**

If ΔλB-only performs similarly or better:

**Prefer ΔλB-only as the final model.**

This strengthens the claim that the model operates using the **single physical FBG sensing channel**.

If ΔλB + time performs substantially better:

Do not hide this.

Instead clearly state:

"Time was retained as an auxiliary temporal input because it improved reconstruction performance."

Also make sure the manuscript does **not** call the model a "single-input" model if time is actually an input.

**TASK 4 — Complete Physics-Loss Ablation**

**Objective**

Demonstrate that the **physics constraint actually contributes to the model performance**.

Currently λ = 0.1, 1 and 10 have been tested.

Complete the experiment with:

**λ values:**

**0, 0.01, 0.1, 1, 10, 100**

**Critical case**

**λ = 0**

means:

No physics loss / data-driven model without the physics constraint.

This is extremely important because it provides the direct comparison:

**Without physics constraint vs with physics constraint**

**Keep everything else identical**

- Same dataset
- Same train/test split
- Same architecture
- Same optimizer
- Same learning rate
- Same epochs
- Same preprocessing
- Same random seed

**Required table**

| **λ** | **MAE (pm)** | **RMSE (pm)** | **R²** |
| ----- | ------------ | ------------- | ------ |
| 0     |              |               |        |
| 0.01  |              |               |        |
| 0.1   |              |               |        |
| 1     |              |               |        |
| 10    |              |               |        |
| 100   |              |               |        |

**Required figure**

Plot:

**λ vs RMSE**

and preferably:

**λ vs MAE**

Use a logarithmic x-axis if appropriate because λ spans several orders of magnitude.

**Important**

Do not choose the best λ only because it produces the lowest error and then ignore the others.

The complete ablation should show the performance trend.

**TASK 5 — Residual / Failure-Case Analysis**

**Objective**

Show **where the PINN makes errors**, rather than reporting only one MAE/RMSE number.

Use the **final selected PINN model**.

**Figure 1 — Measured vs reconstructed wavelength**

Plot:

- Measured ΔλB
- PINN reconstructed ΔλB

Preferably against sample/time.

**Figure 2 — Residual vs wavelength**

Calculate:

Residual = Measured ΔλB − Predicted ΔλB

Plot:

**Residual vs measured ΔλB**

Add a horizontal zero-error reference line.

Check whether errors increase at:

- low wavelength
- high wavelength
- transition regions
- specific operating ranges

**Figure 3 — Residual vs time**

Plot:

**Residual vs time**

Look for:

- temporal drift
- systematic patterns
- periodic errors
- errors concentrated in particular cycles

**Figure 4 — Residual histogram**

Plot distribution of residuals.

Report:

- Mean residual
- Standard deviation
- MAE
- RMSE

**Worst-case analysis**

Identify the **top 10 or top 20 largest absolute errors**.

Create a table:

| **Rank** | **Cycle** | **Time** | **Measured ΔλB** | **Predicted ΔλB** | **Absolute Error** |
| -------- | --------- | -------- | ---------------- | ----------------- | ------------------ |
| 1        |           |          |                  |                   |                    |
| 2        |           |          |                  |                   |                    |
| ...      |           |          |                  |                   |                    |

Then determine whether the worst cases correspond to:

- transition regions
- high strain
- high temperature
- noisy measurements
- cycle boundaries
- unusual operating conditions

**Do not invent an explanation.** If there is no obvious physical reason, state that.

**TASK 6 — Block Bootstrap**

**Objective**

The sensor data are temporally/cycle correlated, so ordinary sample-by-sample bootstrap can underestimate uncertainty.

Use **block bootstrap** instead.

**Recommended procedure**

Use:

**1,000 or 5,000 block-bootstrap resamples**

Preferably create blocks based on:

- experimental cycle, or
- contiguous temporal segments

depending on how the dataset is organized.

**Important**

Do NOT randomly select individual samples independently.

Instead:

Select entire blocks/cycles with replacement.

For each bootstrap sample:

1. Evaluate the model.
2. Calculate MAE.
3. Calculate RMSE.
4. Store the result.

After 1,000/5,000 repetitions calculate:

**95% confidence interval**

For example:

PINN MAE = 1.99 pm  
95% CI = \[lower, upper\] pm

and:

PINN RMSE = 3.05 pm  
95% CI = \[lower, upper\] pm

**Required table**

| **Model** | **Metric** | **Mean** | **95% CI** |
| --------- | ---------- | -------- | ---------- |
| PINN      | MAE        |          |            |
| PINN      | RMSE       |          |            |
| PINN      | R²         |          |            |

If feasible, provide the same for all six models.

**TASK 7 — Repeat Data-Efficiency Experiment**

The existing experiment uses:

- 20%
- 40%
- 60%
- 80%
- 100%

training data.

The problem is that the current results are somewhat non-monotonic.

For example, the current RMSE values were approximately:

20% → 1.24 pm  
40% → 5.02 pm  
60% → 1.82 pm  
80% → 1.71 pm  
100% → 1.12 pm

This could simply be random variation from a single split.

**Preferred approach**

For each training fraction:

**20%, 40%, 60%, 80%, 100%**

run **5 independent random seeds**.

For each fraction calculate:

- Mean MAE
- SD of MAE
- Mean RMSE
- SD of RMSE
- Mean R²
- SD of R²

**Required table**

| **Training data** | **MAE (mean ± SD)** | **RMSE (mean ± SD)** | **R² (mean ± SD)** |
| ----------------- | ------------------- | -------------------- | ------------------ |
| 20%               |                     |                      |                    |
| 40%               |                     |                      |                    |
| 60%               |                     |                      |                    |
| 80%               |                     |                      |                    |
| 100%              |                     |                      |                    |

**Required figure**

Plot:

**Training data (%) vs RMSE**

with error bars representing SD.

**If 5 seeds are computationally expensive**

At minimum, report the existing single-run result honestly as:

"single-run data-efficiency sensitivity analysis"

rather than presenting it as statistically robust evidence.

**TASK 8 — FINAL CONSISTENCY CHECK**

This needs to be done **after all experiments are completed**.

There are currently several numbers in the manuscript that come from different experiments.

For example:

- **0.51 pm RMSE**
- **0.61 pm RMSE**
- **1.12 pm RMSE**
- **1.99 pm MAE**
- **3.05 pm RMSE**

These are not necessarily contradictory because they may come from **different experimental protocols**.

But the manuscript must make this absolutely clear.

**Create a master results sheet**

Before modifying the manuscript, prepare one Excel/CSV table containing:

| **Result**   | **Dataset** | **Train/Test protocol** | **Noise** | **λ** | **Input** | **Purpose** |
| ------------ | ----------- | ----------------------- | --------- | ----- | --------- | ----------- |
| 0.51 pm RMSE |             |                         |           |       |           |             |
| 0.61 pm RMSE |             |                         |           |       |           |             |
| 1.12 pm RMSE |             |                         |           |       |           |             |
| 1.99 pm MAE  |             |                         |           |       |           |             |
| 3.05 pm RMSE |             |                         |           |       |           |             |

For **every number appearing in the manuscript**, record:

1. Which experiment produced it?
2. Which dataset?
3. Which train/test split?
4. Which λ?
5. Whether noise was applied.
6. Whether input was ΔλB or ΔλB + time.
7. Whether it is validation, test, ablation, or robustness result.