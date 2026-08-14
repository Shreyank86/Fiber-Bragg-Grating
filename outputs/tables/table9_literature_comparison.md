# Table 9: Comparison with Existing FBG Strain-Temperature Decoupling Methods

| Method | Single FBG Sensor | Artificial Intelligence | Physics Loss Constraint | Experimental Verification |
| :--- | :--- | :--- | :--- | :--- |
| **Dual FBG Array Matrix Inversion** | No (Requires 2 FBGs) | No | Yes (Analytical) | Yes |
| **FBG + Auxiliary Thermocouple** | No (Requires Probe) | No | Yes (Analytical) | Yes |
| **Standard ANN / MLP** | Yes (Single FBG) | Yes | No (Black-box) | Yes |
| **Support Vector Regression (SVR)**| Yes (Single FBG) | Yes | No (Black-box) | Yes |
| **Random Forest Regression** | Yes (Single FBG) | Yes | No (Black-box) | Yes |
| **Gaussian Process Regression** | Yes (Single FBG) | Yes (Probabilistic) | Partial (Kernel) | Yes |
| **Proposed Measurement-Constrained PINN** | **Yes (Single FBG)** | **Yes (SciML)** | **Yes (Analytical Loss)** | **Yes** |
