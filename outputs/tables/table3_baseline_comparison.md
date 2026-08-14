# Table 3: Machine Learning Baseline Comparison Table

| Model | MAE (pm) | RMSE (pm) | R² | Training Time (s) | Inference Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 147.3706 | 173.6049 | 0.6591 | 0.0008 | 0.0001 |
| **Random Forest** | 1.8645 | 6.7924 | 0.9995 | 0.1938 | 0.0310 |
| **SVR** | 22.9181 | 47.2421 | 0.9748 | 1.4724 | 0.6307 |
| **MLP** | 20.9321 | 44.1278 | 0.9780 | 7.2624 | 0.0031 |
| **Gaussian Process** | 691.0143 | 751.1688 | -5.3814 | 0.5376 | 0.0579 |
| **Proposed PINN** | 0.3897 | 0.4993 | 1.0000 | 5.0680 | 0.0012 |
