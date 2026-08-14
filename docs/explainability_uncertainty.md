# Phase 12 & 13: Explainability & Uncertainty Analysis

**Document Title:** Physical Regularization, Explainability & Uncertainty Comparison  
**Authors:** AI Research Team  

---

## 1. Phase 12: Explainability of Physics-Guided Learning

### 1.1 Why Physics Constraints Improve Performance
Unconstrained black-box neural networks (such as standard MLPs or deep regressors) minimize purely empirical data loss:
$$\mathcal{L}_{\text{data}} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2$$

Because decoupling a single FBG sensor reading ($\Delta\lambda$) into strain ($\varepsilon$) and temperature ($\Delta T$) is ill-posed, infinitely many linear combinations of $(\hat{\varepsilon}, \hat{T})$ satisfy a given target wavelength shift. A standard data-driven network can overfit to random high-frequency sensor noise, predicting physically impossible state combinations (e.g., negative absolute temperatures or extreme strain spikes during thermal equilibrium).

The **Measurement-Constrained PINN** resolves this non-uniqueness by restricting the optimization search space to physically admissible solution manifolds. Embedding the analytical measurement equation $\Delta\lambda = k_\varepsilon \cdot \varepsilon + k_T \cdot \Delta T$ as an explicit loss penalty:
$$\mathcal{L}_{\text{phys}} = \|\Delta\lambda - (k_\varepsilon \cdot \hat{\varepsilon} + k_T \cdot \hat{T})\|^2$$
serves as a strict physical regularizer. It forces the network's hidden representation gradients to align with optical transducer physics.

---

## 2. Phase 13: Uncertainty Discussion (Gaussian Process vs. PINN)

### 2.1 Statistical Uncertainty vs. Physical Consistency

| Model Family | Primary Mechanism | Type of Assurance | Out-of-Distribution Behavior |
| :--- | :--- | :--- | :--- |
| **Gaussian Process Regression (GPR)** | Non-parametric Kernel Statistics | **Statistical Uncertainty Bounds** ($\pm 1.96\sigma$) | Identifies regions of low sample density via high variance $\sigma(x)$. |
| **Measurement-Constrained PINN** | Physics Loss Constraint Regularization | **Physical Consistency Assurance** ($\mathcal{L}_{\text{phys}} \to 0$) | Bounds predictions to physically admissible transducer state manifolds. |

### 2.2 Complementary Advantages & Future Integration

- **Gaussian Processes** excel at providing quantitative standard deviations ($\sigma$) around predictions, indicating statistical confidence. However, GPR kernel computations scale as $\mathcal{O}(N^3)$ with dataset size $N$.
- **PINNs** excel at enforcing physical conservation laws and measurement equations with fast, $\mathcal{O}(1)$ parallel GPU/CPU inference, but deterministic PINNs do not output variance bounds by default.
- **Future Direction**: Combining GPR covariance estimation with PINN physics loss constraints via **Bayesian Physics-Informed Neural Networks (B-PINNs)** or Monte Carlo Dropout will yield both physical consistency guarantees and statistical error bars.
