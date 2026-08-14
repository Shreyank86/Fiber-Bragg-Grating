# Phase 2: Technical Contribution, Problem Positioning & Reframed Scientific Contributions

**Document Title:** Problem Framing & Scientific Contribution Statement  
**Target Paper:** Measurement-Constrained Physics-Guided Inverse Sensing for Single-Sensor FBG Decoupling  

---

## 1. Objective 1: Problem Positioning as an Inverse Measurement Problem

### 1.1 Conceptual Distinction

In scientific machine learning literature, Physics-Informed Neural Networks (PINNs) are predominantly categorized under forward or inverse partial differential equation (PDE) solving:

$$\text{Forward Physics: } \text{PDE } \mathcal{N}[u](x,t) = 0 \longrightarrow \text{Predict unknown continuous field } u(x,t)$$

In contrast, our work positions FBG strain–temperature decoupling as an **Inverse Measurement Problem** governed by sensor transducer physics:

$$\text{Inverse Measurement: } \text{Measured Wavelength Shift } \Delta\lambda_B \longrightarrow \text{Invert unknown internal states } [\varepsilon, \Delta T]$$

$$\begin{pmatrix} \text{Measured } \Delta\lambda \end{pmatrix} = \begin{pmatrix} k_\varepsilon & k_T \end{pmatrix} \begin{pmatrix} \varepsilon \\ \Delta T \end{pmatrix}$$

### 1.2 Scientific Significance

1. **Under-Determined Inversion**: A single scalar wavelength measurement ($\Delta\lambda$) must be decomposed into two unknown independent scalar states ($\varepsilon, \Delta T$), creating an under-determined system at any single measurement instant.
2. **Measurement Physics Constraint**: The analytical transducer equation provides the governing physical constraint that links predicted hidden states back to observable physical quantities.
3. **Bridge between Instrumentation & SciML**: This reframing extends PINNs from fluid dynamics and continuum mechanics into smart optical sensors and instrumentation-oriented physical machine learning.

---

## 2. Objective 2: Scientific Terminology Replacement

To raise the academic quality of the manuscript, non-rigorous terminology like `"Engineering PINN"` is replaced throughout all documentation and text with scientifically precise terms:

- **Primary Term**: **Measurement-Constrained Physics-Informed Neural Network (MC-PINN)**
- **Framework Term**: **Physics-Guided Inverse Sensing Framework**
- **Domain Term**: **Instrumentation-Oriented Scientific Machine Learning**

---

## 3. Objective 3: Rewritten Scientific Contributions

The original generic contribution statements are replaced with four high-impact, scientifically rigorous contribution statements:

### Contribution 1: Measurement-Constrained Inverse Sensing Framework
> **We formulate a physics-informed inverse measurement framework** that embeds the analytical Fiber Bragg Grating (FBG) optical sensing equation directly into the neural network optimization objective, establishing a novel paradigm for inverse sensing without relying on partial differential equation (PDE) spatial grid discretizations.

### Contribution 2: Auxiliary-Hardware-Free Single-Sensor Decoupling
> **We present a single-sensor decoupling architecture** capable of accurately separating mechanical strain ($\varepsilon$) and thermal variations ($\Delta T$) from a single optical channel, eliminating the cost, deployment footprint, and interrogation complexity of auxiliary sensors or dual-FBG arrays.

### Contribution 3: Rigorous Benchmarking & Statistical Validation
> **We establish a comprehensive benchmarking suite** comparing the proposed framework against standard and advanced machine learning algorithms (Linear Regression, Random Forest, Support Vector Regression, Multi-Layer Perceptrons, and Gaussian Process Regression) across 5-fold cross-validation, 1,000-sample bootstrap confidence intervals, noise robustness sweeps, and data-scarcity regimes.

### Contribution 4: Physical Consistency under Extreme Data Noise and Scarcity
> **We demonstrate experimental verification of physical consistency**, proving that physics-guided loss regularization enables graceful accuracy degradation under high optical measurement noise (up to 10% Gaussian noise) and low data availability (down to 20% training data), outperforming standard data-driven models.
