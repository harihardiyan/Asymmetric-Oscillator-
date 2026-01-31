
# Structural Stability of Parity-Broken Asymmetric Quantum Oscillators via Spectral Curvature Matrices

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/Accelerated_by-JAX-orange?logo=google&logoColor=white)
![Physics](https://img.shields.io/badge/Field-Asymmetric_Quantum_Physics-ff69b4)
![License](https://img.shields.io/badge/license-MIT-green)

## 1. Abstract
This repository provides an advanced computational framework to investigate the structural stability and quantum observables of an **Asymmetric Cubic-Sextic Oscillator**. Unlike standard symmetric models, this system incorporates a cubic non-linearity ($\beta x^3$) that breaks parity symmetry ($\hat{P}$), leading to non-trivial ground-state displacements and skewness. Utilizing **JAX-accelerated spectral methods**, the engine maps stability boundaries in the $(\beta, \alpha)$ parameter space, ensuring high-fidelity results through double-precision (x64) calculations and automated bisection algorithms.

## 2. Project Structure (Root Tree)
```text
Asymmetric-Oscillator/
├── .gitignore               # Excludes JAX cache and temporary files
├── LICENSE                  # MIT License (2026)
├── README.md                # Scientific documentation
├── requirements.txt         # Core dependencies (JAX, NumPy)
└── src/                     
    └── asymmetric_cubic_sextic_jax.py  # Principal simulation engine
```

---

<p align="center">
  <b>Author:</b> Hari Hardiyan <br>
  <b>Email:</b> <a href="mailto:lorozloraz@gmail.com">lorozloraz@gmail.com</a> <br><br>
  <b>Lead AI Development:</b> AI Tamer <br>
  <b>Assistant:</b> Microsoft Copilot
</p>

---

## 3. Physical Model: The Asymmetric Hamiltonian
The system investigates a particle governed by a parity-broken potential, defined by the following sextic Hamiltonian:
$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2 + \beta\hat{x}^3 + \alpha\hat{x}^4 + \gamma\hat{x}^6$$

Where:
*   **$\beta \hat{x}^3$**: The cubic asymmetry term responsible for symmetry breaking.
*   **$\alpha \hat{x}^4$**: The quartic parameter, whose critical stability threshold ($\alpha^*$) is numerically determined.
*   **$\gamma \hat{x}^6$**: The sextic stabilization term, ensuring the potential remains bounded for large displacements.

## 4. Methodology
### 4.1 Stability via Curvature Analysis
Stability is rigorously defined by the positive-definiteness of the curvature matrix $K$. In the asymmetric regime, the spatial curvature $K_{xx}$ is sensitive to the first and second moments of the ground state:
$$K_{xx} = m\omega^2 + 6\beta\langle x \rangle + 12\alpha\langle x^2 \rangle + 30\gamma\langle x^4 \rangle$$
The software identifies the boundary where the minimum eigenvalue $\lambda_{min}(K) \to 0$.

### 4.2 Quantum Observables
Beyond stability, the engine computes higher-order spectral moments:
*   **Skewness ($\mathcal{S}$)**: $\langle \hat{x}^3 \rangle / \langle \hat{x}^2 \rangle^{1.5}$, quantifying the asymmetry of the wave function.
*   **Kurtosis ($\mathcal{K}$)**: $\langle \hat{x}^4 \rangle / \langle \hat{x}^2 \rangle^2$, measuring the "fatness" of the distribution tails.
*   **Displacement ($\langle x \rangle$)**: The expectation value of the position operator, which vanishes in symmetric systems but is non-zero here.

## 5. Implementation Highlights
*   **JAX vmap & Bisection:** Parallelized grid scanning for initial boundary detection, followed by a 40-step bisection algorithm to refine $\alpha^*$ with $10^{-4}$ precision.
*   **Non-NaN Collection:** A robust pipeline that filters and collects only valid physical boundaries across the $\beta$ grid.
*   **Memory Optimization:** Uses a lightweight basis size ($N=64$) for rapid phase-boundary mapping without sacrificing spectral convergence in the low-energy regime.

## 6. Example Output & Results
The following metrics represent a typical execution for $\gamma=0.006$ and $\omega=1.0$ across a $\beta$ range of $[-1.2, 1.2]$:

```text
[Boundary count] found=25 of 25 betas
[Boundary] β range=-1.200..1.200 | α* range=-1.663..-1.606
[Observables @ boundary] <x>: mean(|x|)=1.0105e+01 | <x^2>: min=1.1080e+02, max=1.1080e+02 | skew: mean=-1.7764e-17 | kurtosis: mean=1.0000

```

## 7. License
This project is licensed under the **MIT License**.

Copyright (c) 2026 Hari Hardiyan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions...

## 8. Citation
If this research code is used in scientific publications, please cite as:
> **Hardiyan, H. (2026).** *Structural Stability of Parity-Broken Quantum Oscillators via JAX-Accelerated Curvature Matrices.* GitHub: [harihardiyan/Asymmetric-Oscillator-](https://github.com/harihardiyan/Asymmetric-Oscillator-).

---
*Facilitated by AI Tamer and Microsoft Copilot as part of a computational physics development initiative.*

---

