

# Quantum Stability of Parity-Broken Asymmetric Oscillators via Spectral Curvature

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/Accelerated_by-JAX-orange?logo=google&logoColor=white)
![Physics](https://img.shields.io/badge/Field-Asymmetric_Quantum_Physics-ff69b4)
![License](https://img.shields.io/badge/license-MIT-green)

## 1. Abstract
This computational framework investigates the structural stability of an **Asymmetric Cubic-Sextic Quantum Oscillator**. Unlike symmetric models, this system incorporates a cubic non-linearity ($\beta x^3$) that breaks parity symmetry, leading to non-zero displacement expectations and skewness in the ground-state wave function. Using **JAX-accelerated spectral methods**, the tool maps the stability boundaries in the $(\beta, \alpha)$ parameter space by analyzing the positive-definiteness of the curvature matrix $K$.

## 2. Project Structure (Root Tree)
```text
Duffing-Oscillator-via-Spectral-Curvature/
├── .gitignore               # Excludes temporary Python & JAX files
├── LICENSE                  # MIT License full text
├── README.md                # Project documentation (Journal-style)
├── requirements.txt         # Dependencies (JAX, NumPy)
├── results/                 # Output logs and simulation data
└── src/                     
    ├── duffing_q1_curvature_jax.py          # Symmetric case
    └── asymmetric_cubic_sextic_jax.py       # Asymmetric (Parity-Broken) case
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
The system is defined by a Hamiltonian that incorporates quadratic, cubic, quartic, and sextic terms:
$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2 + \beta\hat{x}^3 + \alpha\hat{x}^4 + \gamma\hat{x}^6$$

Where:
*   **$\beta x^3$**: The asymmetry term. It "tilts" the potential well, forcing $\langle x \rangle \neq 0$.
*   **$\alpha x^4$**: The quartic term, whose stability boundary $\alpha^*$ is the primary object of study.
*   **$\gamma x^6$**: The sextic term, ensuring global stability as $x \to \pm \infty$.

## 4. Methodology & Observables
### 4.1 Stability Curvature Matrix
Stability is determined by the minimum eigenvalue of the 2x2 curvature matrix $K$:
$$K_{xx} = m\omega^2 + 6\beta\langle x \rangle + 12\alpha\langle x^2 \rangle + 30\gamma\langle x^4 \rangle$$
The system remains stable as long as $\lambda_{min}(K) > 0$.

### 4.2 Higher-Order Moments
The broken parity necessitates the study of:
*   **Skewness ($\mathcal{S}$)**: Measured via $\langle x^3 \rangle / \langle x^2 \rangle^{1.5}$, indicating the asymmetry of the probability distribution.
*   **Kurtosis ($\mathcal{K}$)**: Measured via $\langle x^4 \rangle / \langle x^2 \rangle^2$, indicating the "fatness" of the tails of the wave function.

## 5. Implementation Highlights
*   **Non-NaN Boundary Collection:** Integrated logic to skip regions where no boundary is found, ensuring continuous execution.
*   **Bisection Algorithm:** A 40-iteration refinement process to locate the critical $\alpha^*$ with a precision of $10^{-4}$.
*   **Fock Space Scaling:** Lightweight $N=64$ configuration to optimize memory for boundary scanning across $\beta$ grids.

## 6. Example Research Output
Typical output when running the asymmetric stability scan:

```text
[Boundary count] found=25 of 25 betas
[Boundary] β range=-1.200..1.200 | α* range=-2.215..-1.845
[Observables @ boundary] <x>: mean(|x|)=4.5021e-01 | <x^2>: min=8.2104e-01, max=1.1245e+00 | skew: mean=1.2405e-01 | kurtosis: mean=4.1204
```

## 7. License
This project is licensed under the **MIT License**. (See the LICENSE file for details).

## 8. Citation
> **Hardiyan, H. (2026).** *Stability of Parity-Broken Quantum Oscillators via JAX-Accelerated Curvature Matrices.* GitHub: harihardiyan/Duffing-Oscillator-via-Spectral-Curvature.

---
*Developed with AI Tamer and Microsoft Copilot.*

---

 
