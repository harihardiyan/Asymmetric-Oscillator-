
# asymmetric_cubic_sextic_curvature_nonan_light_fixed_jax.py
# Full JAX, x64: asymmetric sextic oscillator (parity broken by x^3)
# Non-NaN boundary collection, lightweight, no jit on N-dependent builders

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

hbar = 1.0
m    = 1.0

# =========================
# Operators and Hamiltonian
# =========================
def create_fock_ops(N):
    n = jnp.arange(N)
    s = jnp.sqrt(n[1:])
    a    = jnp.zeros((N, N), dtype=jnp.complex128)
    adag = jnp.zeros((N, N), dtype=jnp.complex128)
    a    = a.at[1:, :-1].set(jnp.diag(s).astype(jnp.complex128))
    adag = adag.at[:-1, 1:].set(jnp.diag(s).astype(jnp.complex128))
    return a, adag

def x_op(N, omega_ref=1.0):
    a, adag = create_fock_ops(N)
    scale = jnp.sqrt(hbar/(2*m*omega_ref))
    return scale * (a + adag)

def p_op(N, omega_ref=1.0):
    a, adag = create_fock_ops(N)
    scale = 1j*jnp.sqrt(m*hbar*omega_ref/2.0)
    return scale * (adag - a)

def H_asym_cubic_sextic(N, omega, beta, alpha, gamma):
    X = x_op(N)
    P = p_op(N)
    X2 = X @ X
    X3 = X2 @ X
    X4 = X2 @ X2
    X6 = X4 @ X2
    H = (P @ P)/(2*m) + 0.5*m*(omega**2)*X2 + beta*X3 + alpha*X4 + gamma*X6
    return H, X, X2, X3, X4

def ground_state(N, omega, beta, alpha, gamma):
    H, X, X2, X3, X4 = H_asym_cubic_sextic(N, omega, beta, alpha, gamma)
    E, U = jnp.linalg.eigh(H)
    psi0 = U[:, 0]
    psi0 = psi0 / jnp.linalg.norm(psi0)
    return X, X2, X3, X4, psi0

def curvature_and_moments(N, omega, beta, alpha, gamma):
    X, X2, X3, X4, psi0 = ground_state(N, omega, beta, alpha, gamma)
    x  = jnp.real(jnp.vdot(psi0, X  @ psi0))
    x2 = jnp.real(jnp.vdot(psi0, X2 @ psi0))
    x3 = jnp.real(jnp.vdot(psi0, X3 @ psi0))
    x4 = jnp.real(jnp.vdot(psi0, X4 @ psi0))
    K_xx = m*(omega**2) + 6.0*beta*x + 12.0*alpha*x2 + 30.0*gamma*x4
    K_pp = 1.0/m
    K = jnp.array([[K_xx, 0.0],[0.0, K_pp]], dtype=jnp.float64)
    return K, x, x2, x3, x4

def eigen_min(K):
    return jnp.min(jnp.linalg.eigvalsh(K))

# =========================
# Validated boundary (no-NaN collection)
# =========================
def find_boundary_alpha_valid(N, omega, beta, gamma, alpha_lo, alpha_hi, n_grid=96):
    alphas = jnp.linspace(alpha_lo, alpha_hi, n_grid)

    def f(a):
        K, *_ = curvature_and_moments(N, omega, beta, a, gamma)
        return eigen_min(K)

    emin = jax.vmap(f)(alphas)
    signs = jnp.sign(emin)
    cross = jnp.where(signs[:-1] * signs[1:] < 0, 1, 0)
    if int(jnp.sum(cross)) == 0:
        return None  # no boundary in bracket

    idx = int(jnp.argmax(cross))
    alo = float(alphas[idx]); ahi = float(alphas[idx+1])

    for _ in range(40):
        amid = 0.5*(alo + ahi)
        K_mid, *_ = curvature_and_moments(N, omega, beta, amid, gamma)
        e_mid = float(eigen_min(K_mid))
        if e_mid >= 0.0:
            ahi = amid
        else:
            alo = amid
        if (ahi - alo) < 1e-4:
            break
    return 0.5*(alo + ahi)

def observables_at(N, omega, beta, alpha, gamma):
    X, X2, X3, X4, psi0 = ground_state(N, omega, beta, alpha, gamma)
    x  = jnp.real(jnp.vdot(psi0, X  @ psi0))
    x2 = jnp.real(jnp.vdot(psi0, X2 @ psi0))
    x3 = jnp.real(jnp.vdot(psi0, X3 @ psi0))
    x4 = jnp.real(jnp.vdot(psi0, X4 @ psi0))
    skew = x3 / jnp.maximum(x2**1.5, 1e-18)
    kurt = x4 / jnp.maximum(x2**2, 1e-18)
    return x, x2, skew, kurt

# =========================
# Main (lightweight, no-NaN)
# =========================
if __name__ == "__main__":
    # Lightweight config to avoid OOM but show a real boundary
    N = 64
    omega = 1.0
    gamma = 0.006
    beta_grid = jnp.linspace(-1.2, 1.2, 25)
    alpha_bracket = (-3.0, 0.3)

    betas_valid = []
    alpha_star  = []
    x_mean_list = []
    x2_list     = []
    skew_list   = []
    kurt_list   = []

    for i in range(beta_grid.shape[0]):
        b = float(beta_grid[i])
        a_star = find_boundary_alpha_valid(N, omega, b, gamma, alpha_bracket[0], alpha_bracket[1], n_grid=96)
        if a_star is None:
            continue  # skip; no boundary for this beta in bracket
        betas_valid.append(b)
        alpha_star.append(float(a_star))
        x, x2, skew, kurt = observables_at(N, omega, b, float(a_star), gamma)
        x_mean_list.append(float(x))
        x2_list.append(float(x2))
        skew_list.append(float(skew))
        kurt_list.append(float(kurt))

    betas_valid = jnp.array(betas_valid)
    alpha_star  = jnp.array(alpha_star)
    x_mean_arr  = jnp.array(x_mean_list)
    x2_arr      = jnp.array(x2_list)
    skew_arr    = jnp.array(skew_list)
    kurt_arr    = jnp.array(kurt_list)

    print(f"[Boundary count] found={betas_valid.shape[0]} of {beta_grid.shape[0]} betas")
    if betas_valid.shape[0] > 0:
        print(f"[Boundary] β range={float(jnp.min(betas_valid)):.3f}..{float(jnp.max(betas_valid)):.3f} | "
              f"α* range={float(jnp.min(alpha_star)):.3f}..{float(jnp.max(alpha_star)):.3f}")
        print(f"[Observables @ boundary] "
              f"<x>: mean(|x|)={float(jnp.mean(jnp.abs(x_mean_arr))):.4e} | "
              f"<x^2>: min={float(jnp.min(x2_arr)):.4e}, max={float(jnp.max(x2_arr)):.4e} | "
              f"skew: mean={float(jnp.mean(skew_arr)):.4e} | "
              f"kurtosis: mean={float(jnp.mean(kurt_arr)):.4f}")
    else:
        print("No boundary found in the given bracket. Consider widening α_bracket or lowering γ.")
