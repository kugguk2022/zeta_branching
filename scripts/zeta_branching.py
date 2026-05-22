import numpy as np
import matplotlib.pyplot as plt
import mpmath
from numpy import vectorize
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ----------------------------------------------------------------------
# 1. Complex-safe Riemann zeta using mpmath.fp (machine-float speed)
# ----------------------------------------------------------------------
def _zeta_complex(s):
    """mpmath.fp.zeta evaluated pointwise — uses C-speed machine floats.
    Returns NaN+NaN*j for arguments that cause overflow (e.g. Re(s) >> 1)."""
    try:
        return complex(mpmath.fp.zeta(s))
    except (OverflowError, ZeroDivisionError, ValueError):
        return complex(float('nan'), float('nan'))

# Vectorised version (accepts numpy arrays)
zeta_c = vectorize(_zeta_complex, otypes=[complex])

# ----------------------------------------------------------------------
# 2. Controllable branch-cut power function
# ----------------------------------------------------------------------
def branch_pow(z, a, alpha, cut_angle=0.0):
    """
    (z - a)^alpha with a branch cut that can be rotated.
    
    cut_angle : float, radians. 0 = cut along negative real axis (standard).
    """
    w = z - a
    mag = np.abs(w) ** alpha
    # Define the phase with a cut at cut_angle + pi
    phase = np.angle(w)
    # Wrap the phase so that the discontinuity lies at cut_angle + pi
    phase_wrapped = (phase - cut_angle + np.pi) % (2 * np.pi) - np.pi + cut_angle
    return mag * np.exp(1j * alpha * phase_wrapped)

# ----------------------------------------------------------------------
# 3. The improved recursive kernel
# ----------------------------------------------------------------------
def xo2_kernel(z, a=0.0, b=1.25, alpha=0.5, N=3, lam=1.0, cut_angle=0.0):
    """
    One step of the botched Zeta-Geometric kernel.

    f(z) = sum_{n=1}^N [ lam / (1 + exp(w)) ]^n,   w = ζ( (z-a)^α + b )
    """
    # Avoid the pole of ζ at s=1
    shifted = branch_pow(z, a, alpha, cut_angle) + b
    # Evaluate ζ for complex argument (pole at shifted==1 will give inf)
    w = zeta_c(shifted)
    # Geometric series: λ * r + (λ r)^2 + ... + (λ r)^N,  r = 1/(1+exp(w))
    r = lam / (1.0 + np.exp(w))
    # Use the closed form for speed and stability
    if np.isclose(r, 1.0).any():
        # Fallback to loop for problematic points
        f_w = np.sum([r**n for n in range(1, N+1)], axis=0)
    else:
        f_w = r * (1 - r**N) / (1 - r)
    return f_w

def xo2_recursion(z, depth, **kwargs):
    """
    Apply xo2_kernel depth times.
    Returns the final value (complex) for each point in z.
    """
    for _ in range(depth):
        z = xo2_kernel(z, **kwargs)
    return z

# ----------------------------------------------------------------------
# 4. Domain colouring plotter
# ----------------------------------------------------------------------
def domain_color(z, title="", ax=None, saturation=0.85, light_range=(0.3, 1.0)):
    """
    Visualise a complex array z with domain colouring.
    Hue = argument, lightness = f(log10(|z|)).
    """
    H = (np.angle(z) + np.pi) / (2 * np.pi)  # map to [0,1]
    L = np.log10(np.abs(z) + 1e-300)
    # Compress lightness into [0,1]
    L_clipped = np.clip(L, -2, 2)
    L_norm = (L_clipped + 2) / 4  # now 0..1
    L_final = light_range[0] + L_norm * (light_range[1] - light_range[0])

    # Replace NaN/Inf with neutral values before colour mapping
    H = np.nan_to_num(H, nan=0.0)
    L_final = np.nan_to_num(L_final, nan=0.0)

    # HSV to RGB
    from matplotlib.colors import hsv_to_rgb
    hsv = np.stack([H, np.full_like(H, saturation), L_final], axis=-1)
    rgb = hsv_to_rgb(hsv)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, origin='lower', extent=[-3, 3, -3, 3])
    ax.set_title(title)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    return ax

# ----------------------------------------------------------------------
# 5. Full example: multi-panel comparison
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Coarser grid for the parameter-sweep panels (fast), fine for the hero shot
    RES_SWEEP = 200
    RES_HERO  = 350

    x_s = np.linspace(-3, 3, RES_SWEEP)
    y_s = np.linspace(-3, 3, RES_SWEEP)
    Xs, Ys = np.meshgrid(x_s, y_s)
    Z_sweep = Xs + 1j * Ys

    x_h = np.linspace(-3, 3, RES_HERO)
    y_h = np.linspace(-3, 3, RES_HERO)
    Xh, Yh = np.meshgrid(x_h, y_h)
    Z_hero = Xh + 1j * Yh

    base_params = dict(a=0.0, b=1.25, N=3, lam=1.0, cut_angle=0.0)

    # ------------------------------------------------------------------
    # Panel A: depth evolution  (alpha=0.5, depth = 1,2,3,4,5,6)
    # ------------------------------------------------------------------
    depths = [1, 2, 3, 4, 5, 6]
    fig_A, axes_A = plt.subplots(2, 3, figsize=(15, 10))
    fig_A.suptitle("Depth evolution   (α = 0.5, N = 3)", fontsize=14)
    for ax, d in zip(axes_A.flat, depths):
        print(f"  [A] depth={d} ...", flush=True)
        Zd = xo2_recursion(Z_sweep.copy(), d, alpha=0.5, **base_params)
        domain_color(Zd, title=f"depth = {d}", ax=ax)
    plt.tight_layout()
    out_a = OUTPUTS_DIR / "zeta_branching_depth_evolution.png"
    fig_A.savefig(out_a, dpi=130)
    print(f"Saved {out_a}")

    # ------------------------------------------------------------------
    # Panel B: alpha sweep  (depth=4, alpha = 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    # ------------------------------------------------------------------
    alphas = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    fig_B, axes_B = plt.subplots(2, 3, figsize=(15, 10))
    fig_B.suptitle("Branch-exponent sweep   (depth = 4, N = 3)", fontsize=14)
    for ax, a in zip(axes_B.flat, alphas):
        print(f"  [B] alpha={a} ...", flush=True)
        Za = xo2_recursion(Z_sweep.copy(), 4, alpha=a, **base_params)
        domain_color(Za, title=f"α = {a}", ax=ax)
    plt.tight_layout()
    out_b = OUTPUTS_DIR / "zeta_branching_alpha_sweep.png"
    fig_B.savefig(out_b, dpi=130)
    print(f"Saved {out_b}")

    # ------------------------------------------------------------------
    # Panel C: cut-angle sweep  (depth=4, alpha=0.5,
    #                            cut_angle = 0, π/4, π/2, 3π/4, π, 5π/4)
    # ------------------------------------------------------------------
    cut_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]
    fig_C, axes_C = plt.subplots(2, 3, figsize=(15, 10))
    fig_C.suptitle("Branch-cut rotation sweep   (depth = 4, α = 0.5)", fontsize=14)
    for ax, ca in zip(axes_C.flat, cut_angles):
        print(f"  [C] cut_angle={ca:.3f} ...", flush=True)
        kw = {**base_params, "cut_angle": ca}
        Zc = xo2_recursion(Z_sweep.copy(), 4, alpha=0.5, **kw)
        domain_color(Zc, title=f"cut = {ca/np.pi:.2f}π", ax=ax)
    plt.tight_layout()
    out_c = OUTPUTS_DIR / "zeta_branching_cut_sweep.png"
    fig_C.savefig(out_c, dpi=130)
    print(f"Saved {out_c}")

    # ------------------------------------------------------------------
    # Panel D: hero shot  (depth=5, alpha=0.5, high-res)
    # ------------------------------------------------------------------
    print("  [D] hero shot (700×700) ...", flush=True)
    Z_final = xo2_recursion(Z_hero.copy(), 5, alpha=0.5, **base_params)
    fig_D, ax_D = plt.subplots(figsize=(9, 8))
    domain_color(Z_final, title="xo2_recursion  depth=5, α=0.5, N=3", ax=ax_D)
    plt.tight_layout()
    out_d = OUTPUTS_DIR / "zeta_branching_hero.png"
    fig_D.savefig(out_d, dpi=150)
    print(f"Saved {out_d}")

    print("\nAll plots saved.")
    plt.show()