"""
zeta_surrogate.py
-----------------
Trains a small MLP to approximate xo2_recursion(z, depth, alpha, cut_angle)
so that full-sweep plots are milliseconds instead of minutes.

Inputs  : [Re(z_in), Im(z_in), alpha, cut_angle / pi]   (4 floats)
Outputs : [Re(z_out), Im(z_out)]                         (2 floats)

Fixed params during training: a=0.0, b=1.25, N=3, lam=1.0
Fixed depth                 : DEPTH = 4
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ── ground-truth kernel ───────────────────────────────────────────────────────
from zeta_branching import xo2_recursion, domain_color

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXED  = dict(a=0.0, b=1.25, N=3, lam=1.0)
DEPTH  = 4
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data generation
# ─────────────────────────────────────────────────────────────────────────────
N_ALPHA = 12          # α ∈ [0.25, 2.0]
N_CUT   = 12          # cut_angle ∈ [0, 5π/4]
N_GRID  = 100         # spatial grid per param combo (100×100 = 10 k pts)

def generate_dataset() -> tuple[np.ndarray, np.ndarray]:
    alphas     = np.linspace(0.25, 2.0, N_ALPHA)
    cut_angles = np.linspace(0.0, 5 * np.pi / 4, N_CUT)

    x = np.linspace(-3, 3, N_GRID)
    y = np.linspace(-3, 3, N_GRID)
    Xg, Yg = np.meshgrid(x, y)
    Z_grid  = Xg + 1j * Yg   # (N_GRID, N_GRID)

    X_list, Y_list = [], []
    total, done = N_ALPHA * N_CUT, 0

    for alpha in alphas:
        for ca in cut_angles:
            kw = {**FIXED, "cut_angle": ca}
            Z_out = xo2_recursion(Z_grid.copy(), DEPTH, alpha=alpha, **kw)

            re_in  = Xg.ravel().astype(np.float32)
            im_in  = Yg.ravel().astype(np.float32)
            # normalise cut_angle to [0,1] for better conditioning
            a_arr  = np.full_like(re_in, alpha,     dtype=np.float32)
            ca_arr = np.full_like(re_in, ca / np.pi, dtype=np.float32)

            re_out = np.real(Z_out).ravel().astype(np.float32)
            im_out = np.imag(Z_out).ravel().astype(np.float32)

            mask = np.isfinite(re_out) & np.isfinite(im_out)
            X_list.append(np.stack([re_in[mask], im_in[mask],
                                    a_arr[mask], ca_arr[mask]], axis=1))
            Y_list.append(np.stack([re_out[mask], im_out[mask]], axis=1))

            done += 1
            print(f"  [{done:3d}/{total}] α={alpha:.2f}  cut={ca/np.pi:.3f}π"
                  f"  valid={mask.sum()}", flush=True)

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# 2. MLP architecture
# ─────────────────────────────────────────────────────────────────────────────
class ZetaSurrogate(nn.Module):
    """4-input → 2-output MLP with residual skip connections."""

    def __init__(self, hidden: int = 256, n_layers: int = 5):
        super().__init__()
        self.input_proj = nn.Linear(4, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            )
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + block(h))          # pre-norm residual
        return self.head(h)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────────────────────────────────────
def train(
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 100,
    batch: int  = 8192,
    lr: float   = 3e-3,
) -> tuple["ZetaSurrogate", dict]:

    x_mean = X.mean(0, keepdims=True)
    x_std  = X.std(0,  keepdims=True) + 1e-8
    y_mean = Y.mean(0, keepdims=True)
    y_std  = Y.std(0,  keepdims=True) + 1e-8

    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    Xt = torch.from_numpy(Xn).to(DEVICE)
    Yt = torch.from_numpy(Yn).to(DEVICE)

    ds = TensorDataset(Xt, Yt)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, pin_memory=False)

    model   = ZetaSurrogate().to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss(delta=1.0)          # robust to outlier pixels

    best_loss = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * len(xb)
        sched.step()
        avg = running / len(Xt)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % 25 == 0:
            print(f"  epoch {ep:3d}/{epochs}  loss={avg:.6f}  best={best_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    stats = dict(
        x_mean=x_mean.astype(np.float32),
        x_std =x_std .astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std =y_std .astype(np.float32),
    )
    return model, stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. Save / Load
# ─────────────────────────────────────────────────────────────────────────────
def save(model: "ZetaSurrogate", stats: dict, path: Path = MODELS_DIR / "zeta_surrogate.pt") -> None:
    torch.save({"model": model.state_dict(), "stats": stats}, path)
    print(f"Saved surrogate → {path}")


def load(path: Path = MODELS_DIR / "zeta_surrogate.pt") -> tuple["ZetaSurrogate", dict]:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model = ZetaSurrogate()
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["stats"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Fast prediction (drop-in for xo2_recursion in plot code)
# ─────────────────────────────────────────────────────────────────────────────
def surrogate_predict(
    Z_grid: np.ndarray,
    alpha: float,
    cut_angle: float,
    model: "ZetaSurrogate",
    stats: dict,
) -> np.ndarray:
    """
    Drop-in replacement for xo2_recursion(Z_grid, DEPTH, alpha=alpha, cut_angle=...).

    Parameters
    ----------
    Z_grid    : 2-D complex numpy array (H, W)
    alpha     : branch exponent
    cut_angle : branch-cut rotation in radians
    model     : loaded ZetaSurrogate
    stats     : normalisation stats from training

    Returns
    -------
    Complex ndarray of shape (H, W)
    """
    H, W = Z_grid.shape
    re_in  = np.real(Z_grid).ravel().astype(np.float32)
    im_in  = np.imag(Z_grid).ravel().astype(np.float32)
    a_arr  = np.full_like(re_in, alpha,          dtype=np.float32)
    ca_arr = np.full_like(re_in, cut_angle / np.pi, dtype=np.float32)

    X  = np.stack([re_in, im_in, a_arr, ca_arr], axis=1)
    Xn = (X - stats["x_mean"]) / stats["x_std"]

    device = next(model.parameters()).device
    with torch.no_grad():
        Yn = model(torch.from_numpy(Xn).to(device)).cpu().numpy()

    Y = Yn * stats["y_std"] + stats["y_mean"]
    return (Y[:, 0] + 1j * Y[:, 1]).reshape(H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Validation plot  (GT vs MLP side-by-side)
# ─────────────────────────────────────────────────────────────────────────────
def validation_plot(model: "ZetaSurrogate", stats: dict) -> None:
    import time

    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    Xg, Yg = np.meshgrid(x, y)
    Z = Xg + 1j * Yg

    # Test on (α, cut_angle) combos NOT in the training grid
    test_cases = [
        (0.4,  0.1 * np.pi),
        (0.9,  0.6 * np.pi),
        (1.25, 1.0 * np.pi),
    ]

    fig, axes = plt.subplots(len(test_cases), 3,
                             figsize=(17, 5 * len(test_cases)))
    fig.suptitle("Surrogate validation  (depth=4)", fontsize=13)

    for row, (alpha, ca) in enumerate(test_cases):
        kw = {**FIXED, "cut_angle": ca}

        t0 = time.perf_counter()
        Z_true = xo2_recursion(Z.copy(), DEPTH, alpha=alpha, **kw)
        t_gt = time.perf_counter() - t0

        t0 = time.perf_counter()
        Z_pred = surrogate_predict(Z.copy(), alpha, ca, model, stats)
        t_mlp = time.perf_counter() - t0

        # pixel-wise absolute error in complex magnitude
        err = np.abs(Z_true - Z_pred)
        err_finite = err[np.isfinite(err)]

        domain_color(Z_true, title=f"GT  α={alpha}  cut={ca/np.pi:.2f}π  ({t_gt:.1f}s)",
                     ax=axes[row, 0])
        domain_color(Z_pred, title=f"MLP α={alpha}  cut={ca/np.pi:.2f}π  ({t_mlp*1000:.1f}ms)",
                     ax=axes[row, 1])

        # error map
        axes[row, 2].imshow(np.clip(err, 0, np.percentile(err_finite, 99)),
                            origin="lower", extent=[-3, 3, -3, 3], cmap="inferno")
        axes[row, 2].set_title(f"|GT−MLP|  median={np.median(err_finite):.3f}")
        axes[row, 2].set_xlabel("Re"); axes[row, 2].set_ylabel("Im")

    plt.tight_layout()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "zeta_surrogate_validation.png"
    fig.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Fast sweep plot using the trained surrogate
# ─────────────────────────────────────────────────────────────────────────────
def fast_sweep_plot(model: "ZetaSurrogate", stats: dict) -> None:
    import time
    RES = 400

    x = np.linspace(-3, 3, RES)
    y = np.linspace(-3, 3, RES)
    Xg, Yg = np.meshgrid(x, y)
    Z = Xg + 1j * Yg

    alphas     = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    cut_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]

    # α sweep
    fig_a, axes_a = plt.subplots(2, 3, figsize=(15, 10))
    fig_a.suptitle(f"Surrogate α sweep  (depth={DEPTH}, 400×400)", fontsize=13)
    t0 = time.perf_counter()
    for ax, alpha in zip(axes_a.flat, alphas):
        Z_pred = surrogate_predict(Z.copy(), alpha, 0.0, model, stats)
        domain_color(Z_pred, title=f"α = {alpha}", ax=ax)
    print(f"α sweep: {time.perf_counter()-t0:.2f}s total")
    plt.tight_layout()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    alpha_out = OUTPUTS_DIR / "zeta_surrogate_alpha_sweep.png"
    fig_a.savefig(alpha_out, dpi=130)
    print(f"Saved {alpha_out}")

    # cut-angle sweep
    fig_c, axes_c = plt.subplots(2, 3, figsize=(15, 10))
    fig_c.suptitle(f"Surrogate cut-angle sweep  (depth={DEPTH}, 400×400)", fontsize=13)
    t0 = time.perf_counter()
    for ax, ca in zip(axes_c.flat, cut_angles):
        Z_pred = surrogate_predict(Z.copy(), 0.5, ca, model, stats)
        domain_color(Z_pred, title=f"cut = {ca/np.pi:.2f}π", ax=ax)
    print(f"cut sweep: {time.perf_counter()-t0:.2f}s total")
    plt.tight_layout()
    cut_out = OUTPUTS_DIR / "zeta_surrogate_cut_sweep.png"
    fig_c.savefig(cut_out, dpi=130)
    print(f"Saved {cut_out}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = MODELS_DIR / "zeta_surrogate.pt"
    DATA_PATH = MODELS_DIR / "zeta_surrogate_data.npz"

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing surrogate from {MODEL_PATH} ...")
        model, stats = load(MODEL_PATH)
    else:
        if os.path.exists(DATA_PATH):
            print(f"Loading cached dataset from {DATA_PATH} ...")
            d = np.load(DATA_PATH)
            X, Y = d["X"], d["Y"]
        else:
            print(f"=== Generating training data  (device: {DEVICE}) ===")
            X, Y = generate_dataset()
            np.savez_compressed(DATA_PATH, X=X, Y=Y)
            print(f"Cached dataset → {DATA_PATH}")
        print(f"\nDataset: {X.shape[0]:,} samples  "
              f"[Re(z), Im(z), α, cut/π] → [Re(z_out), Im(z_out)]")

        print("\n=== Training MLP surrogate ===")
        model, stats = train(X, Y)
        save(model, stats, MODEL_PATH)

    print("\n=== Validation (GT vs MLP) ===")
    validation_plot(model, stats)

    print("\n=== Fast sweep plots ===")
    fast_sweep_plot(model, stats)
