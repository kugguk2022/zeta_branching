"""
branched_mp4_zeta.py
--------------------
Trains a small MLP to approximate xo2_recursion(z, depth, alpha, cut_angle, gamma)
so that full-sweep plots and animations are milliseconds instead of minutes.

Inputs  : [Re(z_in), Im(z_in), alpha, cut_angle/π, gamma]   (5 floats)
Outputs : [Re(z_out), Im(z_out)]                             (2 floats)

Fixed params during training: a=0.0, b=1.25, N=3, lam=1.0
Fixed depth                 : DEPTH = 4
"""

import os, time, itertools, argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")          # headless — avoids Tk errors when saving video
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Try to import the ground‑truth kernel
try:
    from zeta_branching import branch_pow, zeta_c, domain_color
except ImportError:
    raise ImportError("Place zeta_branching.py (with branch_pow, zeta_c and domain_color) in the same folder.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXED  = dict(a=0.0, b=1.25, N=3, lam=1.0)
DEPTH  = 4
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODEL_PATH = MODELS_DIR / "zeta_surrogate_v3.pt"
DATA_PATH  = MODELS_DIR / "zeta_surrogate_v3_data.npz"

# ─────────────────────────────────────────────────────────────────
# 1. Ground‑truth kernel with gamma parameter (no monkey-patching)
# ─────────────────────────────────────────────────────────────────
def xo2_kernel_gamma(z, a=0.0, b=1.25, alpha=0.5, N=3, lam=1.0,
                     cut_angle=0.0, gamma=1.0):
    """
    One kernel step:  r = lam / (1 + exp(w / gamma)),  w = ζ( (z-a)^α + b )
    gamma=1 recovers the original kernel.
    """
    shifted = branch_pow(z, a, alpha, cut_angle) + b
    w = zeta_c(shifted)
    r = lam / (1.0 + np.exp(np.real(w) / gamma))   # real exponent; avoids complex exp blowup
    r = np.where(np.isfinite(r), r, 0.0)
    # Closed-form geometric series, with fallback for r≈1
    close = np.abs(r - 1.0) < 1e-6
    f_w = np.where(
        close,
        np.sum([r**n for n in range(1, N + 1)], axis=0),
        r * (1 - r**N) / (1 - r),
    )
    return f_w

def xo2_recursion_gamma(Z_grid, depth, alpha, cut_angle, gamma):
    """Apply xo2_kernel_gamma `depth` times."""
    z = Z_grid.copy()
    kw = {**FIXED, "alpha": alpha, "cut_angle": cut_angle, "gamma": gamma}
    for _ in range(depth):
        z = xo2_kernel_gamma(z, **kw)
    return z

# ─────────────────────────────────────────────────────────────────
# 2. Data generation (sequential — works on Windows, no pickling issues)
# ─────────────────────────────────────────────────────────────────
def generate_dataset(n_alpha=8, n_cut=8, n_gamma=5, n_grid=64):
    alphas     = np.linspace(0.25, 2.0, n_alpha)
    cut_angles = np.linspace(0.0, 5 * np.pi / 4, n_cut)
    gammas     = np.linspace(0.5, 4.0, n_gamma)
    x = np.linspace(-3, 3, n_grid)
    y = np.linspace(-3, 3, n_grid)
    Xg, Yg = np.meshgrid(x, y)
    Z_grid  = Xg + 1j * Yg

    combos = list(itertools.product(alphas, cut_angles, gammas))
    total  = len(combos)
    X_list, Y_list = [], []

    t0 = time.perf_counter()
    for i, (alpha, ca, gamma) in enumerate(combos, 1):
        Z_out  = xo2_recursion_gamma(Z_grid, DEPTH, alpha, ca, gamma)
        re_in  = Xg.ravel().astype(np.float32)
        im_in  = Yg.ravel().astype(np.float32)
        a_arr  = np.full_like(re_in, alpha,       dtype=np.float32)
        ca_arr = np.full_like(re_in, ca / np.pi,  dtype=np.float32)
        g_arr  = np.full_like(re_in, gamma,        dtype=np.float32)
        re_out = np.real(Z_out).ravel().astype(np.float32)
        im_out = np.imag(Z_out).ravel().astype(np.float32)
        mask   = np.isfinite(re_out) & np.isfinite(im_out)
        X_list.append(np.stack([re_in[mask], im_in[mask],
                                a_arr[mask], ca_arr[mask], g_arr[mask]], axis=1))
        Y_list.append(np.stack([re_out[mask], im_out[mask]], axis=1))
        if i % 20 == 0 or i == total:
            print(f"  [{i:3d}/{total}] α={alpha:.2f} cut={ca/np.pi:.2f}π γ={gamma:.2f}"
                  f"  valid={mask.sum()}", flush=True)

    print(f"Data generation: {time.perf_counter()-t0:.1f}s")
    return np.concatenate(X_list), np.concatenate(Y_list)

# ─────────────────────────────────────────────────────────────────
# 3. MLP architecture with Fourier feature encoding
# ─────────────────────────────────────────────────────────────────
class FourierFeatures(nn.Module):
    """Sinusoidal encoding of the two spatial inputs (Re, Im).
    Fixed random frequencies — not learned — give the MLP cheap
    access to multiple spatial frequency bands."""
    def __init__(self, n_fourier: int = 32, scale: float = 2.5):
        super().__init__()
        B = torch.randn(2, n_fourier) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B          # (..., n_fourier)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (..., 2*n_fourier)


class ZetaSurrogate(nn.Module):
    """5-input [Re, Im, α, cut/π, γ] → 2-output [Re_out, Im_out] MLP.
    Spatial coordinates are lifted with sinusoidal Fourier features."""

    def __init__(self, hidden: int = 256, n_layers: int = 5,
                 n_fourier: int = 32):
        super().__init__()
        self.fourier = FourierFeatures(n_fourier=n_fourier)
        input_dim = 2 * n_fourier + 3   # spatial encoding + (alpha, ca/pi, gamma)
        self.input_proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 5) — [Re, Im, alpha, ca/pi, gamma]
        spatial = x[..., :2]
        params  = x[..., 2:]
        h = self.input_proj(torch.cat([self.fourier(spatial), params], dim=-1))
        for block, norm in zip(self.blocks, self.norms):
            h = norm(h + block(h))
        return self.head(h)

# ─────────────────────────────────────────────────────────────────
# 4. Training with early stopping
# ─────────────────────────────────────────────────────────────────
def train(X, Y, epochs=200, batch=8192, lr=3e-3, valid_split=0.1):
    # Normalisation
    x_mean = X.mean(0, keepdims=True)
    x_std  = X.std(0, keepdims=True) + 1e-8
    y_mean = Y.mean(0, keepdims=True)
    y_std  = Y.std(0, keepdims=True) + 1e-8
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    # Train/val split
    n_total = len(X)
    n_val = int(n_total * valid_split)
    n_train = n_total - n_val
    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    Xt = torch.from_numpy(Xn[train_idx]).float().to(DEVICE)
    Yt = torch.from_numpy(Yn[train_idx]).float().to(DEVICE)
    Xv = torch.from_numpy(Xn[val_idx]).float().to(DEVICE)
    Yv = torch.from_numpy(Yn[val_idx]).float().to(DEVICE)

    train_ds = TensorDataset(Xt, Yt)
    val_ds   = TensorDataset(Xv, Yv)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch * 2, pin_memory=False)

    model = ZetaSurrogate().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                           factor=0.5, patience=10)
    loss_fn = nn.HuberLoss(delta=0.5)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    max_patience = 25

    for ep in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(Xt)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
        val_loss /= len(Xv)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if ep % 25 == 0 or patience_counter >= max_patience:
            print(f"ep {ep:3d} | train {train_loss:.6f} | val {val_loss:.6f} | best {best_val_loss:.6f}")
        if patience_counter >= max_patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    model.eval()
    stats = dict(x_mean=x_mean.astype(np.float32), x_std=x_std.astype(np.float32),
                 y_mean=y_mean.astype(np.float32), y_std=y_std.astype(np.float32))
    return model, stats

# ─────────────────────────────────────────────────────────────────
# 5. Save / load
# ─────────────────────────────────────────────────────────────────
def save(model, stats, path=MODEL_PATH):
    torch.save({"model": model.state_dict(), "stats": stats}, path)
    print(f"Saved surrogate → {path}")

def load(path=MODEL_PATH):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ZetaSurrogate()
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["stats"]

# ─────────────────────────────────────────────────────────────────
# 6. Fast prediction (now accepts gamma)
# ─────────────────────────────────────────────────────────────────
def surrogate_predict(Z_grid, alpha, cut_angle, gamma, model, stats):
    H, W = Z_grid.shape
    re_in  = np.real(Z_grid).ravel().astype(np.float32)
    im_in  = np.imag(Z_grid).ravel().astype(np.float32)
    a_arr  = np.full_like(re_in, alpha,          dtype=np.float32)
    ca_arr = np.full_like(re_in, cut_angle / np.pi, dtype=np.float32)
    g_arr  = np.full_like(re_in, gamma,           dtype=np.float32)
    X  = np.stack([re_in, im_in, a_arr, ca_arr, g_arr], axis=1)
    Xn = (X - stats["x_mean"]) / stats["x_std"]
    device = next(model.parameters()).device
    with torch.no_grad():
        Yn = model(torch.from_numpy(Xn).to(device)).cpu().numpy()
    Y = Yn * stats["y_std"] + stats["y_mean"]
    return (Y[:, 0] + 1j * Y[:, 1]).reshape(H, W)

# ─────────────────────────────────────────────────────────────────
# 7. Validation plot (GT vs surrogate + error)
# ─────────────────────────────────────────────────────────────────
def validation_plot(model, stats, grid_res=200):
    x = np.linspace(-3, 3, grid_res)
    y = np.linspace(-3, 3, grid_res)
    Xg, Yg = np.meshgrid(x, y)
    Z = Xg + 1j * Yg

    test_cases = [
        (0.4,  0.1 * np.pi, 1.0),
        (0.9,  0.6 * np.pi, 2.0),
        (1.25, 1.0 * np.pi, 0.5),
    ]

    fig, axes = plt.subplots(len(test_cases), 3, figsize=(17, 5*len(test_cases)))
    fig.suptitle(f"Surrogate validation  depth={DEPTH}", fontsize=14)

    for row, (alpha, ca, gamma) in enumerate(test_cases):
        t0 = time.perf_counter()
        Z_true = xo2_recursion_gamma(Z.copy(), DEPTH, alpha, ca, gamma)
        t_gt = time.perf_counter() - t0

        t0 = time.perf_counter()
        Z_pred = surrogate_predict(Z.copy(), alpha, ca, gamma, model, stats)
        t_mlp = time.perf_counter() - t0

        err = np.abs(Z_true - Z_pred)
        err_finite = err[np.isfinite(err)]

        domain_color(Z_true, title=f"GT  α={alpha} cut={ca/np.pi:.2f}π γ={gamma} ({t_gt:.1f}s)",
                     ax=axes[row,0])
        domain_color(Z_pred, title=f"MLP α={alpha} cut={ca/np.pi:.2f}π γ={gamma} ({t_mlp*1e3:.1f}ms)",
                     ax=axes[row,1])
        im = axes[row,2].imshow(np.clip(err, 0, np.percentile(err_finite, 99)),
                                origin="lower", extent=[-3,3,-3,3], cmap="inferno")
        axes[row,2].set_title(f"|GT−MLP| median={np.median(err_finite):.3f}")
        plt.colorbar(im, ax=axes[row,2])
    plt.tight_layout()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "surrogate_validation.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved {out_path}")
    plt.show()

# ─────────────────────────────────────────────────────────────────
# 8. Animation sweep (rotation or alpha)
# ─────────────────────────────────────────────────────────────────
def animation_sweep(model, stats, param="cut_angle", grid_res=200, frames=60,
                    alpha=0.75, gamma=1.5, filename="cut_sweep.mp4"):
    x = np.linspace(-3, 3, grid_res)
    y = np.linspace(-3, 3, grid_res)
    Xg, Yg = np.meshgrid(x, y)
    Z = Xg + 1j * Yg

    if param == "cut_angle":
        values = np.linspace(0, 2 * np.pi, frames)
    elif param == "alpha":
        values = np.linspace(0.25, 2.0, frames)
    else:
        raise ValueError("param must be 'cut_angle' or 'alpha'")

    fig, ax = plt.subplots(figsize=(6, 6))

    def _update(i):
        val = values[i]
        if param == "cut_angle":
            Z_pred = surrogate_predict(Z, alpha, val, gamma, model, stats)
            title = f"cut={val/np.pi:.2f}\u03c0  [{i+1}/{frames}]"
        else:
            Z_pred = surrogate_predict(Z, val, 0.0, gamma, model, stats)
            title = f"\u03b1={val:.2f}  [{i+1}/{frames}]"
        ax.clear()
        domain_color(Z_pred, title=title, ax=ax)
        if i % 10 == 0:
            print(f"  frame {i+1}/{frames}", flush=True)
        return ax.get_images()

    anim = FuncAnimation(fig, _update, frames=frames, blit=False)

    output_path = Path(filename)
    if not output_path.is_absolute():
        output_path = OUTPUTS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = False
    # Primary path: matplotlib FFMpegWriter (requires ffmpeg on PATH)
    try:
        writer = FFMpegWriter(fps=15, bitrate=2000)
        anim.save(output_path, writer=writer, dpi=120)
        print(f"Animation saved → {output_path}")
        saved = True
    except Exception as e:
        print(f"FFMpegWriter failed ({e}), trying imageio...")

    # Fallback 1: imageio
    if not saved:
        try:
            import imageio
            # Render frames manually with corrected buffer extraction
            imgs = []
            for i in range(frames):
                _update(i)
                fig.canvas.draw()
                W_px, H_px = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                imgs.append(buf.reshape(H_px, W_px, 4)[..., :3].copy())
            gif_path = output_path.with_suffix(".gif")
            imageio.mimsave(output_path if output_path.suffix == ".mp4" else gif_path,
                            imgs, fps=15)
            print(f"Animation saved → {output_path}")
            saved = True
        except ImportError:
            print("imageio not available.")
        except Exception as e:
            print(f"imageio failed ({e}).")

    # Fallback 2: PIL GIF
    if not saved:
        try:
            from PIL import Image as PILImage
            imgs_pil = []
            for i in range(frames):
                _update(i)
                fig.canvas.draw()
                W_px, H_px = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                rgb = buf.reshape(H_px, W_px, 4)[..., :3].copy()
                imgs_pil.append(PILImage.fromarray(rgb))
            gif_path = output_path.with_suffix(".gif")
            imgs_pil[0].save(gif_path, save_all=True,
                             append_images=imgs_pil[1:], duration=67, loop=0)
            print(f"Saved as GIF → {gif_path}")
            saved = True
        except ImportError:
            print("Pillow not available. Install ffmpeg, imageio[ffmpeg], or Pillow.")

    plt.close(fig)
    if not saved:
        print("Could not save animation — no suitable writer found.")

# ─────────────────────────────────────────────────────────────────
# 9. Optional 3D Riemann‑surface view
# ─────────────────────────────────────────────────────────────────
def riemann_surface_plot(model, stats, alpha=0.5, cut_angle=0.0, gamma=1.0,
                         res=100, mode="real"):
    from mpl_toolkits.mplot3d import Axes3D
    x = np.linspace(-3, 3, res)
    y = np.linspace(-3, 3, res)
    X, Y = np.meshgrid(x, y)
    Z_grid = X + 1j * Y
    Z_out = surrogate_predict(Z_grid, alpha, cut_angle, gamma, model, stats)
    if mode == "real":
        height = np.real(Z_out)
    elif mode == "imag":
        height = np.imag(Z_out)
    else:
        height = np.abs(Z_out)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, height, cmap="viridis", edgecolor='none', alpha=0.9)
    ax.set_title(f"Riemann surface ({mode}) α={alpha} cut={cut_angle/np.pi:.2f}π γ={gamma}")
    plt.colorbar(surf)
    plt.show()

# ─────────────────────────────────────────────────────────────────
# 10. Main execution
# ─────────────────────────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(
        description="branched_mp4_zeta v3 — MLP surrogate for xo2_recursion"
    )
    ap.add_argument("--skip-validation", action="store_true",
                    help="Skip the GT-vs-surrogate validation plot (faster startup)")
    ap.add_argument("--skip-animation", action="store_true",
                    help="Skip animation generation")
    ap.add_argument("--frames", type=int, default=60,
                    help="Number of animation frames (default: 60)")
    ap.add_argument("--grid-res", type=int, default=200,
                    help="Spatial grid resolution for plots (default: 200)")
    ap.add_argument("--param", choices=["cut_angle", "alpha"], default="cut_angle",
                    help="Parameter to sweep in the animation")
    ap.add_argument("--alpha", type=float, default=0.75,
                    help="Fixed alpha for cut_angle sweep (default: 0.75)")
    ap.add_argument("--gamma", type=float, default=1.5,
                    help="Fixed gamma for animations (default: 1.5)")
    ap.add_argument("--retrain", action="store_true",
                    help="Force retraining even if a saved model exists")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if os.path.exists(MODEL_PATH) and not args.retrain:
        print(f"Loading existing surrogate from {MODEL_PATH} ...")
        model, stats = load(MODEL_PATH)
    else:
        if args.retrain and os.path.exists(MODEL_PATH):
            print("--retrain: ignoring existing model.")
        if os.path.exists(DATA_PATH) and not args.retrain:
            print(f"Loading cached dataset from {DATA_PATH} ...")
            d = np.load(DATA_PATH)
            X, Y = d["X"], d["Y"]
        else:
            print("Generating training data...")
            X, Y = generate_dataset()
            np.savez_compressed(DATA_PATH, X=X, Y=Y)
            print(f"Cached dataset → {DATA_PATH}")
        print(f"Training on {X.shape[0]:,} samples (device={DEVICE})")
        model, stats = train(X, Y)
        save(model, stats, MODEL_PATH)

    if not args.skip_validation:
        print("\n=== Validation (GT vs surrogate) ===")
        validation_plot(model, stats, grid_res=min(args.grid_res, 150))

    if not args.skip_animation:
        out = f"{args.param}_sweep.mp4"
        print(f"\n=== Generating animation: {out} ===")
        animation_sweep(
            model, stats,
            param=args.param,
            grid_res=args.grid_res,
            frames=args.frames,
            alpha=args.alpha,
            gamma=args.gamma,
            filename=out,
        )

    # Optional 3D view (uncomment to enable)
    # riemann_surface_plot(model, stats, alpha=0.8, cut_angle=np.pi/4, gamma=1.0, mode="real")