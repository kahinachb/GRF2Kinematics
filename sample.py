"""
sample.py
=========
Generate joint angles conditioned on kinetics using a trained model.

Usage:
    # Generate from a single kinetics .npy file
    python sample.py --checkpoint checkpoints/best.pt \
                     --kinetics /path/to/kinetics.npy \
                     --out generated_joints.npy

    # Generate from all trials in a folder and compare with ground truth
    python sample.py --checkpoint checkpoints/best.pt \
                     --npy-root /path/to/npy \
                     --out-dir generated/ \
                     --plot
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from utils.diffuser_utils  import DDPM, DiffusionTransformer
from dataset import Normalizer


# ─────────────────────────────────────────────────────────────────────────────
# LOAD TRAINED MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """Load model + DDPM from a checkpoint file produced by train.py."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    model = DiffusionTransformer(
        joint_dim  = 29,
        force_dim  = 12,
        embed_dim  = args["embed_dim"],
        nhead      = args["nhead"],
        num_layers = args["num_layers"],
        seq_len    = args["seq_len"],
        dropout    = 0.0,   # disable dropout at inference
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ddpm = DDPM(device   = device,
                n_steps  = args["n_steps"],
                min_beta = args["min_beta"],
                max_beta = args["max_beta"])

    print(f"  [Sample] Checkpoint loaded  (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['best_val_loss']:.6f})")
    return model, ddpm, args


# ─────────────────────────────────────────────────────────────────────────────
# WINDOWED GENERATION FOR A SINGLE TRIAL
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_trial(
    model:         DiffusionTransformer,
    ddpm:          DDPM,
    kinetics_raw:  np.ndarray,
    kinetics_norm: Normalizer,
    joint_norm:    Normalizer,
    seq_len:       int,
    device:        torch.device,
    overlap:       int = 32,
    n_samples:     int = 1,
) -> np.ndarray:
    """
    Generate joint angles for a full trial.

    When n_samples > 1, the reverse diffusion is run n_samples times from
    independent noise realizations, all conditioned on the same kinetics.
    This exploits the multimodal nature of the diffusion model.

    Args:
        n_samples : number of independent samples to generate
    Returns:
        joints_pred : (n_samples, T, 29)  — all samples stacked
                      (n_samples=1) → still (1, T, 29) for consistency
    """
    T     = len(kinetics_raw)
    k_norm = kinetics_norm.transform(kinetics_raw)   # (T, 12)

    all_samples = []   # will collect n_samples arrays of shape (T, 29)

    for sample_idx in range(n_samples):
        pred_acc   = np.zeros((T, 29), dtype=np.float32)
        weight_acc = np.zeros((T, 1),  dtype=np.float32)
        # win        = np.hanning(seq_len).reshape(-1, 1).astype(np.float32)
        win = np.ones((seq_len, 1), dtype=np.float32)

        start = 0
        while start < T:
            end   = min(start + seq_len, T)
            chunk = end - start

            k_chunk          = np.zeros((seq_len, 12), dtype=np.float32)
            k_chunk[:chunk]  = k_norm[start:end]
            k_tensor         = torch.from_numpy(k_chunk).unsqueeze(0).to(device)

            # Each call to generate() draws fresh Gaussian noise → different sample
            x_gen = ddpm.generate(model, k_tensor, joint_dim=29)
            x_gen = joint_norm.inverse_transform(x_gen.squeeze(0).cpu().numpy())

            pred_acc[start:end]   += x_gen[:chunk] * win[:chunk]
            weight_acc[start:end] += win[:chunk]

            if end == T:
                break
            start += seq_len - overlap

        weight_acc = np.where(weight_acc < 1e-8, 1.0, weight_acc)
        all_samples.append(pred_acc / weight_acc)   # (T, 29)

    return np.stack(all_samples, axis=0)   # (n_samples, T, 29)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: QUICK COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _plot_all_samples_section(joints_gt, joints_pred, dof_indices, dof_names,
                              title, n_cols=3):
    """
    Figure showing every individual sample vs GT for a subset of DOFs.

    joints_gt   : (T, 29)            ground truth
    joints_pred : (n_samples, T, 29) all generated samples
    """
    import matplotlib.pyplot as plt

    n_dof     = len(dof_indices)
    n_rows    = -(-n_dof // n_cols)
    T         = joints_pred.shape[1]
    n_samples = joints_pred.shape[0]
    time_s    = np.arange(T) / 100.0

    # Each sample gets a distinct red shade
    sample_colors = plt.cm.Reds(np.linspace(0.30, 0.80, max(n_samples, 2)))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5.5, n_rows * 2.8),
        facecolor="white",
    )
    fig.suptitle(
        f"{title}\n"
        f"{n_samples} samples",
        fontsize=12, fontweight="bold", y=1.002,
    )
    axes_flat = axes.flatten() if n_dof > 1 else [axes]

    for idx, (col, name) in enumerate(zip(dof_indices, dof_names)):
        ax = axes_flat[idx]

        # All individual samples
        for s in range(n_samples):
            ax.plot(time_s, joints_pred[s, :, col],
                    color=sample_colors[s], alpha=0.65,
                    label=f"Sample {s+1}" if n_samples <= 5 else ("Samples" if s == 0 else None))

        # Ground truth on top
        if joints_gt is not None:
            ax.plot(time_s, joints_gt[:, col],
                    color="black", alpha=0.92, label="GT", zorder=10)

        ax.set_title(name, fontsize=9, fontweight="bold", pad=3)
        ax.set_ylabel("rad", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")
        ax.tick_params(labelsize=7.5)

        if idx >= n_dof - n_cols:
            ax.set_xlabel("Time (s)", fontsize=8)

    axes_flat[0].legend(fontsize=7.5, loc="upper left", framealpha=0.7)

    for j in range(n_dof, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def _plot_mean_section(joints_gt, joints_pred, dof_indices, dof_names,
                       title, n_cols=3):
    """
    Figure showing mean prediction ± 1 std vs GT, with per-DOF RMSE badge.

    joints_gt   : (T, 29)
    joints_pred : (n_samples, T, 29)
    """
    import matplotlib.pyplot as plt

    n_dof     = len(dof_indices)
    n_rows    = -(-n_dof // n_cols)
    T         = joints_pred.shape[1]
    n_samples = joints_pred.shape[0]
    time_s    = np.arange(T) / 100.0

    mean_pred = joints_pred.mean(axis=0)   # (T, 29)
    std_pred  = joints_pred.std(axis=0)    # (T, 29)

    if joints_gt is not None:
        rmse_rad = np.sqrt(((joints_gt - mean_pred) ** 2).mean(axis=0))  # (29,)
    else:
        rmse_rad = None

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5.5, n_rows * 2.8),
        facecolor="white",
    )
    fig.suptitle(
        f"{title}\n"
        f"(mean of {n_samples} sample(s))",
        fontsize=12, fontweight="bold", y=1.002,
    )
    axes_flat = axes.flatten() if n_dof > 1 else [axes]

    for idx, (col, name) in enumerate(zip(dof_indices, dof_names)):
        ax = axes_flat[idx]

        # ± 1 std band
        # if n_samples > 1:
        #     ax.fill_between(
        #         time_s,
        #         mean_pred[:, col] - std_pred[:, col],
        #         mean_pred[:, col] + std_pred[:, col],
        #         color="#e74c3c", alpha=0.15, label="± 1 std",
        #     )

        # Mean prediction
        ax.plot(time_s, mean_pred[:, col],
                color="red", label="Mean pred")

        # Ground truth
        if joints_gt is not None:
            ax.plot(time_s, joints_gt[:, col],
                    color="black", alpha=0.92, label="GT", zorder=5)

        ax.set_title(name, fontsize=9, fontweight="bold", pad=3)

        # RMSE badge (rad + degrees)
        if rmse_rad is not None:
            rmse     = rmse_rad[col]
            rmse_deg = np.degrees(rmse)
            ax.text(
                0.98, 0.96,
                f"RMSE  {rmse:.4f} rad\n      {rmse_deg:.2f} °",
                transform=ax.transAxes,
                fontsize=7.5, ha="right", va="top",
                
            )

        ax.set_ylabel("rad", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")
        ax.tick_params(labelsize=7.5)

        if idx >= n_dof - n_cols:
            ax.set_xlabel("Time (s)", fontsize=8)

    axes_flat[0].legend(fontsize=7.5, loc="upper left", framealpha=0.7)

    for j in range(n_dof, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_comparison(joints_gt, joints_pred):
    """
    Display four figures:
        Figure 1 — All samples vs GT — Lower body  (12 DOFs)
        Figure 2 — All samples vs GT — Upper body  (17 DOFs)
        Figure 3 — Mean pred ± std vs GT — Lower body  (with RMSE badges)
        Figure 4 — Mean pred ± std vs GT — Upper body  (with RMSE badges)

    Args:
        joints_gt   : (T, 29)            ground truth
        joints_pred : (n_samples, T, 29) all generated samples
    """
    LOWER_NAMES = [
        "R Hip Flex/Ext",    "R Hip Abd/Add",    "R Hip Int/Ext Rot",
        "R Knee Flex/Ext",   "R Ankle Flex/Ext", "R Ankle Abd/Add",
        "L Hip Flex/Ext",    "L Hip Abd/Add",    "L Hip Int/Ext Rot",
        "L Knee Flex/Ext",   "L Ankle Flex/Ext", "L Ankle Abd/Add",
    ]
    UPPER_NAMES = [
        "Lumbar Flex/Ext",        "Lumbar Lateral Flex",
        "L Clavicle X",
        "L Shoulder Flex/Ext",    "L Shoulder Abd/Add",   "L Shoulder Int/Ext Rot",
        "L Elbow Flex/Ext",       "L Elbow Pron/Sup",
        "Cervical Flex/Ext",      "Cervical Lat Bend",    "Cervical Int/Ext Rot",
        "R Clavicle X",
        "R Shoulder Flex/Ext",    "R Shoulder Abd/Add",   "R Shoulder Int/Ext Rot",
        "R Elbow Flex/Ext",       "R Elbow Pron/Sup",
    ]
    lower_idx = list(range(12))
    upper_idx = list(range(12, 29))

    # ── Figures 1 & 2 — All individual samples vs GT ─────────────────────────
    _plot_all_samples_section(
        joints_gt, joints_pred,
        dof_indices = lower_idx,
        dof_names   = LOWER_NAMES,
        title       = "Lower",
        n_cols      = 3,
    )
    _plot_all_samples_section(
        joints_gt, joints_pred,
        dof_indices = upper_idx,
        dof_names   = UPPER_NAMES,
        title       = "Upper",
        n_cols      = 3,
    )

    # ── Figures 3 & 4 — Mean prediction ± std vs GT (with RMSE) ─────────────
    _plot_mean_section(
        joints_gt, joints_pred,
        dof_indices = lower_idx,
        dof_names   = LOWER_NAMES,
        title       = "Lower",
        n_cols      = 3,
    )
    _plot_mean_section(
        joints_gt, joints_pred,
        dof_indices = upper_idx,
        dof_names   = UPPER_NAMES,
        title       = "Upper ",
        n_cols      = 3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(joints_gt: np.ndarray, joints_pred: np.ndarray) -> dict:
    """
    Compute metrics between ground truth and prediction.

    Args:
        joints_gt   : (T, 29)
        joints_pred : (n_samples, T, 29)

    Metrics are computed on the **mean** prediction across samples.
    Also reports std across samples as a diversity measure.
    """
    mean_pred = joints_pred.mean(axis=0)   # (T, 29)
    mae       = float(np.abs(joints_gt - mean_pred).mean())
    rmse      = float(np.sqrt(((joints_gt - mean_pred) ** 2).mean()))

    corrs = []
    for i in range(joints_gt.shape[1]):
        g, p = joints_gt[:, i], mean_pred[:, i]
        if g.std() > 1e-8 and p.std() > 1e-8:
            corrs.append(float(np.corrcoef(g, p)[0, 1]))
    corr = float(np.mean(corrs)) if corrs else float("nan")

    # Diversity: mean std across samples and DOFs
    diversity = float(joints_pred.std(axis=0).mean()) if joints_pred.shape[0] > 1 else 0.0

    return {"mae": mae, "rmse": rmse, "mean_corr": corr, "diversity": diversity}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate joint angles with trained diffusion model")
    parser.add_argument("--checkpoint",  required=True, help="Path to best.pt from train.py")
    parser.add_argument("--norm-prefix", default=None,
                        help="Prefix for normalizer files (default: same folder as checkpoint)")
    # Single-file mode
    parser.add_argument("--kinetics",   default=None, help="Single kinetics.npy file")
    parser.add_argument("--joints-gt",  default=None, help="Ground-truth all_joints.npy for comparison")
    parser.add_argument("--out",        default="generated_joints.npy", help="Output .npy path")
    # Batch mode
    parser.add_argument("--npy-root",   default=None, help="npy/ root: process all trials")
    parser.add_argument("--out-dir",    default="generated",            help="Output dir for batch mode")
    # Options
    parser.add_argument("--overlap",    type=int,   default=32,   help="Window overlap for blending (frames)")
    parser.add_argument("--n-samples",  type=int,   default=1,
                        help="Number of independent samples to generate per trial "
                             "(> 1 exploits multimodality of the diffusion model)")
    parser.add_argument("--plot",       action="store_true",       help="Plot GT vs prediction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  [Sample] Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model, ddpm, train_args = load_model(args.checkpoint, device)
    seq_len = train_args["seq_len"]

    # ── Load normalizers ──────────────────────────────────────────────────────
    ckpt_dir    = Path(args.checkpoint).parent
    norm_prefix = args.norm_prefix or str(ckpt_dir / "normalizer")

    joint_norm    = Normalizer()
    kinetics_norm = Normalizer()
    joint_norm.load(f"{norm_prefix}_joints.npz")
    kinetics_norm.load(f"{norm_prefix}_kinetics.npz")
    print(f"  [Sample] Normalizers loaded from {norm_prefix}_*.npz")

    # ── Single-file mode ──────────────────────────────────────────────────────
    if args.kinetics:
        k_raw = np.load(args.kinetics).astype(np.float32)
        print(f"  [Sample] Generating {args.n_samples} sample(s) from {args.kinetics}  "
              f"({len(k_raw)} frames = {len(k_raw)/100:.2f} sec)")

        j_pred = generate_trial(model, ddpm, k_raw, kinetics_norm, joint_norm,
                                 seq_len, device, overlap=args.overlap,
                                 n_samples=args.n_samples)
        # j_pred : (n_samples, T, 29)
        np.save(args.out, j_pred)
        print(f"  [Sample] Saved → {args.out}  shape={j_pred.shape}")

        if args.joints_gt:
            j_gt    = np.load(args.joints_gt).astype(np.float32)[:j_pred.shape[1]]
            metrics = compute_metrics(j_gt, j_pred)
            print(f"  [Metrics]  MAE={metrics['mae']:.4f} rad  |  "
                  f"RMSE={metrics['rmse']:.4f} rad  |  "
                  f"MeanCorr={metrics['mean_corr']:.4f}  |  "
                  f"Diversity={metrics['diversity']:.4f} rad"
                  + ("  (n=1)" if args.n_samples == 1 else ""))
            if args.plot:
                plot_comparison(j_gt, j_pred)
        elif args.plot:
            print("  [Warning] --plot without --joints-gt: nothing to compare.")

    # ── Batch mode ────────────────────────────────────────────────────────────
    elif args.npy_root:
        from dataset import _find_trial_paths
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs   = _find_trial_paths(Path(args.npy_root))
        all_metrics = []

        for joints_path, kinetics_path in pairs:
            k_raw  = np.load(kinetics_path).astype(np.float32)
            j_pred = generate_trial(model, ddpm, k_raw, kinetics_norm, joint_norm,
                                     seq_len, device, overlap=args.overlap,
                                     n_samples=args.n_samples)
            # j_pred : (n_samples, T, 29)

            rel      = joints_path.parent.relative_to(args.npy_root)
            save_dir = out_dir / rel
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / "pred_joints.npy", j_pred)

            j_gt    = np.load(joints_path).astype(np.float32)[:j_pred.shape[1]]
            metrics = compute_metrics(j_gt, j_pred)
            all_metrics.append(metrics)
            print(f"  {rel}  |  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
                  f"Corr={metrics['mean_corr']:.4f}  Diversity={metrics['diversity']:.4f}")

            if args.plot:
                plot_comparison(j_gt, j_pred)

        # Aggregate summary
        if all_metrics:
            import json
            avg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
            print(f"\n  ── Aggregate ──────────────────────────────────")
            print(f"  MAE       : {avg['mae']:.4f} rad")
            print(f"  RMSE      : {avg['rmse']:.4f} rad")
            print(f"  MeanCorr  : {avg['mean_corr']:.4f}")
            print(f"  Diversity : {avg['diversity']:.4f} rad")
            with open(out_dir / "metrics.json", "w") as f:
                json.dump({"per_trial": all_metrics, "average": avg}, f, indent=2)
    else:
        print("  [Error] Provide either --kinetics or --npy-root.")


if __name__ == "__main__":
    main()