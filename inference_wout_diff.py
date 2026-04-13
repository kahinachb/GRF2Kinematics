"""
inference.py
============
Inference script for the discriminative MotionPredictor model.
Takes kinetics (.npy) as input and predicts 29 joint angles.

Usage:
    python inference.py --checkpoint checkpoints/best.pt \
                        --kinetics /path/to/kinetics.npy \
                        --joints-gt /path/to/joints.npy \
                        --plot

    # Batch mode: process all trials in a folder
    python inference.py --checkpoint checkpoints/best.pt \
                        --npy-root /path/to/npy \
                        --out-dir results/ \
                        --plot
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from utils.diffuser_utils import MotionPredictor
from dataset import Normalizer


# ─────────────────────────────────────────────────────────────────────────────
# DOF LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

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
LOWER_IDX = list(range(12))
UPPER_IDX  = list(range(12, 29))


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """Load MotionPredictor from a checkpoint produced by train.py."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    model = MotionPredictor(
        joint_dim  = 29,
        force_dim  = 12,
        embed_dim  = args["embed_dim"],
        nhead      = args["nhead"],
        num_layers = args["num_layers"],
        dropout    = 0.0,   # disable dropout at inference
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"  [Inference] Checkpoint loaded  (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['best_val_loss']:.6f})")
    return model, args


# ─────────────────────────────────────────────────────────────────────────────
# WINDOWED PREDICTION FOR A SINGLE TRIAL
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_trial(
    model:         MotionPredictor,
    kinetics_raw:  np.ndarray,
    kinetics_norm: Normalizer,
    joint_norm:    Normalizer,
    seq_len:       int,
    device:        torch.device,
    overlap:       int = 32,
) -> np.ndarray:
    """
    Predict joint angles for a full trial using sliding windows
    with Hanning-weighted overlap-add blending.

    Args:
        kinetics_raw  : (T, 12)  raw (un-normalised) kinetics
        overlap       : number of frames shared between consecutive windows

    Returns:
        joints_pred   : (T, 29)  predicted joint angles (denormalised)
    """
    T      = len(kinetics_raw)
    k_norm = kinetics_norm.transform(kinetics_raw)   # (T, 12)

    pred_acc   = np.zeros((T, 29), dtype=np.float32)
    weight_acc = np.zeros((T, 1),  dtype=np.float32)
    # win        = np.hanning(seq_len).reshape(-1, 1).astype(np.float32)
    win = np.ones((seq_len, 1), dtype=np.float32)

    stride = seq_len - overlap
    start  = 0
    n_wins = 0

    while start < T:
        end   = min(start + seq_len, T)
        chunk = end - start

        # Pad the last window if shorter than seq_len
        k_chunk         = np.zeros((seq_len, 12), dtype=np.float32)
        k_chunk[:chunk] = k_norm[start:end]
        k_tensor        = torch.from_numpy(k_chunk).unsqueeze(0).to(device)  # (1, W, 12)

        j_win = model(k_tensor)                                               # (1, W, 29)
        j_win = joint_norm.inverse_transform(j_win.squeeze(0).cpu().numpy())  # (W, 29)

        pred_acc  [start:end] += j_win[:chunk] * win[:chunk]
        weight_acc[start:end] += win[:chunk]
        n_wins += 1

        if end == T:
            break
        start += stride

    weight_acc = np.where(weight_acc < 1e-8, 1.0, weight_acc)
    print(f"  [Inference] {T} frames processed in {n_wins} windows")
    return pred_acc / weight_acc   # (T, 29)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def _plot_section(joints_gt, joints_pred, dof_indices, dof_names,
                  title, n_cols=3, fps=100):
    """
    One figure: prediction vs GT with per-DOF RMSE badge (rad + degrees).

    joints_gt   : (T, 29) or None
    joints_pred : (T, 29)
    """
    n_dof  = len(dof_indices)
    n_rows = -(-n_dof // n_cols)
    T      = joints_pred.shape[0]
    time_s = np.arange(T) / fps

    rmse_rad = None
    if joints_gt is not None:
        rmse_rad = np.sqrt(((joints_gt - joints_pred) ** 2).mean(axis=0))  # (29,)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5.5, n_rows * 2.8),
        facecolor="white",
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.002)
    axes_flat = axes.flatten() if n_dof > 1 else [axes]

    for idx, (col, name) in enumerate(zip(dof_indices, dof_names)):
        ax = axes_flat[idx]

        ax.plot(time_s, joints_pred[:, col], color="red",   lw=1.5, label="Prediction")
        if joints_gt is not None:
            ax.plot(time_s, joints_gt[:, col], color="black", lw=1.5,
                    alpha=0.85, label="Ground Truth", zorder=5)

        ax.set_title(name, fontsize=9, fontweight="bold", pad=3)

        if rmse_rad is not None:
            rmse     = rmse_rad[col]
            rmse_deg = np.degrees(rmse)
            ax.text(0.98, 0.96,
                    f"RMSE  {rmse:.4f} rad\n      {rmse_deg:.2f} °",
                    transform=ax.transAxes,
                    fontsize=7.5, ha="right", va="top")

        ax.set_ylabel("rad", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_facecolor("white")
        ax.tick_params(labelsize=7.5)

        if idx >= n_dof - n_cols:
            ax.set_xlabel("Time (s)", fontsize=8)

    axes_flat[0].legend(fontsize=8, loc="upper left", framealpha=0.7)

    for j in range(n_dof, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()


def plot_comparison(joints_gt, joints_pred, out_dir=None, trial_name="trial", fps=100):
    """
    Two figures:
        Figure 1 — Lower body  (DOFs 1–12)
        Figure 2 — Upper body  (DOFs 13–29)

    Args:
        joints_gt   : (T, 29) or None
        joints_pred : (T, 29)
        out_dir     : if set, figures are saved; otherwise shown interactively
        trial_name  : prefix for saved filenames
    """
    _plot_section(
        joints_gt, joints_pred,
        dof_indices = LOWER_IDX,
        dof_names   = LOWER_NAMES,
        title       = "Lower Body  (DOFs 1–12)  —  Prediction vs Ground Truth",
        fps         = fps,
    )
    if out_dir:
        plt.savefig(Path(out_dir) / f"{trial_name}_lower_body.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved → {out_dir}/{trial_name}_lower_body.png")
    else:
        plt.show()

    _plot_section(
        joints_gt, joints_pred,
        dof_indices = UPPER_IDX,
        dof_names   = UPPER_NAMES,
        title       = "Upper Body  (DOFs 13–29)  —  Prediction vs Ground Truth",
        fps         = fps,
    )
    if out_dir:
        plt.savefig(Path(out_dir) / f"{trial_name}_upper_body.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Saved → {out_dir}/{trial_name}_upper_body.png")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(joints_gt: np.ndarray, joints_pred: np.ndarray) -> dict:
    mae  = float(np.abs(joints_gt - joints_pred).mean())
    rmse = float(np.sqrt(((joints_gt - joints_pred) ** 2).mean()))
    corrs = []
    for i in range(joints_gt.shape[1]):
        g, p = joints_gt[:, i], joints_pred[:, i]
        if g.std() > 1e-8 and p.std() > 1e-8:
            corrs.append(float(np.corrcoef(g, p)[0, 1]))
    corr = float(np.mean(corrs)) if corrs else float("nan")
    return {"mae": mae, "rmse": rmse, "mean_corr": corr}


def print_metrics(joints_gt, joints_pred):
    metrics   = compute_metrics(joints_gt, joints_pred)
    rmse_dofs = np.sqrt(((joints_gt - joints_pred) ** 2).mean(axis=0))  # (29,)

    print(f"\n{'─'*60}")
    print(f"  GLOBAL METRICS")
    print(f"{'─'*60}")
    print(f"  MAE       : {metrics['mae']:.6f} rad")
    print(f"  RMSE      : {metrics['rmse']:.6f} rad")
    print(f"  Mean Corr : {metrics['mean_corr']:.4f}")

    all_names = LOWER_NAMES + UPPER_NAMES
    print(f"\n  {'DOF':>5}  {'Seg':<6}  {'Name':<26}  RMSE (rad)   RMSE (°)")
    print(f"  {'─'*62}")
    for dof, name in enumerate(all_names):
        seg  = "Lower" if dof < 12 else "Upper"
        rmse = rmse_dofs[dof]
        print(f"  {dof+1:>5}  {seg:<6}  {name:<26}  {rmse:.6f}   {np.degrees(rmse):.3f}")
    print(f"{'─'*60}\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference — MotionPredictor (discriminative)")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to best.pt from train.py")
    parser.add_argument("--norm-prefix", default=None,
                        help="Prefix for normalizer files "
                             "(default: same folder as checkpoint)")
    # Single-file mode
    parser.add_argument("--kinetics",  default=None,
                        help="Single kinetics.npy  (T, 12)")
    parser.add_argument("--joints-gt", default=None,
                        help="Ground-truth joints.npy  (T, 29)  for comparison")
    parser.add_argument("--out",       default="predicted_joints.npy",
                        help="Output .npy path (single-file mode)")
    # Batch mode
    parser.add_argument("--npy-root",  default=None,
                        help="npy/ root: process all trials found recursively")
    parser.add_argument("--out-dir",   default="inference_results",
                        help="Output directory (batch mode)")
    # Options
    parser.add_argument("--overlap",   type=int, default=32,
                        help="Window overlap in frames (default 32)")
    parser.add_argument("--plot",      action="store_true",
                        help="Show / save comparison figures")
    parser.add_argument("--fps",       type=int, default=100,
                        help="Frames per second for time axis (default 100)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*60}")
    print("  MOTION PREDICTOR  —  INFERENCE")
    print(f"  Device : {device}")
    print(f"{'═'*60}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, train_args = load_model(args.checkpoint, device)
    seq_len = train_args["seq_len"]

    # ── Load normalizers ──────────────────────────────────────────────────────
    ckpt_dir    = Path(args.checkpoint).parent
    norm_prefix = args.norm_prefix or str(ckpt_dir / "normalizer")

    joint_norm    = Normalizer()
    kinetics_norm = Normalizer()
    joint_norm   .load(f"{norm_prefix}_joints.npz")
    kinetics_norm.load(f"{norm_prefix}_kinetics.npz")
    print(f"  Normalizers loaded from {norm_prefix}_*.npz\n")

    # ── Single-file mode ──────────────────────────────────────────────────────
    if args.kinetics:
        k_raw = np.load(args.kinetics).astype(np.float32)
        print(f"  Kinetics : {args.kinetics}  "
              f"({len(k_raw)} frames, {len(k_raw)/args.fps:.2f} s)")

        j_pred = predict_trial(model, k_raw, kinetics_norm, joint_norm,
                               seq_len, device, overlap=args.overlap)
        np.save(args.out, j_pred)
        print(f"  ✓ Predictions saved → {args.out}  shape={j_pred.shape}")

        j_gt = None
        if args.joints_gt:
            j_gt = np.load(args.joints_gt).astype(np.float32)
            T    = min(len(j_gt), len(j_pred))
            j_gt, j_pred = j_gt[:T], j_pred[:T]
            print_metrics(j_gt, j_pred)

        if args.plot:
            if j_gt is None:
                print("  [Warning] --plot without --joints-gt: GT will not appear.")
            trial_name = Path(args.kinetics).parent.name
            out_dir    = Path(args.out).parent
            plot_comparison(j_gt, j_pred,
                            out_dir=str(out_dir),
                            trial_name=trial_name,
                            fps=args.fps)

    # ── Batch mode ────────────────────────────────────────────────────────────
    elif args.npy_root:
        from dataset import _find_trial_paths
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pairs   = _find_trial_paths(Path(args.npy_root))
        all_metrics = []

        for joints_path, kinetics_path in pairs:
            k_raw  = np.load(kinetics_path).astype(np.float32)
            j_pred = predict_trial(model, k_raw, kinetics_norm, joint_norm,
                                   seq_len, device, overlap=args.overlap)

            j_gt = np.load(joints_path).astype(np.float32)
            T    = min(len(j_gt), len(j_pred))
            j_gt, j_pred = j_gt[:T], j_pred[:T]

            rel      = joints_path.parent.relative_to(args.npy_root)
            save_dir = out_dir / rel
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / "pred_joints.npy", j_pred)

            metrics = compute_metrics(j_gt, j_pred)
            all_metrics.append(metrics)
            print(f"  {rel}  |  MAE={metrics['mae']:.4f}  "
                  f"RMSE={metrics['rmse']:.4f}  "
                  f"Corr={metrics['mean_corr']:.4f}")

            if args.plot:
                trial_name = str(rel).replace("/", "_")
                plot_comparison(j_gt, j_pred,
                                out_dir=str(save_dir),
                                trial_name=trial_name,
                                fps=args.fps)

        # Aggregate summary
        if all_metrics:
            avg = {k: float(np.mean([m[k] for m in all_metrics]))
                   for k in all_metrics[0]}
            print(f"\n{'─'*52}")
            print(f"  AGGREGATE  ({len(all_metrics)} trials)")
            print(f"{'─'*52}")
            print(f"  MAE       : {avg['mae']:.4f} rad")
            print(f"  RMSE      : {avg['rmse']:.4f} rad")
            print(f"  Mean Corr : {avg['mean_corr']:.4f}")
            print(f"{'─'*52}\n")
            with open(out_dir / "metrics.json", "w") as f:
                json.dump({"per_trial": all_metrics, "average": avg}, f, indent=2)
    else:
        print("  [Error] Provide either --kinetics or --npy-root.")


if __name__ == "__main__":
    main()