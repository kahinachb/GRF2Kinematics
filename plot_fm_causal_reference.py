"""Trace prediction/reference par DOF pour un checkpoint FM causal global."""
import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import train_fm_causal_windows_christine_npz as fm
from train_linear_christine_npz import (JOINT_NAMES, load_pair, split_files,
                                        correlation_columns, discover_variant_files)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("results_fm_causal_windows/window_100/best.pth"))
    p.add_argument("--data-root", type=Path, default=Path("DATA/Christine_synthetic"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results_fm_causal_windows/window_100/reference_figures"))
    p.add_argument("--frame-stride", type=int, default=10)
    p.add_argument("--first-eval-frame", type=int, default=199)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--heads", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def predict(model, loader, device, steps, seed):
    refs, predictions = [], []
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model.eval()
    for condition, target in loader:
        condition = condition.to(device)
        predictions.append(fm.heun(model, condition, steps).cpu().numpy())
        refs.append(target.numpy())
    return np.concatenate(refs), np.concatenate(predictions)


def plot_group(path, time, reference, prediction, indices, title, rmse, cc):
    columns = 2 if len(indices) == 12 else 3
    rows = int(np.ceil(len(indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(16, 3.2 * rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, index in zip(axes, indices):
        axis.plot(time, reference[:, index], color="black", lw=1.2, label="Reference")
        axis.plot(time, prediction[:, index], color="#d62728", lw=1.0,
                  alpha=.9, label="Prediction FM")
        axis.set_title(f"{JOINT_NAMES[index]} | RMSE={rmse[index]:.2f} deg | CC={cc[index]:.2f}",
                       fontsize=9)
        axis.set_ylabel("Angle (deg)")
        axis.grid(alpha=.25)
    for axis in axes[len(indices):]: axis.axis("off")
    for axis in axes[-columns:]: axis.set_xlabel("Temps (s)")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    window = int(checkpoint["window"])
    saved_stats = checkpoint["stats"]
    xm = np.asarray(saved_stats["xm"], dtype=np.float32)
    xs = np.asarray(saved_stats["xs"], dtype=np.float32)
    ym = np.asarray(saved_stats["ym"], dtype=np.float32)
    ys = np.asarray(saved_stats["ys"], dtype=np.float32)

    files = discover_variant_files(args.data_root)
    _, _, test = split_files(files, args.seed, .7, .15)
    test_file = test[0]
    rx, ry = load_pair(test_file, "q", reference=False)
    dataset = fm.CausalWindowDataset([], window, xm, xs, ym, ys,
                                     stride=args.frame_stride,
                                     first_frame=args.first_eval_frame)
    dataset.data = [(rx.astype(np.float32), ry.astype(np.float32))]
    start = max(window - 1, args.first_eval_frame)
    dataset.indices = [(0, t) for t in range(start, len(rx), args.frame_stride)]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    state = checkpoint["model"]
    dim = state["cond_in.weight"].shape[0]
    layers = 1 + max(int(key.split(".")[2]) for key in state
                     if key.startswith("encoder.layers.") and key.split(".")[2].isdigit())
    model = fm.CausalFlowModel(window, dim=dim, heads=args.heads, layers=layers).to(device)
    model.load_state_dict(state)
    ref_norm, pred_norm = predict(model, loader, device, args.steps, args.seed)
    reference = np.degrees(ref_norm * ys + ym)
    prediction = np.degrees(pred_norm * ys + ym)
    rmse = np.sqrt(np.mean((prediction - reference) ** 2, axis=0))
    cc = correlation_columns(reference, prediction)
    time = np.asarray([t for _, t in dataset.indices]) * .01

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_group(args.output_dir / "reference_lower_body.png", time, reference, prediction,
               range(12), f"FM causal global - reference reelle - fenetre {window} frames", rmse, cc)
    plot_group(args.output_dir / "reference_upper_body.png", time, reference, prediction,
               range(12, 29), f"FM causal global - reference reelle - fenetre {window} frames", rmse, cc)
    with (args.output_dir / "reference_metrics_seed42.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["dof", "rmse_deg", "cc"])
        writer.writerows(zip(JOINT_NAMES, rmse, cc))
    np.save(args.output_dir / "reference_prediction_deg.npy", prediction.astype(np.float32))
    np.save(args.output_dir / "reference_target_deg.npy", reference.astype(np.float32))
    print(f"Checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']}, W={window})")
    print(f"Reference: {test_file.name}, {len(reference)} points")
    print(f"RMSE moyenne={rmse.mean():.3f} deg | RMSE globale={np.sqrt(np.mean(rmse**2)):.3f} deg | "
          f"CC median={np.nanmedian(cc):.3f}")
    print(f"Figures: {args.output_dir}")


if __name__ == "__main__":
    main()
