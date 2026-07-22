"""Inference and evaluation for train_fm_full_step2motion_real_feet_improved.py.

Outputs, for every requested trial:
  - raw and (optionally) Butterworth-filtered prediction CSV files;
  - raw and filtered per-joint metric CSV files, averaged across seeds;
  - lower-body and upper-body plots for the seed-0 prediction.

The model definition and GRFM preprocessing helpers are imported from the
matching training script, so a checkpoint cannot silently be evaluated with a
different architecture or preprocessing pipeline.  Both Euler and Heun keep
the same full-trial overlap inpainting procedure.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt

from train_fm_full_step2motion_real_feet_improved import (
    FlowTransformer,
    INFERENCE_STEPS,
    N_JOINTS,
    STRIDE,
    WINDOW_SIZE,
    load_trial,
    window_starts,
)


JOINT_NAMES = [
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",
    "Lumbar_flex_ext", "Lumbar_lateral_flex", "Lcalvicule_x",
    "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
    "Lelbow_flex_ext", "Lelbow_pron_supi",
    "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
    "Rcalvicule_x", "Rshoulder_flex_ext", "Rshoulder_abd_add",
    "Rshoulder_int_ext_rot", "Relbow_flex_ext", "Relbow_pron_supi",
]


def lowpass_prediction(prediction, fs_hz=100.0, cutoff_hz=6.0, order=4):
    """Zero-phase Butterworth filtering of a complete predicted trial."""
    if len(prediction) <= 3 * (2 * order + 1):
        raise ValueError("Prediction is too short for the requested zero-phase filter.")
    sos = butter(order, cutoff_hz, btype="low", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, prediction, axis=0)


def set_inference_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def per_joint_metrics(reference, prediction):
    error = reference - prediction
    return np.sqrt(np.mean(error ** 2, axis=0)), np.mean(np.abs(error), axis=0)


def aggregate_metrics(reference, predictions):
    """Return per-joint mean/std RMSE and MAE across a list of predictions."""
    rmse, mae = zip(*(per_joint_metrics(reference, prediction) for prediction in predictions))
    rmse = np.stack(rmse)
    mae = np.stack(mae)
    return {
        "rmse_mean": rmse.mean(axis=0),
        "rmse_std": rmse.std(axis=0),
        "mae_mean": mae.mean(axis=0),
        "mae_std": mae.std(axis=0),
        "global_rmse": float(np.sqrt(np.mean(rmse ** 2))),
        "global_mae": float(mae.mean()),
    }


def save_metrics(metrics, output_path):
    table = pd.DataFrame({
        "joint": JOINT_NAMES,
        "rmse_mean": metrics["rmse_mean"],
        "rmse_std": metrics["rmse_std"],
        "mae_mean": metrics["mae_mean"],
        "mae_std": metrics["mae_std"],
    })
    table.to_csv(output_path, index=False)


def plot_prediction_groups(reference, prediction, output_dir, subject_name, trial_name, suffix):
    """Create the three plot groups used by the previous inference script."""
    prefix = f"{subject_name}_{trial_name}"

    figure, axes = plt.subplots(6, 2, figsize=(15, 18))
    for row in range(6):
        for column, index in ((0, row), (1, row + 6)):
            rmse = np.sqrt(np.mean((reference[:, index] - prediction[:, index]) ** 2))
            axes[row, column].plot(reference[:, index], "k--", alpha=0.6, label="Reference")
            axes[row, column].plot(prediction[:, index], "r", linewidth=1.5, label="Prediction")
            axes[row, column].set_title(f"{JOINT_NAMES[index]} (RMSE: {rmse:.4f})", fontsize=10)
            axes[row, column].grid(True)
    axes[0, 0].legend()
    figure.suptitle(f"Lower body — {suffix}", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_lower_body_joints_{suffix}.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(18, 10))
    lumbar = [12, 13]
    left_middle = [14, 15, 16]
    left_last = [17, 18, 19]
    for row in range(3):
        if row < len(lumbar):
            index = lumbar[row]
            rmse = np.sqrt(np.mean((reference[:, index] - prediction[:, index]) ** 2))
            axes[row, 0].plot(reference[:, index], "k--", alpha=0.6, label="Reference")
            axes[row, 0].plot(prediction[:, index], "r", linewidth=1.5, label="Prediction")
            axes[row, 0].set_title(f"{JOINT_NAMES[index]} (RMSE: {rmse:.4f})", fontsize=10)
            axes[row, 0].grid(True)
        else:
            axes[row, 0].axis("off")
        for column, index in ((1, left_middle[row]), (2, left_last[row])):
            rmse = np.sqrt(np.mean((reference[:, index] - prediction[:, index]) ** 2))
            axes[row, column].plot(reference[:, index], "k--", alpha=0.6, label="Reference")
            axes[row, column].plot(prediction[:, index], "r", linewidth=1.5, label="Prediction")
            axes[row, column].set_title(f"{JOINT_NAMES[index]} (RMSE: {rmse:.4f})", fontsize=10)
            axes[row, column].grid(True)
    axes[0, 0].legend()
    figure.suptitle(f"Upper body: lumbar and left — {suffix}", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_upper_body_lumbar_left_{suffix}.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(18, 10))
    cervical = [20, 21, 22]
    right_middle = [23, 24, 25]
    right_last = [26, 27, 28]
    for row in range(3):
        for column, index in ((0, cervical[row]), (1, right_middle[row]), (2, right_last[row])):
            rmse = np.sqrt(np.mean((reference[:, index] - prediction[:, index]) ** 2))
            axes[row, column].plot(reference[:, index], "k--", alpha=0.6, label="Reference")
            axes[row, column].plot(prediction[:, index], "r", linewidth=1.5, label="Prediction")
            axes[row, column].set_title(f"{JOINT_NAMES[index]} (RMSE: {rmse:.4f})", fontsize=10)
            axes[row, column].grid(True)
    axes[0, 0].legend()
    figure.suptitle(f"Upper body: cervical and right — {suffix}", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_upper_body_cervical_right_{suffix}.png", dpi=300)
    plt.close(figure)


def load_stats(scalers_path):
    with open(scalers_path, "r") as file:
        stats = {key: torch.tensor(value, dtype=torch.float32) for key, value in json.load(file).items()}
    expected = {"f_mean", "f_std", "j_mean", "j_std"}
    if set(stats) != expected:
        raise ValueError(
            f"Scalers at {scalers_path} do not match the new training script. "
            f"Expected keys {sorted(expected)}, got {sorted(stats)}."
        )
    if stats["f_mean"].numel() != 18 or stats["j_mean"].numel() != N_JOINTS:
        raise ValueError("Scaler dimensions are incompatible with 18 GRFM features and 29 joint DoFs.")
    return stats


def load_ema_model(model_path, device):
    """Load either the simple EMA state dict or a checkpoint containing a model key."""
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = FlowTransformer().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_full_trial_with_solver(
    model,
    kinetics_path,
    joints_path,
    stats,
    device,
    solver="heun",
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    n_steps=INFERENCE_STEPS,
):
    """Generate a full trial with Euler or Heun and exact overlap inpainting.

    This is kept in the inference script so selecting a solver does not alter
    the training script or its checkpoint.  The preprocessing, windows, and
    inpainting are identical for both solvers; only the integration update
    differs.
    """
    solver = solver.lower()
    if solver not in {"euler", "heun"}:
        raise ValueError(f"Unsupported solver {solver!r}; choose 'euler' or 'heun'.")

    model.eval()
    body_weight_cache = {}
    grfm, reference = load_trial(kinetics_path, joints_path, body_weight_cache)
    condition = (torch.from_numpy(grfm) - stats["f_mean"]) / (stats["f_std"] + 1e-6)
    total_frames = len(condition)

    if total_frames < window_size:
        padding = condition[-1:].repeat(window_size - total_frames, 1)
        condition_for_sampling = torch.cat((condition, padding), dim=0)
    else:
        condition_for_sampling = condition

    full_prediction = torch.zeros((len(condition_for_sampling), N_JOINTS), device=device)
    previous_start = None
    nfe_per_window = n_steps * (1 if solver == "euler" else 2)
    print(
        f"  [INF] Sampling Flow Matching ({solver.capitalize()}): "
        f"{total_frames} frames, {n_steps} steps/window, {nfe_per_window} NFE/window"
    )

    for start in window_starts(len(condition_for_sampling), window_size, stride):
        end = start + window_size
        condition_window = condition_for_sampling[start:end].unsqueeze(0).to(device)
        x0 = torch.randn((1, window_size, N_JOINTS), device=device)
        x = x0.clone()

        overlap = 0 if previous_start is None else previous_start + window_size - start
        if overlap < 0:
            raise RuntimeError("Windows must overlap for inpainting.")
        if overlap:
            known_x1 = full_prediction[start:start + overlap].unsqueeze(0)
            x0_known = x0[:, :overlap]

            def interpolated_known(t_value):
                return (1.0 - t_value) * x0_known + t_value * known_x1

        dt = 1.0 / n_steps
        for step in range(n_steps):
            t0 = step / n_steps
            t1 = (step + 1) / n_steps
            if overlap:
                x[:, :overlap] = interpolated_known(t0)
            t0_tensor = torch.tensor([t0], device=device)
            v0 = model(x, t0_tensor, condition_window)

            if solver == "euler":
                x = x + dt * v0
                if overlap:
                    x[:, :overlap] = interpolated_known(t1)
            else:
                t1_tensor = torch.tensor([t1], device=device)
                x_predictor = x + dt * v0
                if overlap:
                    x_predictor[:, :overlap] = interpolated_known(t1)
                v1 = model(x_predictor, t1_tensor, condition_window)
                x = x + 0.5 * dt * (v0 + v1)

        if overlap:
            x[:, :overlap] = known_x1
        full_prediction[start:end] = x.squeeze(0)
        previous_start = start

    prediction = full_prediction[:total_frames].cpu() * stats["j_std"] + stats["j_mean"]
    return reference, prediction.numpy()


def run_inference(
    subject_name,
    trial_name,
    model_path="results_real_feet_improved/fm_biomech_model_best_ema.pth",
    scalers_path="results_real_feet_improved/scalers_concat.json",
    data_root="processed_data_feet",
    output_dir="results_real_feet_improved",
    n_steps=INFERENCE_STEPS,
    n_seeds=3,
    solver="heun",
    apply_lowpass=True,
    cutoff_hz=6.0,
):
    """Run multi-seed full-trial inference for the new feet-frame model."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least one.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least one.")
    solver = solver.lower()
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be either 'euler' or 'heun'.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = Path(data_root) / subject_name / trial_name
    kinetics_path = data_path / "kinetics_feet.npy"
    joints_path = data_path / "all_joints.npy"
    if not kinetics_path.exists() or not joints_path.exists():
        raise FileNotFoundError(f"Expected kinetics_feet.npy and all_joints.npy in {data_path}")

    print("\n" + "=" * 68)
    print("FLOW MATCHING INFERENCE — REAL SQUAT / LOCAL FOOT FRAMES")
    print(f"Subject/trial: {subject_name}/{trial_name}")
    nfe_per_window = n_steps * (1 if solver == "euler" else 2)
    print(
        f"Device: {device} | sampler: {solver.capitalize()} | "
        f"steps: {n_steps} | NFE/window: {nfe_per_window}"
    )
    print(f"Seeds: {n_seeds} | Butterworth: {'on' if apply_lowpass else 'off'}")
    print("=" * 68)

    stats = load_stats(scalers_path)
    model = load_ema_model(model_path, device)
    print(f"[1/4] EMA model loaded: {model_path}")
    print(f"[2/4] Data: {kinetics_path}")

    raw_predictions = []
    filtered_predictions = []
    reference = None
    for seed in range(n_seeds):
        set_inference_seed(seed)
        reference, prediction = predict_full_trial_with_solver(
            model, kinetics_path, joints_path, stats, device,
            solver=solver, n_steps=n_steps,
        )
        raw_predictions.append(prediction)
        if apply_lowpass:
            filtered_predictions.append(lowpass_prediction(prediction, cutoff_hz=cutoff_hz))
        print(f"  seed={seed}: completed")

    raw_metrics = aggregate_metrics(reference, raw_predictions)
    filtered_metrics = aggregate_metrics(reference, filtered_predictions) if apply_lowpass else None

    # Keep Euler and Heun results separate so one run cannot overwrite the other.
    prefix = f"{subject_name}_{trial_name}_feet_improved_{solver}"
    raw_seed0 = raw_predictions[0]
    pd.DataFrame(raw_seed0, columns=JOINT_NAMES).to_csv(
        output_dir / f"{prefix}_prediction_raw.csv", index=False
    )
    save_metrics(raw_metrics, output_dir / f"{prefix}_metrics_raw.csv")
    plot_prediction_groups(
        reference, raw_seed0, output_dir, subject_name, f"{trial_name}_feet_improved_{solver}", "raw"
    )

    print(f"[3/4] Raw RMSE={raw_metrics['global_rmse']:.6f} | MAE={raw_metrics['global_mae']:.6f}")
    if apply_lowpass:
        filtered_seed0 = filtered_predictions[0]
        pd.DataFrame(filtered_seed0, columns=JOINT_NAMES).to_csv(
            output_dir / f"{prefix}_prediction_filtered.csv", index=False
        )
        save_metrics(filtered_metrics, output_dir / f"{prefix}_metrics_filtered.csv")
        plot_prediction_groups(
            reference, filtered_seed0, output_dir, subject_name,
            f"{trial_name}_feet_improved_{solver}", "filtered"
        )
        print(
            f"[4/4] Filtered RMSE={filtered_metrics['global_rmse']:.6f} | "
            f"MAE={filtered_metrics['global_mae']:.6f}"
        )
    else:
        print("[4/4] Filter disabled; only raw outputs were written.")

    summary = {
        "subject": subject_name,
        "trial": trial_name,
        "solver": solver,
        "n_steps": n_steps,
        "nfe_per_window": nfe_per_window,
        "n_seeds": n_seeds,
        "raw_global_rmse": raw_metrics["global_rmse"],
        "raw_global_mae": raw_metrics["global_mae"],
    }
    if filtered_metrics is not None:
        summary["filtered_global_rmse"] = filtered_metrics["global_rmse"]
        summary["filtered_global_mae"] = filtered_metrics["global_mae"]
    with open(output_dir / f"{prefix}_summary.json", "w") as file:
        json.dump(summary, file, indent=2)
    return summary


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference for the improved real-feet Flow Matching model.")
    parser.add_argument("--subject", default="Christine")
    parser.add_argument("--trial", default="Trial110")
    parser.add_argument("--model", default="results_real_feet_improved/fm_biomech_model_best_raw.pth")
    parser.add_argument("--scalers", default="results_real_feet_improved/scalers_concat.json")
    parser.add_argument("--data-root", default="processed_data_feet")
    parser.add_argument("--output-dir", default="results_real_feet_improved")
    parser.add_argument(
        "--solver", choices=("heun", "euler"), default="heun",
        help="ODE solver. Heun is second order; Euler uses half as many network evaluations per step.",
    )
    parser.add_argument("--steps", type=int, default=INFERENCE_STEPS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--no-filter", action="store_true", help="Write and evaluate only the raw prediction.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    result = run_inference(
        subject_name=arguments.subject,
        trial_name=arguments.trial,
        model_path=arguments.model,
        scalers_path=arguments.scalers,
        data_root=arguments.data_root,
        output_dir=arguments.output_dir,
        n_steps=arguments.steps,
        n_seeds=arguments.seeds,
        solver=arguments.solver,
        apply_lowpass=not arguments.no_filter,
        cutoff_hz=arguments.cutoff,
    )
    print("\nSummary:")
    for key, value in result.items():
        print(f"  {key}: {value}")
