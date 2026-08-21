"""Inference for the morphology-conditioned real-feet Flow Matching model.

This script is paired with ``train_fm_full_step2motion_real_feet_morphology.py``.
It extracts the same subject-specific segment lengths from the scaled URDF,
then runs full-trial Euler or Heun inference with overlap inpainting.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_fm_full_step2motion_real_feet_improved as base
from fm_inference_real_feet_improved import (
    JOINT_NAMES,
    aggregate_metrics,
    lowpass_prediction,
    plot_prediction_groups,
    save_metrics,
    set_inference_seed,
)
from train_fm_full_step2motion_real_feet_morphology import (
    MORPHOLOGY_FEATURE_NAMES,
    N_MORPHOLOGY_FEATURES,
    MorphologyFlowTransformer,
    subject_morphology,
)


def load_stats(scalers_path: Path):
    """Load scalers and verify that they belong to the morphology experiment."""
    with open(scalers_path, "r") as file:
        payload = json.load(file)
    required = {
        "f_mean", "f_std", "j_mean", "j_std", "morph_mean", "morph_std",
        "morphology_feature_names",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Morphology scalers at {scalers_path} are missing: {sorted(missing)}")
    if payload["morphology_feature_names"] != list(MORPHOLOGY_FEATURE_NAMES):
        raise ValueError("Morphology feature order differs from the training script.")

    stats = {
        key: torch.tensor(payload[key], dtype=torch.float32)
        for key in ("f_mean", "f_std", "j_mean", "j_std", "morph_mean", "morph_std")
    }
    expected_shapes = {
        "f_mean": (base.N_GRFM_FEATURES,),
        "f_std": (base.N_GRFM_FEATURES,),
        "j_mean": (base.N_JOINTS,),
        "j_std": (base.N_JOINTS,),
        "morph_mean": (N_MORPHOLOGY_FEATURES,),
        "morph_std": (N_MORPHOLOGY_FEATURES,),
    }
    for key, expected_shape in expected_shapes.items():
        if tuple(stats[key].shape) != expected_shape:
            raise ValueError(f"{key} has shape {tuple(stats[key].shape)}, expected {expected_shape}.")
    return stats


def load_morphology_model(model_path: Path, device):
    """Load an EMA or raw state dict from the morphology training script."""
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = MorphologyFlowTransformer().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_full_trial_with_solver(
    model,
    kinetics_path: Path,
    joints_path: Path,
    subject_name: str,
    stats,
    device,
    solver="heun",
    window_size=base.WINDOW_SIZE,
    stride=base.STRIDE,
    n_steps=base.INFERENCE_STEPS,
):
    """Generate a trial with a morphology-conditioned Euler or Heun solver."""
    solver = solver.lower()
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'.")

    model.eval()
    body_weight_cache = {}
    grfm, reference = base.load_trial(kinetics_path, joints_path, body_weight_cache)
    condition = (torch.from_numpy(grfm) - stats["f_mean"]) / (stats["f_std"] + 1e-6)
    morphology = subject_morphology(subject_name, {})
    morphology = torch.from_numpy(morphology)
    morphology = (morphology - stats["morph_mean"]) / (stats["morph_std"] + 1e-6)
    total_frames = len(condition)

    if total_frames < window_size:
        padding = condition[-1:].repeat(window_size - total_frames, 1)
        condition_for_sampling = torch.cat((condition, padding), dim=0)
    else:
        condition_for_sampling = condition

    full_prediction = torch.zeros((len(condition_for_sampling), base.N_JOINTS), device=device)
    previous_start = None
    nfe_per_window = n_steps * (1 if solver == "euler" else 2)
    print(
        f"  [INF] Sampling morphology-conditioned Flow Matching ({solver.capitalize()}): "
        f"{total_frames} frames, {n_steps} steps/window, {nfe_per_window} NFE/window"
    )

    for start in base.window_starts(len(condition_for_sampling), window_size, stride):
        end = start + window_size
        condition_window = condition_for_sampling[start:end].unsqueeze(0).to(device)
        morphology_window = morphology.unsqueeze(0).to(device)
        x0 = torch.randn((1, window_size, base.N_JOINTS), device=device)
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
            velocity0 = model(x, t0_tensor, condition_window, morphology_window)

            if solver == "euler":
                x = x + dt * velocity0
                if overlap:
                    x[:, :overlap] = interpolated_known(t1)
            else:
                t1_tensor = torch.tensor([t1], device=device)
                x_predictor = x + dt * velocity0
                if overlap:
                    x_predictor[:, :overlap] = interpolated_known(t1)
                velocity1 = model(x_predictor, t1_tensor, condition_window, morphology_window)
                x = x + 0.5 * dt * (velocity0 + velocity1)

        if overlap:
            x[:, :overlap] = known_x1
        full_prediction[start:end] = x.squeeze(0)
        previous_start = start

    prediction = full_prediction[:total_frames].cpu() * stats["j_std"] + stats["j_mean"]
    return reference, prediction.numpy()


def run_inference(
    subject_name,
    trial_name,
    model_path="results_real_feet_morphology/fm_morphology_model_best_ema.pth",
    scalers_path="results_real_feet_morphology/scalers_morphology.json",
    data_root=None,
    output_dir="results_real_feet_morphology",
    model_label="ema",
    solver="heun",
    n_steps=base.INFERENCE_STEPS,
    n_seeds=3,
    apply_lowpass=True,
    cutoff_hz=6.0,
):
    """Run multi-seed full-trial inference for the morphology model."""
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least one.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least one.")
    solver = solver.lower()
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be either 'euler' or 'heun'.")
    if not model_label.replace("_", "").replace("-", "").isalnum():
        raise ValueError("model_label may contain only letters, numbers, '_' and '-'.")

    data_root = Path(base.DATA_ROOT if data_root is None else data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = data_root / subject_name / trial_name
    kinetics_path = data_path / "kinetics_feet.npy"
    joints_path = data_path / "all_joints.npy"
    if not kinetics_path.exists() or not joints_path.exists():
        raise FileNotFoundError(f"Expected kinetics_feet.npy and all_joints.npy in {data_path}")

    nfe_per_window = n_steps * (1 if solver == "euler" else 2)
    print("\n" + "=" * 72)
    print("FLOW MATCHING INFERENCE — LOCAL FOOT FRAMES + MORPHOLOGY")
    print(f"Subject/trial: {subject_name}/{trial_name}")
    print(f"Device: {device} | model: {model_label} | solver: {solver.capitalize()} | "
          f"steps: {n_steps} | NFE/window: {nfe_per_window}")
    print(f"Seeds: {n_seeds} | Butterworth: {'on' if apply_lowpass else 'off'}")
    print("=" * 72)

    stats = load_stats(Path(scalers_path))
    model = load_morphology_model(Path(model_path), device)
    print(f"[1/4] Model loaded: {model_path}")
    print(f"[2/4] Data: {kinetics_path}")

    raw_predictions = []
    filtered_predictions = []
    reference = None
    for seed in range(n_seeds):
        set_inference_seed(seed)
        reference, prediction = predict_full_trial_with_solver(
            model, kinetics_path, joints_path, subject_name, stats, device,
            solver=solver, n_steps=n_steps,
        )
        raw_predictions.append(prediction)
        if apply_lowpass:
            filtered_predictions.append(lowpass_prediction(prediction, cutoff_hz=cutoff_hz))
        print(f"  seed={seed}: completed")

    raw_metrics = aggregate_metrics(reference, raw_predictions)
    filtered_metrics = aggregate_metrics(reference, filtered_predictions) if apply_lowpass else None

    prefix = f"{subject_name}_{trial_name}_feet_morphology_{model_label}_{solver}"
    raw_seed0 = raw_predictions[0]
    np.savetxt(output_dir / f"{prefix}_prediction_raw.csv", raw_seed0, delimiter=",", header=",".join(JOINT_NAMES), comments="")
    save_metrics(raw_metrics, output_dir / f"{prefix}_metrics_raw.csv")
    plot_prediction_groups(
        reference, raw_seed0, output_dir, subject_name,
        f"{trial_name}_feet_morphology_{model_label}_{solver}", "raw",
    )
    print(f"[3/4] Raw RMSE={raw_metrics['global_rmse']:.6f} | MAE={raw_metrics['global_mae']:.6f}")

    if apply_lowpass:
        filtered_seed0 = filtered_predictions[0]
        np.savetxt(output_dir / f"{prefix}_prediction_filtered.csv", filtered_seed0, delimiter=",", header=",".join(JOINT_NAMES), comments="")
        save_metrics(filtered_metrics, output_dir / f"{prefix}_metrics_filtered.csv")
        plot_prediction_groups(
            reference, filtered_seed0, output_dir, subject_name,
            f"{trial_name}_feet_morphology_{model_label}_{solver}", "filtered",
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
        "model_label": model_label,
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
    parser = argparse.ArgumentParser(description="Inference for the morphology-conditioned real-feet model.")
    parser.add_argument("--subject", default="Christine")
    parser.add_argument("--trial", default="Trial108")
    parser.add_argument("--model", default="results_realV_feet_morphology/fm_morphology_model_best_ema.pth")
    parser.add_argument("--scalers", default="results_realV_feet_morphology/scalers_morphology.json")
    parser.add_argument("--data-root", default="processed_data_feet", help="Defaults to DATA_ROOT from the training script.")
    parser.add_argument("--output-dir", default="results_realV_feet_morphology")
    parser.add_argument("--model-label", default="ema", help="Label used only in output filenames.")
    parser.add_argument("--solver", choices=("heun", "euler"), default="heun")
    parser.add_argument("--steps", type=int, default=base.INFERENCE_STEPS)
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
        model_label=arguments.model_label,
        solver=arguments.solver,
        n_steps=arguments.steps,
        n_seeds=arguments.seeds,
        apply_lowpass=not arguments.no_filter,
        cutoff_hz=arguments.cutoff,
    )
    print("\nSummary:")
    for key, value in result.items():
        print(f"  {key}: {value}")
