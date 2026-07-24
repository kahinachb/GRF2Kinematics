"""Inference for one Anais-task model trained with ``feet_improved``.

The script is paired with ``train_fm_full_step2motion_anais_feet_improved.py``.
It evaluates one subject/trial using local-foot GRFM only and writes the same
CSV files, joint metrics, plots, and optional Butterworth-filtered results as
the Vinc inference workflow.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fm_inference_real_feet_improved as common
import train_fm_full_step2motion_anais_feet_improved as training


def resolve_paths(task: str, results_dir: Path, model_path, scalers_path, model_label: str):
    if model_label not in {"ema", "raw"}:
        raise ValueError("--model-label must be either 'ema' or 'raw'.")
    model_path = (
        Path(model_path)
        if model_path is not None
        else results_dir / f"fm_anais_task_model_best_{model_label}.pth"
    )
    scalers_path = (
        Path(scalers_path)
        if scalers_path is not None
        else results_dir / "scalers_anais_task.json"
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not scalers_path.exists():
        raise FileNotFoundError(f"Scalers not found: {scalers_path}")
    return model_path, scalers_path


def run_inference(
    subject_name: str,
    trial_name: str,
    task: str = "luyo",
    model_path=None,
    scalers_path=None,
    data_root="processed_data_feet",
    urdf_dir="DATA/10_urdf",
    output_dir=None,
    model_label="ema",
    solver="heun",
    n_steps=20,
    n_seeds=3,
    apply_lowpass=True,
    cutoff_hz=6.0,
):
    """Run multi-seed full-trial inference for an Anais feet_improved model."""
    task = training.canonical_task_name(task)
    if task is None:
        raise ValueError(f"Unsupported task. Choose one of: {', '.join(training.SUPPORTED_TASKS)}")
    if training.canonical_task_name(trial_name) != task:
        raise ValueError(
            f"Trial {trial_name!r} is not a {task!r} trial. "
            "Pass the matching --task value to avoid evaluating a different movement."
        )
    if n_seeds < 1 or n_steps < 1:
        raise ValueError("--seeds and --steps must both be at least one.")
    solver = solver.lower()
    if solver not in {"heun", "euler"}:
        raise ValueError("--solver must be 'heun' or 'euler'.")

    data_root = Path(data_root)
    urdf_dir = Path(urdf_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root.resolve()}")
    if not urdf_dir.exists():
        raise FileNotFoundError(f"URDF directory does not exist: {urdf_dir.resolve()}")
    training.configure_base_urdf(urdf_dir)

    results_dir = Path(output_dir) if output_dir is not None else Path(f"results_anais_{task}_feet_improved")
    model_path, scalers_path = resolve_paths(task, results_dir, model_path, scalers_path, model_label)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_root / subject_name / trial_name
    kinetics_path = data_path / "kinetics_feet.npy"
    joints_path = data_path / "all_joints.npy"
    if not kinetics_path.exists() or not joints_path.exists():
        raise FileNotFoundError(f"Expected kinetics_feet.npy and all_joints.npy in {data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nfe_per_window = n_steps * (1 if solver == "euler" else 2)
    print("\n" + "=" * 72)
    print("FLOW MATCHING INFERENCE — ANAIS / LOCAL FOOT FRAMES")
    print(f"Task, subject/trial: {task} | {subject_name}/{trial_name}")
    print(f"Device: {device} | model: {model_label} | solver: {solver.capitalize()} | "
          f"steps: {n_steps} | NFE/window: {nfe_per_window}")
    print(f"Seeds: {n_seeds} | Butterworth: {'on' if apply_lowpass else 'off'}")
    print("=" * 72)

    stats = common.load_stats(scalers_path)
    model = common.load_ema_model(model_path, device)
    print(f"[1/4] Model loaded: {model_path}")
    print(f"[2/4] Data: {kinetics_path}")

    raw_predictions, filtered_predictions = [], []
    reference = None
    for seed in range(n_seeds):
        common.set_inference_seed(seed)
        reference, prediction = common.predict_full_trial_with_solver(
            model, kinetics_path, joints_path, stats, device,
            solver=solver, n_steps=n_steps,
        )
        raw_predictions.append(prediction)
        if apply_lowpass:
            filtered_predictions.append(common.lowpass_prediction(prediction, cutoff_hz=cutoff_hz))
        print(f"  seed={seed}: completed")

    raw_metrics = common.aggregate_metrics(reference, raw_predictions)
    filtered_metrics = (
        common.aggregate_metrics(reference, filtered_predictions) if apply_lowpass else None
    )
    prefix = f"{subject_name}_{trial_name}_anais_{task}_feet_improved_{model_label}_{solver}"
    raw_seed0 = raw_predictions[0]
    pd.DataFrame(raw_seed0, columns=common.JOINT_NAMES).to_csv(
        results_dir / f"{prefix}_prediction_raw.csv", index=False
    )
    common.save_metrics(raw_metrics, results_dir / f"{prefix}_metrics_raw.csv")
    common.plot_prediction_groups(
        reference, raw_seed0, results_dir, subject_name,
        f"{trial_name}_anais_{task}_feet_improved_{model_label}_{solver}", "raw",
    )
    print(f"[3/4] Raw RMSE={raw_metrics['global_rmse']:.6f} | MAE={raw_metrics['global_mae']:.6f}")

    if filtered_metrics is not None:
        filtered_seed0 = filtered_predictions[0]
        pd.DataFrame(filtered_seed0, columns=common.JOINT_NAMES).to_csv(
            results_dir / f"{prefix}_prediction_filtered.csv", index=False
        )
        common.save_metrics(filtered_metrics, results_dir / f"{prefix}_metrics_filtered.csv")
        common.plot_prediction_groups(
            reference, filtered_seed0, results_dir, subject_name,
            f"{trial_name}_anais_{task}_feet_improved_{model_label}_{solver}", "filtered",
        )
        print(
            f"[4/4] Filtered RMSE={filtered_metrics['global_rmse']:.6f} | "
            f"MAE={filtered_metrics['global_mae']:.6f}"
        )
    else:
        print("[4/4] Filter disabled; only raw outputs were written.")

    summary = {
        "dataset": "Anais subjects01-16",
        "task": task,
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
    with open(results_dir / f"{prefix}_summary.json", "w") as file:
        json.dump(summary, file, indent=2)
    return summary


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="luyo")
    parser.add_argument("--subject", default="subject01")
    parser.add_argument("--trial", default="luyo", help="Use luyo_trimmed for subject12 and luyo2 for subject13.")
    parser.add_argument("--model", type=Path, default=None, help="Override the checkpoint path.")
    parser.add_argument("--scalers", type=Path, default=None, help="Override the scaler JSON path.")
    parser.add_argument("--model-label", choices=("ema", "raw"), default="ema")
    parser.add_argument("--data-root", type=Path, default=Path("processed_data_feet"))
    parser.add_argument("--urdf-dir", type=Path, default=Path("DATA/10_urdf"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--solver", choices=("heun", "euler"), default="heun")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cutoff", type=float, default=6.0)
    parser.add_argument("--no-filter", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    result = run_inference(
        subject_name=arguments.subject,
        trial_name=arguments.trial,
        task=arguments.task,
        model_path=arguments.model,
        scalers_path=arguments.scalers,
        data_root=arguments.data_root,
        urdf_dir=arguments.urdf_dir,
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
