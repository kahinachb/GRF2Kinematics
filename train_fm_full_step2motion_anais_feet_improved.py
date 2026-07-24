"""Flow Matching on one Anais task expressed in local foot frames.

This script is intentionally separate from the Vinc squat experiments.  It
reuses the ``feet_improved`` architecture and preprocessing, but trains only
the selected task from subjects 01--16.  The default task is ``luyo``.

Examples
--------
python train_fm_full_step2motion_anais_feet_improved.py
python train_fm_full_step2motion_anais_feet_improved.py --task lufe
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import train_fm_full_step2motion_real_feet_improved as base


ANAIS_SUBJECT_PATTERN = re.compile(r"subject(?:0[1-9]|1[0-6])$", re.IGNORECASE)
SUPPORTED_TASKS = ("bend", "dyna", "lufe", "luyo", "static2", "walk")


def canonical_task_name(name: str) -> str | None:
    """Map folder-name variants (e.g. ``luyo_trimmed``) to one task name."""
    normalized = name.strip().lower()
    # ``walk - Copie`` is a duplicate of the original trial and must never be
    # selected as independent data.
    if "copie" in normalized:
        return None
    for task in SUPPORTED_TASKS:
        if normalized == task or normalized.startswith(f"{task}_") or normalized.startswith(f"{task}2"):
            return task
    return None


def collect_task_samples(data_root: Path, task: str):
    """Collect exactly one selected-task trial per Anais subject.

    The function accepts the known folder variants ``bend2``, ``lufe2`` and
    ``luyo_trimmed`` while retaining the canonical task label for provenance.
    """
    requested_task = canonical_task_name(task)
    if requested_task is None or requested_task != task.lower():
        raise ValueError(f"Unsupported task {task!r}. Choose one of: {', '.join(SUPPORTED_TASKS)}")

    samples = []
    for subject_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if ANAIS_SUBJECT_PATTERN.fullmatch(subject_dir.name) is None:
            continue
        for task_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            if canonical_task_name(task_dir.name) != requested_task:
                continue
            kinetics_path = task_dir / "kinetics_feet.npy"
            joints_path = task_dir / "all_joints.npy"
            if not kinetics_path.exists() or not joints_path.exists():
                raise FileNotFoundError(
                    f"Missing kinetics_feet.npy or all_joints.npy in {task_dir}"
                )
            samples.append({
                "subject": subject_dir.name.lower(),
                "task": requested_task,
                "source_task": task_dir.name,
                "kinetics": kinetics_path,
                "joints": joints_path,
            })

    counts = Counter(sample["subject"] for sample in samples)
    duplicates = sorted(subject for subject, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(
            "Expected one selected trial per Anais subject; invalid trial count for: "
            f"{duplicates}"
        )
    if len(samples) < 5:
        raise ValueError(f"Only {len(samples)} valid trials found for task {requested_task!r}.")
    return samples


def configure_base_urdf(urdf_dir: Path) -> None:
    """Point the reused body-weight loader to the Anais scaled URDF folder."""
    base.URDF_DIR = urdf_dir
    base.URDF_MESHES_PATH = str(urdf_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="luyo", help=f"One of: {', '.join(SUPPORTED_TASKS)} (default: luyo).")
    parser.add_argument("--data-root", type=Path, default="/lustre/fsn1/projects/rech/vsi/ulm94jm/dataset_grf2kine/processed_data_feet_anais_platefixed")
    parser.add_argument("--urdf-dir", type=Path, default=base.URDF_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=base.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=base.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=base.SEED)
    parser.add_argument("--inference-steps", type=int, default=base.INFERENCE_STEPS)
    return parser.parse_args()


def run_experiment(args):
    task = canonical_task_name(args.task)
    if task is None or task != args.task.lower():
        raise ValueError(f"Unsupported task {args.task!r}. Choose one of: {', '.join(SUPPORTED_TASKS)}")
    data_root = args.data_root
    urdf_dir = args.urdf_dir
    results_dir = args.results_dir or Path(f"results_anais_{task}_feet_improved")
    if args.epochs <= base.WARMUP_EPOCHS:
        raise ValueError(f"--epochs must be greater than the {base.WARMUP_EPOCHS} warm-up epochs.")
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root.resolve()}")
    if not urdf_dir.exists():
        raise FileNotFoundError(f"URDF_DIR does not exist: {urdf_dir.resolve()}")

    base.SEED = args.seed
    configure_base_urdf(urdf_dir)
    base.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_samples = collect_task_samples(data_root, task)
    print(f"Selected Anais task: {task}")
    selected_trials = [f"{sample['subject']}/{sample['source_task']}" for sample in all_samples]
    print(f"Trials ({len(all_samples)}): {selected_trials}")
    # Sixteen subjects give the explicit 12 / 2 / 2 subject split.
    train_samples, val_samples, test_samples = base.split_dataset(
        all_samples, train_ratio=0.75, val_ratio=0.125, seed=args.seed
    )

    def pairs(samples):
        return [(sample["kinetics"], sample["joints"]) for sample in samples]

    train_pairs, val_pairs, test_pairs = map(pairs, (train_samples, val_samples, test_samples))
    stats = base.compute_and_save_stats(train_pairs, results_dir / "scalers_anais_task.json")
    train_dataset = base.BiomechFlowDataset(train_pairs, stats=stats)
    val_dataset = base.BiomechFlowDataset(val_pairs, stats=stats)
    test_dataset = base.BiomechFlowDataset(test_pairs, stats=stats)
    print(f"Windows: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = base.FlowTransformer().to(device)
    ema = base.ModelEMA(model, decay=base.EMA_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=base.WARMUP_EPOCHS),
            CosineAnnealingLR(optimizer, T_max=args.epochs - base.WARMUP_EPOCHS, eta_min=1e-6),
        ],
        milestones=[base.WARMUP_EPOCHS],
    )

    print(f"Device: {device}")
    print(f"Trainable parameters: {base.count_parameters(model):,}")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_ema_path = results_dir / "fm_anais_task_model_best_ema.pth"
    best_raw_path = results_dir / "fm_anais_task_model_best_raw.pth"

    for epoch in range(args.epochs):
        model.train()
        sum_squared_error, total_values = 0.0, 0
        for grfm, joints in train_loader:
            grfm = grfm.to(device, non_blocking=True)
            joints = joints.to(device, non_blocking=True)
            batch_size = joints.shape[0]
            x0 = torch.randn_like(joints)
            t = torch.rand((batch_size, 1, 1), device=device)
            xt = t * joints + (1.0 - t) * x0
            target_velocity = joints - x0

            optimizer.zero_grad(set_to_none=True)
            predicted_velocity = model(xt, t.view(batch_size), grfm)
            loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
            sum_squared_error += torch.sum((predicted_velocity.detach() - target_velocity) ** 2).item()
            total_values += target_velocity.numel()

        train_loss = sum_squared_error / total_values
        val_loss = base.validation_loss(ema.model, val_loader, device)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ema.model.state_dict(), best_ema_path)
            torch.save(model.state_dict(), best_raw_path)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"train={train_loss:.6f} | val_ema={val_loss:.6f}"
        )

    torch.save(model.state_dict(), results_dir / "fm_anais_task_model_final_raw.pth")
    torch.save(ema.model.state_dict(), results_dir / "fm_anais_task_model_final_ema.pth")

    # Keep the same qualitative outputs as the feet_improved experiment.
    model.load_state_dict(torch.load(best_ema_path, map_location=device))
    model.eval()
    example_grfm, example_joints = test_dataset[random.randrange(len(test_dataset))]
    sample_prediction = base.sample_heun(
        model, example_grfm.unsqueeze(0).to(device), n_steps=args.inference_steps
    ).cpu().squeeze(0)
    sample_prediction = sample_prediction * stats["j_std"] + stats["j_mean"]
    sample_reference = example_joints * stats["j_std"] + stats["j_mean"]
    base.plot_joints(sample_reference.numpy(), sample_prediction.numpy(), 0, 12, results_dir / "inference_test_joints_1_12.png")
    base.plot_joints(sample_reference.numpy(), sample_prediction.numpy(), 12, 29, results_dir / "inference_test_joints_13_29.png")

    full_reference, full_prediction = base.predict_full_trial(
        model, *random.choice(test_pairs), stats, device, n_steps=args.inference_steps
    )
    base.plot_joints(full_reference, full_prediction, 0, 12, results_dir / "full_trial_joints_1_12.png")
    base.plot_joints(full_reference, full_prediction, 12, 29, results_dir / "full_trial_joints_13_29.png")

    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation (EMA, fixed noise/time)")
    plt.xlabel("Epoch")
    plt.ylabel("Flow Matching MSE")
    plt.title(f"Loss history — Anais {task}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "loss_curve.png")
    plt.close()

    with open(results_dir / "experiment_config.json", "w") as file:
        json.dump({
            "base_training_script": "train_fm_full_step2motion_real_feet_improved.py",
            "dataset": "Anais subjects01-16",
            "task": task,
            "data_root": str(data_root),
            "urdf_dir": str(urdf_dir),
            "train_subjects": sorted({sample["subject"] for sample in train_samples}),
            "val_subjects": sorted({sample["subject"] for sample in val_samples}),
            "test_subjects": sorted({sample["subject"] for sample in test_samples}),
            "source_trials": [f"{sample['subject']}/{sample['source_task']}" for sample in all_samples],
            "epochs": args.epochs,
            "seed": args.seed,
        }, file, indent=2)
    print(f"\n[FINISH] Anais {task} feet_improved experiment saved in {results_dir}")


if __name__ == "__main__":
    run_experiment(parse_args())
