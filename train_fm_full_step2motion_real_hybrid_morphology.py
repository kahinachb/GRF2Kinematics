"""Hybrid global- and foot-frame Flow Matching with static morphology.

This is an ablation of the local-foot morphology model.  The target, subject
split, optimiser, EMA, and sampler stay unchanged.  The sole modelling change
is that the Transformer receives both representations of the same bilateral
wrench:

    [kinetics_glob (18), kinetics_feet (18)] + morphology (12 static values)

The global representation preserves absolute lateral directions, while the
foot representation retains local foot/ankle information.  No weighted loss is
used here, so results can be attributed to the hybrid representation itself.
"""

import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

import train_fm_full_step2motion_real_feet_improved as base
import train_fm_full_step2motion_real_feet_morphology as morphology
from utils.utils import is_squat_task_only


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
DATA_ROOT = morphology.DATA_ROOT
URDF_DIR = morphology.URDF_DIR
RESULTS_DIR = Path("results_real_hybrid_morphology")

WINDOW_SIZE = base.WINDOW_SIZE
STRIDE = base.STRIDE
BATCH_SIZE = base.BATCH_SIZE
NUM_WORKERS = base.NUM_WORKERS
EPOCHS = base.EPOCHS
LEARNING_RATE = base.LEARNING_RATE
WEIGHT_DECAY = base.WEIGHT_DECAY
WARMUP_EPOCHS = base.WARMUP_EPOCHS
EMA_DECAY = base.EMA_DECAY
INFERENCE_STEPS = base.INFERENCE_STEPS
SEED = base.SEED

N_SINGLE_FRAME_FEATURES = base.N_GRFM_FEATURES       # 18
N_CONDITION_FEATURES = 2 * N_SINGLE_FRAME_FEATURES   # 36 = global + feet
N_JOINTS = base.N_JOINTS
N_MORPHOLOGY_FEATURES = morphology.N_MORPHOLOGY_FEATURES
MORPHOLOGY_FEATURE_NAMES = morphology.MORPHOLOGY_FEATURE_NAMES


def load_hybrid_trial(global_path: Path, feet_path: Path, joints_path: Path,
                      body_weight_cache: dict[str, float]):
    """Load aligned global/local GRFM views and one common joint target."""
    global_grfm, global_joints = base.load_trial(global_path, joints_path, body_weight_cache)
    feet_grfm, feet_joints = base.load_trial(feet_path, joints_path, body_weight_cache)
    if global_grfm.shape != feet_grfm.shape:
        raise ValueError(
            f"Global/local GRFM shapes differ in {global_path.parent}: "
            f"{global_grfm.shape} vs {feet_grfm.shape}."
        )
    if not np.array_equal(global_joints, feet_joints):
        raise ValueError(f"Joint targets do not align between global and feet files in {global_path.parent}.")
    condition = np.concatenate((global_grfm, feet_grfm), axis=1).astype(np.float32, copy=False)
    if condition.shape[1] != N_CONDITION_FEATURES:
        raise RuntimeError(f"Expected {N_CONDITION_FEATURES} hybrid features, got {condition.shape}.")
    return condition, global_joints


class HybridMorphologyDataset(Dataset):
    """Windowed hybrid GRFM/joint samples with static morphology per window."""
    def __init__(self, samples, stats, window_size=WINDOW_SIZE):
        self.samples = []
        body_weight_cache = {}
        for sample in samples:
            condition, joints = load_hybrid_trial(
                sample["global"], sample["feet"], sample["joints"], body_weight_cache
            )
            morphology_values = np.asarray(sample["morphology"], dtype=np.float32)
            if morphology_values.shape != (N_MORPHOLOGY_FEATURES,):
                raise ValueError(
                    f"Expected {N_MORPHOLOGY_FEATURES} morphology values for "
                    f"{sample['subject']}, got {morphology_values.shape}."
                )
            for start in range(0, len(condition) - window_size + 1, window_size // 2):
                self.samples.append((
                    condition[start:start + window_size],
                    joints[start:start + window_size],
                    morphology_values,
                ))
        if not self.samples:
            raise ValueError("The dataset has no complete windows.")
        self.stats = stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        condition, joints, morphology_values = self.samples[index]
        condition = torch.from_numpy(condition)
        joints = torch.from_numpy(joints)
        morphology_values = torch.from_numpy(morphology_values)
        condition = (condition - self.stats["condition_mean"]) / (self.stats["condition_std"] + 1e-6)
        joints = (joints - self.stats["j_mean"]) / (self.stats["j_std"] + 1e-6)
        morphology_values = (morphology_values - self.stats["morph_mean"]) / (self.stats["morph_std"] + 1e-6)
        return condition, joints, morphology_values


def compute_and_save_stats(train_samples, output_path: Path):
    """Fit all scalers on training data only; static scalers weight subjects equally."""
    print("\n[INFO] Calcul des scalers hybrides sur le train set uniquement...")
    body_weight_cache = {}
    all_conditions, all_joints = [], []
    morphologies_by_subject = {}
    for sample in train_samples:
        condition, joints = load_hybrid_trial(
            sample["global"], sample["feet"], sample["joints"], body_weight_cache
        )
        all_conditions.append(condition)
        all_joints.append(joints)
        morphology_values = np.asarray(sample["morphology"], dtype=np.float32)
        existing = morphologies_by_subject.setdefault(sample["subject"], morphology_values)
        if not np.allclose(existing, morphology_values):
            raise ValueError(f"Morphology changed between trials for {sample['subject']}.")

    condition_cat = np.vstack(all_conditions)
    joints_cat = np.vstack(all_joints)
    morphology_cat = np.vstack(list(morphologies_by_subject.values()))
    morphology_std = morphology_cat.std(axis=0)
    if np.any(morphology_std < 1e-6):
        constant = [
            MORPHOLOGY_FEATURE_NAMES[index]
            for index, value in enumerate(morphology_std) if value < 1e-6
        ]
        raise ValueError(f"Constant morphology feature(s) in train subjects: {constant}")

    stats = {
        "condition_mean": condition_cat.mean(axis=0),
        "condition_std": condition_cat.std(axis=0),
        "j_mean": joints_cat.mean(axis=0),
        "j_std": joints_cat.std(axis=0),
        "morph_mean": morphology_cat.mean(axis=0),
        "morph_std": morphology_std,
    }
    payload = {
        **{key: value.tolist() for key, value in stats.items()},
        "condition_layout": "[global: RF,R M,R CoP,L F,L M,L CoP][feet: same order]",
        "morphology_feature_names": list(MORPHOLOGY_FEATURE_NAMES),
        "morphology_units": "metres_before_standardisation",
    }
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in stats.items()}


class HybridMorphologyFlowTransformer(morphology.MorphologyFlowTransformer):
    """Morphology model with distinct global and local GRFM memory tokens."""
    def __init__(self, embed_dim=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__(
            morphology_dim=N_MORPHOLOGY_FEATURES,
            embed_dim=embed_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        # The inherited six embeddings remain dedicated to local-foot inputs.
        self.embed_global_F_R = nn.Linear(3, embed_dim)
        self.embed_global_M_R = nn.Linear(3, embed_dim)
        self.embed_global_CoP_R = nn.Linear(3, embed_dim)
        self.embed_global_F_L = nn.Linear(3, embed_dim)
        self.embed_global_M_L = nn.Linear(3, embed_dim)
        self.embed_global_CoP_L = nn.Linear(3, embed_dim)
        self.global_condition_segment = nn.Parameter(torch.randn(6, embed_dim) * 0.02)

    def forward(self, x, t, condition, morphology_values):
        batch_size, window_size, _ = x.shape
        if condition.shape[:2] != (batch_size, window_size) or condition.shape[2] != N_CONDITION_FEATURES:
            raise ValueError(
                f"Expected condition shape ({batch_size}, {window_size}, {N_CONDITION_FEATURES}), "
                f"got {tuple(condition.shape)}."
            )
        if morphology_values.dim() == 1:
            morphology_values = morphology_values.unsqueeze(0).expand(batch_size, -1)
        if morphology_values.shape != (batch_size, N_MORPHOLOGY_FEATURES):
            raise ValueError(
                f"Expected morphology shape ({batch_size}, {N_MORPHOLOGY_FEATURES}), "
                f"got {tuple(morphology_values.shape)}."
            )
        morphology_embedding = self.morphology_encoder(
            morphology_values.to(dtype=x.dtype)
        ).unsqueeze(1)

        if isinstance(t, (int, float)):
            t = torch.full((batch_size,), float(t), device=x.device)
        elif t.dim() == 0:
            t = t.float().unsqueeze(0).expand(batch_size)
        time_embedding = self.time_mlp(t).unsqueeze(1)

        target_blocks = [
            self.embed_Rleg(x[:, :, 0:6]),
            self.embed_Lleg(x[:, :, 6:12]),
            self.embed_Upper(x[:, :, 12:29]),
        ]
        target_blocks = [
            self.pos_encoder(block) + self.target_segment[index]
            for index, block in enumerate(target_blocks)
        ]
        target = torch.cat(target_blocks, dim=1) + time_embedding

        global_condition = condition[:, :, :N_SINGLE_FRAME_FEATURES]
        feet_condition = condition[:, :, N_SINGLE_FRAME_FEATURES:]
        global_blocks = [
            self.embed_global_F_R(global_condition[:, :, 0:3]),
            self.embed_global_M_R(global_condition[:, :, 3:6]),
            self.embed_global_CoP_R(global_condition[:, :, 6:9]),
            self.embed_global_F_L(global_condition[:, :, 9:12]),
            self.embed_global_M_L(global_condition[:, :, 12:15]),
            self.embed_global_CoP_L(global_condition[:, :, 15:18]),
        ]
        local_blocks = [
            self.embed_F_R(feet_condition[:, :, 0:3]),
            self.embed_M_R(feet_condition[:, :, 3:6]),
            self.embed_CoP_R(feet_condition[:, :, 6:9]),
            self.embed_F_L(feet_condition[:, :, 9:12]),
            self.embed_M_L(feet_condition[:, :, 12:15]),
            self.embed_CoP_L(feet_condition[:, :, 15:18]),
        ]
        global_blocks = [
            self.pos_encoder(block) + self.global_condition_segment[index]
            for index, block in enumerate(global_blocks)
        ]
        local_blocks = [
            self.pos_encoder(block) + self.condition_segment[index]
            for index, block in enumerate(local_blocks)
        ]
        memory = torch.cat(global_blocks + local_blocks, dim=1)
        memory = memory + time_embedding + morphology_embedding

        output = self.transformer(tgt=target, memory=memory)
        right_leg = self.out_Rleg(output[:, 0:window_size])
        left_leg = self.out_Lleg(output[:, window_size:2 * window_size])
        upper_body = self.out_Upper(output[:, 2 * window_size:3 * window_size])
        return torch.cat((right_leg, left_leg, upper_body), dim=2)


@torch.no_grad()
def sample_heun_hybrid(model, condition, morphology_values, n_steps=INFERENCE_STEPS):
    """Second-order sampler for hybrid-conditioned windows."""
    batch_size, window_size, _ = condition.shape
    x = torch.randn((batch_size, window_size, N_JOINTS), device=condition.device)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t0 = step / n_steps
        t1 = (step + 1) / n_steps
        t0_tensor = torch.full((batch_size,), t0, device=condition.device)
        t1_tensor = torch.full((batch_size,), t1, device=condition.device)
        velocity0 = model(x, t0_tensor, condition, morphology_values)
        x_predictor = x + dt * velocity0
        velocity1 = model(x_predictor, t1_tensor, condition, morphology_values)
        x = x + 0.5 * dt * (velocity0 + velocity1)
    return x


@torch.no_grad()
def predict_full_trial_hybrid(model, global_path, feet_path, joints_path, morphology_values,
                              stats, device, window_size=WINDOW_SIZE, stride=STRIDE,
                              n_steps=INFERENCE_STEPS):
    """Full-trial hybrid Heun inference with exact overlap inpainting."""
    model.eval()
    body_weight_cache = {}
    condition, reference = load_hybrid_trial(global_path, feet_path, joints_path, body_weight_cache)
    condition = (torch.from_numpy(condition) - stats["condition_mean"]) / (stats["condition_std"] + 1e-6)
    morphology_values = torch.as_tensor(morphology_values, dtype=torch.float32)
    morphology_values = (morphology_values - stats["morph_mean"]) / (stats["morph_std"] + 1e-6)
    total_frames = len(condition)

    if total_frames < window_size:
        padding = condition[-1:].repeat(window_size - total_frames, 1)
        condition_for_sampling = torch.cat((condition, padding), dim=0)
    else:
        condition_for_sampling = condition

    full_prediction = torch.zeros((len(condition_for_sampling), N_JOINTS), device=device)
    previous_start = None
    print(f"  [INF] Sampling hybrid Flow Matching (Heun): {total_frames} frames, {n_steps} steps/window")
    for start in base.window_starts(len(condition_for_sampling), window_size, stride):
        end = start + window_size
        condition_window = condition_for_sampling[start:end].unsqueeze(0).to(device)
        morphology_window = morphology_values.unsqueeze(0).to(device)
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
            t1_tensor = torch.tensor([t1], device=device)
            velocity0 = model(x, t0_tensor, condition_window, morphology_window)
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


def validation_loss(model, data_loader, device):
    """Deterministic EMA validation loss for hybrid inputs."""
    model.eval()
    total_squared_error = 0.0
    total_values = 0
    generator_device = device.type
    with torch.no_grad():
        for batch_index, (condition, joints, morphology_values) in enumerate(data_loader):
            condition = condition.to(device, non_blocking=True)
            joints = joints.to(device, non_blocking=True)
            morphology_values = morphology_values.to(device, non_blocking=True)
            batch_size = joints.shape[0]
            generator = torch.Generator(device=generator_device).manual_seed(SEED + batch_index)
            x0 = torch.randn(joints.shape, generator=generator, device=device)
            t = torch.rand((batch_size, 1, 1), generator=generator, device=device)
            xt = t * joints + (1.0 - t) * x0
            target_velocity = joints - x0
            prediction = model(xt, t.view(batch_size), condition, morphology_values)
            total_squared_error += torch.sum((prediction - target_velocity) ** 2).item()
            total_values += target_velocity.numel()
    return total_squared_error / total_values


def run_experiment():
    base.set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not Path(DATA_ROOT).exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {Path(DATA_ROOT).resolve()}")
    if not Path(URDF_DIR).exists():
        raise FileNotFoundError(f"URDF_DIR does not exist: {Path(URDF_DIR).resolve()}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = []
    morphology_cache = {}
    for subject_dir in sorted(path for path in Path(DATA_ROOT).iterdir() if path.is_dir()):
        subject_name = subject_dir.name
        if subject_name.lower() not in morphology.VINC_SUBJECTS:
            continue
        for task_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            task_name = task_dir.name
            if not is_squat_task_only(subject_name, task_name):
                continue
            global_path = task_dir / "kinetics_glob.npy"
            feet_path = task_dir / "kinetics_feet.npy"
            joints_path = task_dir / "all_joints.npy"
            if global_path.exists() and feet_path.exists() and joints_path.exists():
                all_samples.append({
                    "subject": subject_name,
                    "task": task_name,
                    "global": global_path,
                    "feet": feet_path,
                    "joints": joints_path,
                    "morphology": morphology.subject_morphology(subject_name, morphology_cache),
                })

    print(f"Nombre total d'essais squat : {len(all_samples)}")
    train_samples, val_samples, test_samples = base.split_dataset(all_samples, seed=SEED)
    morphology.save_morphology_table(all_samples, RESULTS_DIR / "morphology_features_metres.json")
    stats = compute_and_save_stats(train_samples, RESULTS_DIR / "scalers_hybrid_morphology.json")

    train_dataset = HybridMorphologyDataset(train_samples, stats=stats)
    val_dataset = HybridMorphologyDataset(val_samples, stats=stats)
    test_dataset = HybridMorphologyDataset(test_samples, stats=stats)
    print(f"Windows: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print("Condition: 18 global GRFM + 18 local-foot GRFM + static morphology")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    model = HybridMorphologyFlowTransformer().to(device)
    ema = base.ModelEMA(model, decay=EMA_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS),
            CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6),
        ],
        milestones=[WARMUP_EPOCHS],
    )

    print(f"Device: {device}")
    print(f"Trainable parameters: {base.count_parameters(model):,}")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        sum_squared_error = 0.0
        total_values = 0
        for condition, joints, morphology_values in train_loader:
            condition = condition.to(device, non_blocking=True)
            joints = joints.to(device, non_blocking=True)
            morphology_values = morphology_values.to(device, non_blocking=True)
            batch_size = joints.shape[0]
            x0 = torch.randn_like(joints)
            t = torch.rand((batch_size, 1, 1), device=device)
            xt = t * joints + (1.0 - t) * x0
            target_velocity = joints - x0

            optimizer.zero_grad(set_to_none=True)
            predicted_velocity = model(xt, t.view(batch_size), condition, morphology_values)
            loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            sum_squared_error += torch.sum((predicted_velocity.detach() - target_velocity) ** 2).item()
            total_values += target_velocity.numel()

        train_loss = sum_squared_error / total_values
        val_loss = validation_loss(ema.model, val_loader, device)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ema.model.state_dict(), RESULTS_DIR / "fm_hybrid_morphology_model_best_ema.pth")
            torch.save(model.state_dict(), RESULTS_DIR / "fm_hybrid_morphology_model_best_raw.pth")

        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | lr={learning_rate:.2e} | "
            f"train={train_loss:.6f} | val_ema={val_loss:.6f}"
        )

    torch.save(model.state_dict(), RESULTS_DIR / "fm_hybrid_morphology_model_final_raw.pth")
    torch.save(ema.model.state_dict(), RESULTS_DIR / "fm_hybrid_morphology_model_final_ema.pth")

    model.load_state_dict(torch.load(RESULTS_DIR / "fm_hybrid_morphology_model_best_ema.pth", map_location=device))
    model.eval()
    example_condition, example_joints, example_morphology = test_dataset[random.randrange(len(test_dataset))]
    sample_prediction = sample_heun_hybrid(
        model,
        example_condition.unsqueeze(0).to(device),
        example_morphology.unsqueeze(0).to(device),
    ).cpu().squeeze(0)
    sample_prediction = sample_prediction * stats["j_std"] + stats["j_mean"]
    sample_reference = example_joints * stats["j_std"] + stats["j_mean"]
    base.plot_joints(sample_reference.numpy(), sample_prediction.numpy(), 0, 12, RESULTS_DIR / "inference_test_joints_1_12.png")
    base.plot_joints(sample_reference.numpy(), sample_prediction.numpy(), 12, 29, RESULTS_DIR / "inference_test_joints_13_29.png")

    test_sample = random.choice(test_samples)
    full_reference, full_prediction = predict_full_trial_hybrid(
        model,
        test_sample["global"],
        test_sample["feet"],
        test_sample["joints"],
        test_sample["morphology"],
        stats,
        device,
    )
    base.plot_joints(full_reference, full_prediction, 0, 12, RESULTS_DIR / "full_trial_joints_1_12.png")
    base.plot_joints(full_reference, full_prediction, 12, 29, RESULTS_DIR / "full_trial_joints_13_29.png")

    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation (EMA, fixed noise/time)")
    plt.xlabel("Epoch")
    plt.ylabel("Flow Matching MSE")
    plt.title("Loss history — hybrid global + feet + morphology")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png")
    plt.close()

    with open(RESULTS_DIR / "experiment_config.json", "w") as file:
        json.dump({
            "base_training_script": "train_fm_full_step2motion_real_feet_morphology.py",
            "data_root": str(DATA_ROOT),
            "urdf_dir": str(URDF_DIR),
            "condition_layout": "global_18_then_feet_18",
            "morphology_feature_names": list(MORPHOLOGY_FEATURE_NAMES),
            "weighted_loss": False,
            "epochs": EPOCHS,
            "seed": SEED,
        }, file, indent=2)
    print(f"\n[FINISH] Hybrid morphology experiment saved in {RESULTS_DIR}")


if __name__ == "__main__":
    run_experiment()
