"""Flow Matching: real squat data expressed in local foot frames.

This script is intentionally independent from the existing training scripts.
It trains only the seven Vinc squat trials, keeps body-weight normalization,
and adds: block/segment embeddings, deterministic EMA validation, a warm-up +
cosine learning-rate schedule, and Heun sampling.
"""

import copy
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

from utils.model_utils import build_human_model
# from utils.utils import is_squat_task_only


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/lustre/fsn1/projects/rech/vsi/ulm94jm/dataset_grf2kine/processed_data_feet_Vinc")
RESULTS_DIR = Path("results_realV_feet_improved")
URDF_DIR = Path("DATA/10_urdf")
URDF_MESHES_PATH = "DATA/10_urdf"

WINDOW_SIZE = 128
STRIDE = 64
BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 250
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 10
EMA_DECAY = 0.999
INFERENCE_STEPS = 20  # Heun uses two network evaluations per step.
SEED = 42

GRAVITY_M_S2 = 9.81
N_GRFM_FEATURES = 18
N_JOINTS = 29


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def split_list(items, train_ratio=0.7, val_ratio=0.15):
    n_items = len(items)
    n_train = int(n_items * train_ratio)
    n_val = int(n_items * val_ratio)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def split_dataset(all_samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split by subject when several subjects are available."""
    unique_subjects = sorted({sample["subject"] for sample in all_samples})
    if not unique_subjects:
        raise ValueError("No samples were found.")

    rng = random.Random(seed)
    if len(unique_subjects) > 1:
        rng.shuffle(unique_subjects)
        train_subjects, val_subjects, test_subjects = split_list(
            unique_subjects, train_ratio, val_ratio
        )
        train_samples = [sample for sample in all_samples if sample["subject"] in train_subjects]
        val_samples = [sample for sample in all_samples if sample["subject"] in val_subjects]
        test_samples = [sample for sample in all_samples if sample["subject"] in test_subjects]
        print("\n[SPLIT PAR SUJET]")
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Val subjects   ({len(val_subjects)}): {val_subjects}")
        print(f"Test subjects  ({len(test_subjects)}): {test_subjects}")
    else:
        unique_tasks = sorted({sample["task"] for sample in all_samples})
        rng.shuffle(unique_tasks)
        train_tasks, val_tasks, test_tasks = split_list(unique_tasks, train_ratio, val_ratio)
        train_samples = [sample for sample in all_samples if sample["task"] in train_tasks]
        val_samples = [sample for sample in all_samples if sample["task"] in val_tasks]
        test_samples = [sample for sample in all_samples if sample["task"] in test_tasks]

    print("=" * 40)
    print(f"Train trials : {len(train_samples)}")
    print(f"Val trials   : {len(val_samples)}")
    print(f"Test trials  : {len(test_samples)}")
    print("=" * 40)
    if not train_samples or not val_samples or not test_samples:
        raise ValueError("One split is empty; adjust the ratios or add more trials.")
    return train_samples, val_samples, test_samples


def body_weight_newtons(kinetics_path: Path, cache: dict[str, float]) -> float:
    """Read the subject mass from its scaled URDF and cache the body weight."""
    subject = Path(kinetics_path).parent.parent.name
    if subject not in cache:
        urdf_path = URDF_DIR / f"{subject}_scaled.urdf"
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found for {subject}: {urdf_path}")
        model_h = build_human_model(str(urdf_path), URDF_MESHES_PATH)[0]
        mass_kg = float(pin.computeTotalMass(model_h))
        if mass_kg <= 0:
            raise ValueError(f"Invalid URDF mass for {subject}: {mass_kg} kg")
        cache[subject] = mass_kg * GRAVITY_M_S2
        print(f"[BW] {subject}: {mass_kg:.2f} kg ({cache[subject]:.2f} N)")
    return cache[subject]


def normalize_grfm_by_body_weight(grfm: np.ndarray, body_weight_n: float) -> np.ndarray:
    """Normalize force and moment components; leave the three CoP components in metres.

    Each foot block follows [F(3), M(3), CoP(3)].  CoPz is intentionally kept:
    it is informative in a local foot frame.
    """
    normalized = np.asarray(grfm, dtype=np.float32).copy()
    for offset in (0, 9):
        normalized[:, offset:offset + 6] /= body_weight_n
    return normalized


def load_trial(kinetics_path: Path, joints_path: Path, body_weight_cache: dict[str, float]):
    grfm = np.load(kinetics_path).astype(np.float32)
    joints = np.load(joints_path).astype(np.float32)
    if grfm.ndim != 2 or grfm.shape[1] != N_GRFM_FEATURES:
        raise ValueError(f"Expected GRFM shape (T, {N_GRFM_FEATURES}), got {grfm.shape}: {kinetics_path}")
    if joints.ndim != 2 or joints.shape[1] < 6 + N_JOINTS:
        raise ValueError(f"Expected at least {6 + N_JOINTS} joint columns, got {joints.shape}: {joints_path}")
    if len(grfm) != len(joints):
        raise ValueError(f"GRFM/joint frame mismatch: {len(grfm)} vs {len(joints)} in {kinetics_path.parent}")
    grfm = normalize_grfm_by_body_weight(grfm, body_weight_newtons(kinetics_path, body_weight_cache))
    return grfm, joints[:, 6:6 + N_JOINTS]


class BiomechFlowDataset(Dataset):
    def __init__(self, file_pairs, window_size=WINDOW_SIZE, stats=None):
        self.samples = []
        body_weight_cache = {}
        for kinetics_path, joints_path in file_pairs:
            grfm, joints = load_trial(kinetics_path, joints_path, body_weight_cache)
            for start in range(0, len(grfm) - window_size + 1, window_size // 2):
                self.samples.append((grfm[start:start + window_size], joints[start:start + window_size]))
        self.stats = stats
        if not self.samples:
            raise ValueError("The dataset has no complete windows.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        grfm, joints = self.samples[index]
        grfm = torch.from_numpy(grfm)
        joints = torch.from_numpy(joints)
        if self.stats is not None:
            grfm = (grfm - self.stats["f_mean"]) / (self.stats["f_std"] + 1e-6)
            joints = (joints - self.stats["j_mean"]) / (self.stats["j_std"] + 1e-6)
        return grfm, joints


def compute_and_save_stats(file_pairs, save_path: Path):
    print("\n[INFO] Calcul des scalers sur le train set...")
    body_weight_cache = {}
    all_grfm, all_joints = [], []
    for kinetics_path, joints_path in file_pairs:
        grfm, joints = load_trial(kinetics_path, joints_path, body_weight_cache)
        all_grfm.append(grfm)
        all_joints.append(joints)
    grfm_cat = np.vstack(all_grfm)
    joints_cat = np.vstack(all_joints)
    stats = {
        "f_mean": grfm_cat.mean(axis=0),
        "f_std": grfm_cat.std(axis=0),
        "j_mean": joints_cat.mean(axis=0),
        "j_std": joints_cat.std(axis=0),
    }
    with open(save_path, "w") as file:
        json.dump({key: value.tolist() for key, value in stats.items()}, file)
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in stats.items()}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * scale)
        pe[0, :, 1::2] = torch.cos(position * scale)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SinusoidalTimeEmbeddings(nn.Module):
    """Continuous t in [0, 1], internally scaled once to the 0--1000 range."""
    def __init__(self, dim, time_scale=1000.0):
        super().__init__()
        self.dim = dim
        self.time_scale = time_scale

    def forward(self, time):
        time = time.float() * self.time_scale
        half_dim = self.dim // 2
        scale = math.log(10000.0) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim, device=time.device, dtype=torch.float32) * -scale)
        embedding = time[:, None] * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class FlowTransformer(nn.Module):
    """Conditioned Flow Matching transformer with explicit body/sensor blocks."""
    def __init__(self, embed_dim=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embed_Rleg = nn.Linear(6, embed_dim)
        self.embed_Lleg = nn.Linear(6, embed_dim)
        self.embed_Upper = nn.Linear(17, embed_dim)
        self.embed_F_R = nn.Linear(3, embed_dim)
        self.embed_M_R = nn.Linear(3, embed_dim)
        self.embed_CoP_R = nn.Linear(3, embed_dim)
        self.embed_F_L = nn.Linear(3, embed_dim)
        self.embed_M_L = nn.Linear(3, embed_dim)
        self.embed_CoP_L = nn.Linear(3, embed_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.target_segment = nn.Parameter(torch.randn(3, embed_dim) * 0.02)
        self.condition_segment = nn.Parameter(torch.randn(6, embed_dim) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=256,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_Rleg = nn.Linear(embed_dim, 6)
        self.out_Lleg = nn.Linear(embed_dim, 6)
        self.out_Upper = nn.Linear(embed_dim, 17)

    def forward(self, x, t, condition):
        batch_size, window_size, _ = x.shape
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

        condition_blocks = [
            self.embed_F_R(condition[:, :, 0:3]),
            self.embed_M_R(condition[:, :, 3:6]),
            self.embed_CoP_R(condition[:, :, 6:9]),
            self.embed_F_L(condition[:, :, 9:12]),
            self.embed_M_L(condition[:, :, 12:15]),
            self.embed_CoP_L(condition[:, :, 15:18]),
        ]
        condition_blocks = [
            self.pos_encoder(block) + self.condition_segment[index]
            for index, block in enumerate(condition_blocks)
        ]
        memory = torch.cat(condition_blocks, dim=1) + time_embedding

        output = self.transformer(tgt=target, memory=memory)
        right_leg = self.out_Rleg(output[:, 0:window_size])
        left_leg = self.out_Lleg(output[:, window_size:2 * window_size])
        upper_body = self.out_Upper(output[:, 2 * window_size:3 * window_size])
        return torch.cat((right_leg, left_leg, upper_body), dim=2)


class ModelEMA:
    """A separate EMA model; raw and averaged weights remain available together."""
    def __init__(self, model: nn.Module, decay=EMA_DECAY):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_parameter, parameter in zip(self.model.parameters(), model.parameters()):
            ema_parameter.mul_(self.decay).add_(parameter, alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


@torch.no_grad()
def sample_heun(model, condition, n_steps=INFERENCE_STEPS):
    """Second-order integration from Gaussian noise at t=0 to data at t=1."""
    batch_size, window_size, _ = condition.shape
    x = torch.randn((batch_size, window_size, N_JOINTS), device=condition.device)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t0 = step / n_steps
        t1 = (step + 1) / n_steps
        t0_tensor = torch.full((batch_size,), t0, device=condition.device)
        t1_tensor = torch.full((batch_size,), t1, device=condition.device)
        v0 = model(x, t0_tensor, condition)
        x_predictor = x + dt * v0
        v1 = model(x_predictor, t1_tensor, condition)
        x = x + 0.5 * dt * (v0 + v1)
    return x


def window_starts(length: int, window_size: int, stride: int):
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    final_start = length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.no_grad()
def predict_full_trial(model, kinetics_path, joints_path, stats, device,
                       window_size=WINDOW_SIZE, stride=STRIDE, n_steps=INFERENCE_STEPS):
    """Generate a complete trial with exact overlap inpainting and tail coverage."""
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
    print(f"  [INF] Sampling Flow Matching (Heun): {total_frames} frames, {n_steps} steps/window")

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
            t1_tensor = torch.tensor([t1], device=device)
            v0 = model(x, t0_tensor, condition_window)
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


def plot_joints(reference, prediction, start, end, filename):
    n_joints = end - start
    ncols = 3
    nrows = math.ceil(n_joints / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()
    for index, axis in enumerate(axes):
        joint_index = start + index
        if joint_index >= end:
            axis.axis("off")
            continue
        axis.plot(reference[:, joint_index], "k--", label="Reference")
        axis.plot(prediction[:, joint_index], "r", label="Prediction")
        axis.set_title(f"Joint {joint_index + 1}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def validation_loss(model, data_loader, device):
    """A deterministic Flow Matching validation estimate for model selection."""
    model.eval()
    total_squared_error = 0.0
    total_values = 0
    generator_device = device.type
    with torch.no_grad():
        for batch_index, (grfm, joints) in enumerate(data_loader):
            grfm = grfm.to(device, non_blocking=True)
            joints = joints.to(device, non_blocking=True)
            batch_size = joints.shape[0]
            generator = torch.Generator(device=generator_device).manual_seed(SEED + batch_index)
            x0 = torch.randn(joints.shape, generator=generator, device=device)
            t = torch.rand((batch_size, 1, 1), generator=generator, device=device)
            xt = t * joints + (1.0 - t) * x0
            target_velocity = joints - x0
            prediction = model(xt, t.view(batch_size), grfm)
            total_squared_error += torch.sum((prediction - target_velocity) ** 2).item()
            total_values += target_velocity.numel()
    return total_squared_error / total_values


def run_experiment():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT.resolve()}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for subject_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
        subject_name = subject_dir.name.lower()
        for task_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            task_name = task_dir.name.lower()
            # if not is_squat_task_only(subject_name, task_name):
            #     continue
            kinetics_path = task_dir / "kinetics_feet.npy"
            joints_path = task_dir / "all_joints.npy"
            if kinetics_path.exists() and joints_path.exists():
                all_samples.append({
                    "subject": subject_name,
                    "task": task_name,
                    "kinetics": kinetics_path,
                    "joints": joints_path,
                })

    print(f"Nombre total d'essais squat : {len(all_samples)}")
    train_samples, val_samples, test_samples = split_dataset(all_samples, seed=SEED)

    def pairs(samples):
        return [(sample["kinetics"], sample["joints"]) for sample in samples]

    train_pairs = pairs(train_samples)
    val_pairs = pairs(val_samples)
    test_pairs = pairs(test_samples)
    stats = compute_and_save_stats(train_pairs, RESULTS_DIR / "scalers_concat.json")

    train_dataset = BiomechFlowDataset(train_pairs, stats=stats)
    val_dataset = BiomechFlowDataset(val_pairs, stats=stats)
    test_dataset = BiomechFlowDataset(test_pairs, stats=stats)
    print(f"Windows: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    model = FlowTransformer().to(device)
    ema = ModelEMA(model)
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
    print(f"Trainable parameters: {count_parameters(model):,}")
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        sum_squared_error = 0.0
        total_values = 0
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
        val_loss = validation_loss(ema.model, val_loader, device)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ema.model.state_dict(), RESULTS_DIR / "fm_biomech_model_best_ema.pth")
            torch.save(model.state_dict(), RESULTS_DIR / "fm_biomech_model_best_raw.pth")

        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | lr={learning_rate:.2e} | "
            f"train={train_loss:.6f} | val_ema={val_loss:.6f}"
        )

    torch.save(model.state_dict(), RESULTS_DIR / "fm_biomech_model_final_raw.pth")
    torch.save(ema.model.state_dict(), RESULTS_DIR / "fm_biomech_model_final_ema.pth")

    model.load_state_dict(torch.load(RESULTS_DIR / "fm_biomech_model_best_ema.pth", map_location=device))
    model.eval()
    example_grfm, example_joints = test_dataset[random.randrange(len(test_dataset))]
    sample_prediction = sample_heun(model, example_grfm.unsqueeze(0).to(device)).cpu().squeeze(0)
    sample_prediction = sample_prediction * stats["j_std"] + stats["j_mean"]
    sample_reference = example_joints * stats["j_std"] + stats["j_mean"]
    plot_joints(sample_reference.numpy(), sample_prediction.numpy(), 0, 12, RESULTS_DIR / "inference_test_joints_1_12.png")
    plot_joints(sample_reference.numpy(), sample_prediction.numpy(), 12, 29, RESULTS_DIR / "inference_test_joints_13_29.png")

    full_reference, full_prediction = predict_full_trial(
        model, *random.choice(test_pairs), stats, device
    )
    plot_joints(full_reference, full_prediction, 0, 12, RESULTS_DIR / "full_trial_joints_1_12.png")
    plot_joints(full_reference, full_prediction, 12, 29, RESULTS_DIR / "full_trial_joints_13_29.png")

    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation (EMA, fixed noise/time)")
    plt.xlabel("Epoch")
    plt.ylabel("Flow Matching MSE")
    plt.title("Loss history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png")
    plt.close()
    print(f"\n[FINISH] Results saved in {RESULTS_DIR}")


if __name__ == "__main__":
    run_experiment()
