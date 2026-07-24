"""Flow Matching on real squat data in local foot frames, with morphology.

This is a separate experiment from
``train_fm_full_step2motion_real_feet_improved.py``.  It keeps that script's
data split, GRFM preprocessing, EMA, scheduler, and base architecture, then
adds a small static conditioning branch fed by segment lengths extracted from
each subject's scaled URDF.

The static morphology vector is added to the *condition memory* of the
Transformer, rather than being appended to the 18 time-varying GRFM channels.
This keeps the physical meanings of the GRFM channels unchanged.
"""

import json
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

import train_fm_full_step2motion_real_feet_improved as base
# from utils.utils import is_squat_task_only, manual_mapping


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
# morphology conditioning.
DATA_ROOT = base.DATA_ROOT
URDF_DIR = base.URDF_DIR
RESULTS_DIR = Path("results_realV_feet_morphology")

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
N_GRFM_FEATURES = base.N_GRFM_FEATURES
N_JOINTS = base.N_JOINTS

# All values are lengths in metres before standardisation with train subjects.
MORPHOLOGY_FEATURE_NAMES = (
    "pelvis_width_m",
    "right_thigh_length_m",
    "right_shank_length_m",
    "right_foot_length_m",
    "left_thigh_length_m",
    "left_shank_length_m",
    "left_foot_length_m",
    "trunk_length_m",
    "right_upperarm_length_m",
    "right_forearm_length_m",
    "left_upperarm_length_m",
    "left_forearm_length_m",
)
N_MORPHOLOGY_FEATURES = len(MORPHOLOGY_FEATURE_NAMES)
# The experiment is intentionally restricted to the seven Vinc subjects.
VINC_SUBJECTS = frozenset(manual_mapping)


def _joint_origins(urdf_path: Path) -> dict[str, np.ndarray]:
    """Return URDF joint-origin translations indexed by joint name."""
    root = ET.parse(urdf_path).getroot()
    origins = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        origin = joint.find("origin")
        if name is None or origin is None or origin.get("xyz") is None:
            continue
        values = np.fromstring(origin.get("xyz"), sep=" ", dtype=np.float32)
        if values.shape != (3,):
            raise ValueError(f"Invalid xyz origin for joint {name!r} in {urdf_path}")
        origins[name] = values
    return origins


def _find_scaled_urdf(subject_name: str) -> Path:
    """Resolve a scaled URDF without relying on subject-name capitalisation."""
    subject_key = subject_name.lower()
    matches = [
        path for path in Path(URDF_DIR).glob("*_scaled.urdf")
        if path.stem.removesuffix("_scaled").lower() == subject_key
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one scaled URDF for {subject_name!r} in {URDF_DIR}; "
            f"found {len(matches)}."
        )
    return matches[0]


def extract_morphology_features(urdf_path: Path) -> np.ndarray:
    """Extract bilateral segment lengths and trunk/pelvis dimensions from a URDF.

    Thigh, shank, and upper-arm lengths are joint-to-joint vectors.  Foot
    length is the calcaneus-to-toe marker distance.  Forearm length is the
    elbow-to-mid-wrist distance, where the mid-wrist is the mean of the two
    wrist markers.  These definitions are robust to the virtual zero-length
    links used by the scaled human URDFs.
    """
    origins = _joint_origins(urdf_path)

    def origin(name: str) -> np.ndarray:
        if name not in origins:
            raise KeyError(f"Joint {name!r} is missing from {urdf_path}")
        return origins[name]

    def length(vector: np.ndarray) -> float:
        value = float(np.linalg.norm(vector))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid morphology length {value} from {urdf_path}")
        return value

    right_wrist = 0.5 * (origin("joint_r_lwrist_study") + origin("joint_r_mwrist_study"))
    left_wrist = 0.5 * (origin("joint_L_lwrist_study") + origin("joint_L_mwrist_study"))

    features = np.asarray([
        length(origin("left_hip_Z") - origin("right_hip_Z")),
        length(origin("right_knee_Z")),
        length(origin("right_ankle_Z")),
        length(origin("joint_r_toe_study") - origin("joint_r_calc_study")),
        length(origin("left_knee_Z")),
        length(origin("left_ankle_Z")),
        length(origin("joint_L_toe_study") - origin("joint_L_calc_study")),
        length(origin("middle_cervical_Z")),
        length(origin("right_elbow_Z")),
        length(right_wrist),
        length(origin("left_elbow_Z")),
        length(left_wrist),
    ], dtype=np.float32)
    if features.shape != (N_MORPHOLOGY_FEATURES,):
        raise RuntimeError("Morphology feature dimension is inconsistent.")
    return features


def subject_morphology(subject_name: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    """Load and cache a subject's raw morphology vector in metres."""
    key = subject_name.lower()
    if key not in cache:
        cache[key] = extract_morphology_features(_find_scaled_urdf(subject_name))
    return cache[key].copy()


def save_morphology_table(samples, output_path: Path) -> None:
    """Save raw, interpretable morphology values for experiment provenance."""
    subject_values = {}
    for sample in samples:
        subject_values[sample["subject"]] = [float(value) for value in sample["morphology"]]
    payload = {
        "units": "metres",
        "feature_names": list(MORPHOLOGY_FEATURE_NAMES),
        "subjects": dict(sorted(subject_values.items())),
    }
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)


class MorphologyFlowDataset(Dataset):
    """Windowed GRFM/joint samples with one static morphology vector per window."""
    def __init__(self, samples, stats, window_size=WINDOW_SIZE):
        self.samples = []
        body_weight_cache = {}
        for sample in samples:
            grfm, joints = base.load_trial(
                sample["kinetics"], sample["joints"], body_weight_cache
            )
            morphology = np.asarray(sample["morphology"], dtype=np.float32)
            if morphology.shape != (N_MORPHOLOGY_FEATURES,):
                raise ValueError(
                    f"Expected {N_MORPHOLOGY_FEATURES} morphology values for "
                    f"{sample['subject']}, got {morphology.shape}."
                )
            for start in range(0, len(grfm) - window_size + 1, window_size // 2):
                self.samples.append((
                    grfm[start:start + window_size],
                    joints[start:start + window_size],
                    morphology,
                ))
        if not self.samples:
            raise ValueError("The dataset has no complete windows.")
        self.stats = stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        grfm, joints, morphology = self.samples[index]
        grfm = torch.from_numpy(grfm)
        joints = torch.from_numpy(joints)
        morphology = torch.from_numpy(morphology)
        grfm = (grfm - self.stats["f_mean"]) / (self.stats["f_std"] + 1e-6)
        joints = (joints - self.stats["j_mean"]) / (self.stats["j_std"] + 1e-6)
        morphology = (morphology - self.stats["morph_mean"]) / (self.stats["morph_std"] + 1e-6)
        return grfm, joints, morphology


def compute_and_save_stats(train_samples, output_path: Path):
    """Fit GRFM/joint scalers on training frames and morphology scalers on subjects."""
    print("\n[INFO] Calcul des scalers sur le train set uniquement...")
    body_weight_cache = {}
    all_grfm, all_joints = [], []
    morphologies_by_subject = {}
    for sample in train_samples:
        grfm, joints = base.load_trial(sample["kinetics"], sample["joints"], body_weight_cache)
        all_grfm.append(grfm)
        all_joints.append(joints)
        morphology = np.asarray(sample["morphology"], dtype=np.float32)
        existing = morphologies_by_subject.setdefault(sample["subject"], morphology)
        if not np.allclose(existing, morphology):
            raise ValueError(f"Morphology changed between trials for {sample['subject']}.")

    grfm_cat = np.vstack(all_grfm)
    joints_cat = np.vstack(all_joints)
    morphology_cat = np.vstack(list(morphologies_by_subject.values()))
    morphology_std = morphology_cat.std(axis=0)
    if np.any(morphology_std < 1e-6):
        constant_features = [
            MORPHOLOGY_FEATURE_NAMES[index]
            for index, value in enumerate(morphology_std) if value < 1e-6
        ]
        raise ValueError(
            "At least one morphology feature is constant across training subjects: "
            f"{constant_features}. Remove it or change the split."
        )

    stats = {
        "f_mean": grfm_cat.mean(axis=0),
        "f_std": grfm_cat.std(axis=0),
        "j_mean": joints_cat.mean(axis=0),
        "j_std": joints_cat.std(axis=0),
        "morph_mean": morphology_cat.mean(axis=0),
        "morph_std": morphology_std,
    }
    payload = {
        **{key: value.tolist() for key, value in stats.items()},
        "morphology_feature_names": list(MORPHOLOGY_FEATURE_NAMES),
        "morphology_units": "metres_before_standardisation",
    }
    with open(output_path, "w") as file:
        json.dump(payload, file, indent=2)
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in stats.items()}


class MorphologyFlowTransformer(base.FlowTransformer):
    """The improved FlowTransformer with static subject morphology conditioning."""
    def __init__(self, morphology_dim=N_MORPHOLOGY_FEATURES, embed_dim=128,
                 nhead=4, num_layers=2, dropout=0.1):
        super().__init__(embed_dim=embed_dim, nhead=nhead, num_layers=num_layers, dropout=dropout)
        self.morphology_encoder = nn.Sequential(
            nn.Linear(morphology_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x, t, condition, morphology):
        batch_size, window_size, _ = x.shape
        if morphology.dim() == 1:
            morphology = morphology.unsqueeze(0).expand(batch_size, -1)
        if morphology.shape != (batch_size, N_MORPHOLOGY_FEATURES):
            raise ValueError(
                f"Expected morphology shape ({batch_size}, {N_MORPHOLOGY_FEATURES}), "
                f"got {tuple(morphology.shape)}."
            )
        morphology_embedding = self.morphology_encoder(morphology.to(dtype=x.dtype)).unsqueeze(1)

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
        memory = torch.cat(condition_blocks, dim=1) + time_embedding + morphology_embedding

        output = self.transformer(tgt=target, memory=memory)
        right_leg = self.out_Rleg(output[:, 0:window_size])
        left_leg = self.out_Lleg(output[:, window_size:2 * window_size])
        upper_body = self.out_Upper(output[:, 2 * window_size:3 * window_size])
        return torch.cat((right_leg, left_leg, upper_body), dim=2)


@torch.no_grad()
def sample_heun_morphology(model, condition, morphology, n_steps=INFERENCE_STEPS):
    """Second-order Flow Matching sampler for one or more morphology-conditioned windows."""
    batch_size, window_size, _ = condition.shape
    x = torch.randn((batch_size, window_size, N_JOINTS), device=condition.device)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t0 = step / n_steps
        t1 = (step + 1) / n_steps
        t0_tensor = torch.full((batch_size,), t0, device=condition.device)
        t1_tensor = torch.full((batch_size,), t1, device=condition.device)
        velocity0 = model(x, t0_tensor, condition, morphology)
        x_predictor = x + dt * velocity0
        velocity1 = model(x_predictor, t1_tensor, condition, morphology)
        x = x + 0.5 * dt * (velocity0 + velocity1)
    return x


@torch.no_grad()
def predict_full_trial_morphology(model, kinetics_path, joints_path, morphology, stats,
                                  device, window_size=WINDOW_SIZE, stride=STRIDE,
                                  n_steps=INFERENCE_STEPS):
    """Full-trial Heun inference with the same overlap inpainting as the base model."""
    model.eval()
    body_weight_cache = {}
    grfm, reference = base.load_trial(kinetics_path, joints_path, body_weight_cache)
    condition = (torch.from_numpy(grfm) - stats["f_mean"]) / (stats["f_std"] + 1e-6)
    morphology = torch.as_tensor(morphology, dtype=torch.float32)
    morphology = (morphology - stats["morph_mean"]) / (stats["morph_std"] + 1e-6)
    total_frames = len(condition)

    if total_frames < window_size:
        padding = condition[-1:].repeat(window_size - total_frames, 1)
        condition_for_sampling = torch.cat((condition, padding), dim=0)
    else:
        condition_for_sampling = condition

    full_prediction = torch.zeros((len(condition_for_sampling), N_JOINTS), device=device)
    previous_start = None
    print(f"  [INF] Sampling morphology-conditioned Flow Matching (Heun): {total_frames} frames")
    for start in base.window_starts(len(condition_for_sampling), window_size, stride):
        end = start + window_size
        condition_window = condition_for_sampling[start:end].unsqueeze(0).to(device)
        morphology_window = morphology.unsqueeze(0).to(device)
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
    """Deterministic EMA validation loss, including static morphology."""
    model.eval()
    total_squared_error = 0.0
    total_values = 0
    generator_device = device.type
    with torch.no_grad():
        for batch_index, (grfm, joints, morphology) in enumerate(data_loader):
            grfm = grfm.to(device, non_blocking=True)
            joints = joints.to(device, non_blocking=True)
            morphology = morphology.to(device, non_blocking=True)
            batch_size = joints.shape[0]
            generator = torch.Generator(device=generator_device).manual_seed(SEED + batch_index)
            x0 = torch.randn(joints.shape, generator=generator, device=device)
            t = torch.rand((batch_size, 1, 1), generator=generator, device=device)
            xt = t * joints + (1.0 - t) * x0
            target_velocity = joints - x0
            prediction = model(xt, t.view(batch_size), grfm, morphology)
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
        if subject_name.lower() not in VINC_SUBJECTS:
            continue
        for task_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            task_name = task_dir.name
            # if not is_squat_task_only(subject_name, task_name):
            #     continue
            kinetics_path = task_dir / "kinetics_feet.npy"
            joints_path = task_dir / "all_joints.npy"
            if kinetics_path.exists() and joints_path.exists():
                # Only resolve an URDF after a trial has passed the same squat
                # filter as the baseline script. 
                morphology = subject_morphology(subject_name, morphology_cache)
                all_samples.append({
                    "subject": subject_name,
                    "task": task_name,
                    "kinetics": kinetics_path,
                    "joints": joints_path,
                    "morphology": morphology,
                })

    print(f"Nombre total d'essais squat : {len(all_samples)}")
    train_samples, val_samples, test_samples = base.split_dataset(all_samples, seed=SEED)
    save_morphology_table(all_samples, RESULTS_DIR / "morphology_features_metres.json")
    stats = compute_and_save_stats(train_samples, RESULTS_DIR / "scalers_morphology.json")

    train_dataset = MorphologyFlowDataset(train_samples, stats=stats)
    val_dataset = MorphologyFlowDataset(val_samples, stats=stats)
    test_dataset = MorphologyFlowDataset(test_samples, stats=stats)
    print(f"Windows: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"Morphology features ({N_MORPHOLOGY_FEATURES}): {', '.join(MORPHOLOGY_FEATURE_NAMES)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    model = MorphologyFlowTransformer().to(device)
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
        for grfm, joints, morphology in train_loader:
            grfm = grfm.to(device, non_blocking=True)
            joints = joints.to(device, non_blocking=True)
            morphology = morphology.to(device, non_blocking=True)
            batch_size = joints.shape[0]
            x0 = torch.randn_like(joints)
            t = torch.rand((batch_size, 1, 1), device=device)
            xt = t * joints + (1.0 - t) * x0
            target_velocity = joints - x0

            optimizer.zero_grad(set_to_none=True)
            predicted_velocity = model(xt, t.view(batch_size), grfm, morphology)
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
            torch.save(ema.model.state_dict(), RESULTS_DIR / "fm_morphology_model_best_ema.pth")
            torch.save(model.state_dict(), RESULTS_DIR / "fm_morphology_model_best_raw.pth")

        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | lr={learning_rate:.2e} | "
            f"train={train_loss:.6f} | val_ema={val_loss:.6f}"
        )

    torch.save(model.state_dict(), RESULTS_DIR / "fm_morphology_model_final_raw.pth")
    torch.save(ema.model.state_dict(), RESULTS_DIR / "fm_morphology_model_final_ema.pth")

    # Qualitative test plots use the best EMA checkpoint, as in the base script.
    model.load_state_dict(torch.load(RESULTS_DIR / "fm_morphology_model_best_ema.pth", map_location=device))
    model.eval()
    example_grfm, example_joints, example_morphology = test_dataset[random.randrange(len(test_dataset))]
    sample_prediction = sample_heun_morphology(
        model,
        example_grfm.unsqueeze(0).to(device),
        example_morphology.unsqueeze(0).to(device),
    ).cpu().squeeze(0)
    sample_prediction = sample_prediction * stats["j_std"] + stats["j_mean"]
    sample_reference = example_joints * stats["j_std"] + stats["j_mean"]
    base.plot_joints(
        sample_reference.numpy(), sample_prediction.numpy(), 0, 12,
        RESULTS_DIR / "inference_test_joints_1_12.png",
    )
    base.plot_joints(
        sample_reference.numpy(), sample_prediction.numpy(), 12, 29,
        RESULTS_DIR / "inference_test_joints_13_29.png",
    )

    test_sample = random.choice(test_samples)
    full_reference, full_prediction = predict_full_trial_morphology(
        model,
        test_sample["kinetics"],
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
    plt.title("Loss history — morphology-conditioned model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "loss_curve.png")
    plt.close()

    with open(RESULTS_DIR / "experiment_config.json", "w") as file:
        json.dump({
            "base_training_script": "train_fm_full_step2motion_real_feet_improved.py",
            "data_root": str(DATA_ROOT),
            "urdf_dir": str(URDF_DIR),
            "morphology_feature_names": list(MORPHOLOGY_FEATURE_NAMES),
            "epochs": EPOCHS,
            "seed": SEED,
        }, file, indent=2)
    print(f"\n[FINISH] Morphology experiment saved in {RESULTS_DIR}")


if __name__ == "__main__":
    run_experiment()
