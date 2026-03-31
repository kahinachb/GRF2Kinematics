"""
dataset.py
==========
Dataset for the motion diffusion model.

Scans the npy/ output folder produced by convert_to_npy.py and builds a
sliding-window dataset:
    X  (joints)   : (T, 29)  — all_joints.npy
    C  (condition): (T, 12)  — kinetics.npy

Normalization:
    Mean and std are computed on the training split and saved as .npz so
    that the same statistics can be reused at inference time.

Folder structure expected (output of convert_to_npy.py):
    <npy_root>/<dataset>/<subject>/<task>/all_joints.npy
    <npy_root>/<dataset>/<subject>/<task>/kinetics.npy
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZER
# ─────────────────────────────────────────────────────────────────────────────

class Normalizer:
    """Z-score normalizer. Fit on training data, applied to all splits."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std:  Optional[np.ndarray] = None

    def fit(self, data: np.ndarray):
        """Compute mean/std over axis 0 (frames × DOFs concatenated)."""
        self.mean = data.mean(axis=0, keepdims=True).astype(np.float32)
        self.std  = data.std(axis=0,  keepdims=True).astype(np.float32)
        # Avoid division by zero for constant channels
        self.std  = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)
        print(f"  [Normalizer] Saved → {path}")

    def load(self, path: str):
        data      = np.load(path)
        self.mean = data["mean"].astype(np.float32)
        self.std  = data["std"].astype(np.float32)

    def to_tensor(self, device):
        """Return (mean, std) as torch tensors on the given device."""
        return (
            torch.tensor(self.mean, device=device),
            torch.tensor(self.std,  device=device),
        )


# ─────────────────────────────────────────────────────────────────────────────
# TRIAL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _find_trial_paths(npy_root: Path) -> List[Tuple[Path, Path]]:
    """Return all (all_joints.npy, kinetics.npy) pairs under npy_root."""
    pairs = []
    for joints_path in sorted(npy_root.rglob("all_joints.npy")):
        kinetics_path = joints_path.parent / "kinetics.npy"
        if kinetics_path.exists():
            pairs.append((joints_path, kinetics_path))
    if not pairs:
        raise FileNotFoundError(
            f"No (all_joints.npy, kinetics.npy) pairs found under {npy_root}.\n"
            "Run convert_to_npy.py first."
        )
    return pairs


def load_all_trials(npy_root: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load all trials and return two lists: joints, kinetics."""
    pairs    = _find_trial_paths(npy_root)
    joints_l, kinetics_l = [], []
    print(f"  [Dataset] Found {len(pairs)} trial(s) under {npy_root}")
    for jp, kp in pairs:
        j = np.load(jp).astype(np.float32)   # (T, 29)
        k = np.load(kp).astype(np.float32)   # (T, 12)
        # Truncate to the shortest to keep them aligned
        n = min(len(j), len(k))
        joints_l.append(j[:n])
        kinetics_l.append(k[:n])
    return joints_l, kinetics_l


# ─────────────────────────────────────────────────────────────────────────────
# SLIDING-WINDOW DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MotionDiffusionDataset(Dataset):
    """
    Sliding-window dataset that returns (joints_window, kinetics_window).

    Each sample is a fixed-length window of `seq_len` frames.
    Overlapping windows are created with a stride of `stride` frames.

    Args:
        joints_list   : list of (T_i, 29) arrays
        kinetics_list : list of (T_i, 12) arrays
        seq_len       : window length in frames (128 → 1.28 s @ 100 Hz)
        stride        : step between consecutive windows (default = seq_len // 2)
        joint_norm    : fitted Normalizer for joints  (None = no normalization)
        kinetics_norm : fitted Normalizer for kinetics
    """

    def __init__(
        self,
        joints_list:   List[np.ndarray],
        kinetics_list: List[np.ndarray],
        seq_len:       int = 128,
        stride:        Optional[int] = None,
        joint_norm:    Optional[Normalizer] = None,
        kinetics_norm: Optional[Normalizer] = None,
    ):
        self.seq_len      = seq_len
        self.stride       = stride or seq_len // 2
        self.joint_norm   = joint_norm
        self.kinetics_norm= kinetics_norm

        # Build list of (trial_idx, start_frame) index pairs
        self.index: List[Tuple[int, int]] = []
        self.joints_list   = joints_list
        self.kinetics_list = kinetics_list

        for i, j in enumerate(joints_list):
            n_frames = len(j)
            if n_frames < seq_len:
                continue   # skip trials shorter than one window
            start = 0
            while start + seq_len <= n_frames:
                self.index.append((i, start))
                start += self.stride

        print(f"  [Dataset] {len(self.index)} windows  "
              f"(seq_len={seq_len}, stride={self.stride}, "
              f"trials={len(joints_list)})")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        trial_idx, start = self.index[idx]
        end = start + self.seq_len

        joints   = self.joints_list[trial_idx][start:end].copy()    # (T, 29)
        kinetics = self.kinetics_list[trial_idx][start:end].copy()  # (T, 12)

        if self.joint_norm is not None:
            joints   = self.joint_norm.transform(joints)
        if self.kinetics_norm is not None:
            kinetics = self.kinetics_norm.transform(kinetics)

        return (
            torch.from_numpy(joints),    # (T, 29)  float32
            torch.from_numpy(kinetics),  # (T, 12)  float32
        )


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY: build train / val datasets + normalizers in one call
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(
    npy_root:       str,
    seq_len:        int   = 128,
    stride:         Optional[int] = None,
    val_ratio:      float = 0.15,
    norm_save_path: Optional[str] = None,
) -> Tuple["MotionDiffusionDataset", "MotionDiffusionDataset",
           Normalizer, Normalizer]:
    """
    Load all trials, fit normalizers on the training split, and return
    (train_dataset, val_dataset, joint_normalizer, kinetics_normalizer).

    Args:
        npy_root       : path to the npy/ folder produced by convert_to_npy.py
        seq_len        : window length in frames
        stride         : stride between windows (default: seq_len // 2)
        val_ratio      : fraction of trials used for validation
        norm_save_path : if given, save normalizers as <path>_joints.npz / _kinetics.npz
    """
    npy_root = Path(npy_root)
    joints_list, kinetics_list = load_all_trials(npy_root)

    # Trial-level train / val split (split by trial, not by frame)
    n_total = len(joints_list)
    n_val   = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    # Shuffle consistently
    rng   = np.random.default_rng(seed=42)
    order = rng.permutation(n_total).tolist()
    train_idx = order[:n_train]
    val_idx   = order[n_train:]

    train_j = [joints_list[i]   for i in train_idx]
    train_k = [kinetics_list[i] for i in train_idx]
    val_j   = [joints_list[i]   for i in val_idx]
    val_k   = [kinetics_list[i] for i in val_idx]

    print(f"  [Dataset] Train trials: {len(train_j)} | Val trials: {len(val_j)}")

    # Fit normalizers on training data only
    joint_norm    = Normalizer()
    kinetics_norm = Normalizer()
    joint_norm.fit(np.concatenate(train_j, axis=0))      # (N_train_frames, 29)
    kinetics_norm.fit(np.concatenate(train_k, axis=0))   # (N_train_frames, 12)

    if norm_save_path is not None:
        joint_norm.save(f"{norm_save_path}_joints.npz")
        kinetics_norm.save(f"{norm_save_path}_kinetics.npz")

    train_ds = MotionDiffusionDataset(train_j, train_k, seq_len, stride,
                                       joint_norm, kinetics_norm)
    val_ds   = MotionDiffusionDataset(val_j,   val_k,   seq_len, stride,
                                       joint_norm, kinetics_norm)
    return train_ds, val_ds, joint_norm, kinetics_norm