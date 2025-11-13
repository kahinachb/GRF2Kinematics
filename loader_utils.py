# utils.py
import os
import glob
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Subject split (leave-subjects-out)
# -----------------------------
def train_test_split_by_subject(
    data_dir: str, test_subjects: Optional[List[str]] = None, test_ratio: float = 0.2, exclude: Optional[List[str]] = None, seed: int = 42) -> Tuple[List[str], List[str]]:
    
    """
    Split data by subject for training and testing.
    
    Args:
        data_dir: Directory containing processed data
        test_subjects: List of subject names to use for testing
        test_ratio: Ratio of subjects to use for testing (if test_subjects not provided)
    
    Returns:
        train_subjects, test_subjects lists
    """
    data_path = Path(data_dir)
    all_subjects = [d.name for d in data_path.iterdir() if d.is_dir()]
    if exclude:
        all_subjects = [s for s in all_subjects if s not in exclude]
    rng = np.random.default_rng(seed)
    rng.shuffle(all_subjects)

    if test_subjects is None:
        # Randomly select test subjects
        num_test = max(1, int(len(all_subjects) * test_ratio))
        test_subjects = all_subjects[:num_test]
        train_subjects = all_subjects[num_test:]
    else:
        train_subjects = [s for s in all_subjects if s not in test_subjects]
    return train_subjects, test_subjects


# -----------------------------
# Core sequence slicing (shared)
# -----------------------------
def slice_trial_into_sequences(
    forces: np.ndarray, joints: np.ndarray, seq_len: int, stride: int
) -> List[Dict]:
    """
    Slice one trial into fixed-length windows with a sliding stride.
    No padding; tail shorter than seq_len is dropped (same behavior for all models).
    Returns list of dicts with 'forces' and 'joints' arrays.
    """
    T_f= len(forces)
    T_j = len(joints)
    if T_j == T_f :
        T = T_j
    else : 
        print("shape issue, forces vs joints")
    forces, joints = forces[:T], joints[:T]
    out = []
    for i in range(0, T - seq_len + 1, stride):
        out.append({
            "forces": forces[i:i+seq_len],    # (L, F)
            "joints": joints[i:i+seq_len],    # (L, J)
        })
    return out


# -----------------------------
# Shared dataset (identical for LSTM & Diffusion)
# -----------------------------
class SequenceDataset(Dataset):
    """
    Loads (forces, joints) sequences using the exact same slicing and scalers for
    both LSTM and Diffusion models.

    Directory layout:
      data_dir/Subject/Trial/forces.npy  [T, F]
      data_dir/Subject/Trial/joints.npy  [T, J]
    """
    def __init__(
        self,
        data_dir: str,
        seq_len: int = 50,
        stride: int = 10,
        normalize: bool = True,
        scaler_dir: str = "./scalers_shared",
        is_training: bool = True,
        subjects_filter: Optional[List[str]] = None,
        scale_forces: bool = True,
        scale_joints: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.normalize = normalize
        self.is_training = is_training
        self.subjects_filter = set(subjects_filter) if subjects_filter else None
        self.scale_forces = scale_forces
        self.scale_joints = scale_joints

        # Collect sequences
        self.samples: List[Dict] = []
        for subj_dir in sorted(self.data_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            if self.subjects_filter and subj_dir.name not in self.subjects_filter:
                continue
            for trial_dir in sorted(subj_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                fpath = trial_dir / "forces.npy"
                jpath = trial_dir / "joints.npy"
                if not (fpath.exists() and jpath.exists()):
                    continue
                forces = np.load(fpath)
                joints = np.load(jpath)
                seqs = slice_trial_into_sequences(forces, joints, self.seq_len, self.stride)
                for s in seqs:
                    self.samples.append({
                        "forces": s["forces"].astype(np.float32),
                        "joints": s["joints"].astype(np.float32),
                        "subject": subj_dir.name,
                        "trial": trial_dir.name,
                    })

        # Scalers (fit on train, reused on val/test)
        self.scaler_dir = Path(scaler_dir)
        self.scaler_dir.mkdir(exist_ok=True, parents=True)
        self.force_scaler_path = self.scaler_dir / "force_scaler.pkl"
        self.joint_scaler_path = self.scaler_dir / "joint_scaler.pkl"
        self.force_scaler = StandardScaler()
        self.joint_scaler = StandardScaler()

        if self.normalize:
            if self.is_training:
                # Fit on *train* only
                if self.scale_forces:
                    all_f = np.concatenate([s["forces"] for s in self.samples], axis=0)
                    self.force_scaler.fit(all_f)
                    with open(self.force_scaler_path, "wb") as f:
                        pickle.dump(self.force_scaler, f)
                if self.scale_joints:
                    all_j = np.concatenate([s["joints"] for s in self.samples], axis=0)
                    self.joint_scaler.fit(all_j)
                    with open(self.joint_scaler_path, "wb") as f:
                        pickle.dump(self.joint_scaler, f)
            else:
                # Load existing scalers
                if self.scale_forces and self.force_scaler_path.exists():
                    with open(self.force_scaler_path, "rb") as f:
                        self.force_scaler = pickle.load(f)
                if self.scale_joints and self.joint_scaler_path.exists():
                    with open(self.joint_scaler_path, "rb") as f:
                        self.joint_scaler = pickle.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        f = s["forces"]
        j = s["joints"]
        if self.normalize:
            if self.scale_forces:
                f = self.force_scaler.transform(f)
            if self.scale_joints:
                j = self.joint_scaler.transform(j)
        # Return consistent shapes: [L, F] and [L, J]
        return torch.from_numpy(f), torch.from_numpy(j)


# -----------------------------
# Helper: build shared loaders
# -----------------------------
def build_loaders_shared(
    data_dir: str,
    seq_len: int,
    stride_train: int,
    stride_val: int,
    batch_size: int,
    normalize: bool,
    scaler_dir: str,
    test_ratio: float,
    exclude_subjects: Optional[List[str]],
    num_workers: int = 0,
    seed: int = 42,
    scale_forces: bool = True,
    scale_joints: bool = True, ) -> Tuple[SequenceDataset, SequenceDataset, DataLoader, DataLoader, int, int, List[str], List[str]]:
    

    train_subs, test_subs = train_test_split_by_subject(
        data_dir, test_subjects=None, test_ratio=test_ratio, exclude=exclude_subjects, seed=seed)

    train_ds = SequenceDataset(
        data_dir, seq_len, stride_train, normalize, scaler_dir, True,
        subjects_filter=train_subs, scale_forces=scale_forces, scale_joints=scale_joints)
    
    val_ds = SequenceDataset(
        data_dir, seq_len, stride_val, normalize, scaler_dir, False,
        subjects_filter=test_subs, scale_forces=scale_forces, scale_joints=scale_joints)
    

    # Infer feature dims from one sample
    f0, j0 = train_ds[0]
    Fd = f0.shape[1]; Jd = j0.shape[1]

    #loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_ds, val_ds, train_loader, val_loader, Fd, Jd, train_subs, test_subs


def visualize_collected(collected, max_joints=12, save_dir=None, prefix="val_epoch", epoch=1):
    """
    Pour chaque batch collecté:
      - on trace pour 1..K séquences (samples_per_batch) et pour min(J, max_joints) joints
      - colonnes: [GT, Noisy, True noise, Pred noise, x0 reconstruit]
    Si save_dir est donné, on sauvegarde les figures; sinon on fait plt.show().
    """
    for item in collected:
        gt         = item["gt"]          # [S, L, J]
        noisy      = item["noisy"]
        true_noise = item["true_noise"]
        pred_noise = item["pred_noise"]
        x0_pred    = item["x0_pred"]
        timesteps  = item["timesteps"]   # len S
        bidx       = item["batch_idx"]

        S, L, J = gt.shape
        Jvis = min(J, max_joints)

        for s in range(S):
            fig, axes = plt.subplots(Jvis, 5, figsize=(18, 2.4*Jvis))
            if Jvis == 1:
                axes = axes.reshape(1, 5)  # sécurité si Jvis=1

            for j in range(Jvis):
                axes[j, 0].plot(gt[s, :, j]);         axes[j, 0].set_title(f"GT (t={timesteps[s]})", fontsize=9)
                axes[j, 1].plot(noisy[s, :, j]);      axes[j, 1].set_title("Noisy", fontsize=9)
                axes[j, 2].plot(true_noise[s, :, j]); axes[j, 2].set_title("True noise", fontsize=9)
                axes[j, 3].plot(pred_noise[s, :, j]); axes[j, 3].set_title("Pred noise", fontsize=9)
                axes[j, 4].plot(x0_pred[s, :, j]);    axes[j, 4].set_title("x0 reconstruit", fontsize=9)
                for c in range(5):
                    axes[j, c].grid(True)

            plt.tight_layout()
            if save_dir is None:
                plt.show()
            else:
                out = f"{save_dir}/{prefix}_e{epoch:03d}_batch{bidx:04d}_sample{s:02d}.png"
                plt.savefig(out, dpi=130)
                plt.close(fig)


class ForceToJointDataset(Dataset):
    def __init__(self, root_dir, subjects, seq_len=None):
        self.samples = []
        root = Path(root_dir)

        for subj in subjects:
            subj_path = root / subj
            if not subj_path.is_dir():
                continue

            trials = [p for p in subj_path.iterdir() if p.is_dir()]
            for trial in trials:
                f_path = trial / "forces.npy"
                j_path = trial / "joints.npy"
                if f_path.exists() and j_path.exists():
                    self.samples.append((str(f_path), str(j_path)))

        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f_path, j_path = self.samples[idx]
        forces = np.load(f_path)    # (T, 12)
        joints = np.load(j_path)    # (T, 12)

        # Option: crop/pad windows
        if self.seq_len is not None:
            T = forces.shape[0]
            if T >= self.seq_len:
                start = np.random.randint(0, T - self.seq_len + 1)
                forces = forces[start:start+self.seq_len]
                joints = joints[start:start+self.seq_len]
            else:
                pad = self.seq_len - T
                forces = np.pad(forces, ((0, pad), (0, 0)), mode="edge")
                joints = np.pad(joints, ((0, pad), (0, 0)), mode="edge")

        return torch.from_numpy(forces).float(), torch.from_numpy(joints).float()