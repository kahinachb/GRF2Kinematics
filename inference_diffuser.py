"""
Inference and Visualization Script (Diffusion)
- Loads a trained diffusion checkpoint (with stored config/dims)
- Uses the shared scalers from utils.py (same as training)
- Two modes:
  1) sliding-last-step: windows with stride=1; keep last frame of each sampled window
  2) batch-seq: exact same slicing as training (seq_len/stride); concatenate predictions
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

# Reuse model & diffusion classes from your training script
from train_diffuser import DenoiserUNet1D, Diffusion
# Reuse the exact same slicing from your shared utils
from utils import slice_trial_into_sequences

JOINT_NAMES = [
    'Left Hip Z', 'Left Hip X', 'Left Hip Y',
    'Left Knee Z', 'Left Ankle Z', 'Left Ankle X',
    'Right Hip Z', 'Right Hip X', 'Right Hip Y',
    'Right Knee Z', 'Right Ankle Z', 'Right Ankle X'
]
FORCE_NAMES = [
    'Left Fx', 'Left Fy', 'Left Fz',
    'Left Mx', 'Left My', 'Left Mz',
    'Right Fx', 'Right Fy', 'Right Fz',
    'Right Mx', 'Right My', 'Right Mz'
]


# ============== helpers ==============

def load_scaler(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """Same style as LSTM metrics: overall + per-joint + correlations."""
    mse_per_joint = ((y_true - y_pred) ** 2).mean(axis=0)
    rmse_per_joint = np.sqrt(mse_per_joint)
    mae_per_joint = np.abs(y_true - y_pred).mean(axis=0)

    overall_mse = float(mse_per_joint.mean())
    overall_rmse = float(np.sqrt(mse_per_joint.mean()))
    overall_mae = float(mae_per_joint.mean())

    corrs = []
    for j in range(y_true.shape[1]):
        if np.std(y_true[:, j]) < 1e-8 or np.std(y_pred[:, j]) < 1e-8:
            corrs.append(0.0)
        else:
            c = np.corrcoef(y_true[:, j], y_pred[:, j])[0, 1]
            corrs.append(float(c) if np.isfinite(c) else 0.0)

    return dict(
        overall_mse=overall_mse,
        overall_rmse=overall_rmse,
        overall_mae=overall_mae,
        mse_per_joint=mse_per_joint,
        rmse_per_joint=rmse_per_joint,
        mae_per_joint=mae_per_joint,
        correlations=np.array(corrs, dtype=np.float32)
    )


def print_metrics(metrics: Dict, joint_names: List[str], degrees: bool = True):
    print("\n" + "="*60)
    print("EVALUATION METRICS (Diffusion)")
    print("="*60)
    print(f"Overall (rad): MSE={metrics['overall_mse']:.6f} | RMSE={metrics['overall_rmse']:.6f} | MAE={metrics['overall_mae']:.6f}")

    if degrees:
        deg = 180/np.pi
        print(f"Overall (deg): RMSE={metrics['overall_rmse']*deg:.3f} | MAE={metrics['overall_mae']*deg:.3f}")

    print(f"\nPer-Joint Metrics:")
    print(f"{'Joint':<20} {'RMSE(deg)':>12} {'MAE(deg)':>12} {'Corr':>10}")
    print("-"*56)
    deg = 180/np.pi
    for i, name in enumerate(joint_names):
        rmse_d = metrics['rmse_per_joint'][i] * deg
        mae_d  = metrics['mae_per_joint'][i]  * deg
        corr   = metrics['correlations'][i]
        print(f"{name:<20} {rmse_d:>12.4f} {mae_d:>12.4f} {corr:>10.3f}")
    print("="*60)


def plot_sequence(y_true: np.ndarray, y_pred: np.ndarray,
                  title: str = "Diffusion Force-to-Angle Prediction",
                  joint_names: Optional[List[str]] = None,
                  in_degrees: bool = True):
    if joint_names is None:
        joint_names = JOINT_NAMES

    if in_degrees:
        y_true_plot = y_true * (180/np.pi)
        y_pred_plot = y_pred * (180/np.pi)
        ylab = "Angle (deg)"
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        ylab = "Angle (rad)"

    T, J = y_true_plot.shape
    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)

    t = np.arange(T)
    for j, (ax, name) in enumerate(zip(axes.flat, joint_names)):
        ax.plot(t, y_true_plot[:, j], label="True", alpha=0.7)
        ax.plot(t, y_pred_plot[:, j], label="Pred", alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    return fig


# ============== core class ==============

class DiffusionInference:
    def __init__(self,
                 model_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 scaler_dir: str = "./scalers_shared",
                 normalize_forces: bool = True,
                 normalize_joints: bool = True,
                 sample_steps: int = 50,
                 eta: float = 0.0):
        """
        Args:
            model_path: checkpoint from training (best_diffusion.pth)
            device: cuda / cpu
            scaler_dir: directory where force/joint scalers are stored by utils
            normalize_forces: True if forces were normalized at training
            normalize_joints: True if joints were normalized at training
            sample_steps: DDIM steps for sampling (20-100 is typical)
            eta: >0 for stochastic DDIM; 0 for deterministic DDIM
        """
        self.device = device
        self.model, self.diffusion, self.cfg, self.dims = self._load_model_and_diffusion(model_path)
        self.model.eval()
        self.sample_steps = sample_steps
        self.eta = eta

        self.force_scaler = load_scaler(Path(scaler_dir)/"force_scaler.pkl") if normalize_forces else None
        self.joint_scaler = load_scaler(Path(scaler_dir)/"joint_scaler.pkl") if normalize_joints else None

        self.normalize_forces = normalize_forces
        self.normalize_joints = normalize_joints

    def _load_model_and_diffusion(self, model_path: str):
        ckpt = torch.load(model_path, map_location=self.device)
        cfg = ckpt.get("config", {})
        dims = ckpt.get("dims", None)  # {"F": Fd, "J": Jd}

        # Fallbacks if missing
        base_dim = cfg.get("base_dim", 128)
        time_dim = cfg.get("time_dim", 128)
        n_blocks = cfg.get("n_blocks", 6)
        timesteps = cfg.get("timesteps", 1000)

        # Build model: need joint_dim and cond_dim
        if dims is None:
            raise RuntimeError("Checkpoint missing 'dims'. Retrain saving {'dims': {'F': Fd, 'J': Jd}}.")
        Fd, Jd = int(dims["F"]), int(dims["J"])

        model = DenoiserUNet1D(joint_dim=Jd, cond_dim=Fd,
                               base_dim=base_dim, time_dim=time_dim, n_blocks=n_blocks).to(self.device)
        model.load_state_dict(ckpt["state_dict"])

        diffusion = Diffusion(timesteps=timesteps, beta_schedule="cosine")
        return model, diffusion, cfg, dims

    @torch.no_grad()
    def _sample_window(self, forces_window: np.ndarray) -> np.ndarray:
        """
        Sample one full sequence given a force window [L, F] -> [L, J] (normalized joint space).
        """
        x = forces_window.astype(np.float32)
        if self.normalize_forces and self.force_scaler is not None:
            x = self.force_scaler.transform(x)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)  # [1, L, F]
        y = self.diffusion.sample(self.model, xt, steps=self.sample_steps, eta=self.eta, device=self.device)  # [1, L, J]
        y = y.squeeze(0).cpu().numpy()  # [L, J]
        return y

    def _inverse_joints(self, y_norm: np.ndarray) -> np.ndarray:
        """Inverse-transform joints from normalized space to radians if scaler is available."""
        if self.normalize_joints and self.joint_scaler is not None and y_norm.size > 0:
            B = y_norm.shape[0]
            J = y_norm.shape[1]
            y = self.joint_scaler.inverse_transform(y_norm.reshape(-1, J)).reshape(B, J)
            return y
        return y_norm

    # --------- public APIs ---------
    def run_sliding_last_step(self,
                              forces: np.ndarray,
                              joints: np.ndarray,
                              seq_len: int,
                              batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding stride=1 windows; for each window, sample a full sequence and take only the last frame.
        Returns aligned arrays in raw units (radians): y_true [T-L+1, J], y_pred [T-L+1, J]
        """
        T = min(len(forces), len(joints))
        forces = forces[:T]; joints = joints[:T]
        starts = np.arange(seq_len - 1, T)  # last index of each window

        preds_chunks = []
        # Batch the windows for speed
        for i in range(0, len(starts), batch_size):
            batch_idxs = starts[i:i+batch_size]
            windows = []
            for end_idx in batch_idxs:
                win = forces[end_idx - seq_len + 1 : end_idx + 1]   # [L, F]
                if self.normalize_forces and self.force_scaler is not None:
                    win = self.force_scaler.transform(win.astype(np.float32))
                windows.append(win)
            x = torch.from_numpy(np.stack(windows, axis=0)).to(self.device)  # [B, L, F]
            y_seq = self.diffusion.sample(self.model, x, steps=self.sample_steps, eta=self.eta, device=self.device)  # [B, L, J]
            y_last = y_seq[:, -1, :].cpu().numpy()  # [B, J]
            preds_chunks.append(y_last)

        y_pred_norm = np.vstack(preds_chunks) if preds_chunks else np.zeros((0, joints.shape[1]), dtype=np.float32)
        # inverse-transform to radians (if joints were normalized during training)
        y_pred = self._inverse_joints(y_pred_norm)

        # Align ground truth
        y_true = joints[seq_len - 1 : ]  # [T-L+1, J]
        return y_true, y_pred

    def run_batch_seq(self,
                      forces: np.ndarray,
                      joints: np.ndarray,
                      seq_len: int,
                      stride: int,
                      batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use the same slicing as training: slice into windows (seq_len, stride),
        sample each window (batched), and concatenate in temporal order.
        Returns concatenated arrays (radians): y_true_cat [N*L, J], y_pred_cat [N*L, J]
        """
        seqs = slice_trial_into_sequences(forces, joints, seq_len, stride)
        if not seqs:
            return np.zeros((0, joints.shape[1]), dtype=np.float32), np.zeros((0, joints.shape[1]), dtype=np.float32)

        y_true_list = []
        y_pred_list = []

        # Batch windows
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            f_batch = np.stack([s["forces"].astype(np.float32) for s in batch], axis=0)  # [B, L, F]
            j_batch = np.stack([s["joints"].astype(np.float32) for s in batch], axis=0)  # [B, L, J]

            if self.normalize_forces and self.force_scaler is not None:
                # apply per-window normalization row-wise
                B, L, Fd = f_batch.shape
                f_batch = f_batch.reshape(-1, Fd)
                f_batch = self.force_scaler.transform(f_batch)
                f_batch = f_batch.reshape(B, L, Fd)

            x = torch.from_numpy(f_batch).to(self.device)
            y_seq = self.diffusion.sample(self.model, x, steps=self.sample_steps, eta=self.eta, device=self.device)  # [B, L, J]
            y_seq = y_seq.cpu().numpy()  # normalized space

            # flatten B*L for concat
            B, L, J = y_seq.shape
            y_pred_list.append(y_seq.reshape(B*L, J))
            y_true_list.append(j_batch.reshape(B*L, J))

        y_pred_norm = np.concatenate(y_pred_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0)

        # inverse-transform predictions to radians if needed
        if self.normalize_joints and self.joint_scaler is not None and y_pred_norm.size > 0:
            J = y_pred_norm.shape[1]
            y_pred = self.joint_scaler.inverse_transform(y_pred_norm.reshape(-1, J)).reshape(-1, J)
        else:
            y_pred = y_pred_norm
        return y_true, y_pred


# ============== CLI ==============

def main():
    ap = argparse.ArgumentParser(description="Inference for Diffusion model using shared utils")
    ap.add_argument("--model_path", type=str, default="./checkpoints_diff/best_diffusion.pth")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--subject", type=str, required=True)
    ap.add_argument("--trial", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scaler_dir", type=str, default="./scalers_diff")
    ap.add_argument("--normalize_forces", action="store_true", default=True)
    ap.add_argument("--normalize_joints", action="store_true", default=True)

    ap.add_argument("--mode", type=str, choices=["sliding-last-step", "batch-seq"], default="sliding-last-step")
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--stride", type=int, default=10, help="Used in batch-seq; ignored in sliding-last-step")
    ap.add_argument("--sample_steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--plot", action="store_true",default=True)

    args = ap.parse_args()

    # Load data files
    trial_dir = Path(args.data_dir) / args.subject / args.trial
    fpath = trial_dir / "forces.npy"
    jpath = trial_dir / "joints.npy"
    if not fpath.exists() or not jpath.exists():
        raise FileNotFoundError(f"Missing files: {fpath} or {jpath}")

    forces = np.load(fpath)   # [T, F]
    joints = np.load(jpath)   # [T, J]
    print(f"Loaded: forces {forces.shape}, joints {joints.shape}")

    infer = DiffusionInference(
        model_path=args.model_path,
        device=args.device,
        scaler_dir=args.scaler_dir,
        normalize_forces=args.normalize_forces,
        normalize_joints=args.normalize_joints,
        sample_steps=args.sample_steps,
        eta=args.eta
    )

    if args.mode == "sliding-last-step":
        y_true, y_pred = infer.run_sliding_last_step(
            forces, joints, seq_len=args.seq_len, batch_size=args.batch_size
        )
        title = f"Diffusion – sliding last-step (L={args.seq_len}, stride=1, steps={args.sample_steps}, eta={args.eta})"
    else:
        y_true, y_pred = infer.run_batch_seq(
            forces, joints, seq_len=args.seq_len, stride=args.stride, batch_size=args.batch_size
        )
        title = f"Diffusion – batch sequences (L={args.seq_len}, stride={args.stride}, steps={args.sample_steps}, eta={args.eta})"

    print(f"Aligned arrays: y_true {y_true.shape} | y_pred {y_pred.shape}")

    # Metrics (in radians)
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, JOINT_NAMES, degrees=True)

    if args.plot and y_true.shape[0] > 0:
        fig = plot_sequence(y_true, y_pred, title=title, joint_names=JOINT_NAMES, in_degrees=True)
        plt.show()


if __name__ == "__main__":
    main()
