#!/usr/bin/env python3
"""
Inference and Visualization (Diffusion + LSTM denoiser)
- Imports model + diffusion from training code
- Uses shared scalers saved during training
- Modes:
  1) sliding-last-step: predict only the last frame of each L-window (stride=1)
  2) batch-seq: sample full sequences with the same slicing as training
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from train_diffuser import LSTMDiffusionDenoiser, Diffusion 
from loader_utils import slice_trial_into_sequences  
from inference_utils import *

# ==========================
# Diffusion Inference
# ==========================
class DiffusionInference:
    def __init__(self,
                 ckpt_path: str = "./checkpoints_diff/best_diffusion_lstm.pth",
                 scaler_dir: str = "./scalers_diff",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 sample_steps: int = 50,
                 eta: float = 0.0):
        """
        Args:
            ckpt_path: path to diffusion checkpoint saved by training script
            scaler_dir: folder containing force_scaler.pkl and joint_scaler.pkl
            device: cpu / cuda
            sample_steps: DDIM steps
            eta: DDIM stochasticity (0 = deterministic)
        """
        self.device = device
        self.sample_steps = sample_steps
        self.eta = eta

        self.model, self.config, self.FdJd = self._load_model(ckpt_path)
        self.force_scaler, self.joint_scaler = self._load_scalers(scaler_dir)
        self.diffusion = Diffusion(
            timesteps=self.config.get("timesteps", 1000),
            beta_schedule="linear"
        )

        self.joint_names = JOINT_NAMES
        self.force_names = FORCE_NAMES

    # ----- IO -----
    def _load_model(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg = ckpt.get("config", {})
        dims = ckpt.get("dims", {})
        Fd = int(dims["F"])
        Jd = int(dims["J"])
        print(cfg)

        model = LSTMDiffusionDenoiser(
            joint_dim=Jd,
            cond_dim=Fd,
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
            bidirectional=cfg.get("bidirectional", False),
            time_dim=cfg.get("time_dim", 128),
            use_film=cfg.get("use_film", True)
        ).to(self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        print(f"Loaded diffusion denoiser. hparams={cfg} | dims={dims}")
        return model, cfg, (Fd, Jd)

    def _load_scalers(self, scaler_dir: str):
        scaler_dir = Path(scaler_dir)
        fsc = jsc = None
        fpath = scaler_dir / "force_scaler.pkl"
        jpath = scaler_dir / "joint_scaler.pkl"
        if fpath.exists():
            with open(fpath, "rb") as f:
                fsc = pickle.load(f)
            print(f"Loaded force scaler: {fpath}")
        else:
            print("WARNING: force scaler not found (proceeding without).")
        if jpath.exists():
            with open(jpath, "rb") as f:
                jsc = pickle.load(f)
            print(f"Loaded joint scaler: {jpath}")
        else:
            print("WARNING: joint scaler not found (outputs will remain normalized).")
        return fsc, jsc

    # ----- core sampling -----
    @torch.no_grad()
    def _sample_window(self, forces_win: np.ndarray,y_true_window) -> np.ndarray:
        """
        forces_win: [L, F] in original units.
        Returns:
            pred_joints: [L, J] (inverse-transformed if scaler available)
        """
        x = forces_win.astype(np.float32)
        y = y_true_window.astype(np.float32)
        if self.force_scaler is not None:
            x = self.force_scaler.transform(x)
        
        if self.joint_scaler is not None:
            y = self.force_scaler.transform(y)

        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)  # [1, L, F]
        yt = torch.from_numpy(x).unsqueeze(0).to(self.device)

        # samples = sample_from_gt_with_visualization(
        # self.model, self.diffusion, xt, yt)
        
        samples = self.diffusion.sample_with_visualization(
            model=self.model,
            cond=xt,
            y =yt,
            steps=self.sample_steps,
            eta=self.eta,
            device=self.device
        )  # [1, L, J] in normalized joint space
        print("norm mean/std:", samples.mean().item(), samples.std().item())  # should be O(1)

        arr = samples.squeeze(0).cpu().numpy()  # [L, J]

        if self.joint_scaler is not None:
            L, J = arr.shape
            arr = self.joint_scaler.inverse_transform(arr.reshape(-1, J)).reshape(L, J)

        return arr

    def predict_sequence(self, forces_seq: np.ndarray) -> np.ndarray:
        """
        Sample a full sequence (L, F) -> (L, J) with DDIM.
        """
        return self._sample_window(forces_seq)

    def predict_last_frame(self, forces_window: np.ndarray,y_true_window) -> np.ndarray:
        """
        Sample a window (L, F) -> (J,), returning ONLY the LAST FRAME
        """
        y_full = self._sample_window(forces_window,y_true_window)  # [L, J]
        return y_full[-1]                             # [J]

    def run_sliding_last_step(self,
                              forces: np.ndarray,
                              joints: np.ndarray,
                              seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding window with stride=1, predicting the last frame each time.
        Returns:
            y_true_aligned: [T - L + 1, J]
            y_pred_aligned: [T - L + 1, J]
        """
        T_j = len(joints)
        T_f = len(forces)
        if T_j == T_f:
            T=T_j
        else: 
            print("shape issues")
        forces = forces[:]
        joints = joints[:]

        preds = []
        for i in range(seq_len - 1, T):
            window = forces[i - seq_len + 1 : i + 1]      # [L, F]
            y_true_window = joints[i - seq_len + 1 : i + 1]
            y_last = self.predict_last_frame(window,y_true_window)   
            preds.append(y_last)
        y_pred = np.vstack(preds)                         # [T-L+1, J]
        y_true = joints[seq_len - 1 : ]                   # [T-L+1, J]
  

        return y_true, y_pred

    def run_batch_seq(self,
                      forces: np.ndarray,
                      joints: np.ndarray,
                      seq_len: int,
                      stride: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict full sequences using the same slicing as training:
        - windows of length seq_len
        - sliding with stride
        Concats predictions & targets in temporal order.
        """
        seqs = slice_trial_into_sequences(forces, joints, seq_len, stride)
        y_true_list, y_pred_list = [], []
        for s in seqs:
            f = s["forces"]  # [L, F]
            j = s["joints"]  # [L, J]
            y = self.predict_sequence(f)
            y_true_list.append(j)
            y_pred_list.append(y)
        y_true = np.concatenate(y_true_list, axis=0) if y_true_list else np.zeros((0, joints.shape[1]))
        y_pred = np.concatenate(y_pred_list, axis=0) if y_pred_list else np.zeros((0, joints.shape[1]))
        return y_true, y_pred


def main():
    ap = argparse.ArgumentParser(description="Inference for Diffusion LSTM using shared utils")
    ap.add_argument("--ckpt_path", type=str, default="./checkpoints_diff/best_diffusion_lstm.pth")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--subject", type=str, required=True)
    ap.add_argument("--trial", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scaler_dir", type=str, default="./scalers_diff")

    # Inference behavior
    ap.add_argument("--mode", type=str, choices=["sliding-last-step", "batch-seq"], default="sliding-last-step")
    ap.add_argument("--seq_len", type=int, default=50, help="Window length used at training")
    ap.add_argument("--stride", type=int, default=1, help="Stride for batch-seq mode; ignored for sliding-last-step")

    # Sampler params
    ap.add_argument("--sample_steps", type=int, default=1000)
    ap.add_argument("--eta", type=float, default=0.0)

    # Viz
    ap.add_argument("--plot", action="store_true", default=True)

    args = ap.parse_args()
    device = args.device

    # Load files
    trial_dir = Path(args.data_dir) / args.subject / args.trial
    fpath = trial_dir / "forces.npy"
    jpath = trial_dir / "joints.npy"
    if not fpath.exists() or not jpath.exists():
        raise FileNotFoundError(f"Missing files: {fpath} or {jpath}")

    forces = np.load(fpath)   # [T, F] in original units
    joints = np.load(jpath)   # [T, J] in original units
    print(f"Loaded: forces {forces.shape}, joints {joints.shape}")

    infer = DiffusionInference(
        ckpt_path=args.ckpt_path,
        scaler_dir=args.scaler_dir,
        device=device,
        sample_steps=args.sample_steps,
        eta=args.eta
    )

    # Run inference in requested mode
    if args.mode == "sliding-last-step":
        y_true, y_pred = infer.run_sliding_last_step(forces, joints, seq_len=args.seq_len)
        title = f"Diffusion | Sliding last-step (L={args.seq_len}, stride=1)"
    else:
        y_true, y_pred = infer.run_batch_seq(forces, joints, seq_len=args.seq_len, stride=args.stride)
        title = f"Diffusion | Batch sequences (L={args.seq_len}, stride={args.stride})"

    print(f"Aligned arrays: y_true {y_true.shape} | y_pred {y_pred.shape}")

    # Metrics
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, degrees=True)

    # Plot
    if args.plot and y_true.shape[0] > 0:
        fig = plot_sequence(y_true, y_pred, title=title, in_degrees=True)
        plt.show()


if __name__ == "__main__":
    main()
