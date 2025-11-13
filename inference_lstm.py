"""
Inference and Visualization Script (LSTM/BiLSTM)
- Loads a trained model checkpoint (with stored hparams)
- Uses the shared scalers from utils.py (same as training)
- Supports two modes:
  1) sliding-last-step: predict the last frame for every step with stride=1
  2) batch-seq: predict full sequences using the same seq_len/stride as training (optional)
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from train_lstm import ForcesTojointsLSTM
from loader_utils import slice_trial_into_sequences
from inference_utils import *



# ==========================
# Inference helper
# ==========================
class LSTMInference:
    def __init__(self,
                 model_path: str,
                 scaler_dir: str = "./scalers",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 normalize_forces: bool = True):
        """
        Args:
            model_path: path to the LSTM/BiLSTM checkpoint saved by training script
            scaler_dir: directory where utils' shared scalers were saved
            device: cpu / cuda
            normalize_forces: set True if training used normalized forces
        """
        self.device = device
        self.model, self.hparams = self._load_model(model_path)
        self.force_scaler = self._load_force_scaler(scaler_dir) if normalize_forces else None
        self.normalize_forces = normalize_forces

        self.joint_names = JOINT_NAMES
        self.force_names = FORCE_NAMES

    # --------- IO ----------
    def _load_model(self, model_path: str) -> Tuple[nn.Module, Dict]:
        ckpt = torch.load(model_path, map_location=self.device)

        # Retrieve hparams from checkpoint (stored in training)
        hp = ckpt.get('model_hparams', {})
        print(hp)
        # Backward-compatible defaults
        # hp.setdefault('input_size', 12)
        # hp.setdefault('hidden_size', 128)
        # hp.setdefault('output_size', 12)
        # hp.setdefault('num_layers', 2)
        # hp.setdefault('dropout', 0.2)
        hp.setdefault('bidirectional', False)  # if you added this flag later

        model = ForcesTojointsLSTM(**hp)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(self.device).eval()

        epoch = ckpt.get('epoch', 'N/A')
        val_loss = ckpt.get('val_loss', 'N/A')
        print(f"Loaded model @epoch={epoch}, val_loss={val_loss}, hparams={hp}")
        return model, hp

    def _load_force_scaler(self, scaler_dir: str):
        scaler_dir = Path(scaler_dir)
        with open(scaler_dir / "force_scaler.pkl", "rb") as f:
            force_scaler = pickle.load(f)
        print(f"Loaded force scaler from: {scaler_dir/'force_scaler.pkl'}")
        return force_scaler

    # --------- core preds ----------
    @torch.no_grad()
    def predict_sequence(self, forces_seq: np.ndarray) -> np.ndarray:
        """
        Predict joints for one full window (L, F) -> (L, J).
        Uses the model in a single forward pass.
        """
        x = forces_seq.astype(np.float32)
        if self.normalize_forces and self.force_scaler is not None:
            x = self.force_scaler.transform(x)

        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)  # [1, L, F]
        yt = self.model(xt)                                    # [1, L, J]
        y = yt.squeeze(0).cpu().numpy()                        # [L, J]
        return y

    @torch.no_grad()
    def predict_last_frame(self, forces_window: np.ndarray) -> np.ndarray:
        """
        Predict only the last frame from a window (L, F) -> (J,)
        """
        x = forces_window.astype(np.float32)
        if self.normalize_forces and self.force_scaler is not None:
            x = self.force_scaler.transform(x)

        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)  # [1, L, F]
        yt = self.model(xt)                                    # [1, L, J]
        y_last = yt[:, -1, :].squeeze(0).cpu().numpy()         # [J]
        return y_last

    # --------- end-to-end helpers ----------
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
        T = min(len(forces), len(joints))
        forces = forces[:T]
        joints = joints[:T]

        preds = []
        for i in range(seq_len - 1, T):
            window = forces[i - seq_len + 1 : i + 1]     # [L, F]
            y_last = self.predict_last_frame(window)     # [J]
            preds.append(y_last)
        y_pred = np.vstack(preds)                        # [T-L+1, J]
        y_true = joints[seq_len - 1 : ]                  # [T-L+1, J]
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


# ==========================
# CLI
# ==========================
def main():
    ap = argparse.ArgumentParser(description="Inference for LSTM/BiLSTM using shared utils")
    ap.add_argument("--model_path", type=str, default="./checkpoints_lstm_local/best_model.pth")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--subject", type=str, required=True)
    ap.add_argument("--trial", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scaler_dir", type=str, default="./scalers_lstm_local")
    ap.add_argument("--normalize_forces", action="store_true", default=True)

    # How to infer
    ap.add_argument("--mode", type=str, choices=["sliding-last-step", "batch-seq"], default="sliding-last-step")
    ap.add_argument("--seq_len", type=int, default=10, help="Window length used at training")
    ap.add_argument("--stride", type=int, default=1,  help="Stride for batch-seq mode (use training stride); ignored for sliding-last-step")
    ap.add_argument("--plot", action="store_true",  default=True, help="Show prediction plot")

    args = ap.parse_args()

    # Load files
    trial_dir = Path(args.data_dir) / args.subject / args.trial
    fpath = trial_dir / "forces.npy"
    jpath = trial_dir / "joints.npy"
    if not fpath.exists() or not jpath.exists():
        raise FileNotFoundError(f"Missing files: {fpath} or {jpath}")

    forces = np.load(fpath)   # [T, F]
    joints = np.load(jpath)   # [T, J]
    print(f"Loaded: forces {forces.shape}, joints {joints.shape}")

    # Inference
    infer = LSTMInference(
        model_path=args.model_path,
        scaler_dir=args.scaler_dir,
        device=args.device,
        normalize_forces=args.normalize_forces
    )

    if args.mode == "sliding-last-step":
        y_true, y_pred = infer.run_sliding_last_step(forces, joints, seq_len=args.seq_len)
        title = f"Sliding last-step (L={args.seq_len}, stride=1)"
    else:
        y_true, y_pred = infer.run_batch_seq(forces, joints, seq_len=args.seq_len, stride=args.stride)
        title = f"Batch sequences (L={args.seq_len}, stride={args.stride})"

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
