import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from utils.diffuser_utils import DDPM
import torch.nn as nn


# ==========================================
# 2. INFERENCE FUNCTION
# ==========================================

def predict_full_trial(model, ddpm, f_path, j_path, stats, device, 
                       window_size=128, stride=64, inference_steps=50,n_samples=1):
    """
    Generate predictions for a full trial using sliding windows.
    
    Args:
        model: Trained DiffusionTransformer
        ddpm: DDPM object
        f_path: Path to forces.npy file
        j_path: Path to joints.npy file (ground truth for comparison)
        stats: Dictionary with normalization statistics
        device: torch device
        window_size: Size of prediction window (default 128)
        stride: Step size between windows (default 64)
        inference_steps: Number of denoising steps (default 50)
    
    Returns:
        j_raw: Ground truth joint angles
        pred_full: Predicted joint angles
    """
    model.eval()
    
    # Load data
    f_raw = np.load(f_path).astype(np.float32)
    j_raw = np.load(j_path).astype(np.float32)

    j_raw = j_raw[:, :]
 
    # Normalize forces
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)
    
    T = f_norm.shape[0]
    full_pred = torch.zeros((T, 35)).to(device)
    count_map = torch.zeros((T, 35)).to(device)

    print(f"  [INFO] Predicting trial with {T} frames...")
    print(f"  [INFO] Using window_size={window_size}, stride={stride}")
    
    num_windows = 0
    all_preds = []

    for s in range(n_samples):
        full_pred = torch.zeros((T, 35)).to(device)
        count_map = torch.zeros((T, 35)).to(device)

        for start in range(0, T - window_size, stride):
            end = start + window_size
            f_win = f_norm[start:end].unsqueeze(0).to(device)

            curr_j = torch.randn((1, window_size, 35)).to(device)  # 🔥 différent à chaque sample

            step_size = ddpm.n_steps // inference_steps

            for t_idx in reversed(range(0, ddpm.n_steps, step_size)):
                with torch.no_grad():
                    curr_j = ddpm.sample_reverse_selfs(model, curr_j, t_idx, f_win)

            full_pred[start:end] += curr_j.squeeze(0)
            count_map[start:end] += 1.0

        final_pred = full_pred / torch.clamp(count_map, min=1.0)
        final_pred = (final_pred.cpu() * stats['j_s']) + stats['j_m']

        all_preds.append(final_pred.numpy())
    
    return j_raw, np.stack(all_preds)

# ==========================================
# 2. ARCHITECTURE
# ==========================================
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ---> ADDED: Sinusoidal Timestep Embedding Class <---
class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: 1D tensor of shape (batch_size,)
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionTransformer(nn.Module):
    def __init__(self, joint_dim=35, force_dim=18, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
        self.joint_embed = nn.Linear(joint_dim, embed_dim) 
        self.force_embed = nn.Linear(force_dim, embed_dim)
        
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ) 
        
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=1000) 
        
        # ---> CHANGED: Encoder to Decoder for Cross-Attention <---
        layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(embed_dim, joint_dim)

    def forward(self, x, t, cond):
        # 1. Safeguard: Ensure t is a 1D tensor (handles inference loop integers)
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=torch.long).expand(x.shape[0])
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # 2. Process Time Embedding
        t_emb = self.time_mlp(t).unsqueeze(1) 
        
        # 3. Process Target Sequence (Noisy Joints + Time + Position)
        x_emb = self.joint_embed(x) + t_emb 
        x_emb = self.pos_encoder(x_emb) 
        
        # 4. Process Memory Sequence (Forces + Position)
        # Note: Time-series forces also need positional awareness!
        cond_emb = self.force_embed(cond)
        cond_emb = self.pos_encoder(cond_emb)
        
        # 5. Transformer Decoder (tgt=Joints, memory=Forces)
        out = self.transformer(tgt=x_emb, memory=cond_emb)
        
        return self.output_layer(out)


# ==========================================
# 3. MAIN INFERENCE SCRIPT
# ==========================================

def run_inference(subject_name, trial_name, model_path, scalers_path, 
                  data_root="./processed_data", output_dir="./inference_results"):
    """
    Run inference on a specific subject and trial.
    
    Args:
        subject_name: e.g., "Subject01" or "Squat_01"
        trial_name: e.g., "task1" or "trial1"
        model_path: Path to saved .pth model
        scalers_path: Path to scalers.json
        data_root: Root directory containing processed data
        output_dir: Where to save results
    """
    output_dir = Path(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"INFERENCE ON: {subject_name} / {trial_name}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # ===== 1. LOAD SCALERS =====
    print("[1/5] Loading scalers...")
    with open(scalers_path, 'r') as f:
        scalers_dict = json.load(f)
    
    stats = {k: torch.tensor(v).float() for k, v in scalers_dict.items()}
    print(f"  ✓ Loaded normalization statistics")
    
    # ===== 2. LOAD MODEL =====
    print("\n[2/5] Loading model...")
    model = DiffusionTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  ✓ Loaded model from {model_path}")
    
    # ===== 3. INITIALIZE DDPM =====
    print("\n[3/5] Initializing DDPM...")
    ddpm = DDPM(device, n_steps=1000)
    print(f"  ✓ DDPM initialized with {ddpm.n_steps} steps")
    
    # ===== 4. LOCATE DATA FILES =====
    print("\n[4/5] Locating data files...")
    # data_path = Path(data_root) / subject_name / trial_name
    data_path = Path(data_root) / subject_name / f"{trial_name}"
    
    f_path = data_path / "kinetics_feet.npy"
    j_path = data_path / "all_joints.npy"
    
    if not f_path.exists():
        raise FileNotFoundError(f"Forces file not found: {f_path}")
    if not j_path.exists():
        raise FileNotFoundError(f"Joints file not found: {j_path}")
    
    print(f"  ✓ Found forces: {f_path}")
    print(f"  ✓ Found joints: {j_path}")
    
    # ===== 5. RUN INFERENCE =====
    print("\n[5/5] Running inference...")
    j_ref, j_preds = predict_full_trial(
        model, ddpm, f_path, j_path, stats, device,
        window_size=128, stride=64, inference_steps=1000,
        n_samples=10   
    )

    # ===== BEST SAMPLE SELECTION =====
    print("\n[6/7] Selecting best sample...")

    mse_samples = []
    rmse_samples = []

    for s in range(j_preds.shape[0]):
        mse_s = np.mean((j_ref - j_preds[s])**2)
        rmse_s = np.sqrt(mse_s)
        mse_samples.append(mse_s)
        rmse_samples.append(rmse_s)

    best_idx = np.argmin(mse_samples)

    j_pred_best = j_preds[best_idx]

    print(f"Best sample: {best_idx}")
    print(f"Best MSE: {mse_samples[best_idx]:.6f}")
    print(f"Best RMSE: {rmse_samples[best_idx]:.6f}")
        
    
    # ===== 7. VISUALIZE =====
    fig, axes = plt.subplots(7, 1, figsize=(14, 18))

    for i in range(0, 6):
        ax = axes[i]
        
        ax.plot(j_ref[:, i], 'k--', label='GT', alpha=0.6)

        for s in range(j_preds.shape[0]):
            ax.plot(j_preds[s, :, i], alpha=0.2)

        ax.plot(j_pred_best[:, i], 'r', linewidth=2, label='Best')

        ax.set_title(f"Joint {i}")
        ax.grid(True)

    axes[0].legend()
    plt.suptitle("Freeflyer joints (0–5)")
    plt.tight_layout()
    plt.savefig(output_dir / "freeflyer_joints.png")
    plt.close()

    fig, axes = plt.subplots(12, 1, figsize=(14, 20))

    for idx, i in enumerate(range(6, 18)):
        ax = axes[idx]

        ax.plot(j_ref[:, i], 'k--', alpha=0.6)

        for s in range(j_preds.shape[0]):
            ax.plot(j_preds[s, :, i], alpha=0.15)

        ax.plot(j_pred_best[:, i], 'r', linewidth=2)

        ax.set_title(f"Joint {i}")
        ax.grid(True)

    plt.suptitle("Lower body joints (6–17)")
    plt.tight_layout()
    plt.savefig(output_dir / "lower_body_joints.png")
    plt.close()

    fig, axes = plt.subplots(17, 1, figsize=(14, 22))

    for idx, i in enumerate(range(18, 35)):
        ax = axes[idx]

        ax.plot(j_ref[:, i], 'k--', alpha=0.6)

        for s in range(j_preds.shape[0]):
            ax.plot(j_preds[s, :, i], alpha=0.15)

        ax.plot(j_pred_best[:, i], 'r', linewidth=2)

        ax.set_title(f"Joint {i}")
        ax.grid(True)

    plt.suptitle("Upper body joints (18–34)")
    plt.tight_layout()
    plt.savefig(output_dir / "upper_body_joints.png")
    plt.close()
    # plt.show()


# ==========================================
# 4. EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    SUBJECT_NAME = "Jeremy"     
    TRIAL_NAME = "Trial111"           
          
    
    MODEL_PATH = "./training_res/results_full_ff_real/diffusion_biomech_model_concat.pth"
    SCALERS_PATH = "./training_res/results_full_ff_real/scalers_concat.json"
    DATA_ROOT = "/datasets/GRF2Kine/processed_data_feet"
    OUTPUT_DIR = "./inference_results_transfo_full_ff"
    
    # ===== RUN INFERENCE =====
    run_inference(
        subject_name=SUBJECT_NAME,
        trial_name=TRIAL_NAME,
        model_path=MODEL_PATH,
        scalers_path=SCALERS_PATH,
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR
    )