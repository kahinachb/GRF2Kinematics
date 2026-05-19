import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch.nn as nn
import math

# ==========================================
# 1. CFM INFERENCE FUNCTION (SLIDING WINDOW)
# ==========================================

def predict_full_trial(model, f_path, j_path, stats, device, 
                       window_size=128, stride=64, inference_steps=25):
    """
    Generate predictions for a full trial using Flow Matching (Euler ODE integration).
    
    Args:
        model: Trained CFM DiffusionTransformer
        f_path: Path to kinetics.npy file
        j_path: Path to all_joints.npy file (ground truth for comparison)
        stats: Dictionary with normalization statistics
        device: torch device
        window_size: Size of prediction window (default 128)
        stride: Step size between windows (default 64)
        inference_steps: Number of Euler integration steps (default 25 is ideal for CFM)
    
    Returns:
        j_raw: Ground truth joint angles
        pred_full: Predicted joint angles
    """
    model.eval()
    
    # Load data
    f_raw = np.load(f_path).astype(np.float32)
    j_raw = np.load(j_path).astype(np.float32)

    # Focus unique sur les 12 DOFs du bas du corps
    j_raw = j_raw[:, 7:19]
 
    # Normalize forces
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)
    
    T = f_norm.shape[0]
    full_pred = torch.zeros((T, 12)).to(device)
    count_map = torch.zeros((T, 12)).to(device)

    print(f"  [INFO] Predicting trial with {T} frames using Flow Matching...")
    print(f"  [INFO] Window_size={window_size}, stride={stride}, CFM Euler steps={inference_steps}")
    
    dt = 1.0 / inference_steps
    num_windows = 0
    
    for start in range(0, T - window_size, stride):
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        
        # 1. Départ depuis un bruit Gaussien pur (x_0) à t = 0
        curr_x = torch.randn((1, window_size, 12)).to(device)
        
        # 2. Intégration d'Euler (Inférence CFM)
        for i in range(inference_steps):
            t_val = i / inference_steps
            # Crucial : t doit être un float32 continu entre 0 et 1
            t = torch.ones((1,), device=device, dtype=torch.float32) * t_val
            
            with torch.no_grad():
                # Le modèle prédit la vitesse (v)
                v = model(curr_x, t, f_win)
                
            # x_{t+dt} = x_t + v * dt
            curr_x = curr_x + v * dt
        
        # Accumulate predictions
        full_pred[start:end] += curr_x.squeeze(0)
        count_map[start:end] += 1.0
        num_windows += 1
    
    print(f"  [INFO] Processed {num_windows} windows")
    
    # Average overlapping predictions
    final_pred = full_pred / torch.clamp(count_map, min=1.0)
    
    # Denormalize
    final_pred = (final_pred.cpu() * stats['j_s']) + stats['j_m']
    
    return j_raw, final_pred.numpy()


# ==========================================
# 2. ARCHITECTURE (Compatible Flow Matching)
# ==========================================

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


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DiffusionTransformer(nn.Module):
    def __init__(self, joint_dim=12, force_dim=18, embed_dim=256, nhead=8, num_layers=4):
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
        layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, joint_dim)

    def forward(self, x, t, cond):
        # --- FIX POUR LE FLOW MATCHING : t reste un float32 ---
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=torch.float32).expand(x.shape[0])
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # On "étire" t (0->1 devient 0->1000) pour l'embedding sinusoidal classique
        t_input = t * 1000.0
        t_emb = self.time_mlp(t_input).unsqueeze(1) 
        
        x_emb = self.joint_embed(x) + t_emb 
        x_emb = self.pos_encoder(x_emb) 
        
        cond_emb = self.force_embed(cond)
        cond_emb = self.pos_encoder(cond_emb)
        
        out = self.transformer(tgt=x_emb, memory=cond_emb)
        return self.output_layer(out)


# ==========================================
# 3. MAIN INFERENCE SCRIPT
# ==========================================

def run_inference(subject_name, trial_name, model_path, scalers_path, 
                  data_root="./processed_data", output_dir="./inference_results"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"CFM INFERENCE ON: {subject_name} / {trial_name}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # ===== 1. LOAD SCALERS =====
    print("[1/4] Loading scalers...")
    with open(scalers_path, 'r') as f:
        scalers_dict = json.load(f)
    stats = {k: torch.tensor(v).float() for k, v in scalers_dict.items()}
    print(f"  ✓ Loaded normalization statistics")
    
    # ===== 2. LOAD MODEL =====
    print("\n[2/4] Loading CFM model...")
    model = DiffusionTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  ✓ Loaded model from {model_path}")
    
    # ===== 3. LOCATE DATA FILES =====
    print("\n[3/4] Locating data files...")
    data_path = Path(data_root) / subject_name / f"{trial_name}"
    f_path = data_path / "kinetics.npy"
    j_path = data_path / "all_joints.npy"
    
    if not f_path.exists(): raise FileNotFoundError(f"Forces file not found: {f_path}")
    if not j_path.exists(): raise FileNotFoundError(f"Joints file not found: {j_path}")
    print(f"  ✓ Found forces: {f_path}\n  ✓ Found joints: {j_path}")
    
    # ===== 4. RUN INFERENCE =====
    print("\n[4/4] Running CFM inference...")
    # Changement majeur : inference_steps=25 au lieu de 1000 (Gain de vitesse x40 !)
    j_ref, j_pred = predict_full_trial(
        model, f_path, j_path, stats, device,
        window_size=128, stride=64, inference_steps=25
    )
    
    # ===== 5. SAVE RESULTS =====
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_name = f"{subject_name}_{trial_name}_cfm_prediction.npy"
    np.save(output_path / save_name, j_pred)
    print(f"\n  ✓ Saved predictions to: {output_path / save_name}")
    
    # ===== 6. VISUALIZE =====
    print("\nCreating visualization...")
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    
    for i, ax in enumerate(axes.flatten()):
        ax.plot(j_ref[:, i], 'k--', alpha=0.6, linewidth=1.5, label='Ground Truth')
        ax.plot(j_pred[:, i], 'r', linewidth=1.5, label='CFM Prediction')
        ax.set_title(f"Joint {i+1}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='upper right')
    
    plt.suptitle(f"Conditional Flow Matching Inference: {subject_name} / {trial_name}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_name = f"{subject_name}_{trial_name}_cfm_comparison.png"
    plt.savefig(output_path / plot_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot to: {output_path / plot_name}")
    
    # ===== 7. COMPUTE METRICS =====
    print("\n" + "="*60)
    print("RESULTS SUMMARY (CFM)")
    print("="*60)
    
    mse = np.mean((j_ref - j_pred)**2)
    mae = np.mean(np.abs(j_ref - j_pred))
    rmse = np.sqrt(mse)
    
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    print("\nPer-joint RMSE:")
    for i in range(12):
        joint_rmse = np.sqrt(np.mean((j_ref[:, i] - j_pred[:, i])**2))
        print(f"  Joint {i+1:2d}: {joint_rmse:.6f}")
    print("\n" + "="*60)


# ==========================================
# 4. RUN EXEMPLE
# ==========================================

if __name__ == "__main__":
    
    SUBJECT_NAME = "s1"     
    TRIAL_NAME = "squat_variant_980_dz-0.080_dx+0.023_dy-0.017"     

    MODEL_PATH = "./results_FM/cfm_best_model.pth"
    SCALERS_PATH = "./results_FM/scalers_concat.json"
    DATA_ROOT = "/datasets/GRF2Kine/synth_npy"
    OUTPUT_DIR = "./inference_results_CFM"
    
    run_inference(
        subject_name=SUBJECT_NAME,
        trial_name=TRIAL_NAME,
        model_path=MODEL_PATH,
        scalers_path=SCALERS_PATH,
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR
    )