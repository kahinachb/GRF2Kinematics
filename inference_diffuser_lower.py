import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from utils.diffuser_utils import DDPM, DiffusionTransformer
import torch.nn as nn


# ==========================================
# 2. INFERENCE FUNCTION
# ==========================================

def predict_full_trial(model, ddpm, f_path, j_path, stats, device, 
                       window_size=128, stride=64, inference_steps=50):
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

    j_raw = j_raw[:, 6:18]
    # cols = list(range(6)) + list(range(9, 15))
    # f_raw = f_raw[:,cols]

    
    
    # Normalize forces
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)
    
    T = f_norm.shape[0]
    full_pred = torch.zeros((T, 12)).to(device)
    count_map = torch.zeros((T, 12)).to(device)

    print(f"  [INFO] Predicting trial with {T} frames...")
    print(f"  [INFO] Using window_size={window_size}, stride={stride}")
    
    num_windows = 0
    for start in range(0, T - window_size, stride):
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        
        # Start from pure noise
        curr_j = torch.randn((1, window_size, 12)).to(device)
        
        # Reverse diffusion
        step_size = ddpm.n_steps // inference_steps
        
        for t_idx in reversed(range(0, ddpm.n_steps, step_size)):
            with torch.no_grad():
                curr_j = ddpm.sample_reverse(model, curr_j, t_idx, f_win)
        
        # Accumulate predictions
        full_pred[start:end] += curr_j.squeeze(0)
        count_map[start:end] += 1.0
        num_windows += 1
    
    print(f"  [INFO] Processed {num_windows} windows")
    
    # Average overlapping predictions
    final_pred = full_pred / torch.clamp(count_map, min=1.0)
    
    # Denormalize
    final_pred = (final_pred.cpu() * stats['j_s']) + stats['j_m']
    
    return j_raw, final_pred.numpy()

class DiffusionTransformer(nn.Module):
    def __init__(self, joint_dim=12, force_dim=18, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
        self.joint_embed = nn.Linear(joint_dim, embed_dim) #input embeddings
        self.force_embed = nn.Linear(force_dim, embed_dim)
        self.time_embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)) #time embedding, Encodes the diffusion timestep 
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, joint_dim)

    def forward(self, x, t, cond):
        t_emb = self.time_embed(t.view(-1, 1)).unsqueeze(1)
        x_emb = self.joint_embed(x) + self.force_embed(cond) + t_emb #All information is blended into the same 256-dimensional space
        return self.output_layer(self.transformer(x_emb))


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
    
    f_path = data_path / "kinetics.npy"
    j_path = data_path / "all_joints.npy"
    
    if not f_path.exists():
        raise FileNotFoundError(f"Forces file not found: {f_path}")
    if not j_path.exists():
        raise FileNotFoundError(f"Joints file not found: {j_path}")
    
    print(f"  ✓ Found forces: {f_path}")
    print(f"  ✓ Found joints: {j_path}")
    
    # ===== 5. RUN INFERENCE =====
    print("\n[5/5] Running inference...")
    j_ref, j_pred = predict_full_trial(
        model, ddpm, f_path, j_path, stats, device,
        window_size=128, stride=64, inference_steps=1000
    )
    
    # ===== 6. SAVE RESULTS =====
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    save_name = f"{subject_name}_{trial_name}_prediction.npy"
    np.save(output_path / save_name, j_pred)
    print(f"\n  ✓ Saved predictions to: {output_path / save_name}")
    
    # ===== 7. VISUALIZE =====
    print("\n[6/6] Creating visualization...")
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    
    for i, ax in enumerate(axes.flatten()):
        ax.plot(j_ref[:, i], 'k--', alpha=0.6, linewidth=1.5, label='Ground Truth')
        ax.plot(j_pred[:, i], 'r', linewidth=1.5, label='Prediction')
        ax.set_title(f"Joint {i+1}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')
    
    plt.suptitle(f"Inference: {subject_name} / {trial_name}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_name = f"{subject_name}_{trial_name}_comparison.png"
    plt.savefig(output_path / plot_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot to: {output_path / plot_name}")
    
    # ===== 8. COMPUTE METRICS =====
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    mse = np.mean((j_ref - j_pred)**2)
    mae = np.mean(np.abs(j_ref - j_pred))
    rmse = np.sqrt(mse)
    
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    # Per-joint errors
    print("\nPer-joint RMSE:")
    for i in range(12):
        joint_rmse = np.sqrt(np.mean((j_ref[:, i] - j_pred[:, i])**2))
        print(f"  Joint {i+1:2d}: {joint_rmse:.6f}")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE!")
    print("="*60 + "\n")


# ==========================================
# 4. EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    SUBJECT_NAME = "npy_synth"     
    TRIAL_NAME = "Trial111"           
    
    MODEL_PATH = "./results_feet/diffusion_biomech_model_concat.pth"
    SCALERS_PATH = "./results_feet/scalers_concat.json"
    # DATA_ROOT = "./processed_data_global"
    DATA_ROOT = "./DATA/synth"
    OUTPUT_DIR = "./inference_results_feet"
    
    # ===== RUN INFERENCE =====
    run_inference(
        subject_name=SUBJECT_NAME,
        trial_name=TRIAL_NAME,
        model_path=MODEL_PATH,
        scalers_path=SCALERS_PATH,
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR
    )