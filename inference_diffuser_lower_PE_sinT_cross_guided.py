import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from utils.diffuser_utils import DDPM
import torch.nn as nn
import pandas as pd
import pytorch_kinematics as pk
import example_robot_data as robex
import re
# the EXACT link names from URDF that represent the feet
RIGHT_FOOT_LINK = "right_foot" 
LEFT_FOOT_LINK = "left_foot"   

def compute_physics_energy_basic(chain, n_pk, index_mapping, pred_joints_norm, ff_tensor, forces, stats, device):
    # 1. Denormalize joints
    j_s = stats['j_s'][:].to(device) 
    j_m = stats['j_m'][:].to(device)
    pred_joints_raw = (pred_joints_norm * j_s) + j_m
    
    batch_size, seq_len, _ = pred_joints_raw.shape
    
    # 2. Map 12 DDPM joints to full URDF size
    q_pk = torch.zeros((batch_size, seq_len, n_pk), device=device, dtype=torch.float32)
    for ddpm_idx, pk_idx in index_mapping:
        q_pk[:, :, pk_idx] = pred_joints_raw[:, :, ddpm_idx]
        
    q_pk_flat = q_pk.view(-1, n_pk)
    
    # 3. Apply FreeFlyer (w, x, y, z order for PK)
    ff_flat = ff_tensor.view(-1, 7)
    pos_base = ff_flat[:, :3]
    quat_xyzw = ff_flat[:, 3:]
    quat_wxyz = torch.cat([quat_xyzw[:, 3:], quat_xyzw[:, :3]], dim=1)
    
    base_transform = pk.Transform3d(pos=pos_base, rot=quat_wxyz, device=device)
    
    # 4. Forward Kinematics
    relative_transforms = chain.forward_kinematics(q_pk_flat)
    
    r_world = base_transform.compose(relative_transforms[RIGHT_FOOT_LINK])
    l_world = base_transform.compose(relative_transforms[LEFT_FOOT_LINK])
    
    r_foot_pos = r_world.get_matrix()[:, :3, 3].view(batch_size, seq_len, 3)
    l_foot_pos = l_world.get_matrix()[:, :3, 3].view(batch_size, seq_len, 3)
    
    # 5. Penalties
    # Sinking Loss (Z < 0)

    floor_offset = 0.08
    r_sink = torch.nn.functional.relu(-(r_foot_pos[:, :, 2] - floor_offset)) ** 2
    l_sink = torch.nn.functional.relu(-(l_foot_pos[:, :, 2] - floor_offset)) ** 2
    loss_sink = r_sink.mean() + l_sink.mean()
    
    # Skating Loss (Velocity in X and Y ONLY)
    # Notice the :2 slicing at the end to grab only X and Y coordinates!
    r_vel_xy = r_foot_pos[:, 1:, :2] - r_foot_pos[:, :-1, :2]
    l_vel_xy = l_foot_pos[:, 1:, :2] - l_foot_pos[:, :-1, :2]
    
    # ---> IMPORTANT: Update these indices to match Fz in your forces array! <---
    # Assuming forces shape is [batch, seq, 18]. 
    FZ_RIGHT_IDX = 2 
    FZ_LEFT_IDX = 11
    
    # Grab denormalized forces to check the 20N threshold
    f_s = stats['f_s'].to(device)
    f_m = stats['f_m'].to(device)
    forces_raw = (forces * f_s) + f_m
    
    r_fz = forces_raw[:, :-1, FZ_RIGHT_IDX]
    l_fz = forces_raw[:, :-1, FZ_LEFT_IDX]
    
    r_contact = (r_fz > 5.0).float()
    l_contact = (l_fz > 5.0).float()
    
    # Calculate skating loss using only the 2D velocity
    r_skate = (r_vel_xy ** 2).sum(dim=-1) * r_contact
    l_skate = (l_vel_xy ** 2).sum(dim=-1) * l_contact

    loss_skate = r_skate.mean() + l_skate.mean()
    total_loss = loss_skate + loss_sink
    return total_loss, loss_skate, loss_sink

def compute_physics_energy(chain, n_pk, index_mapping, pred_joints_norm, ff_tensor, forces, stats, device):
    # 1. Denormalize joints
    j_s = stats['j_s'][:12].to(device) # Only the 12 joint scalers
    j_m = stats['j_m'][:12].to(device)
    pred_joints_raw = (pred_joints_norm * j_s) + j_m
    
    batch_size, seq_len, _ = pred_joints_raw.shape
    
    # 2. Map 12 DDPM joints to full URDF size
    q_pk = torch.zeros((batch_size, seq_len, n_pk), device=device, dtype=torch.float32)
    for ddpm_idx, pk_idx in index_mapping:
        q_pk[:, :, pk_idx] = pred_joints_raw[:, :, ddpm_idx]
        
    q_pk_flat = q_pk.view(-1, n_pk)
    
    # 3. Apply FreeFlyer (w, x, y, z order for PK)
    ff_flat = ff_tensor.view(-1, 7)
    pos_base = ff_flat[:, :3]
    quat_xyzw = ff_flat[:, 3:]
    quat_wxyz = torch.cat([quat_xyzw[:, 3:], quat_xyzw[:, :3]], dim=1)
    
    base_transform = pk.Transform3d(pos=pos_base, rot=quat_wxyz, device=device)
    
    # 4. Forward Kinematics
    relative_transforms = chain.forward_kinematics(q_pk_flat)
    
    r_world = base_transform.compose(relative_transforms[RIGHT_FOOT_LINK])
    l_world = base_transform.compose(relative_transforms[LEFT_FOOT_LINK])
    
    r_foot_pos = r_world.get_matrix()[:, :3, 3].view(batch_size, seq_len, 3)
    l_foot_pos = l_world.get_matrix()[:, :3, 3].view(batch_size, seq_len, 3)
    
    # 5. Penalties
    # ---> EXTRACT FORCES & CONTACT MASKS <---
    FZ_RIGHT_IDX = 2 
    FZ_LEFT_IDX = 11 
    
    f_s = stats['f_s'].to(device)
    f_m = stats['f_m'].to(device)
    forces_raw = (forces * f_s) + f_m
    
    r_fz = forces_raw[:, :-1, FZ_RIGHT_IDX]
    l_fz = forces_raw[:, :-1, FZ_LEFT_IDX]
    
    # We pad the masks to match the seq_len (forces are 127, joints are 128)
    r_contact = torch.cat([r_fz > 20.0, (r_fz[:, -1:] > 20.0)], dim=1).float()
    l_contact = torch.cat([l_fz > 20.0, (l_fz[:, -1:] > 20.0)], dim=1).float()
    
    # Expand masks to match XYZ dimensions: [batch, seq_len, 3]
    r_contact_mask = r_contact.unsqueeze(-1)
    l_contact_mask = l_contact.unsqueeze(-1)

    # ---> 1. THE ABSOLUTE ANCHOR PENALTY (X, Y, Z) <---
    # Find the mean X, Y, Z position of the foot ONLY when it is planted
    # (We add 1e-6 to avoid dividing by zero if the foot is completely in the air)
    r_mean_pos = (r_foot_pos * r_contact_mask).sum(dim=1, keepdim=True) / (r_contact_mask.sum(dim=1, keepdim=True) + 1e-6)
    l_mean_pos = (l_foot_pos * l_contact_mask).sum(dim=1, keepdim=True) / (l_contact_mask.sum(dim=1, keepdim=True) + 1e-6)
    
    # Penalize the squared distance between the current position and the anchor position
    # The mask ensures we ONLY penalize the foot if Fz > 20N
    r_anchor_loss = (((r_foot_pos - r_mean_pos) ** 2) * r_contact_mask).mean()
    l_anchor_loss = (((l_foot_pos - l_mean_pos) ** 2) * l_contact_mask).mean()
    loss_anchor = r_anchor_loss + l_anchor_loss

    # ---> 2. SINKING PENALTY (Z < 0) <---
    # We still need this! If the foot is in the air (Fz < 20N), the anchor loss is disabled.
    # But we still don't want the airborne foot clipping through the floor!
    floor_offset = 0.00 
    r_sink = torch.nn.functional.relu(-(r_foot_pos[:, :, 2] - floor_offset)) ** 2
    l_sink = torch.nn.functional.relu(-(l_foot_pos[:, :, 2] - floor_offset)) ** 2
    loss_sink = r_sink.mean() + l_sink.mean()
    total_loss = loss_anchor + loss_sink
    return total_loss, loss_anchor, loss_sink

# ==========================================
# 2. INFERENCE FUNCTION
# ==========================================
def predict_full_trial(model, ddpm, f_path, j_path,j_path_FF, stats, device, 
                       chain, n_pk, index_mapping,
                       window_size=128, stride=64, inference_steps=50,
                       guidance_scale=0.1):      
    model.eval()
    
    f_raw = np.load(f_path).astype(np.float32)
    j_full_raw = np.load(j_path).astype(np.float32) # Load all columns first
    ff_raw = pd.read_csv(j_path_FF,skiprows=1).iloc[:,:7].to_numpy() #get freeflyer from csv file
    print(ff_raw.shape)
    # Extract  Joints (Cols 6 to 17)
    j_raw = j_full_raw[:, 6:18]
 
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)
    ff_tensor_full = torch.from_numpy(ff_raw).float().to(device) # Send FF to GPU
    
    T = f_norm.shape[0]
    full_pred = torch.zeros((T, 12)).to(device)
    count_map = torch.zeros((T, 12)).to(device)
    
    for start in range(0, T - window_size, stride):
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        ff_win = ff_tensor_full[start:end].unsqueeze(0) # ---> Get FF window
        
        curr_j = torch.randn((1, window_size, 12)).to(device)
        step_size = ddpm.n_steps // inference_steps
        
        # for t_idx in reversed(range(0, ddpm.n_steps, step_size)):
            
        #     # ---> PHYSICS GUIDANCE START <---
        #     curr_j = curr_j.detach().requires_grad_(True)
            
        #     with torch.enable_grad():
        #         # We calculate energy directly on the noisy state (a simplified Universal Guidance approach)
        #         # It tells the noise which direction to shift to respect the floor
        #         energy, l_skate,l_sink = compute_physics_energy_basic(
        #             chain, n_pk, index_mapping, curr_j, ff_win, f_win, stats, device
        #         )
        #         # if start == 0 and t_idx % 10 == 0: # Printing every 10 steps for a clean log
        #         #     print(f"  [Win 0] Step {t_idx:02d} | Total: {energy.item():.5f} | Anchor: {l_skate.item():.5f} | Sink: {l_sink.item():.5f}")
        #         # If energy > 0, calculate gradients
        #         if energy.item() > 0:
        #             grad = torch.autograd.grad(energy, curr_j)[0]
        #         else:
        #             grad = torch.zeros_like(curr_j)
                    
        #     with torch.no_grad():
        #         # Normal DDPM step
        #         curr_j = ddpm.sample_reverse_selfs(model, curr_j, t_idx, f_win)
        #         # Apply guidance push!
        #         curr_j = curr_j - (guidance_scale * grad)
            # ---> PHYSICS GUIDANCE END <---

        for t_idx in reversed(range(0, ddpm.n_steps, step_size)):
            
            # ---> PHYSICS GUIDANCE START <---
            curr_j = curr_j.detach().requires_grad_(True)
            
            with torch.enable_grad():
                # 1. Ask the model for its clean guess (x_0)
                pred_x0 = model(curr_j, t_idx, f_win)
                
                # 2. Calculate energy on the CLEAN GUESS, not the noise!
                energy, l_skate, l_sink = compute_physics_energy_basic(
                    chain, n_pk, index_mapping, pred_x0, ff_win, f_win, stats, device
                )
                
                # Optional: Print loss to track it
                # if start == 0 and t_idx % 10 == 0: 
                #     print(f"  [Win 0] Step {t_idx:02d} | Total: {energy.item():.5f}")
                
                if energy.item() > 0:
                    # 3. Calculate gradient w.r.t the noisy state (curr_j)
                    grad = torch.autograd.grad(energy, curr_j)[0]
                    
                    # 4. CRITICAL: Normalize the gradient to prevent explosion
                    grad_norm = torch.norm(grad, p=2, dim=-1, keepdim=True) + 1e-8
                    grad = grad / grad_norm
                else:
                    grad = torch.zeros_like(curr_j)
                    
            with torch.no_grad():
                # 5. ORDER OF OPERATIONS: Apply the guidance nudge FIRST
                curr_j_guided = curr_j - (guidance_scale * grad)
                
                # 6. Take the normal DDPM step using the nudged state
                curr_j = ddpm.sample_reverse_selfs(model, curr_j_guided, t_idx, f_win)
                
        full_pred[start:end] += curr_j.squeeze(0)
        count_map[start:end] += 1.0
        print(f"  [INFO] Processed windows")
    
    # Average overlapping predictions
    final_pred = full_pred / torch.clamp(count_map, min=1.0)
    
    # Denormalize (Make sure to only use the 12 lower limb scalers)
    final_pred = (final_pred.cpu() * stats['j_s'][:]) + stats['j_m'][:]
    
    return j_raw, final_pred.numpy()
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
    
    # j_path_FF= "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/generated_human_like_motions_csv_new/generated_human_like_motions_csv/joint_filtered_squat_variant_980_dz-0.080_dx+0.023_dy-0.017.csv"
    # j_path_FF= f"DATA/generated_human_like_motions_csv_new/generated_human_like_motions_csv/joints_filtered_{trial_name}.csv"

    j_path_FF= f"DATA/Vinc/{subject_name}/{trial_name}/joints_filtered_FF.csv"
    if not f_path.exists():
        raise FileNotFoundError(f"Forces file not found: {f_path}")
    if not j_path.exists():
        raise FileNotFoundError(f"Joints file not found: {j_path}")
    
    print(f"  ✓ Found forces: {f_path}")
    print(f"  ✓ Found joints: {j_path}")
    
    # ===== 5. RUN INFERENCE =====
   ### Setup PK Chain 
    print("\n[3.5/5] Initializing PyTorch Kinematics (Generic Human)...")
    
    # Load the clean generic human model directly from the library
    # human = robex.human.HumanLoader(height=1.55, weight=68, gender='male').robot
    
    # with open(human.urdf, 'r') as f:
    #     urdf_data = f.read()

    urdf_path=f"DATA/urdf_scaled/Vinc/{subject_name}_scaled.urdf"
    print(f"Loading URDF from: {urdf_path}")

    with open(urdf_path, 'r', encoding='utf-8') as f:
        urdf_data = f.read()

    # 1. On supprime la première ligne (la déclaration XML qui posait le 1er problème)
    urdf_data = re.sub(r'<\?xml[^>]+\?>', '', urdf_data)

    # 2. On supprime les balises <texture> mal formées (le 2ème problème)
    urdf_data = re.sub(r'<texture[^>]*>', '', urdf_data)

    # 3. On convertit la chaîne propre en bytes pour PyTorch Kinematics
    urdf_data_bytes = urdf_data.encode('utf-8')

        
    # Build the differentiable chain
    chain = pk.build_chain_from_urdf(urdf_data).to(device=device)
    
    pk_joint_names = chain.get_joint_parameter_names()
    n_pk = len(pk_joint_names)
    
    # Map 12 predicted joints to the URDF
    ddpm_joint_names = [
        "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot", "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",
        "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot", "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add"
    ]
    
    # Use dictionary mapping
    mapping = {
        'left_hip_Z': 'Lhip_flex_ext', 'left_hip_X': 'Lhip_abd_add', 'left_hip_Y': 'Lhip_int_ext_rot',
        'left_knee_Z': 'Lknee_flex_ext', 'left_ankle_Z': 'Lankle_flex_ext', 'left_ankle_X': 'Lankle_abd_add',
        'right_hip_Z': 'Rhip_flex_ext', 'right_hip_X': 'Rhip_abd_add', 'right_hip_Y': 'Rhip_int_ext_rot',
        'right_knee_Z': 'Rknee_flex_ext', 'right_ankle_Z': 'Rankle_flex_ext', 'right_ankle_X': 'Rankle_abd_add',
    }
    
    index_mapping = []
    for pk_idx, pk_name in enumerate(pk_joint_names):
        csv_name = mapping.get(pk_name)
        if csv_name and csv_name in ddpm_joint_names:
            ddpm_idx = ddpm_joint_names.index(csv_name)
            index_mapping.append((ddpm_idx, pk_idx))
            
    print(f"  ✓ Mapped {len(index_mapping)} joints to URDF")
    
    j_ref, j_pred = predict_full_trial(
        model, ddpm, f_path, j_path, j_path_FF, stats, device,
        chain=chain, n_pk=n_pk, index_mapping=index_mapping, 
        window_size=128, stride=64, inference_steps=1000,
        guidance_scale=0.00 # <--- Start with 5.0, increase if still skating
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
    # SUBJECT_NAME = "s1"     
    # TRIAL_NAME = "squat_variant_980_dz-0.080_dx+0.023_dy-0.017"     

    SUBJECT_NAME = "Jeremy"     
    TRIAL_NAME = "Trial111"           
          
    
    # MODEL_PATH = "./training_res/synth_res/results_PE_sin_cross_fixed/diffusion_biomech_model_concat.pth"
    # SCALERS_PATH = "./training_res/synth_res/results_PE_sin_cross_fixed/scalers_concat.json"
    # DATA_ROOT = "./DATA/synth_npy/"

    MODEL_PATH = "./training_res/results_PE_sin_cross_real_selfs/diffusion_biomech_model_best.pth"
    SCALERS_PATH = "./training_res/results_PE_sin_cross_real_selfs/scalers_concat.json"
    DATA_ROOT = "/datasets/GRF2Kine/processed_data_feet"
    OUTPUT_DIR = "./inference_results_PE_sin_cross_selfs_best_guided"
    
    # ===== RUN INFERENCE =====
    run_inference(
        subject_name=SUBJECT_NAME,
        trial_name=TRIAL_NAME,
        model_path=MODEL_PATH,
        scalers_path=SCALERS_PATH,
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR
    )
