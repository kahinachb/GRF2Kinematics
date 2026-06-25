import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch.nn as nn
import math
import pandas as pd
# ==========================================
# 1. ARCHITECTURE (Flow Matching)
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
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionTransformer(nn.Module):
    def __init__(self, embed_dim=256, nhead=8, num_layers=2):
        super().__init__()

        self.embed_Rleg = nn.Linear(6, embed_dim)
        self.embed_Lleg = nn.Linear(6, embed_dim)
        self.embed_Upper = nn.Linear(17, embed_dim)
        
        self.embed_F_R = nn.Linear(3, embed_dim)
        self.embed_M_R = nn.Linear(3, embed_dim)
        self.embed_CoP_R = nn.Linear(3, embed_dim)
        
        self.embed_F_L = nn.Linear(3, embed_dim)
        self.embed_M_L = nn.Linear(3, embed_dim)
        self.embed_CoP_L = nn.Linear(3, embed_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ) 
        
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=2000) 
        
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=512,   
            activation="gelu",    
            batch_first=True, 
            norm_first=True, 
            dropout=0.25
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        
        self.out_Rleg = nn.Linear(embed_dim, 6)
        self.out_Lleg = nn.Linear(embed_dim, 6)
        self.out_Upper = nn.Linear(embed_dim, 17)

    def forward(self, x, t, cond):
        B, W, _ = x.shape
        
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=torch.long).expand(x.shape[0])
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        t_emb = self.time_mlp(t).unsqueeze(1) 
        
        x_R = x[:, :, 0:6]
        x_L = x[:, :, 6:12]
        x_U = x[:, :, 12:29]
        
        emb_R = self.embed_Rleg(x_R)
        emb_L = self.embed_Lleg(x_L)
        emb_U = self.embed_Upper(x_U)
        
        x_emb = torch.cat([emb_R, emb_L, emb_U], dim=1)
        x_emb = x_emb + t_emb
        x_emb = self.pos_encoder(x_emb) 
        
        emb_FR = self.embed_F_R(cond[:, :, 0:3])
        emb_MR = self.embed_M_R(cond[:, :, 3:6])
        emb_CoPR = self.embed_CoP_R(cond[:, :, 6:9])
        
        emb_FL = self.embed_F_L(cond[:, :, 9:12])
        emb_ML = self.embed_M_L(cond[:, :, 12:15])
        emb_CoPL = self.embed_CoP_L(cond[:, :, 15:18])
        
        cond_emb = torch.cat([emb_FR, emb_MR, emb_CoPR, emb_FL, emb_ML, emb_CoPL], dim=1)
        cond_emb = cond_emb + t_emb
        cond_emb = self.pos_encoder(cond_emb)
        
        out = self.transformer(tgt=x_emb, memory=cond_emb)
        
        out_R = self.out_Rleg(out[:, 0:W, :])
        out_L = self.out_Lleg(out[:, W:2*W, :])
        out_U = self.out_Upper(out[:, 2*W:3*W, :])
        
        return torch.cat([out_R, out_L, out_U], dim=2)


# ==========================================
# 2. INFERENCE FUNCTION (ODE SOLVER)
# ==========================================
def predict_full_trial(model, f_path, j_path, stats, device, window_size=128, stride=64, n_steps=20):
    """
    Inférence avec Inpainting Autorégressif et solveur Euler (Flow Matching)
    Garantit une continuité temporelle parfaite.
    """
    model.eval()
    f_raw = np.load(f_path).astype(np.float32)
    
    f_raw = np.concatenate(
        [f_raw[:,-9:], f_raw[:, :9]],
        axis=1
    )   
    
    j_raw = np.load(j_path).astype(np.float32)[:, 6:]
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)
    
    T = f_norm.shape[0]
    full_pred = torch.zeros((T, 29)).to(device)

    print(f"  [INF] Sampling trial complet ({T} frames) avec {n_steps} steps (Flow Matching)...")
    
    overlap_size = window_size - stride
    dt = 1.0 / n_steps
    
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        
        has_context = (start > 0)
        
        # 1. On part d'un bruit pur fixe pour toute la fenêtre
        x_0 = torch.randn((1, window_size, 29)).to(device)
        x_t = x_0.clone()
        
        if has_context:
            # On récupère le contexte déjà généré
            known_x1 = full_pred[start : start + overlap_size].unsqueeze(0)
        
        # 2. Boucle d'intégration ODE
        for i in range(n_steps):
            t_val = i / n_steps
            t_tensor = torch.tensor([t_val * 1000.0], device=device)
            
            if has_context:
                # Inpainting : interpolation rigoureuse sur la zone de chevauchement
                forced_xt = t_val * known_x1 + (1.0 - t_val) * x_0[:, :overlap_size, :]
                x_t[:, :overlap_size, :] = forced_xt
            
            with torch.no_grad():
                v_pred = model(x_t, t_tensor, f_win)
            
            x_t = x_t + v_pred * dt
        
        if has_context:
            x_t[:, :overlap_size, :] = known_x1
            
        full_pred[start:end] = x_t.squeeze(0)

    final_pred = (full_pred.cpu() * stats['j_s']) + stats['j_m']
    return j_raw, final_pred.numpy()


# ==========================================
# 3. MAIN INFERENCE SCRIPT
# ==========================================
def run_inference(subject_name, trial_name, model_path, scalers_path, 
                  data_root="./processed_data", output_dir="./inference_results"):
    
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
    
    # ===== 3. LOCATE DATA FILES =====
    print("\n[3/5] Locating data files...")
    data_path = Path(data_root) / subject_name / f"{trial_name}"
    
    f_path = data_path / "kinetics_glob.npy"
    j_path = data_path / "all_joints.npy"
    
    # f_path = data_path / "kinetics_deltaf.npy"
    # j_path = data_path / "all_joints_deltaf.npy"
    if not f_path.exists():
        raise FileNotFoundError(f"Forces file not found: {f_path}")
    if not j_path.exists():
        raise FileNotFoundError(f"Joints file not found: {j_path}")
    
    print(f"  ✓ Found forces: {f_path}")
    print(f"  ✓ Found joints: {j_path}")
    
    # ===== 4. RUN INFERENCE =====
    print("\n[4/5] Running inference...")
    j_ref, j_preds = predict_full_trial(
        model, f_path, j_path, stats, device,
        window_size=128, stride=64, n_steps=20
    )

    # ===== SAVE PREDICTION CSV =====

    joint_names = [
        "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
        "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
        "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
        "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",

        "Lumbar_flex_ext", "Lumbar_lateral_flex",
        "Lcalvicule_x",
        "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
        "Lelbow_flex_ext", "Lelbow_pron_supi",

        "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
        "Rcalvicule_x",
        "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot",
        "Relbow_flex_ext", "Relbow_pron_supi"
    ]

    df_pred = pd.DataFrame(
        j_preds,
        columns=joint_names
    )

    csv_path = output_dir / f"{subject_name}_{trial_name}_prediction.csv"
    df_pred.to_csv(csv_path, index=False)

    print(f"  ✓ Prediction saved: {csv_path}")


    # ===== 5. EVALUATION =====
    print("\n[5/5] Evaluating and Plotting...")

    # Calcul de l'erreur globale sur toute la trajectoire
    mse_global = np.mean((j_ref - j_preds)**2)
    rmse_global = np.sqrt(mse_global)

    print(f"  ✓ Global MSE : {mse_global:.6f}")
    print(f"  ✓ Global RMSE: {rmse_global:.6f}")
    
    # ===== VISUALIZATION =====
    lower_names = [
        "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
        "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
        "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
        "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add"
    ]

    upper_names = [
        "Lumbar_flex_ext", "Lumbar_lateral_flex",
        "Lcalvicule_x",
        "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
        "Lelbow_flex_ext", "Lelbow_pron_supi",
        "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
        "Rcalvicule_x",
        "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot",
        "Relbow_flex_ext", "Relbow_pron_supi"
    ]

    # --- FIGURE 1 : Lower body (Jambes) ---
    # Colonne 0 : Right (6 DOFs) | Colonne 1 : Left (6 DOFs)
    fig, axes = plt.subplots(6, 2, figsize=(15, 18))
    lower_right_idx = [0, 1, 2, 3, 4, 5]
    lower_left_idx = [6, 7, 8, 9, 10, 11]

    for row in range(6):
        # Jambe droite
        i = lower_right_idx[row]
        rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i])**2))
        axes[row, 0].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")   
        axes[row, 0].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
        axes[row, 0].set_title(f"{lower_names[i]} (RMSE: {rmse:.4f})", fontsize=10)
        axes[row, 0].grid(True)

        # Jambe gauche
        i = lower_left_idx[row]
        rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i])**2))
        axes[row, 1].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")   
        axes[row, 1].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
        axes[row, 1].set_title(f"{lower_names[i]} (RMSE: {rmse:.4f})", fontsize=10)
        axes[row, 1].grid(True)

    axes[0, 0].legend()
    plt.suptitle("Lower body joints (Right vs Left)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f"{subject_name}_lower_body_joints.png", dpi=300)
    plt.close()


    # --- FIGURE 2 : Upper body part 1 (Lumbar & Left side) ---
    # Colonne 0 : Lumbar (2 DOFs + cases vides) | Colonne 1 : Left (6 DOFs)
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    lumbar_idx = [12, 13]
    left_col1_idx = [14, 15, 16] # Lcalvicule_x, Lshoulder_flex_ext, Lshoulder_abd_add
    left_col2_idx = [17, 18, 19] # Lshoulder_int_ext_rot, Lelbow_flex_ext, Lelbow_pron_supi

    for row in range(3):
        # Lumbar (seulement 2 DOFs, on cache la 3ème case)
        if row < len(lumbar_idx):
            i = lumbar_idx[row]
            local_i = i - 12
            rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i])**2))
            axes[row, 0].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")
            axes[row, 0].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
            axes[row, 0].set_title(f"{upper_names[local_i]} (RMSE: {rmse:.4f})", fontsize=10)
            axes[row, 0].grid(True)
            if row == 0: axes[row, 0].legend()
        else:
            axes[row, 0].axis('off') # Cache le sous-graphe vide

        # Left side - Première partie
        i = left_col1_idx[row]
        local_i = i - 12
        rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i])**2))
        axes[row, 1].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")
        axes[row, 1].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
        axes[row, 1].set_title(f"{upper_names[local_i]} (RMSE: {rmse:.4f})", fontsize=10)
        axes[row, 1].grid(True)

        # Left side - Deuxième partie
        i = left_col2_idx[row]
        local_i = i - 12
        rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i])**2))
        axes[row, 2].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")
        axes[row, 2].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
        axes[row, 2].set_title(f"{upper_names[local_i]} (RMSE: {rmse:.4f})", fontsize=10)
        axes[row, 2].grid(True)

    plt.suptitle("Upper body joints - Lumbar & Left side", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f"{subject_name}_upper_body_lumbar_left.png", dpi=300)
    plt.close()


    # --- FIGURE 3 : Upper body part 2 (Cervical & Right side) ---
    # Colonne 0 : Cervical (3 DOFs) | Colonne 1 & 2 : Right diviso en 2 (3 DOFs chacun)
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    cervical_idx = [20, 21, 22]
    right_col1_idx = [23, 24, 25] # Rcalvicule_x, Rshoulder_flex_ext, Rshoulder_abd_add
    right_col2_idx = [26, 27, 28] # Rshoulder_int_ext_rot, Relbow_flex_ext, Relbow_pron_supi

    for row in range(3):
        # Cervical
        i = cervical_idx[row]
        local_i = i - 12
        rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i])**2))
        axes[row, 0].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")
        axes[row, 0].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
        axes[row, 0].set_title(f"{upper_names[local_i]} (RMSE: {rmse:.4f})", fontsize=10)
        axes[row, 0].grid(True)
        if row == 0: axes[row, 0].legend()

        # Right side - Première partie
        i = right_col1_idx[row]
        local_i = i - 12
        rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i])**2))
        axes[row, 1].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")
        axes[row, 1].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
        axes[row, 1].set_title(f"{upper_names[local_i]} (RMSE: {rmse:.4f})", fontsize=10)
        axes[row, 1].grid(True)

        # Right side - Deuxième partie
        i = right_col2_idx[row]
        local_i = i - 12
        rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i])**2))
        axes[row, 2].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")
        axes[row, 2].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
        axes[row, 2].set_title(f"{upper_names[local_i]} (RMSE: {rmse:.4f})", fontsize=10)
        axes[row, 2].grid(True)

    plt.suptitle("Upper body joints - Cervical & Right side", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f"{subject_name}_upper_body_cervical_right.png", dpi=300)
    plt.close()
    plt.show()
    
    print(f"\n[FINISH] Plots saved to {output_dir}")


# ==========================================
# 4. EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    SUBJECT_NAME = "Jeremy"     
    TRIAL_NAME = "Trial111"           
          
    MODEL_PATH = "./results_full_step2motion_fm/fm_biomech_model_best.pth"
    SCALERS_PATH = "./results_full_step2motion_fm/scalers_concat.json"
    
    DATA_ROOT = "processed_data_feet"
    OUTPUT_DIR = "./results_full_step2motion_fm"

    # SUBJECT_NAME = "subject_11"     
    # TRIAL_NAME = "variant_008_dz+0.025_dx-0.025_dy+0.030"           
          
    # MODEL_PATH = "./results_full_step2motion_fm/fm_biomech_model_best.pth"
    # SCALERS_PATH = "./results_full_step2motion_fm/scalers_concat.json"
    # DATA_ROOT = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/synth_npy_all"
    # OUTPUT_DIR = "./results_full_step2motion_fm"
    
    
    # ===== RUN INFERENCE =====
    run_inference(
        subject_name=SUBJECT_NAME,
        trial_name=TRIAL_NAME,
        model_path=MODEL_PATH,
        scalers_path=SCALERS_PATH,
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR
    )