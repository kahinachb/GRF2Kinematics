import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch.nn as nn
import math
import pandas as pd
from scipy.spatial.transform import Rotation as R
N_FF = 6
N_TARGET = 35

def integrate_ff_trajectory(deltas_array, init_pos, init_quat):
    """
    Fonction utilitaire pour intégrer une trajectoire locale en trajectoire absolue.
    deltas_array : tableau de forme (N, >6) où les 6 premières cols sont delta_pos (3) et delta_rotvec (3)
    """
    current_pos = np.array(init_pos)
    current_rot = R.from_quat(init_quat)
    
    trajectory = []
    
    for t in range(len(deltas_array)):
        delta_pos = deltas_array[t, 0:3]
        delta_rotvec = deltas_array[t, 3:6]
        
        # Intégration
        next_pos = current_pos + current_rot.apply(delta_pos)
        next_rot = current_rot * R.from_rotvec(delta_rotvec)
        next_quat = next_rot.as_quat()
        
        trajectory.append([
            next_pos[0], next_pos[1], next_pos[2],
            next_quat[0], next_quat[1], next_quat[2], next_quat[3]
        ])
        
        current_pos = next_pos
        current_rot = next_rot
        
    return np.array(trajectory)

# ==========================================
# 1. ARCHITECTURE (Flow Matching)
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
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
    """Temps continu t in [0,1]. Le facteur d'échelle est appliqué UNE fois ici."""
    def __init__(self, dim, time_scale=1000.0):
        super().__init__()
        self.dim = dim
        self.time_scale = time_scale

    def forward(self, time):
        time = time.float() * self.time_scale
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FlowTransformer(nn.Module):
    def __init__(self, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
        # --- cibles : 4 blocs (FreeFlyer + 3 blocs corporels) ---
        self.embed_FF = nn.Linear(6, embed_dim)     
        self.embed_Rleg = nn.Linear(6, embed_dim)
        self.embed_Lleg = nn.Linear(6, embed_dim)
        self.embed_Upper = nn.Linear(17, embed_dim)
        # --- capteurs (mémoire) : 6 blocs ---
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
            nn.Linear(embed_dim, embed_dim),
        )
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=2000)

        # embeddings de segment appris : 4 cibles (FF, R, L, U) + 6 capteurs
        self.seg_target = nn.Parameter(torch.randn(4, embed_dim) * 0.02)
        self.seg_cond = nn.Parameter(torch.randn(6, embed_dim) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=512,
            activation="gelu", batch_first=True, norm_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)

        self.out_FF = nn.Linear(embed_dim, 6)       # <-- NOUVEAU : tête free-flyer
        self.out_Rleg = nn.Linear(embed_dim, 6)
        self.out_Lleg = nn.Linear(embed_dim, 6)
        self.out_Upper = nn.Linear(embed_dim, 17)

    def forward(self, x, t, cond):
        B, W, _ = x.shape

        if isinstance(t, (int, float)):
            t = torch.full((B,), float(t), device=x.device)
        elif t.dim() == 0:
            t = t.float().unsqueeze(0).expand(B)
        t_emb = self.time_mlp(t).unsqueeze(1)  # (B, 1, D)

        # TARGET : 4 blocs -> PE temporel commun par bloc + segment embedding
        emb_FF = self.pos_encoder(self.embed_FF(x[:, :, 0:6]))    + self.seg_target[0]
        emb_R  = self.pos_encoder(self.embed_Rleg(x[:, :, 6:12])) + self.seg_target[1]
        emb_L  = self.pos_encoder(self.embed_Lleg(x[:, :, 12:18])) + self.seg_target[2]
        emb_U  = self.pos_encoder(self.embed_Upper(x[:, :, 18:35])) + self.seg_target[3]
        x_emb = torch.cat([emb_FF, emb_R, emb_L, emb_U], dim=1) + t_emb   # (B, 4W, D)

        # MEMORY : 6 blocs capteurs
        cond_blocks = [
            self.embed_F_R(cond[:, :, 0:3]),
            self.embed_M_R(cond[:, :, 3:6]),
            self.embed_CoP_R(cond[:, :, 6:9]),
            self.embed_F_L(cond[:, :, 9:12]),
            self.embed_M_L(cond[:, :, 12:15]),
            self.embed_CoP_L(cond[:, :, 15:18]),
        ]
        cond_blocks = [self.pos_encoder(b) + self.seg_cond[i] for i, b in enumerate(cond_blocks)]
        cond_emb = torch.cat(cond_blocks, dim=1) + t_emb                  # (B, 6W, D)

        out = self.transformer(tgt=x_emb, memory=cond_emb)               # (B, 4W, D)

        # Reconstruction des 35 DoF, même ordre que l'entrée
        out_FF = self.out_FF(out[:, 0:W, :])
        out_R  = self.out_Rleg(out[:, W:2 * W, :])
        out_L  = self.out_Lleg(out[:, 2 * W:3 * W, :])
        out_U  = self.out_Upper(out[:, 3 * W:4 * W, :])
        return torch.cat([out_FF, out_R, out_L, out_U], dim=2) 


# ==========================================
# 2. INFERENCE FUNCTION (ODE SOLVER)
# ==========================================
# f_raw = np.concatenate(
    #     [f_raw[:,-9:], f_raw[:, :9]],
    #     axis=1
    # ) 
@torch.no_grad()
def predict_full_trial(model, f_path, j_path, stats, device, window_size=128, stride=64, n_steps=20):
    model.eval()
    f_raw = np.load(f_path).astype(np.float32)
    f_raw = np.concatenate(
        [f_raw[:,-9:], f_raw[:, :9]],
        axis=1
    ) 
    j_raw = np.load(j_path).astype(np.float32)          # (T, 35), FF inclus
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)

    T = f_norm.shape[0]
    full_pred = torch.zeros((T, N_TARGET), device=device)
    overlap = window_size - stride
    dt = 1.0 / n_steps

    print(f"  [INF] Sampling Flow Matching (Heun) complet ({T} frames, {n_steps} steps/win)...")

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        has_ctx = start > 0

        x0 = torch.randn((1, window_size, N_TARGET), device=device)
        x = x0.clone()

        if has_ctx:
            known_x1 = full_pred[start:start + overlap].unsqueeze(0)
            x0_known = x0[:, :overlap, :]
            interp = lambda tv: (1.0 - tv) * x0_known + tv * known_x1
            x[:, :overlap, :] = interp(0.0)

        for i in range(n_steps):
            t0 = i / n_steps
            t1 = (i + 1) / n_steps
            if has_ctx:
                x[:, :overlap, :] = interp(t0)
            t0v = torch.full((1,), t0, device=device)
            t1v = torch.full((1,), t1, device=device)

            v1 = model(x, t0v, f_win)
            x_pred = x + v1 * dt
            if has_ctx:
                x_pred[:, :overlap, :] = interp(t1)
            v2 = model(x_pred, t1v, f_win)
            x = x + 0.5 * (v1 + v2) * dt

        if has_ctx:
            x[:, :overlap, :] = known_x1

        full_pred[start:end] = x.squeeze(0)

    final_pred = (full_pred.cpu() * stats['j_s']) + stats['j_m']
    return j_raw, final_pred.numpy()

@torch.no_grad()
def predict_full_trial_euler(model, f_path, j_path, stats, device, window_size=128, stride=64, n_steps=20):
    """
    Inférence avec Inpainting Autorégressif et solveur Euler (Flow Matching)
    Garantit une continuité temporelle parfaite.
    """
    model.eval()
    f_raw = np.load(f_path).astype(np.float32)
    
    # f_raw = np.concatenate([f_raw[:,-9:], f_raw[:, :9]], axis=1)   
    
    j_raw = np.load(j_path).astype(np.float32)
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)
    
    T = f_norm.shape[0]
    full_pred = torch.zeros((T, N_TARGET)).to(device)

    print(f"  [INF] Sampling trial complet ({T} frames) avec {n_steps} steps (Flow Matching - Euler)...")
    
    overlap_size = window_size - stride
    dt = 1.0 / n_steps
    
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        
        has_context = (start > 0)
        
        # 1. On part d'un bruit pur
        x_0 = torch.randn((1, window_size, N_TARGET)).to(device)
        x_t = x_0.clone()
        
        if has_context:
            known_x1 = full_pred[start : start + overlap_size].unsqueeze(0)
            # Fonction d'interpolation (comme dans Heun)
            interp = lambda tv: (1.0 - tv) * x_0[:, :overlap_size, :] + tv * known_x1
            x_t[:, :overlap_size, :] = interp(0.0)
        
        # 2. Boucle d'intégration ODE (Euler)
        for i in range(n_steps):
            t_val = i / n_steps
            # CORRECTION : t dans [0, 1], le scaling x1000 est géré dans le modèle !
            t_tensor = torch.full((1,), t_val, device=device)
            
            if has_context:
                x_t[:, :overlap_size, :] = interp(t_val)
            
            v_pred = model(x_t, t_tensor, f_win)
            
            x_t = x_t + v_pred * dt
        
        # 3. Verrouillage final de la zone de chevauchement à t=1
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
    model = FlowTransformer().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  ✓ Loaded model from {model_path}")
    
    # ===== 3. LOCATE DATA FILES =====
    print("\n[3/5] Locating data files...")
    data_path = Path(data_root) / subject_name / f"{trial_name}"
    
    # f_path = data_path / "kinetics_glob.npy"
    # j_path = data_path / "all_joints.npy"
    
    f_path = data_path / "kinetics_deltaf.npy"
    j_path = data_path / "all_joints_deltaf.npy"
    if not f_path.exists():
        raise FileNotFoundError(f"Forces file not found: {f_path}")
    if not j_path.exists():
        raise FileNotFoundError(f"Joints file not found: {j_path}")
    
    print(f"  ✓ Found forces: {f_path}")
    print(f"  ✓ Found joints: {j_path}")
    
    # ===== 4. RUN INFERENCE =====
    print("\n[4/5] Running inference...")
    j_ref, j_preds = predict_full_trial_euler(
        model, f_path, j_path, stats, device,
        window_size=128, stride=64, n_steps=20
    )

    # ===== SAVE PREDICTION CSV =====

    joint_names = [
        "delta_x","delta_y","delta_z","delta_rx","delta_ry","delta_rz",
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

    print(f"  ✓ delta Prediction saved: {csv_path}")

    
    # path_joint = f"DATA/Vinc/{subject_name}/{trial_name}/joints_filtered_FF.csv"
    path_joint = f"DATA/generated_data/{subject_name}_squat_{trial_name}_q.csv"
    q_ref_df = pd.read_csv(path_joint)
    position_initiale = q_ref_df.iloc[0, 0:3].values
    quaternion_initial = q_ref_df.iloc[0, 3:7].values

    true_abs_ff = q_ref_df.iloc[1:1+len(j_ref), 0:7].values
    ref_abs_ff = integrate_ff_trajectory(j_ref, position_initiale, quaternion_initial)
    
    # Intégration de la PRÉDICTION
    pred_abs_ff = integrate_ff_trajectory(j_preds, position_initiale, quaternion_initial)


    # Hypothèse : on commence à l'origine (0,0,0) sans rotation initiale
    # current_pos = np.array([0.0, 0.0, 0.0])
    # current_rot = R.from_quat([0.0, 0.0, 0.0, 1.0]) # [x, y, z, w]

    current_pos = np.array(position_initiale, dtype=np.float32)
    current_rot = R.from_quat(quaternion_initial) # [x, y, z, w]

    abs_ff_trajectory = []

    for t in range(len(j_preds)):
        delta_pos = j_preds[t, 0:3]
        delta_rotvec = j_preds[t, 3:6]
        
        # p_next = p_t + R_t * delta_p_local
        next_pos = current_pos + current_rot.apply(delta_pos)
        
        # R_next = R_t * R_local
        r_local = R.from_rotvec(delta_rotvec)
        next_rot = current_rot * r_local
        
        # quaternions au format (x, y, z, w)
        next_quat = next_rot.as_quat()
        
        abs_ff_trajectory.append([
            next_pos[0], next_pos[1], next_pos[2],
            next_quat[0], next_quat[1], next_quat[2], next_quat[3]
        ])
        
        current_pos = next_pos
        current_rot = next_rot

    arr_abs_ff = np.array(abs_ff_trajectory)
    
    arr_other_joints = j_preds[:, 6:]
    
    arr_abs_final = np.hstack([arr_abs_ff, arr_other_joints])

    joint_names_abs = [
        "FF_X", "FF_Y", "FF_Z", "FF_quatx", "FF_quaty", "FF_quatz", "FF_quatw"
    ] + joint_names[6:] 

    df_pred_abs = pd.DataFrame(arr_abs_final, columns=joint_names_abs)
    csv_path_abs = output_dir / f"{subject_name}_{trial_name}_prediction_absolute_euler.csv"
    df_pred_abs.to_csv(csv_path_abs, index=False)
    
    print(f"  ✓ Absolute Prediction saved: {csv_path_abs}")


    # ===== 5. EVALUATION =====
    print("\n[5/5] Evaluating and Plotting...")

    # Calcul de l'erreur globale sur toute la trajectoire
    mse_global = np.mean((j_ref - j_preds)**2)
    rmse_global = np.sqrt(mse_global)

    print(f"  ✓ Global MSE : {mse_global:.6f}")
    print(f"  ✓ Global RMSE: {rmse_global:.6f}")
    
    # ===== VISUALIZATION =====
    ff_names = ["delta_x","delta_y","delta_z","delta_rx","delta_ry","delta_rz"]
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

    abs_ff_names = ["FF_X", "FF_Y", "FF_Z", "FF_quatx", "FF_quaty", "FF_quatz", "FF_quatw"]

    fig, axes = plt.subplots(7, 1, figsize=(15, 20))
    for i in range(7):
        # On calcule le RMSE de la prédiction par rapport à la VRAIE trajectoire du CSV
        rmse_pred = np.sqrt(np.mean((true_abs_ff[:, i] - pred_abs_ff[:, i])**2))
        
        # 1. Le VRAI Freeflyer du CSV (en fond, ligne épaisse verte et transparente)
        axes[i].plot(true_abs_ff[:, i], 'g-', linewidth=4, alpha=0.3, label="True Raw FF (CSV)")
        
        # 2. Le Freeflyer reconstruit par intégration de ta référence (deltas)
        # S'il est parfait, il va se superposer exactement sur la ligne verte !
        axes[i].plot(ref_abs_ff[:, i], 'k--', alpha=0.8, label="Reference (Integrated)")   
        
        # 3. Le Freeflyer prédit par ton modèle
        axes[i].plot(pred_abs_ff[:, i], 'r', linewidth=2, label="Prediction (Integrated)")
        
        axes[i].set_title(f"Absolute {abs_ff_names[i]} (Pred RMSE: {rmse_pred:.4f})", fontsize=10)
        axes[i].grid(True)
        if i == 0:
            axes[i].legend()
            
    plt.suptitle("Absolute Freeflyer Trajectory (True vs Ref Int vs Pred Int)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f"{subject_name}_absolute_ff_euler.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(6, 1, figsize=(15, 18))
    ff_idx = [0, 1, 2, 3, 4, 5]
    for row in range(6):
        i = ff_idx[row]
        rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i])**2))
        axes[row].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")   
        axes[row].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
        axes[row].set_title(f"{ff_names[i]} (RMSE: {rmse:.4f})", fontsize=10)
        axes[row].grid(True)
    
    plt.suptitle("Freeflyer variation)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f"{subject_name}_ff_euler.png", dpi=300)
    plt.close()

    # --- FIGURE 1 : Lower body (Jambes) ---
    # Colonne 0 : Right (6 DOFs) | Colonne 1 : Left (6 DOFs)
    fig, axes = plt.subplots(6, 2, figsize=(15, 18))
    lower_right_idx = [6, 7, 8, 9, 10, 11]
    lower_left_idx = [12, 13, 14, 15, 16, 17]

    for row in range(6):
        # Jambe droite
        i = lower_right_idx[row]
        local_i = i - 6
        rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i])**2))
        axes[row, 0].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")   
        axes[row, 0].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
        axes[row, 0].set_title(f"{lower_names[local_i]} (RMSE: {rmse:.4f})", fontsize=10)
        axes[row, 0].grid(True)

        # Jambe gauche
        i = lower_left_idx[row]
        rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i])**2))
        axes[row, 1].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")   
        axes[row, 1].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
        axes[row, 1].set_title(f"{lower_names[local_i]} (RMSE: {rmse:.4f})", fontsize=10)
        axes[row, 1].grid(True)

    axes[0, 0].legend()
    plt.suptitle("Lower body joints (Right vs Left)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / f"{subject_name}_lower_body_joints_euler.png", dpi=300)
    plt.close()


    # --- FIGURE 2 : Upper body part 1 (Lumbar & Left side) ---
    # Colonne 0 : Lumbar (2 DOFs + cases vides) | Colonne 1 : Left (6 DOFs)
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    lumbar_idx = [12, 13]
    left_col1_idx = [18, 19, 20] # Lcalvicule_x, Lshoulder_flex_ext, Lshoulder_abd_add
    left_col2_idx = [21, 22, 23] # Lshoulder_int_ext_rot, Lelbow_flex_ext, Lelbow_pron_supi

    for row in range(3):
        # Lumbar (seulement 2 DOFs, on cache la 3ème case)
        if row < len(lumbar_idx):
            i = lumbar_idx[row]
            local_i = i - 18
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
    plt.savefig(output_dir / f"{subject_name}_upper_body_lumbar_left_euler.png", dpi=300)
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
    plt.savefig(output_dir / f"{subject_name}_upper_body_cervical_right_euler.png", dpi=300)
    plt.close()
    plt.show()
    
    print(f"\n[FINISH] Plots saved to {output_dir}")


# ==========================================
# 4. EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    # SUBJECT_NAME = "Jeremy"     
    # TRIAL_NAME = "Trial111"           
          
    # MODEL_PATH = "./results_full_step2motion_fm_ff_weighted/fm_biomech_model_best.pth"
    # SCALERS_PATH = "./results_full_step2motion_fm_ff_weighted/scalers_concat.json"
    
    # DATA_ROOT = "processed_data_feet"
    # OUTPUT_DIR = "./results_full_step2motion_fm_ff_weighted"

    SUBJECT_NAME = "subject_01"     
    TRIAL_NAME = "variant_087_dz-0.100_dx+0.025_dy-0.030"   

    MODEL_PATH = "./results_full_step2motion_fm_ff_weighted/fm_biomech_model_best.pth"
    SCALERS_PATH = "./results_full_step2motion_fm_ff_weighted/scalers_concat.json"
    DATA_ROOT = "DATA/synth_npy_all"
    OUTPUT_DIR = "./results_full_step2motion_fm_ff_weighted"
    
    
    # ===== RUN INFERENCE =====
    run_inference(
        subject_name=SUBJECT_NAME,
        trial_name=TRIAL_NAME,
        model_path=MODEL_PATH,
        scalers_path=SCALERS_PATH,
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR
    )