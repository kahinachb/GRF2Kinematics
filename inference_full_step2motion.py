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
def predict_full_trial(model, ddpm, f_path, j_path, stats, device, window_size=128, stride=64):
    """
    Inférence avec Inpainting Autorégressif (Inspiré de Step2Motion / RePaint)
    Garantit une continuité temporelle parfaite sans faire de moyenne.
    """
    model.eval()
    f_raw = np.load(f_path).astype(np.float32)
    j_raw = np.load(j_path).astype(np.float32)[:, 6:]
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)
    
    T = f_norm.shape[0]
    
    # Plus besoin de count_map, on va écraser/ajouter proprement
    full_pred = torch.zeros((T, 29)).to(device)

    print(f"  [INF] Sampling trial complet ({T} frames) avec Inpainting Autorégressif...")
    
    # Taille de la zone de chevauchement (le contexte qu'on connaît déjà)
    overlap_size = window_size - stride
    
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        
        # On part d'un bruit pur pour la fenêtre courante
        curr_j = torch.randn((1, window_size, 29)).to(device)
        
        # Est-ce qu'on a déjà généré une fenêtre précédente ?
        has_context = (start > 0)
        
        if has_context:
            # On récupère la fin de la prédiction précédente (les poses propres x_0)
            known_x0 = full_pred[start : start + overlap_size].unsqueeze(0)
        
        # Boucle de diffusion inverse
        for t_idx in reversed(range(ddpm.n_steps)):
            with torch.no_grad():
                
                if has_context:
                    # 1. On "rebruite" notre contexte connu jusqu'au niveau t
                    t_tensor = torch.tensor([t_idx], device=device)
                    noise_for_known = torch.randn_like(known_x0)
                    known_xt = ddpm.sample_forward(known_x0, t_tensor, noise_for_known)
                    
                    # 2. On FORCE le début de la fenêtre actuelle à être égal à ce contexte bruité
                    curr_j[:, :overlap_size, :] = known_xt
                
                # 3. Le modèle fait un pas de débruitage sur toute la fenêtre
                curr_j = ddpm.sample_reverse_selfs(model, curr_j, t_idx, f_win)
        
        # Nettoyage final pour t=0 : on s'assure que le raccord est strictement parfait
        if has_context:
            curr_j[:, :overlap_size, :] = known_x0
            
        # On sauvegarde la fenêtre dans le tenseur final
        full_pred[start:end] = curr_j.squeeze(0)

    # Dénormalisation finale
    final_pred = (full_pred.cpu() * stats['j_s']) + stats['j_m']
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
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
class DiffusionTransformer(nn.Module):
    def __init__(self, embed_dim=256, nhead=8, num_layers=2):
        super().__init__()

        # ==========================================
        # 1. BODY PARTITIONING (Cibles / Postures)
        # ==========================================
        # Right Leg (6), Left Leg (6), Upper Body (17) -> Total 29
        self.embed_Rleg = nn.Linear(6, embed_dim)
        self.embed_Lleg = nn.Linear(6, embed_dim)
        self.embed_Upper = nn.Linear(17, embed_dim)
        
        # ==========================================
        # 2. SENSOR PARTITIONING (Mémoire / Plateformes)
        # ==========================================
        # Right: F(3), M(3), CoP(3) | Left: F(3), M(3), CoP(3) -> Total 18
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
        
        # On augmente le max_len car notre séquence concaténée sera plus longue (jusqu'à 6*W)
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
        self.transformer = nn.TransformerDecoder(layer, num_layers=2)
        
        # Output projections pour reconstituer les 29 joints
        self.out_Rleg = nn.Linear(embed_dim, 6)
        self.out_Lleg = nn.Linear(embed_dim, 6)
        self.out_Upper = nn.Linear(embed_dim, 17)

    def forward(self, x, t, cond):
        B, W, _ = x.shape
        
        # 1. Time Embedding
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=torch.long).expand(x.shape[0])
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        t_emb = self.time_mlp(t).unsqueeze(1) 
        
        # ==========================================
        # 2. TARGET: Découpage de la Posture (x)
        # ==========================================
        # Indexation basée sur ta liste de 29 joints
        x_R = x[:, :, 0:6]
        x_L = x[:, :, 6:12]
        x_U = x[:, :, 12:29]
        
        emb_R = self.embed_Rleg(x_R)
        emb_L = self.embed_Lleg(x_L)
        emb_U = self.embed_Upper(x_U)
        
        # Concaténation temporelle -> Forme: (B, 3*W, embed_dim)
        x_emb = torch.cat([emb_R, emb_L, emb_U], dim=1)
        x_emb = x_emb + t_emb
        x_emb = self.pos_encoder(x_emb) 
        
        # ==========================================
        # 3. MEMORY: Découpage des Forces (cond)
        # ==========================================
        # Indexation: R_F(0:3), R_M(3:6), R_CoP(6:9), L_F(9:12), L_M(12:15), L_CoP(15:18)
        emb_FR = self.embed_F_R(cond[:, :, 0:3])
        emb_MR = self.embed_M_R(cond[:, :, 3:6])
        emb_CoPR = self.embed_CoP_R(cond[:, :, 6:9])
        
        emb_FL = self.embed_F_L(cond[:, :, 9:12])
        emb_ML = self.embed_M_L(cond[:, :, 12:15])
        emb_CoPL = self.embed_CoP_L(cond[:, :, 15:18])
        
        # Concaténation temporelle -> Forme: (B, 6*W, embed_dim)
        cond_emb = torch.cat([emb_FR, emb_MR, emb_CoPR, emb_FL, emb_ML, emb_CoPL], dim=1)
        cond_emb = cond_emb + t_emb
        cond_emb = self.pos_encoder(cond_emb)
        
        # ==========================================
        # 4. Attention Croisée Multi-Modalités
        # ==========================================
        # Le transformer gère automatiquement le croisement des 3 blocs corporels avec les 6 blocs capteurs
        out = self.transformer(tgt=x_emb, memory=cond_emb)
        
        # ==========================================
        # 5. Reconstruction
        # ==========================================
        # On redécoupe la sortie (B, 3*W, embed_dim) en 3 blocs de taille W
        out_R = self.out_Rleg(out[:, 0:W, :])
        out_L = self.out_Lleg(out[:, W:2*W, :])
        out_U = self.out_Upper(out[:, 2*W:3*W, :])
        
        # On reforme le vecteur final de 29 DoFs
        return torch.cat([out_R, out_L, out_U], dim=2)

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
    
    f_path = data_path / "kinetics_deltaf.npy"
    j_path = data_path / "all_joints_deltaf.npy"
    
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
        window_size=128, stride=64, 
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
    lower_names = [
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",]

    upper_names = [
    "Lumbar_flex_ext", "Lumbar_lateral_flex",
    "Lcalvicule_x",
    "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
    "Lelbow_flex_ext", "Lelbow_pron_supi",
    "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
    "Rcalvicule_x",
    "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot",
    "Relbow_flex_ext", "Relbow_pron_supi",]
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, i in enumerate(range(12)):
        ax = axes[idx]

        ax.plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")   
        ax.plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
        ax.set_title(lower_names[idx], fontsize=10)
        ax.grid(True)

    axes[0].legend()

    plt.suptitle("Lower body joints", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "lower_body_joints.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(5, 4, figsize=(20, 16))
    axes = axes.flatten()

    for idx, i in enumerate(range(12, 29)):
        ax = axes[idx]

        ax.plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")
        ax.plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")

        ax.set_title(upper_names[idx], fontsize=10)
        ax.grid(True)

    # Supprime les sous-graphes inutilisés (20 cases pour 17 articulations)
    for k in range(len(upper_names), len(axes)):
        fig.delaxes(axes[k])

    axes[0].legend()

    plt.suptitle("Upper body joints", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "upper_body_joints.png", dpi=300)
    plt.close()


# ==========================================
# 4. EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    # SUBJECT_NAME = "Jeremy"     
    # TRIAL_NAME = "Trial111"           
          
    # MODEL_PATH = "./results_full_step2motion_corr/diffusion_biomech_model_best.pth"
    # SCALERS_PATH = "./results_full_step2motion_corr/scalers_concat.json"
    # DATA_ROOT = "processed_data_feet"
    # OUTPUT_DIR = "./results_full_step2motion_corr"

    SUBJECT_NAME = "subject_11"     
    TRIAL_NAME = "variant_008_dz+0.025_dx-0.025_dy+0.030"           
          
    MODEL_PATH = "./results_full_step2motion_corr/diffusion_biomech_model_best.pth"
    SCALERS_PATH = "./results_full_step2motion_corr/scalers_concat.json"
    DATA_ROOT = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/synth_npy_all"
    OUTPUT_DIR = "./results_full_step2motion_corr"
    
    
    # ===== RUN INFERENCE =====
    run_inference(
        subject_name=SUBJECT_NAME,
        trial_name=TRIAL_NAME,
        model_path=MODEL_PATH,
        scalers_path=SCALERS_PATH,
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR
    )