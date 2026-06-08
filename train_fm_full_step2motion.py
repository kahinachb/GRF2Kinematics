import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import json

def sample_euler(model, f_cond, n_steps=20):
    """
    Solveur Euler basique pour générer une fenêtre depuis le bruit.
    """
    B, W, _ = f_cond.shape
    device = f_cond.device
    
    # On part du bruit (t=0)
    x = torch.randn((B, W, 29), device=device)
    dt = 1.0 / n_steps
    
    for i in range(n_steps):
        t_val = i / n_steps
        # Toujours multiplier par 1000 pour le TimeEmbedding du transformer
        t_tensor = torch.full((B,), t_val * 1000.0, device=device)
        
        with torch.no_grad():
            v_pred = model(x, t_tensor, f_cond)
        
        # Euler step : position = position + vitesse * temps
        x = x + v_pred * dt
        
    # x est maintenant à t=1 (la pose prédite)
    return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def plot_joints(ref, pred, start, end, filename):
    n = end - start
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    axes = axes.flatten()

    for i in range(len(axes)):
        j_idx = start + i
        if j_idx >= end:
            axes[i].axis('off')
            continue

        axes[i].plot(ref[:, j_idx], 'k--')
        axes[i].plot(pred[:, j_idx], 'r')
        axes[i].set_title(f"Joint {j_idx+1}")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ==========================================
# 1. DATASET 
# ==========================================
class BiomechDiffusionDataset(Dataset):
    def __init__(self, file_list, window_size=128, stats=None):
        self.samples = []
        for f_path, j_path in file_list:
            f_data, j_data = np.load(f_path).astype(np.float32), np.load(j_path).astype(np.float32)
            j_data = j_data[:, 6:]
            
            for i in range(0, len(f_data) - window_size + 1, window_size // 2):
                self.samples.append((f_data[i:i+window_size], j_data[i:i+window_size]))
        self.stats = stats

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        f, j = self.samples[idx]
        f, j = torch.from_numpy(f), torch.from_numpy(j)
        if self.stats:
            f = (f - self.stats['f_m']) / (self.stats['f_s'] + 1e-6)
            j = (j - self.stats['j_m']) / (self.stats['j_s'] + 1e-6)
        return f, j

def compute_and_save_stats(file_list, save_path):
    print("\n[INFO] Calcul des scalers sur le Train Set...")
    all_f, all_j = [], []
    for f_p, j_p in file_list:
        all_f.append(np.load(f_p)); all_j.append(np.load(j_p))
    f_cat, j_cat = np.vstack(all_f), np.vstack(all_j)
    j_cat= j_cat[:, 6:]
    
    stats = {
        'f_m': f_cat.mean(axis=0), 'f_s': f_cat.std(axis=0),
        'j_m': j_cat.mean(axis=0), 'j_s': j_cat.std(axis=0)
    }
    serializable_stats = {k: v.tolist() for k, v in stats.items()}
    with open(save_path, 'w') as f:
        json.dump(serializable_stats, f)
    return {k: torch.tensor(v).float() for k, v in stats.items()}

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
# 3. FONCTION INFERENCE COMPLETE (SLIDING WINDOW)
# ==========================================
def predict_full_trial(model, f_path, j_path, stats, device, window_size=128, stride=64, n_steps=20):
    model.eval()
    f_raw = np.load(f_path).astype(np.float32)
    j_raw = np.load(j_path).astype(np.float32)[:, 6:]
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)
    
    T = f_norm.shape[0]
    full_pred = torch.zeros((T, 29)).to(device)

    print(f"  [INF] Sampling Flow Matching complet ({T} frames, {n_steps} steps/win)...")
    
    overlap_size = window_size - stride
    dt = 1.0 / n_steps
    
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        
        has_context = (start > 0)
        
        # 1. On fixe le bruit de départ (x_0) pour TOUTE la fenêtre
        x_0 = torch.randn((1, window_size, 29), device=device)
        x_t = x_0.clone()
        
        if has_context:
            # On récupère les vraies poses générées au tour précédent (c'est notre x_1 partiel)
            known_x1 = full_pred[start : start + overlap_size].unsqueeze(0)
        
        # Boucle d'intégration ODE (Euler)
        for i in range(n_steps):
            t_val = i / n_steps
            t_tensor = torch.tensor([t_val * 1000.0], device=device)
            
            if has_context:
                # 2. INPAINTING : On force la partie chevauchante à suivre la trajectoire théorique exacte
                # Interpolation entre notre bruit fixe (x_0) et le contexte connu (known_x1)
                forced_xt = t_val * known_x1 + (1.0 - t_val) * x_0[:, :overlap_size, :]
                x_t[:, :overlap_size, :] = forced_xt
            
            # 3. Le modèle prédit le champ de vecteurs global
            with torch.no_grad():
                v_pred = model(x_t, t_tensor, f_win)
            
            # 4. Mise à jour de l'état
            x_t = x_t + v_pred * dt
        
        # Nettoyage final pour s'assurer que le raccord est strictement parfait
        if has_context:
            x_t[:, :overlap_size, :] = known_x1
            
        full_pred[start:end] = x_t.squeeze(0)

    final_pred = (full_pred.cpu() * stats['j_s']) + stats['j_m']
    return j_raw, final_pred.numpy()

# ==========================================
# 4. RUN EXPERIMENT
# ==========================================
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("/datasets/GRF2Kine/synth_npy_all")

    results_dir = Path("results_full_step2motion_fm")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []

    subjects = sorted([d for d in data_root.iterdir() if d.is_dir()])

    for subject_dir in subjects:

        subject_name = subject_dir.name.lower()

        task_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]

        for task_dir in task_dirs:

            task_name = task_dir.name.lower()


            kinetics_file = task_dir / "kinetics_deltaf.npy"
            joints_file = task_dir / "all_joints_deltaf.npy"

            if kinetics_file.exists() and joints_file.exists():

                all_samples.append({
                    "subject": subject_name,
                    "task": task_name,
                    "kinetics": kinetics_file,
                    "joints": joints_file
                })


    print(f"Nombre total de samples squat : {len(all_samples)}")
    
    unique_subjects = sorted(list(set(s["subject"] for s in all_samples)))
    # print(unique_subjects)
    random.seed(42)
    random.shuffle(unique_subjects)


    def split_list(lst, train_ratio=0.7, val_ratio=0.15):

        n = len(lst)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = lst[:n_train]
        val = lst[n_train:n_train + n_val]
        test = lst[n_train + n_val:]

        return train, val, test


    train_subjects, val_subjects, test_subjects = split_list(
        unique_subjects,
        train_ratio=0.7,
        val_ratio=0.15
    )


    train_subs = [s for s in all_samples if s["subject"] in train_subjects]
    val_subs = [s for s in all_samples if s["subject"] in val_subjects]
    test_subs = [s for s in all_samples if s["subject"] in test_subjects]

    print(f"\n[SPLIT SUMMARY]")
    print(f"Train subjects : {len(train_subjects)}")
    print(f"Train subjects   : {train_subjects}")
    print(f"Val subjects   : {len(val_subjects)}")
    print(f"Val subjects   : {val_subjects}")
    print(f"Test subjects  : {len(test_subjects)}")
    print(f"Test subjects   : {test_subjects}")
    print(f"{'='*30}")

    def get_pairs(samples):

        pairs = []

        for s in samples:

            f = s["kinetics"]
            j = s["joints"]

            if f.exists() and j.exists():
                pairs.append((f, j))

        return pairs


    train_pairs = get_pairs(train_subs)
    # print("train_pairs", train_pairs)
    stats = compute_and_save_stats(train_pairs, results_dir /"scalers_concat.json")
    
    train_loader = DataLoader(BiomechDiffusionDataset(train_pairs, stats=stats), batch_size=64, shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
    val_loader = DataLoader(BiomechDiffusionDataset(get_pairs(val_subs), stats=stats), batch_size=64,num_workers=8,pin_memory=True)

    model = DiffusionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2) 
    
    print(f"Nombre de paramètres : {count_parameters(model):,}")
    train_losses, val_losses = [], []

    epochs = 500
    print(f"\n[START] Entraînement fm...")
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for f, j in train_loader:
            f, j = f.to(device), j.to(device)
            B = j.shape[0]
            
            # 1. Définition de la source (bruit) et cible (donnée)
            x_1 = j  # La cible (tes données biomécaniques)
            x_0 = torch.randn_like(x_1) # La source (bruit gaussien pur)
            
            # 2. Tirage du temps t entre 0 et 1
            # On utilise (B, 1, 1) pour que la multiplication broadcast correctement sur (W, 29)
            t = torch.rand((B, 1, 1), device=device)
            
            # 3. Interpolation linéaire (Rectified Flow)
            x_t = t * x_1 + (1 - t) * x_0
            
            # 4. Cible du réseau : la dérivée (vitesse)
            v_target = x_1 - x_0
            
            optimizer.zero_grad()
            
            # Astuce pour le Time Embedding : 
            # Ton SinusoidalTimeEmbeddings attend de grands nombres (comme les steps 0-1000).
            # On multiplie t (qui est entre 0 et 1) par 1000 pour conserver ton architecture intacte.
            t_model = (t.view(B) * 1000.0).to(torch.float32)
            
            # Le modèle prédit la vitesse
            v_pred = model(x_t, t_model, f)
            
            # MSE Loss est le standard pour le Flow Matching
            loss = nn.MSELoss()(v_pred, v_target) 
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for f, j in val_loader:
                f, j = f.to(device), j.to(device)
                B = j.shape[0]
                
                # 1. Définition de la cible (donnée) et de la source (bruit pur)
                x_1 = j 
                x_0 = torch.randn_like(x_1)
                
                # 2. Tirage du temps t entre 0 et 1
                t = torch.rand((B, 1, 1), device=device)
                
                # 3. Interpolation linéaire (Rectified Flow) vers l'état t
                x_t = t * x_1 + (1 - t) * x_0
                
                # 4. La vitesse théorique parfaite (ce qu'on cherche à approcher)
                v_target = x_1 - x_0
                
                # Adaptation du format du temps pour ton SinusoidalTimeEmbeddings
                t_model = (t.squeeze() * 1000.0).to(torch.float32)
                
                # 5. Le modèle prédit la vitesse
                v_pred = model(x_t, t_model, f)
                
                # 6. Calcul de la loss (Mean Squared Error)
                loss = nn.MSELoss()(v_pred, v_target)
                v_loss += loss.item()
        
        train_losses.append(t_loss/len(train_loader))
        val_losses.append(v_loss/len(val_loader))
        print(f"Epoch {epoch:02d} | Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}")

        avg_train_loss = t_loss / len(train_loader)
        avg_val_loss = v_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            print(f"--> Validation loss improved from {best_val_loss:.6f} to {avg_val_loss:.6f}. Saving model...")
            best_val_loss = avg_val_loss
            
            # Save the model state dict specifically as the "best" model
            torch.save(model.state_dict(), results_dir / "fm_biomech_model_best.pth")

    torch.save(model.state_dict(), results_dir /"fm_biomech_model_concat.pth")

    # --- INFERENCE SUR TEST (FENÊTRE) ---
    print("[INF] Génération d'un exemple de test...")
    print("\n[INFO] Chargement du meilleur modèle pour l'inférence...")
    model.load_state_dict(torch.load(results_dir / "fm_biomech_model_best.pth"))
    test_ds = BiomechDiffusionDataset(get_pairs(test_subs), stats=stats)
    f_in, j_ref = test_ds[random.randint(0, len(test_ds)-1)]
    f_in = f_in.unsqueeze(0).to(device)
    
    curr_j = sample_euler(model, f_in, n_steps=20)

    pred = (curr_j.cpu().squeeze(0) * stats['j_s']) + stats['j_m']
    ref = (j_ref * stats['j_s']) + stats['j_m']

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(ref[:, i], 'k--', label='Ref')
        ax.plot(pred[:, i], 'r', label='Pred')
        ax.set_title(f"Joint {i+1}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(results_dir / "inference_test_joints_1_12.png")
    plt.close()

    fig, axes = plt.subplots(6, 3, figsize=(15, 18))  # 18 slots (17 utilisés)

    for i, ax in enumerate(axes.flatten()):
        j_idx = i + 12
        if j_idx >= 29:
            ax.axis('off')
            continue

        ax.plot(ref[:, j_idx], 'k--', label='Ref')
        ax.plot(pred[:, j_idx], 'r', label='Pred')
        ax.set_title(f"Joint {j_idx+1}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(results_dir / "inference_test_joints_13_29.png")
    plt.close()

    # --- INFERENCE COMPLÈTE ---
    print("\n[INF] Inférence sur un essai COMPLET...")
    test_pairs = get_pairs(test_subs)
    random_trial = random.choice(test_pairs)
    print("random_trial for test", random_trial)
    ref_full, pred_full = predict_full_trial(model, random_trial[0], random_trial[1], stats, device, n_steps=20)

    plot_joints(ref_full, pred_full, 0, 12, results_dir/"fig1.png")
    plot_joints(ref_full, pred_full, 12, 29, results_dir/"fig2.png")
    
    plt.figure(); plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
    plt.title("Loss History"); plt.legend(); plt.savefig(results_dir /"loss_curve_concat.png"); plt.close()
    print(f"\n[FINISH] Résultats sauvegardés.{results_dir}")

if __name__ == "__main__":
    run_experiment()

