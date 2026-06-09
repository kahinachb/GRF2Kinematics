import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import json
from utils.diffuser_utils import DDPM 
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# EMA (Exponential Moving Average) des poids.
# En diffusion, échantillonner avec les poids EMA améliore nettement la
# qualité et la stabilité. On entraîne avec les poids "bruts", mais on
# valide / sauvegarde / infère avec les poids lissés.
# ---------------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters() if p.requires_grad
        }
        self.backup = {}
 
    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
 
    def apply_shadow(self, model):
        """Charge les poids EMA dans le modèle (en sauvegardant les poids courants)."""
        self.backup = {
            name: p.detach().clone()
            for name, p in model.named_parameters() if p.requires_grad
        }
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name])
 
    def restore(self, model):
        """Restaure les poids d'entraînement après une parenthèse EMA."""
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}
 

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
    def __init__(self, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
 
        # --- Partitionnement corps (cibles) : Rleg(6), Lleg(6), Upper(17) = 29 ---
        self.embed_Rleg = nn.Linear(6, embed_dim)
        self.embed_Lleg = nn.Linear(6, embed_dim)
        self.embed_Upper = nn.Linear(17, embed_dim)
 
        # --- Partitionnement capteurs (mémoire) : R{F,M,CoP} + L{F,M,CoP} = 18 ---
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
 
        # PE appliqué PAR BLOC (longueur W) -> max_len doit juste être >= window_size
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=2000)
 
        # --- Embeddings de segment appris ---
        # 3 segments cibles (Rleg, Lleg, Upper), 6 segments capteurs (FR,MR,CoPR,FL,ML,CoPL)
        self.seg_target = nn.Parameter(torch.randn(3, embed_dim) * 0.02)
        self.seg_cond = nn.Parameter(torch.randn(6, embed_dim) * 0.02)
 
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=512,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.1,                       # 0.25 -> 0.1
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)  # num_layers effectif
 
        self.out_Rleg = nn.Linear(embed_dim, 6)
        self.out_Lleg = nn.Linear(embed_dim, 6)
        self.out_Upper = nn.Linear(embed_dim, 17)
 
    def forward(self, x, t, cond):
        B, W, _ = x.shape
 
        # 1. Time embedding (global, broadcast sur toutes les positions)
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=torch.long).expand(x.shape[0])
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        t_emb = self.time_mlp(t).unsqueeze(1)  # (B, 1, D)
 
        # 2. TARGET : 3 blocs corporels -> PE temporel commun + segment embedding
        emb_R = self.pos_encoder(self.embed_Rleg(x[:, :, 0:6]))   + self.seg_target[0]
        emb_L = self.pos_encoder(self.embed_Lleg(x[:, :, 6:12]))  + self.seg_target[1]
        emb_U = self.pos_encoder(self.embed_Upper(x[:, :, 12:29])) + self.seg_target[2]
 
        x_emb = torch.cat([emb_R, emb_L, emb_U], dim=1) + t_emb   # (B, 3W, D)
 
        # 3. MEMORY : 6 blocs capteurs -> PE temporel commun + segment embedding
        cond_blocks = [
            self.embed_F_R(cond[:, :, 0:3]),
            self.embed_M_R(cond[:, :, 3:6]),
            self.embed_CoP_R(cond[:, :, 6:9]),
            self.embed_F_L(cond[:, :, 9:12]),
            self.embed_M_L(cond[:, :, 12:15]),
            self.embed_CoP_L(cond[:, :, 15:18]),
        ]
        cond_blocks = [self.pos_encoder(b) + self.seg_cond[i] for i, b in enumerate(cond_blocks)]
        cond_emb = torch.cat(cond_blocks, dim=1) + t_emb          # (B, 6W, D)
 
        # 4. Attention croisée multi-modalités
        out = self.transformer(tgt=x_emb, memory=cond_emb)
 
        # 5. Reconstruction des 29 DoF
        out_R = self.out_Rleg(out[:, 0:W, :])
        out_L = self.out_Lleg(out[:, W:2*W, :])
        out_U = self.out_Upper(out[:, 2*W:3*W, :])
        return torch.cat([out_R, out_L, out_U], dim=2)
# ==========================================
# 3. FONCTION INFERENCE COMPLETE (SLIDING WINDOW)
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
# 4. RUN EXPERIMENT
# ==========================================
def run_experiment():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("/datasets/GRF2Kine/synth_npy_all")
 
    results_dir = Path("results_full_step2motion_corr")
    results_dir.mkdir(parents=True, exist_ok=True)
 
    # ----------------------------- DATA --------------------------------
    all_samples = []
    subjects = sorted([d for d in data_root.iterdir() if d.is_dir()])
    for subject_dir in subjects:
        subject_name = subject_dir.name.lower()
        for task_dir in [d for d in subject_dir.iterdir() if d.is_dir()]:
            kinetics_file = task_dir / "kinetics_deltaf.npy"
            joints_file = task_dir / "all_joints_deltaf.npy"
            if kinetics_file.exists() and joints_file.exists():
                all_samples.append({
                    "subject": subject_name,
                    "task": task_dir.name.lower(),
                    "kinetics": kinetics_file,
                    "joints": joints_file,
                })
 
    print(f"Nombre total de samples : {len(all_samples)}")
 
    unique_subjects = sorted(list(set(s["subject"] for s in all_samples)))
    random.shuffle(unique_subjects)  # seedé par set_seed
 
    def split_list(lst, train_ratio=0.7, val_ratio=0.15):
        n = len(lst)
        n_train, n_val = int(n * train_ratio), int(n * val_ratio)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]
 
    train_subjects, val_subjects, test_subjects = split_list(unique_subjects)
 
    train_subs = [s for s in all_samples if s["subject"] in train_subjects]
    val_subs = [s for s in all_samples if s["subject"] in val_subjects]
    test_subs = [s for s in all_samples if s["subject"] in test_subjects]
 
    print(f"\n[SPLIT] train={len(train_subjects)} | val={len(val_subjects)} | test={len(test_subjects)}")
 
    def get_pairs(samples):
        return [(s["kinetics"], s["joints"]) for s in samples
                if s["kinetics"].exists() and s["joints"].exists()]
 
    train_pairs = get_pairs(train_subs)
    stats = compute_and_save_stats(train_pairs, results_dir / "scalers_concat.json")
 
    train_loader = DataLoader(
        BiomechDiffusionDataset(train_pairs, stats=stats),
        batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        BiomechDiffusionDataset(get_pairs(val_subs), stats=stats),
        batch_size=64, shuffle=False, num_workers=8, pin_memory=True,  # shuffle=False : ordre déterministe pour la val
    )
 
    # --------------------------- MODELE --------------------------------
    ddpm = DDPM(device, n_steps=1000)
    model = DiffusionTransformer(num_layers=4).to(device)  # num_layers effectif après le fix de la classe
    ema = EMA(model, decay=0.999)
 
    # lr plus raisonnable pour un Transformer + warmup + cosine
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    epochs = 500
    warmup_epochs = 10
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )
    criterion = nn.SmoothL1Loss()  # robuste comme L1 mais lisse près de 0 ; mets nn.L1Loss() pour rester à l'identique
 
    print(f"Nombre de paramètres : {count_parameters(model):,}")
    print(f"\n[START] Entraînement DDPM ({epochs} epochs)...")
 
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
 
    for epoch in range(epochs):
        # ---------------------- TRAIN ----------------------
        model.train()
        running = 0.0
        for f, j in train_loader:
            f = f.to(device, non_blocking=True)
            j = j.to(device, non_blocking=True)
 
            t = torch.randint(0, ddpm.n_steps, (j.shape[0],), device=device)
            noise = torch.randn_like(j)
            j_noisy = ddpm.sample_forward(j, t, noise)
 
            optimizer.zero_grad()
            pred_x0 = model(j_noisy, t, f)          # le modèle prédit x_0
            loss = criterion(pred_x0, j)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)                        # MAJ EMA après chaque step
            running += loss.item()
        train_loss = running / len(train_loader)
 
        # ------------------ VALIDATION (EMA) ----------------
        # On valide avec les poids EMA + un bruit/t FIXES par batch
        # -> la val loss devient comparable d'une epoch à l'autre
        ema.apply_shadow(model)
        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for bi, (f, j) in enumerate(val_loader):
                f = f.to(device, non_blocking=True)
                j = j.to(device, non_blocking=True)
 
                g = torch.Generator(device=device).manual_seed(1234 + bi)
                t = torch.randint(0, ddpm.n_steps, (j.shape[0],), generator=g, device=device)
                noise = torch.randn(j.shape, generator=g, device=device)
                j_noisy = ddpm.sample_forward(j, t, noise)
 
                pred_x0 = model(j_noisy, t, f)
                vrun += criterion(pred_x0, j).item()
        val_loss = vrun / len(val_loader)
 
        # Sauvegarde du best PENDANT que les poids EMA sont chargés
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),     # = poids EMA
                "ema": ema.shadow,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, results_dir / "diffusion_biomech_model_best.pth")
 
        ema.restore(model)  # on rend les poids d'entraînement au modèle
        scheduler.step()
 
        # ------------------ LOG (un seul append) ------------
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        tag = "  <-- best" if improved else ""
        print(f"Epoch {epoch:03d} | lr {lr_now:.2e} | train {train_loss:.6f} | val(EMA) {val_loss:.6f}{tag}")
 
    # Sauvegarde finale (poids bruts + shadow EMA)
    torch.save(
        {"model": model.state_dict(), "ema": ema.shadow},
        results_dir / "diffusion_biomech_model_final.pth",
    )
 
    # ======================= INFERENCE =================================
    # On infère avec le MEILLEUR checkpoint (poids EMA)
    print("\n[INFO] Chargement du meilleur modèle (EMA) pour l'inférence...")
    ckpt = torch.load(results_dir / "diffusion_biomech_model_best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
 
    # --- INFERENCE FENÊTRE ---
    test_ds = BiomechDiffusionDataset(get_pairs(test_subs), stats=stats)
    f_in, j_ref = test_ds[random.randint(0, len(test_ds) - 1)]
    f_in = f_in.unsqueeze(0).to(device)
 
    curr_j = torch.randn((1, 128, 29)).to(device)
    for t_idx in reversed(range(ddpm.n_steps)):
        with torch.no_grad():
            curr_j = ddpm.sample_reverse_selfs(model, curr_j, t_idx, f_in)
 
    pred = (curr_j.cpu().squeeze(0) * stats["j_s"]) + stats["j_m"]
    ref = (j_ref * stats["j_s"]) + stats["j_m"]
 
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(ref[:, i], "k--", label="Ref")
        ax.plot(pred[:, i], "r", label="Pred")
        ax.set_title(f"Joint {i+1}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "inference_test_joints_1_12.png")
    plt.close()
 
    fig, axes = plt.subplots(6, 3, figsize=(15, 18))
    for i, ax in enumerate(axes.flatten()):
        j_idx = i + 12
        if j_idx >= 29:
            ax.axis("off")
            continue
        ax.plot(ref[:, j_idx], "k--", label="Ref")
        ax.plot(pred[:, j_idx], "r", label="Pred")
        ax.set_title(f"Joint {j_idx+1}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "inference_test_joints_13_29.png")
    plt.close()
 
    # --- INFERENCE ESSAI COMPLET ---
    print("\n[INF] Inférence sur un essai COMPLET...")
    test_pairs = get_pairs(test_subs)
    random_trial = random.choice(test_pairs)
    print("random_trial for test", random_trial)
    ref_full, pred_full = predict_full_trial(model, ddpm, random_trial[0], random_trial[1], stats, device)
    plot_joints(ref_full, pred_full, 0, 12, results_dir / "fig1.png")
    plot_joints(ref_full, pred_full, 12, 29, results_dir / "fig2.png")
 
    # --- COURBE DE LOSS (une seule série par split) ---
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val (EMA)")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(results_dir / "loss_curve_concat.png")
    plt.close()
    print(f"\n[FINISH] Résultats sauvegardés dans {results_dir}")
 
 
if __name__ == "__main__":
    run_experiment()

