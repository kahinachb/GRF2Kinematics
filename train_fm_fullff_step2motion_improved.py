import math
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


# =====================================================================
#   Free-flyer (delta de la base flottante) : 6
#    29  (R-leg 6 | L-leg 6 | Upper 17)
#   -> total 35
# Layout du vecteur cible (= ordre brut du fichier all_joints_deltaf.npy) :
#   [ FF(0:6) | R-leg(6:12) | L-leg(12:18) | Upper(18:35) ]
# =====================================================================
N_FF = 6
N_TARGET = 35


# =====================================================================
# Reproductibilité
# =====================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# EMA
# =====================================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


def plot_joints(ref, pred, start, end, filename, label="Joint"):
    n = end - start
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for i in range(len(axes)):
        j_idx = start + i
        if j_idx >= end:
            axes[i].axis('off'); continue
        axes[i].plot(ref[:, j_idx], 'k--')
        axes[i].plot(pred[:, j_idx], 'r')
        axes[i].set_title(f"{label} {j_idx+1}")
    plt.tight_layout(); plt.savefig(filename); plt.close()


# =====================================================================
# 1. DATASET  — on GARDE les 35 colonnes 
# =====================================================================
class BiomechDataset(Dataset):
    def __init__(self, file_list, window_size=128, stats=None):
        self.samples = []
        for f_path, j_path in file_list:
            f_data = np.load(f_path).astype(np.float32)
            j_data = np.load(j_path).astype(np.float32)        # (T, 35) : FF + 29 joints
            for i in range(0, len(f_data) - window_size + 1, window_size // 2):
                self.samples.append((f_data[i:i + window_size], j_data[i:i + window_size]))
        self.stats = stats

    def __len__(self):
        return len(self.samples)

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
    f_cat, j_cat = np.vstack(all_f), np.vstack(all_j)       # j_cat : (N, 35), FF inclus
    stats = {
        'f_m': f_cat.mean(axis=0), 'f_s': f_cat.std(axis=0),
        'j_m': j_cat.mean(axis=0), 'j_s': j_cat.std(axis=0),
    }
    with open(save_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f)
    return {k: torch.tensor(v).float() for k, v in stats.items()}


# =====================================================================
# 2. ARCHITECTURE
# =====================================================================
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
        return torch.cat([out_FF, out_R, out_L, out_U], dim=2)           # (B, W, 35)


# =====================================================================
# 3. SAMPLER (Heun) — t=0:bruit -> t=1:donnée
# =====================================================================
@torch.no_grad()
def sample_heun(model, f_cond, n_steps=20):
    B, W, _ = f_cond.shape
    device = f_cond.device
    x = torch.randn((B, W, N_TARGET), device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t0 = i / n_steps
        t1 = (i + 1) / n_steps
        t0v = torch.full((B,), t0, device=device)
        t1v = torch.full((B,), t1, device=device)
        v1 = model(x, t0v, f_cond)
        x_pred = x + v1 * dt
        v2 = model(x_pred, t1v, f_cond)
        x = x + 0.5 * (v1 + v2) * dt
    return x


# =====================================================================
# 4. INFÉRENCE COMPLÈTE — inpainting Flow Matching
# =====================================================================
@torch.no_grad()
def predict_full_trial(model, f_path, j_path, stats, device, window_size=128, stride=64, n_steps=20):
    model.eval()
    f_raw = np.load(f_path).astype(np.float32)
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


# =====================================================================
# 5. RUN EXPERIMENT
# =====================================================================
def run_experiment():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("/datasets/GRF2Kine/synth_npy_all")
    results_dir = Path("results_full_step2motion_fm_ff_weighted")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for subject_dir in sorted([d for d in data_root.iterdir() if d.is_dir()]):
        for task_dir in [d for d in subject_dir.iterdir() if d.is_dir()]:
            kin = task_dir / "kinetics_deltaf.npy"
            jts = task_dir / "all_joints_deltaf.npy"
            if kin.exists() and jts.exists():
                all_samples.append({"subject": subject_dir.name.lower(), "kinetics": kin, "joints": jts})

    print(f"Nombre total de samples : {len(all_samples)}")
    unique_subjects = sorted(list(set(s["subject"] for s in all_samples)))
    random.shuffle(unique_subjects)

    def split_list(lst, train_ratio=0.7, val_ratio=0.15):
        n = len(lst); n_tr, n_va = int(n * train_ratio), int(n * val_ratio)
        return lst[:n_tr], lst[n_tr:n_tr + n_va], lst[n_tr + n_va:]

    train_subjects, val_subjects, test_subjects = split_list(unique_subjects)
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
        return [(s["kinetics"], s["joints"]) for s in samples if s["kinetics"].exists() and s["joints"].exists()]

    train_pairs = get_pairs(train_subs)
    stats = compute_and_save_stats(train_pairs, results_dir / "scalers_concat.json")

    train_loader = DataLoader(BiomechDataset(train_pairs, stats=stats),
                              batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(BiomechDataset(get_pairs(val_subs), stats=stats),
                            batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    model = FlowTransformer(num_layers=4).to(device)
    ema = EMA(model, decay=0.999)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    epochs = 250
    warmup = 10
    scheduler = SequentialLR(
        optimizer,
        schedulers=[LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
                    CosineAnnealingLR(optimizer, T_max=epochs - warmup, eta_min=1e-6)],
        milestones=[warmup],
    )
    criterion = nn.MSELoss()

    print(f"Nombre de paramètres : {count_parameters(model):,}")
    weights = torch.ones(35, device=device)
    weights[0:6] = 5.0  # 5x plus d'importance pour les 6 DoFs du FF
    weights = weights.view(1, 1, 35) 

    print(f"\n[START] Entraînement Flow Matching (Weighted MSE pour FF)...")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for f, j in train_loader:
            f = f.to(device, non_blocking=True)
            j = j.to(device, non_blocking=True)        # (B, W, 35)
            B = j.shape[0]

            x_1 = j
            x_0 = torch.randn_like(x_1)
            t = torch.rand((B, 1, 1), device=device)
            x_t = t * x_1 + (1.0 - t) * x_0
            v_target = x_1 - x_0

            optimizer.zero_grad()
            v_pred = model(x_t, t.view(B), f)
            
            # --- WEIGHTED MSE (Train) ---
            raw_loss = (v_pred - v_target) ** 2
            weighted_loss = raw_loss * weights
            loss = weighted_loss.mean()
            # ----------------------------

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
            running += loss.item()
            
        train_loss = running / len(train_loader)

        ema.apply_shadow(model)
        model.eval()
        vrun = 0.0
        with torch.no_grad():
            for bi, (f, j) in enumerate(val_loader):
                f = f.to(device, non_blocking=True)
                j = j.to(device, non_blocking=True)
                B = j.shape[0]

                g = torch.Generator(device=device).manual_seed(1234 + bi)
                x_1 = j
                x_0 = torch.randn(j.shape, generator=g, device=device)
                t = torch.rand((B, 1, 1), generator=g, device=device)
                x_t = t * x_1 + (1.0 - t) * x_0
                v_target = x_1 - x_0

                v_pred = model(x_t, t.view(B), f)
                
                # --- WEIGHTED MSE (Validation) ---
                raw_loss = (v_pred - v_target) ** 2
                weighted_loss = raw_loss * weights
                val_batch_loss = weighted_loss.mean()
                # ---------------------------------
                
                vrun += val_batch_loss.item()
                
        val_loss = vrun / len(val_loader)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch, "model": model.state_dict(), "ema": ema.shadow,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, results_dir / "fm_biomech_model_best.pth")

        ema.restore(model)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        tag = "  <-- best" if improved else ""
        print(f"Epoch {epoch:03d} | lr {lr_now:.2e} | train {train_loss:.6f} | val(EMA) {val_loss:.6f}{tag}")

    torch.save({"model": model.state_dict(), "ema": ema.shadow}, results_dir / "fm_biomech_model_final.pth")

    # ============================ INFERENCE ============================
    print("\n[INFO] Chargement du meilleur modèle (EMA) pour l'inférence...")
    ckpt = torch.load(results_dir / "fm_biomech_model_best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # --- fenêtre ---
    test_ds = BiomechDataset(get_pairs(test_subs), stats=stats)
    f_in, j_ref = test_ds[random.randint(0, len(test_ds) - 1)]
    f_in = f_in.unsqueeze(0).to(device)
    curr_j = sample_heun(model, f_in, n_steps=20)

    pred = (curr_j.cpu().squeeze(0) * stats['j_s']) + stats['j_m']     # (W, 35)
    ref = (j_ref * stats['j_s']) + stats['j_m']

    plot_joints(ref, pred, 0, 6,   results_dir / "win_freeflyer.png", label="FreeFlyer")
    plot_joints(ref, pred, 6, 18,  results_dir / "win_legs.png")
    plot_joints(ref, pred, 18, 35, results_dir / "win_upper.png")

    # --- essai complet ---
    print("\n[INF] Inférence sur un essai COMPLET...")
    random_trial = random.choice(get_pairs(test_subs))
    print("random_trial for test", random_trial)
    ref_full, pred_full = predict_full_trial(model, random_trial[0], random_trial[1], stats, device, n_steps=20)
    plot_joints(ref_full, pred_full, 0, 6,   results_dir / "fig_freeflyer.png", label="FreeFlyer")
    plot_joints(ref_full, pred_full, 6, 18,  results_dir / "fig_legs.png")
    plot_joints(ref_full, pred_full, 18, 35, results_dir / "fig_upper.png")

    plt.figure(); plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val (EMA)")
    plt.title("Loss History"); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(results_dir / "loss_curve_concat.png"); plt.close()
    print(f"\n[FINISH] Résultats sauvegardés dans {results_dir}")


if __name__ == "__main__":
    run_experiment()