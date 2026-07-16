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



def split_list(lst, train_ratio=0.7, val_ratio=0.15):
    n = len(lst)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = lst[:n_train]
    val = lst[n_train:n_train + n_val]
    test = lst[n_train + n_val:]

    return train, val, test


def split_dataset(all_samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Si plusieurs sujets -> split par sujet.
    Si un seul sujet -> split par essai (task).

    Parameters
    ----------
    all_samples : list of dict
        Chaque élément contient au minimum :
        {
            "subject": ...,
            "task": ...,
            ...
        }

    Returns
    -------
    train_samples, val_samples, test_samples
    """

    unique_subjects = sorted(set(s["subject"] for s in all_samples))

    random.seed(seed)

    # ------------------------------------------------------------------
    # Cas 1 : plusieurs sujets -> split par sujet
    # ------------------------------------------------------------------
    if len(unique_subjects) > 1:

        random.shuffle(unique_subjects)

        train_subjects, val_subjects, test_subjects = split_list(
            unique_subjects,
            train_ratio,
            val_ratio,
        )

        train_samples = [s for s in all_samples if s["subject"] in train_subjects]
        val_samples = [s for s in all_samples if s["subject"] in val_subjects]
        test_samples = [s for s in all_samples if s["subject"] in test_subjects]

        print("\n[SPLIT PAR SUJET]")
        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Val subjects   ({len(val_subjects)}): {val_subjects}")
        print(f"Test subjects  ({len(test_subjects)}): {test_subjects}")

    # ------------------------------------------------------------------
    # Cas 2 : un seul sujet -> split par essai (task)
    # ------------------------------------------------------------------
    else:

        unique_tasks = sorted(set(s["task"] for s in all_samples))
        random.shuffle(unique_tasks)

        train_tasks, val_tasks, test_tasks = split_list(
            unique_tasks,
            train_ratio,
            val_ratio,
        )

        train_samples = [s for s in all_samples if s["task"] in train_tasks]
        val_samples = [s for s in all_samples if s["task"] in val_tasks]
        test_samples = [s for s in all_samples if s["task"] in test_tasks]

        print("\n[SPLIT PAR ESSAI]")
        print(f"Train trials ({len(train_tasks)}): {train_tasks}")
        print(f"Val trials   ({len(val_tasks)}): {val_tasks}")
        print(f"Test trials  ({len(test_tasks)}): {test_tasks}")

    print("=" * 40)
    print(f"Train samples : {len(train_samples)}")
    print(f"Val samples   : {len(val_samples)}")
    print(f"Test samples  : {len(test_samples)}")
    print("=" * 40)

    return train_samples, val_samples, test_samples

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =====================================================================
# EMA (cf. version DDPM) — on échantillonne avec les poids lissés
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


def plot_joints(ref, pred, start, end, filename):
    n = end - start
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()
    for i in range(len(axes)):
        j_idx = start + i
        if j_idx >= end:
            axes[i].axis('off'); continue
        axes[i].plot(ref[:, j_idx], 'k--')
        axes[i].plot(pred[:, j_idx], 'r')
        axes[i].set_title(f"Joint {j_idx+1}")
    plt.tight_layout(); plt.savefig(filename); plt.close()


# =====================================================================
# 1. DATASET
# =====================================================================
class BiomechDiffusionDataset(Dataset):
    def __init__(self, file_list, window_size=128, stats=None):
        self.samples = []
        for f_path, j_path in file_list:
            f_data = np.load(f_path).astype(np.float32)
            j_data = np.load(j_path).astype(np.float32)[:, 6:]
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
    f_cat, j_cat = np.vstack(all_f), np.vstack(all_j)
    j_cat = j_cat[:, 6:]
    stats = {
        'f_m': f_cat.mean(axis=0), 'f_s': f_cat.std(axis=0),
        'j_m': j_cat.mean(axis=0), 'j_s': j_cat.std(axis=0),
    }
    with open(save_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f)
    return {k: torch.tensor(v).float() for k, v in stats.items()}


# =====================================================================
# 2. ARCHITECTURE (avec segment embeddings + PE par bloc + temps CONTINU)
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
    """Temps CONTINU t in [0,1]. Le facteur d'échelle (time_scale) est appliqué
    UNE SEULE FOIS ici -> plus de ×1000 dispersé dans le code, plus de risque
    de désync entre training et inférence. Les appelants passent toujours t in [0,1]."""
    def __init__(self, dim, time_scale=1000.0):
        super().__init__()
        self.dim = dim
        self.time_scale = time_scale

    def forward(self, time):
        time = time.float() * self.time_scale  # [0,1] -> [0, time_scale]
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
        # --- corps (cibles) ---
        self.embed_Rleg = nn.Linear(6, embed_dim)
        self.embed_Lleg = nn.Linear(6, embed_dim)
        self.embed_Upper = nn.Linear(17, embed_dim)
        # --- capteurs (mémoire) ---
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

        # embeddings de segment appris
        self.seg_target = nn.Parameter(torch.randn(3, embed_dim) * 0.02)
        self.seg_cond = nn.Parameter(torch.randn(6, embed_dim) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=512,
            activation="gelu", batch_first=True, norm_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)

        self.out_Rleg = nn.Linear(embed_dim, 6)
        self.out_Lleg = nn.Linear(embed_dim, 6)
        self.out_Upper = nn.Linear(embed_dim, 17)

    def forward(self, x, t, cond):
        B, W, _ = x.shape

        # temps continu -> garder en float (pas de cast en long !)
        if isinstance(t, (int, float)):
            t = torch.full((B,), float(t), device=x.device)
        elif t.dim() == 0:
            t = t.float().unsqueeze(0).expand(B)
        t_emb = self.time_mlp(t).unsqueeze(1)  # (B, 1, D)

        # TARGET : PE temporel commun par bloc + segment embedding
        emb_R = self.pos_encoder(self.embed_Rleg(x[:, :, 0:6]))   + self.seg_target[0]
        emb_L = self.pos_encoder(self.embed_Lleg(x[:, :, 6:12]))  + self.seg_target[1]
        emb_U = self.pos_encoder(self.embed_Upper(x[:, :, 12:29])) + self.seg_target[2]
        x_emb = torch.cat([emb_R, emb_L, emb_U], dim=1) + t_emb

        # MEMORY
        cond_blocks = [
            self.embed_F_R(cond[:, :, 0:3]),
            self.embed_M_R(cond[:, :, 3:6]),
            self.embed_CoP_R(cond[:, :, 6:9]),
            self.embed_F_L(cond[:, :, 9:12]),
            self.embed_M_L(cond[:, :, 12:15]),
            self.embed_CoP_L(cond[:, :, 15:18]),
        ]
        cond_blocks = [self.pos_encoder(b) + self.seg_cond[i] for i, b in enumerate(cond_blocks)]
        cond_emb = torch.cat(cond_blocks, dim=1) + t_emb

        out = self.transformer(tgt=x_emb, memory=cond_emb)

        out_R = self.out_Rleg(out[:, 0:W, :])
        out_L = self.out_Lleg(out[:, W:2 * W, :])
        out_U = self.out_Upper(out[:, 2 * W:3 * W, :])
        return torch.cat([out_R, out_L, out_U], dim=2)


# =====================================================================
# 3. SAMPLERS (Heun, 2e ordre) — convention t=0:bruit  ->  t=1:donnée
# =====================================================================
@torch.no_grad()
def sample_heun(model, f_cond, n_steps=20):
    """Intègre dx/dt = v(x,t) de t=0 (bruit) à t=1 (donnée) avec Heun.
    Heun évalue le champ jusqu'à t=1 et réduit fortement l'erreur d'intégration
    par rapport à Euler, à budget de pas égal."""
    B, W, _ = f_cond.shape
    device = f_cond.device
    x = torch.randn((B, W, 29), device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t0 = i / n_steps
        t1 = (i + 1) / n_steps
        t0v = torch.full((B,), t0, device=device)
        t1v = torch.full((B,), t1, device=device)
        v1 = model(x, t0v, f_cond)              # prédiction
        x_pred = x + v1 * dt
        v2 = model(x_pred, t1v, f_cond)         # correction
        x = x + 0.5 * (v1 + v2) * dt
    return x


# =====================================================================
# 4. INFÉRENCE COMPLÈTE — inpainting style RePaint adapté au Flow Matching
# =====================================================================
@torch.no_grad()
def predict_full_trial(model, f_path, j_path, stats, device, window_size=128, stride=64, n_steps=20):
    model.eval()
    f_raw = np.load(f_path).astype(np.float32)
    j_raw = np.load(j_path).astype(np.float32)[:, 6:]
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)

    T = f_norm.shape[0]
    full_pred = torch.zeros((T, 29), device=device)
    overlap = window_size - stride
    dt = 1.0 / n_steps

    print(f"  [INF] Sampling Flow Matching (Heun) complet ({T} frames, {n_steps} steps/win)...")

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        has_ctx = start > 0

        x0 = torch.randn((1, window_size, 29), device=device)  # bruit source FIXE de la fenêtre
        x = x0.clone()

        if has_ctx:
            known_x1 = full_pred[start:start + overlap].unsqueeze(0)  # poses propres du tour précédent
            x0_known = x0[:, :overlap, :]
            # le long du chemin rectiligne, l'état connu à l'instant t est analytique :
            interp = lambda tv: (1.0 - tv) * x0_known + tv * known_x1
            x[:, :overlap, :] = interp(0.0)

        for i in range(n_steps):
            t0 = i / n_steps
            t1 = (i + 1) / n_steps
            if has_ctx:
                x[:, :overlap, :] = interp(t0)          # on force la zone connue (avant prédiction)
            t0v = torch.full((1,), t0, device=device)
            t1v = torch.full((1,), t1, device=device)

            v1 = model(x, t0v, f_win)
            x_pred = x + v1 * dt
            if has_ctx:
                x_pred[:, :overlap, :] = interp(t1)     # zone connue cohérente pour le correcteur
            v2 = model(x_pred, t1v, f_win)
            x = x + 0.5 * (v1 + v2) * dt

        if has_ctx:
            x[:, :overlap, :] = known_x1                # raccord strictement exact en position

        full_pred[start:end] = x.squeeze(0)

    final_pred = (full_pred.cpu() * stats['j_s']) + stats['j_m']
    return j_raw, final_pred.numpy()


# =====================================================================
# 5. RUN EXPERIMENT
# =====================================================================
def run_experiment():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("/lustre/fsn1/projects/rech/vsi/ulm94jm/dataset_grf2kine/synth_npy_102")
    results_dir = Path("results_full_102_improved")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- DATA --------------------------------
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


    train_subjects, val_subjects, test_subjects = split_dataset(
    all_samples,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
)
    train_subs = [s for s in all_samples if s["subject"] in train_subjects]
    val_subs = [s for s in all_samples if s["subject"] in val_subjects]
    test_subs = [s for s in all_samples if s["subject"] in test_subjects]
    

    def get_pairs(samples):
        return [(s["kinetics"], s["joints"]) for s in samples if s["kinetics"].exists() and s["joints"].exists()]

    train_pairs = get_pairs(train_subjects)
    stats = compute_and_save_stats(train_pairs, results_dir / "scalers_concat.json")

    train_loader = DataLoader(BiomechDiffusionDataset(train_pairs, stats=stats),
                              batch_size=64, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(BiomechDiffusionDataset(get_pairs(val_subjects), stats=stats),
                            batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # --------------------------- MODELE --------------------------------
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
    print(f"\n[START] Entraînement Flow Matching ({epochs} epochs)...")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # ------------------------ TRAIN ------------------------
        model.train()
        running = 0.0
        for f, j in train_loader:
            f = f.to(device, non_blocking=True)
            j = j.to(device, non_blocking=True)
            B = j.shape[0]

            x_1 = j                              # cible : donnée propre
            x_0 = torch.randn_like(x_1)          # source : bruit
            t = torch.rand((B, 1, 1), device=device)   # t in [0,1]
            x_t = t * x_1 + (1.0 - t) * x_0      # interpolation rectiligne
            v_target = x_1 - x_0                 # vitesse cible (constante le long du chemin)

            optimizer.zero_grad()
            v_pred = model(x_t, t.view(B), f)    # t in [0,1] : le scaling se fait dans l'embedding
            loss = criterion(v_pred, v_target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)
            running += loss.item()
        train_loss = running / len(train_loader)

        # --------------------- VALIDATION (EMA, bruit/t figés) ---------------------
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
                vrun += criterion(v_pred, v_target).item()
        val_loss = vrun / len(val_loader)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),    # poids EMA (chargés actuellement)
                "ema": ema.shadow,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
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
    test_ds = BiomechDiffusionDataset(get_pairs(test_subjects), stats=stats)
    f_in, j_ref = test_ds[random.randint(0, len(test_ds) - 1)]
    f_in = f_in.unsqueeze(0).to(device)
    curr_j = sample_heun(model, f_in, n_steps=20)

    pred = (curr_j.cpu().squeeze(0) * stats['j_s']) + stats['j_m']
    ref = (j_ref * stats['j_s']) + stats['j_m']

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(ref[:, i], 'k--', label='Ref'); ax.plot(pred[:, i], 'r', label='Pred')
        ax.set_title(f"Joint {i+1}"); ax.legend()
    plt.tight_layout(); plt.savefig(results_dir / "inference_test_joints_1_12.png"); plt.close()

    fig, axes = plt.subplots(6, 3, figsize=(15, 18))
    for i, ax in enumerate(axes.flatten()):
        j_idx = i + 12
        if j_idx >= 29:
            ax.axis('off'); continue
        ax.plot(ref[:, j_idx], 'k--', label='Ref'); ax.plot(pred[:, j_idx], 'r', label='Pred')
        ax.set_title(f"Joint {j_idx+1}"); ax.legend()
    plt.tight_layout(); plt.savefig(results_dir / "inference_test_joints_13_29.png"); plt.close()

    # --- essai complet ---
    print("\n[INF] Inférence sur un essai COMPLET...")
    random_trial = random.choice(get_pairs(test_subjects))
    print("random_trial for test", random_trial)
    ref_full, pred_full = predict_full_trial(model, random_trial[0], random_trial[1], stats, device, n_steps=20)
    plot_joints(ref_full, pred_full, 0, 12, results_dir / "fig1.png")
    plot_joints(ref_full, pred_full, 12, 29, results_dir / "fig2.png")

    plt.figure(); plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val (EMA)")
    plt.title("Loss History"); plt.xlabel("Epoch"); plt.legend()
    plt.savefig(results_dir / "loss_curve_concat.png"); plt.close()
    print(f"\n[FINISH] Résultats sauvegardés dans {results_dir}")


if __name__ == "__main__":
    run_experiment()