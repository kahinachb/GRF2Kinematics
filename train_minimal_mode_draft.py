import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. Config
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique"  

JOINTS_TRAIN_FILE  = os.path.join(DATA_DIR, "joints_train.npy")
WRENCH_TRAIN_FILE  = os.path.join(DATA_DIR, "wrench_train.npy")
JOINTS_VAL_FILE    = os.path.join(DATA_DIR, "joints_val.npy")
WRENCH_VAL_FILE    = os.path.join(DATA_DIR, "wrench_val.npy")
JOINTS_TEST_FILE   = os.path.join(DATA_DIR, "joints_test.npy")
WRENCH_TEST_FILE   = os.path.join(DATA_DIR, "wrench_test.npy")

# Diffusion hyperparams
T = 1000
BETA_START = 1e-4
BETA_END = 0.02

# Training hyperparams
BATCH_SIZE = 128
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

# Sampling / eval
N_SAMPLES_PER_COND = 200  # nb de samples par My pour la trajectoire de test
output_dir = "minimal_model_local"
out_dir= Path(output_dir)
out_dir.mkdir(exist_ok=True)
MODEL_SAVE_PATH = "minimal_model_local/cond_diffusion_model.pt"


# ============================================================
# 2. Dataset
# ============================================================

class JointsWrenchDataset(Dataset):
    def __init__(self, joints, wrench):
        """
        joints: (N,2)
        wrench: (N,) ou (N,1) -> My
        """
        assert joints.shape[0] == wrench.shape[0]
        self.joints = joints.astype(np.float32)
        self.wrench = wrench.reshape(-1, 1).astype(np.float32)

    def __len__(self):
        return self.joints.shape[0]

    def __getitem__(self, idx):
        q = self.joints[idx]      # (2,)
        My = self.wrench[idx]     # (1,)
        return q, My


# ============================================================
# 3. Sinusoidal time embedding
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (batch,) -> time step
        output: (batch, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


# ============================================================
# 4. MLP diffusion conditionnel
# ============================================================

class CondDiffusionMLP(nn.Module):
    def __init__(self, x_dim=2, cond_dim=1, time_dim=32, hidden_dim=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        in_dim = x_dim + cond_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x_noisy, t, cond):
        """
        x_noisy: (B,2)
        t: (B,)
        cond: (B,1) -> My normalisé
        """
        t_emb = self.time_embed(t)  # (B, time_dim)
        h = torch.cat([x_noisy, cond, t_emb], dim=-1)
        eps_pred = self.net(h)
        return eps_pred


# ============================================================
# 5. Beta schedule
# ============================================================

def make_beta_schedule(T, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, T)


def prepare_diffusion_coeffs(T, beta_start, beta_end, device):
    betas = make_beta_schedule(T, beta_start, beta_end).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


# ============================================================
# 6. Chargement data + normalisation (basée sur TRAIN)
# ============================================================

def load_splits_and_stats():
    joints_train = np.load(JOINTS_TRAIN_FILE)   # (N_train,2)
    wrench_train = np.load(WRENCH_TRAIN_FILE)   # (N_train,) ou (N_train,1)
    joints_val   = np.load(JOINTS_VAL_FILE)
    wrench_val   = np.load(WRENCH_VAL_FILE)
    joints_test  = np.load(JOINTS_TEST_FILE)
    wrench_test  = np.load(WRENCH_TEST_FILE)

    train_dataset = JointsWrenchDataset(joints_train, wrench_train)
    val_dataset   = JointsWrenchDataset(joints_val, wrench_val)
    test_dataset  = JointsWrenchDataset(joints_test, wrench_test)

    # Stats uniquement sur le train
    all_joints_train = joints_train.astype(np.float32)
    all_wrench_train = wrench_train.reshape(-1, 1).astype(np.float32)

    joints_mean = all_joints_train.mean(axis=0, keepdims=True)   # (1,2)
    joints_std  = all_joints_train.std(axis=0, keepdims=True) + 1e-8
    wrench_mean = all_wrench_train.mean(axis=0, keepdims=True)   # (1,1)
    wrench_std  = all_wrench_train.std(axis=0, keepdims=True) + 1e-8

    norm_stats = {
        "joints_mean": joints_mean,
        "joints_std": joints_std,
        "wrench_mean": wrench_mean,
        "wrench_std": wrench_std,
    }

    return train_dataset, val_dataset, test_dataset, norm_stats


def collate_with_normalization(batch, norm_stats):
    joints_mean = norm_stats["joints_mean"]
    joints_std  = norm_stats["joints_std"]
    wrench_mean = norm_stats["wrench_mean"]
    wrench_std  = norm_stats["wrench_std"]

    qs = []
    Mys = []
    for q, My in batch:
        qs.append(q)
        Mys.append(My)

    qs = np.stack(qs, axis=0)    # (B,2)
    Mys = np.stack(Mys, axis=0)  # (B,1)

    qs_norm = (qs - joints_mean) / joints_std
    Mys_norm = (Mys - wrench_mean) / wrench_std

    qs_norm = torch.from_numpy(qs_norm).float()
    Mys_norm = torch.from_numpy(Mys_norm).float()

    return qs_norm, Mys_norm


# ============================================================
# 7. Training
# ============================================================

def train_model():
    train_dataset, val_dataset, test_dataset, norm_stats = load_splits_and_stats()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=lambda batch: collate_with_normalization(batch, norm_stats),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda batch: collate_with_normalization(batch, norm_stats),
    )

    betas, alphas, alpha_bars = prepare_diffusion_coeffs(
        T, BETA_START, BETA_END, DEVICE
    )

    model = CondDiffusionMLP(
        x_dim=2, cond_dim=1, time_dim=32, hidden_dim=128
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---------- TRAIN ----------
        model.train()
        running_train_loss = 0.0
        n_train_batches = 0

        for qs_norm, Mys_norm in train_loader:
            qs_norm = qs_norm.to(DEVICE)     # (B,2)
            Mys_norm = Mys_norm.to(DEVICE)   # (B,1)
            B = qs_norm.shape[0]

            t = torch.randint(0, T, (B,), device=DEVICE)   # (B,)
            alpha_bar_t = alpha_bars[t].view(-1, 1)        # (B,1)

            epsilon = torch.randn_like(qs_norm)
            x_t = torch.sqrt(alpha_bar_t) * qs_norm + torch.sqrt(1.0 - alpha_bar_t) * epsilon

            eps_pred = model(x_t, t, Mys_norm)
            loss = torch.mean((eps_pred - epsilon) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = running_train_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)

        # ---------- VAL ----------
        model.eval()
        running_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for qs_norm, Mys_norm in val_loader:
                qs_norm = qs_norm.to(DEVICE)
                Mys_norm = Mys_norm.to(DEVICE)
                B = qs_norm.shape[0]
                t = torch.randint(0, T, (B,), device=DEVICE)
                alpha_bar_t = alpha_bars[t].view(-1, 1)
                epsilon = torch.randn_like(qs_norm)
                x_t = torch.sqrt(alpha_bar_t) * qs_norm + torch.sqrt(1.0 - alpha_bar_t) * epsilon
                eps_pred = model(x_t, t, Mys_norm)
                loss = torch.mean((eps_pred - epsilon) ** 2)
                running_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = running_val_loss / max(1, n_val_batches)
        val_losses.append(avg_val_loss)

        print(
            f"[Epoch {epoch:03d}/{NUM_EPOCHS}] "
            f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f}"
        )

    # Sauvegarde
    save_dict = {
        "model_state_dict": model.state_dict(),
        "norm_stats": norm_stats,
        "betas": betas.cpu().numpy(),
        "alphas": alphas.cpu().numpy(),
        "alpha_bars": alpha_bars.cpu().numpy(),
        "config": {
            "T": T,
            "BETA_START": BETA_START,
            "BETA_END": BETA_END,
        },
    }
    torch.save(save_dict, MODEL_SAVE_PATH)
    print(f"Modèle sauvegardé dans {MODEL_SAVE_PATH}")

    # Courbe de loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (pred_noise vs true_noise)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("minimal_model_local/loss_curves.png", dpi=150)
    print("Courbe de loss sauvegardée dans loss_curves.png")

    return model, norm_stats, betas, alphas, alpha_bars, test_dataset


# ============================================================
# 8. Sampling et evaluate conditionnel sur une trajectoire de test
# ============================================================
def evaluate_static_multisamples(
       model,
    norm_stats,
    betas,
    alphas,
    alpha_bars,
    n_examples=4,
    n_samples_per_cond=4,
):
    """
    Évalue le modèle sur quelques poses de test (statiques),
    en générant plusieurs solutions possibles pour une même valeur de My.

    - Choisit n_examples indices dans le set de test.
    - Pour chaque My_test[i], on génère n_samples_per_cond échantillons de (q1,q2).
    - Pour chaque exemple, on trace 2 subplots :
        * q1 samples vs index + q1 GT
        * q2 samples vs index + q2 GT
    """
    print("=== Static multi-sample evaluation on test set ===")

    # Limites de génération du dataset (en rad)
    q1_min, q1_max = -np.pi / 4.0, +np.pi / 4.0       # [-45°, +45°]
    q2_min, q2_max = -np.pi / 2.0, +np.pi / 2.0       # [-90°, +90°]


    # Charger test (non normalisé)
    joints_val = np.load(JOINTS_TEST_FILE).astype(np.float32)   # (N_val, 2)
    wrench_val = np.load(WRENCH_TEST_FILE).astype(np.float32)
    wrench_val = wrench_val.reshape(-1, 1)                      # (N_val, 1)

    N_val = joints_val.shape[0]
    print(f"Nombre de poses de test : {N_val}")

    # Choix des exemples à visualiser
    n_examples = min(n_examples, N_val)
    idxs = np.random.choice(N_val, size=n_examples, replace=False)

    joints_mean = norm_stats["joints_mean"]
    joints_std  = norm_stats["joints_std"]
    wrench_mean = norm_stats["wrench_mean"]
    wrench_std  = norm_stats["wrench_std"]

    betas = betas.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alpha_bars = alpha_bars.to(DEVICE)

    model.eval()

    # Figure avec n_examples lignes et 2 colonnes (q1, q2)
    fig, axes = plt.subplots(
        n_examples, 2,
        figsize=(8, 3 * n_examples),
        squeeze=False,
        sharex=False,
    )

    samples_idx = np.arange(n_samples_per_cond)

    with torch.no_grad():
        for i_subplot, idx in enumerate(idxs):
            print(f"Exemple {i_subplot+1}/{n_examples} (idx={idx})")
            ax_q1 = axes[i_subplot, 0]
            ax_q2 = axes[i_subplot, 1]

            My = wrench_val[idx:idx+1]  # (1,1)
            q_gt = joints_val[idx]      # (2,)

            # Normalisation de My
            My_norm = (My - wrench_mean) / wrench_std
            My_norm = torch.from_numpy(My_norm.astype(np.float32)).to(DEVICE)
            My_norm = My_norm.repeat(n_samples_per_cond, 1)  # (n_samples, 1)

            # Bruit initial x_T ~ N(0,I)
            x_t = torch.randn(n_samples_per_cond, 2, device=DEVICE)

            # Reverse diffusion DDPM
            for t_step in reversed(range(T)):
                t_tensor = torch.full(
                    (n_samples_per_cond,),
                    t_step,
                    device=DEVICE,
                    dtype=torch.long,
                )
                alpha_t = alphas[t_step]
                alpha_bar_t = alpha_bars[t_step]
                beta_t = betas[t_step]

                eps_theta = model(x_t, t_tensor, My_norm)

                if t_step > 0:
                    z = torch.randn_like(x_t)
                else:
                    z = torch.zeros_like(x_t)

                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
                sigma_t = torch.sqrt(beta_t)

                x_t = coef1 * (x_t - coef2 * eps_theta) + sigma_t * z

            # x0 normalisé -> dénormalisé en rad
            x0 = x_t.cpu().numpy()                      # (n_samples, 2)
            x0_denorm = x0 * joints_std + joints_mean   # (n_samples, 2)

            # ===========================
            #  RMSE
            # ===========================
            err_q1 = np.abs(x0_denorm[:, 0] - q_gt[0])             
            err_q2 = np.abs(x0_denorm[:, 1] - q_gt[1])      


            # meilleur sample pour q1
            best_idx_q1 = np.argmin(err_q1)
            best_q1 = x0_denorm[best_idx_q1, 0]

            # meilleur sample pour q2
            best_idx_q2 = np.argmin(err_q2)
            best_q2 = x0_denorm[best_idx_q2, 1]

            # moyenne 
            mean_q1 = x0_denorm[:, 0].mean()
            mean_q2 = x0_denorm[:, 1].mean()

            # ===========================
            #  PLOTS
            # ===========================

            # ---- q1 ----
            ax_q1.scatter(
                samples_idx,
                x0_denorm[:, 0],
                s=8,
                alpha=0.3,
                label="q1 samples",
            )
            # meilleure solution
            ax_q1.scatter(
                best_idx_q1,
                best_q1,
                s=30,
                marker="o",
                color="green",
                label="best RMSE",
            )
            # moyenne
            ax_q1.scatter(
                np.mean(samples_idx),
                mean_q1,
                s=30,
                marker="o",
                color="darkblue",
                label="mean",
            )
            # GT (ligne)
            ax_q1.axhline(
                y=q_gt[0],
                color="red",
                linestyle="--",
                linewidth=2,
                label="GT q1",
            )

            #limite q1
            if i_subplot == 0:
                ax_q1.axhline(
                    y=q1_min,
                    color="black",
                    linestyle="--",
                    linewidth=3,
                    label="q1 min",
                )
                ax_q1.axhline(
                    y=q1_max,
                    color="black",
                    linestyle="--",
                    linewidth=3,
                    label="q1 max",
                )
            else:
                ax_q1.axhline(y=q1_min, color="Black", linestyle="--", linewidth=3)
                ax_q1.axhline(y=q1_max, color="Black", linestyle="--", linewidth=3)


            ax_q1.set_ylabel("q1 (rad)")
            ax_q1.grid(True)
            if i_subplot == 0:
                ax_q1.legend()

            # ---- q2 ----
            ax_q2.scatter(
                samples_idx,
                x0_denorm[:, 1],
                s=8,
                alpha=0.3,
                label="q2 samples",
            )
            # meilleure solution
            ax_q2.scatter(
                best_idx_q2,
                best_q2,
                s=30,
                marker="o",
                color="green",
                label="best RMSE",
            )
            # moyenne
            ax_q2.scatter(
                np.mean(samples_idx),
                mean_q2,
                s=30,
                marker="o",
                color="darkblue",
                label="mean",
            )
            # GT
            ax_q2.axhline(
                y=q_gt[1],
                color="red",
                linestyle="--",
                linewidth=2,
                label="GT q2",
            )

            # limites q2
            if i_subplot == 0:
                ax_q2.axhline(
                    y=q2_min,
                    color="Black",
                    linestyle="--",
                    linewidth=3,
                    label="q2 min",
                )
                ax_q2.axhline(
                    y=q2_max,
                    color="Black",
                    linestyle="--",
                    linewidth=3,
                    label="q2 max",
                )
            else:
                ax_q2.axhline(y=q2_min, color="Black", linestyle="--", linewidth=3)
                ax_q2.axhline(y=q2_max, color="Black", linestyle="--", linewidth=3)

            ax_q2.set_ylabel("q2 (rad)")
            ax_q2.set_xlabel("sample index")
            ax_q2.grid(True)
            if i_subplot == 0:
                ax_q2.legend()

            ax_q1.set_title(f"idx={idx}, My={My[0,0]:.3f} Nm, best_abs_error_q1={err_q1[best_idx_q1]:.3f},best_abs_error_q2={err_q2[best_idx_q2]:.3f} ")

    plt.tight_layout()
    out_path = out_dir / "static_multisamples_test_q1_q2_rmse.png"
    plt.savefig(out_path, dpi=150)
    print(f"Figure multi-échantillons (avec RMSE) sauvegardée dans {out_path}")

def evaluate_test_best_over_dataset(
    model,
    norm_stats,
    betas,
    alphas,
    alpha_bars,
    n_samples_per_cond=200,
    max_test_points=None,
):
    """
    Pour chaque échantillon du test set (poses statiques) :

      - on génère n_samples_per_cond solutions (q1,q2) pour My_test[i]
      - on sélectionne, pour chaque DOF séparément :
          * q1_best[i] = sample avec |q1_pred - q1_gt| minimal
          * q2_best[i] = sample avec |q2_pred - q2_gt| minimal

    Puis on trace :
      - subplot 1 : q1_best vs q1_gt (en fonction de l'index de sample)
      - subplot 2 : q2_best vs q2_gt
    """

    print("=== Global evaluation on test set: best per sample (q1, q2) ===")

    # Charger TEST brut (non normalisé)
    joints_test = np.load(JOINTS_TEST_FILE).astype(np.float32)   # (N_test, 2)
    wrench_test = np.load(WRENCH_TEST_FILE).astype(np.float32)
    wrench_test = wrench_test.reshape(-1, 1)                     # (N_test, 1)

    N_test = joints_test.shape[0]
    print(f"Nombre total de poses de test : {N_test}")

    # Optionnel : sous-échantillonner pour pas exploser le temps de sampling
    if max_test_points is not None and max_test_points < N_test:
        idxs = np.random.choice(N_test, size=max_test_points, replace=False)
        idxs = np.sort(idxs)
        print(f"On évalue sur un sous-ensemble de {max_test_points} points.")
    else:
        idxs = np.arange(N_test)

    N_eval = len(idxs)

    joints_mean = norm_stats["joints_mean"]
    joints_std  = norm_stats["joints_std"]
    wrench_mean = norm_stats["wrench_mean"]
    wrench_std  = norm_stats["wrench_std"]

    betas = betas.to(DEVICE)
    alphas = alphas.to(DEVICE)
    alpha_bars = alpha_bars.to(DEVICE)

    # Pour stocker les meilleurs prédicteurs
    best_q1 = np.zeros(N_eval, dtype=np.float32)
    best_q2 = np.zeros(N_eval, dtype=np.float32)
    gt_q1   = np.zeros(N_eval, dtype=np.float32)
    gt_q2   = np.zeros(N_eval, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for k, idx in enumerate(idxs):
            if (k + 1) % 50 == 0 or (k + 1) == N_eval:
                print(f"  Sample {k+1}/{N_eval}")

            q_gt = joints_test[idx]           # (2,)
            My   = wrench_test[idx:idx+1]     # (1,1)

            gt_q1[k] = q_gt[0]
            gt_q2[k] = q_gt[1]

            # Normalisation de My
            My_norm = (My - wrench_mean) / wrench_std
            My_norm = torch.from_numpy(My_norm.astype(np.float32)).to(DEVICE)
            My_norm = My_norm.repeat(n_samples_per_cond, 1)   # (n_samples,1)

            # x_T ~ N(0, I)
            x_t = torch.randn(n_samples_per_cond, 2, device=DEVICE)

            # Reverse diffusion DDPM
            for t_step in reversed(range(T)):
                t_tensor = torch.full(
                    (n_samples_per_cond,),
                    t_step,
                    device=DEVICE,
                    dtype=torch.long,
                )
                alpha_t = alphas[t_step]
                alpha_bar_t = alpha_bars[t_step]
                beta_t = betas[t_step]

                eps_theta = model(x_t, t_tensor, My_norm)

                if t_step > 0:
                    z = torch.randn_like(x_t)
                else:
                    z = torch.zeros_like(x_t)

                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
                sigma_t = torch.sqrt(beta_t)

                x_t = coef1 * (x_t - coef2 * eps_theta) + sigma_t * z

            # Dénormalisation -> (n_samples,2)
            x0 = x_t.cpu().numpy()
            x0_denorm = x0 * joints_std + joints_mean

            # --- erreurs absolues par DOF ---
            err_q1 = np.abs(x0_denorm[:, 0] - q_gt[0])   # (n_samples,)
            err_q2 = np.abs(x0_denorm[:, 1] - q_gt[1])   # (n_samples,)

            best_idx_q1 = np.argmin(err_q1)
            best_idx_q2 = np.argmin(err_q2)

            best_q1[k] = x0_denorm[best_idx_q1, 0]
            best_q2[k] = x0_denorm[best_idx_q2, 1]


    # ===========================
    best_q = np.stack([best_q1, best_q2], axis=1)  # (N_eval, 2)

    save_dir = out_dir
    np.save(save_dir / "best_q_test.npy", best_q)
    np.savetxt(save_dir / "best_q_test.csv", best_q, delimiter=",",
               header="q1_best_rad,q2_best_rad", comments="")

    print(f"Best q predictions saved to :")
    print(f"  -> {save_dir / 'best_q_test.npy'}")
    print(f"  -> {save_dir / 'best_q_test.csv'}")

    sample_idx = np.arange(N_eval)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # q1
    axes[0].scatter(sample_idx, gt_q1, label="q1 GT", color="red",  linewidth=1.5)
    axes[0].scatter(sample_idx, best_q1, label="q1 best", s=10, alpha=0.7)
    axes[0].set_ylabel("q1 (rad)")
    axes[0].set_title("Best q1 per test sample vs GT")
    axes[0].grid(True)
    axes[0].legend()

    # q2
    axes[1].scatter(sample_idx, gt_q2, label="q2 GT", color="red", linewidth=1.5)
    axes[1].scatter(sample_idx, best_q2, label="q2 best", s=10, alpha=0.7)
    axes[1].set_ylabel("q2 (rad)")
    axes[1].set_xlabel("test sample index")
    axes[1].set_title("Best q2 per test sample vs GT")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    out_path = out_dir / "best_over_test_q1_q2.png"
    plt.savefig(out_path, dpi=150)
    print(f"Figure globale sauvegardée dans {out_path}")

    
    return {
        "idxs": idxs,
        "gt_q1": gt_q1,
        "gt_q2": gt_q2,
        "best_q1": best_q1,
        "best_q2": best_q2,
    }


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":
    model, norm_stats, betas, alphas, alpha_bars, test_dataset = train_model()
    evaluate_static_multisamples(
        model,
        norm_stats,
        betas,
        alphas,
        alpha_bars,
        n_examples=6,                 # nombre de My différents à visualiser
        n_samples_per_cond=N_SAMPLES_PER_COND,  # nombre de solutions par My
    )

    eval_results = evaluate_test_best_over_dataset(
        model,
        norm_stats,
        betas,
        alphas,
        alpha_bars,
        n_samples_per_cond=N_SAMPLES_PER_COND,
        max_test_points=None,   # par ex. 500 
    )