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

DATA_DIR = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model"  

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
NUM_EPOCHS = 500
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

# Sampling / eval
N_SAMPLES_PER_COND = 200  # nb de samples par My pour la trajectoire de test
output_dir = "minimal_model"
out_dir= Path(output_dir)
out_dir.mkdir(exist_ok=True)
MODEL_SAVE_PATH = "minimal_model/cond_diffusion_model.pt"


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
    plt.savefig("minimal_model/loss_curves.png", dpi=150)
    print("Courbe de loss sauvegardée dans loss_curves.png")

    return model, norm_stats, betas, alphas, alpha_bars, test_dataset


# ============================================================
# 8. Sampling conditionnel sur une trajectoire de test
# ============================================================

def sample_trajectory_from_model(model, My_traj, norm_stats,
                                 betas, alphas, alpha_bars,
                                 n_samples_per_cond=200):
    """
    My_traj: np.array (N_test,1) -> My non normalisé pour chaque pas de temps
    Retourne :
        samples: (N_test, n_samples_per_cond, 2) en rad
        mean_traj: (N_test, 2)
        std_traj: (N_test, 2)
    """
    model.eval()
    with torch.no_grad():
        joints_mean = norm_stats["joints_mean"]
        joints_std  = norm_stats["joints_std"]
        wrench_mean = norm_stats["wrench_mean"]
        wrench_std  = norm_stats["wrench_std"]

        N_test = My_traj.shape[0]

        betas = betas.to(DEVICE)
        alphas = alphas.to(DEVICE)
        alpha_bars = alpha_bars.to(DEVICE)

        all_samples = []

        for i in range(N_test):
            My = My_traj[i:i+1]  # (1,1)
            My_norm = (My - wrench_mean) / wrench_std
            My_norm = torch.from_numpy(My_norm.astype(np.float32)).to(DEVICE)
            My_norm = My_norm.repeat(n_samples_per_cond, 1)  # (n_samples,1)

            x_t = torch.randn(n_samples_per_cond, 2, device=DEVICE)

            for t_step in reversed(range(T)):
                t = torch.full((n_samples_per_cond,), t_step, device=DEVICE, dtype=torch.long)
                alpha_t = alphas[t_step]
                alpha_bar_t = alpha_bars[t_step]
                beta_t = betas[t_step]

                eps_theta = model(x_t, t, My_norm)

                if t_step > 0:
                    z = torch.randn_like(x_t)
                else:
                    z = torch.zeros_like(x_t)

                coef1 = 1.0 / torch.sqrt(alpha_t)
                coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
                sigma_t = torch.sqrt(beta_t)

                x_t = coef1 * (x_t - coef2 * eps_theta) + sigma_t * z

            x0 = x_t.cpu().numpy()                   # (n_samples,2) normalisé
            x0_denorm = x0 * joints_std + joints_mean  # (n_samples,2) en rad
            all_samples.append(x0_denorm)

        all_samples = np.stack(all_samples, axis=0)  # (N_test, n_samples, 2)

        mean_traj = all_samples.mean(axis=1)         # (N_test,2)
        std_traj  = all_samples.std(axis=1)          # (N_test,2)

        return all_samples, mean_traj, std_traj


def evaluate_on_test_trajectory(model, norm_stats, betas, alphas, alpha_bars):
    # On charge la trajectoire test brute (ordre = temps)
    joints_test = np.load(JOINTS_TEST_FILE).astype(np.float32)   # (N_test,2)
    wrench_test = np.load(WRENCH_TEST_FILE).astype(np.float32).reshape(-1, 1)  # (N_test,1)

    # Génération de samples conditionnés en suivant la trajectoire de My
    samples, mean_traj, std_traj = sample_trajectory_from_model(
        model,
        wrench_test,
        norm_stats,
        betas,
        alphas,
        alpha_bars,
        n_samples_per_cond=N_SAMPLES_PER_COND,
    )

    # RMSE entre mean_traj et trajectoire GT
    diff = mean_traj - joints_test
    mse_q1 = np.mean(diff[:, 0] ** 2)
    mse_q2 = np.mean(diff[:, 1] ** 2)
    rmse_q1 = np.sqrt(mse_q1)
    rmse_q2 = np.sqrt(mse_q2)

    print(f"RMSE q1 (rad) sur la trajectoire test : {rmse_q1:.6f}")
    print(f"RMSE q2 (rad) sur la trajectoire test : {rmse_q2:.6f}")

    # Figures de trajectoire
    t = np.arange(joints_test.shape[0])

    # q1
    plt.figure(figsize=(10, 4))
    plt.plot(t, joints_test[:, 0], label="q1 GT", color="black")
    plt.plot(t, mean_traj[:, 0], label="q1 mean pred", color="blue")
    plt.fill_between(
        t,
        mean_traj[:, 0] - std_traj[:, 0],
        mean_traj[:, 0] + std_traj[:, 0],
        color="blue",
        alpha=0.2,
        label="q1 ±1σ"
    )
    plt.xlabel("time index")
    plt.ylabel("q1 (rad)")
    plt.title("Trajectoire q1 : GT vs mean prédiction diffusion")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("minimal_model/traj_q1.png", dpi=150)

    # q2
    plt.figure(figsize=(10, 4))
    plt.plot(t, joints_test[:, 1], label="q2 GT", color="black")
    plt.plot(t, mean_traj[:, 1], label="q2 mean pred", color="red")
    plt.fill_between(
        t,
        mean_traj[:, 1] - std_traj[:, 1],
        mean_traj[:, 1] + std_traj[:, 1],
        color="red",
        alpha=0.2,
        label="q2 ±1σ"
    )
    plt.xlabel("time index")
    plt.ylabel("q2 (rad)")
    plt.title("Trajectoire q2 : GT vs mean prédiction diffusion")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("minimal_model/traj_q2.png", dpi=150)

    print("Figures de trajectoire sauvegardées : traj_q1.png, traj_q2.png")


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":
    model, norm_stats, betas, alphas, alpha_bars, test_dataset = train_model()
    evaluate_on_test_trajectory(model, norm_stats, betas, alphas, alpha_bars)
    print("Terminé.")
