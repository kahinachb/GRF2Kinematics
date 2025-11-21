from loader_utils import ForceToJointDataset, train_test_split_by_subject
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        """
        t: (B,) entiers [0, T_diff-1]
        retourne: (B, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.lin(emb)

# ======================
# 2. BLOCS TCN 1D
# ======================

class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.act(x + residual)

# ======================
# 3. RESEAU EPSILON (TCN)
# ======================

class EpsilonNetTCN(nn.Module):
    def __init__(self, d_angles=12, d_forces=12, d_model=128, n_blocks=6):
        super().__init__()
        self.d_angles = d_angles
        self.d_forces = d_forces
        self.in_proj = nn.Linear(d_angles + d_forces, d_model)
        self.time_emb = TimestepEmbedding(d_model)

        self.blocks = nn.ModuleList()
        dilations = [2 ** i for i in range(n_blocks)]  # 1,2,4,8,16,32...
        for d in dilations:
            self.blocks.append(ResBlock1D(d_model, kernel_size=3, dilation=d))

        self.out_conv = nn.Conv1d(d_model, d_angles, kernel_size=1)

    def forward(self, q_noisy, forces, t):
        """
        q_noisy : (B, T, 12)  (angles bruités, normalisés)
        forces  : (B, T, 12)  (forces normalisées)
        t       : (B,)
        retourne : (B, T, 12) = prédiction du bruit sur les angles
        """
        x = torch.cat([q_noisy, forces], dim=-1)  # (B, T, 24)
        x = self.in_proj(x)                      # (B, T, d_model)

        # Ajout time embedding (broadcast sur T)
        t_emb = self.time_emb(t)                 # (B, d_model)
        t_emb = t_emb.unsqueeze(1)               # (B, 1, d_model)
        x = x + t_emb                            # (B, T, d_model)

        # Passage en (B, C, T) pour Conv1d
        x = x.permute(0, 2, 1)                   # (B, d_model, T)

        for block in self.blocks:
            x = block(x)

        out = self.out_conv(x)                   # (B, d_angles, T)
        out = out.permute(0, 2, 1)               # (B, T, d_angles)
        return out

# ======================
# 4. SCHEDULER DIFFUSION
# ======================

class DiffusionScheduler:
    def __init__(self, num_steps=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_steps = num_steps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
        )

    def q_sample(self, x0, t, noise=None):
        """
        x0 : (B, T, D)
        t  : (B,) indices [0, num_steps-1]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)  # (B, 1, 1)
        return torch.sqrt(alphas_cumprod_t) * x0 + torch.sqrt(1 - alphas_cumprod_t) * noise

# ======================
# 5. SAMPLING COMPLET
# ======================

@torch.no_grad()
def sample_angles(model, scheduler, forces_norm, num_steps=None):
    """
    forces_norm : (B, T, 12) normalisées
    retourne : (B, T, 12) angles générés (toujours normalisés)
    """
    model.eval()
    device = forces_norm.device
    B, T, D = forces_norm.shape

    if num_steps is None:
        num_steps = scheduler.num_steps

    # initialisation : bruit pur
    x = torch.randn(B, T, D, device=device)

    for i in reversed(range(num_steps)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        eps = model(x, forces_norm, t)

        beta_t = scheduler.betas[i]
        alpha_t = scheduler.alphas[i]
        alpha_bar_t = scheduler.alphas_cumprod[i]
        alpha_bar_prev = scheduler.alphas_cumprod_prev[i]

        if i > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps) \
            + torch.sqrt(beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * noise

    return x  # (B, T, D) normalisé

# ======================
# 6. NORMALISATION (compute + save)
# ======================

def compute_normalization_stats(data_dir: str, subjects: List[str]):
    """
    Parcourt toutes les forces.npy et joints.npy des sujets train,
    calcule mean et std par canal pour forces et angles.
    """
    root = Path(data_dir)
    forces_sum = None
    forces_sumsq = None
    joints_sum = None
    joints_sumsq = None
    n_total = 0

    for subj in subjects:
        subj_path = root / subj
        if not subj_path.is_dir():
            continue
        for trial in subj_path.iterdir():
            if not trial.is_dir():
                continue
            f_path = trial / "forces.npy"
            j_path = trial / "joints.npy"
            if not (f_path.exists() and j_path.exists()):
                continue

            forces = np.load(f_path)  # (T, Df)
            joints = np.load(j_path)  # (T, Dq)

            if forces_sum is None:
                Df = forces.shape[1]
                Dq = joints.shape[1]
                forces_sum = np.zeros(Df, dtype=np.float64)
                forces_sumsq = np.zeros(Df, dtype=np.float64)
                joints_sum = np.zeros(Dq, dtype=np.float64)
                joints_sumsq = np.zeros(Dq, dtype=np.float64)

            forces_sum += forces.sum(axis=0)
            forces_sumsq += (forces ** 2).sum(axis=0)
            joints_sum += joints.sum(axis=0)
            joints_sumsq += (joints ** 2).sum(axis=0)
            n_total += forces.shape[0]

    forces_mean = forces_sum / n_total
    joints_mean = joints_sum / n_total
    forces_var = forces_sumsq / n_total - forces_mean ** 2
    joints_var = joints_sumsq / n_total - joints_mean ** 2
    eps = 1e-8
    forces_std = np.sqrt(np.maximum(forces_var, eps))
    joints_std = np.sqrt(np.maximum(joints_var, eps))

    stats = {
        "forces_mean": forces_mean,
        "forces_std": forces_std,
        "joints_mean": joints_mean,
        "joints_std": joints_std,
    }
    return stats

# ======================
# 7. MAIN TRAINING SCRIPT
# ======================

def main():
    data_dir = "/datasets/GRF2Kine/processed_data"
    seq_len = 128
    batch_size = 8
    num_epochs = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Split sujets ----
    train_subjects, val_subjects = train_test_split_by_subject(data_dir, test_ratio=0.2)
    print("Train subjects:", train_subjects)
    print("Val subjects:", val_subjects)

    # ---- Stats de normalisation sur TRAIN uniquement ----
    stats = compute_normalization_stats(data_dir, train_subjects)
    forces_mean = torch.tensor(stats["forces_mean"], dtype=torch.float32, device=device)
    forces_std  = torch.tensor(stats["forces_std"], dtype=torch.float32, device=device)
    joints_mean = torch.tensor(stats["joints_mean"], dtype=torch.float32, device=device)
    joints_std  = torch.tensor(stats["joints_std"], dtype=torch.float32, device=device)

    # ---- Datasets & loaders ----
    train_dataset = ForceToJointDataset(data_dir, subjects=train_subjects, seq_len=seq_len)
    val_dataset   = ForceToJointDataset(data_dir, subjects=val_subjects, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # ---- Modèle & scheduler & optim ----
    model = EpsilonNetTCN().to(device)
    scheduler = DiffusionScheduler(num_steps=200, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    ckpt_path = "diffusion_force2joints.pt"

    # ---- Boucle d'entraînement ----
    for epoch in range(num_epochs):
        model.train()
        train_loss_acc = 0.0
        n_train = 0

        for forces, joints in train_loader:
            forces = forces.to(device)  # (B, T, 12)
            joints = joints.to(device)  # (B, T, 12)

            # Normalisation
            forces_norm = (forces - forces_mean) / forces_std
            joints_norm = (joints - joints_mean) / joints_std

            B = forces.shape[0]
            t = torch.randint(0, scheduler.num_steps, (B,), device=device)

            noise = torch.randn_like(joints_norm)
            q_noisy = scheduler.q_sample(joints_norm, t, noise)

            noise_pred = model(q_noisy, forces_norm, t)
            loss = torch.mean((noise - noise_pred) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_acc += loss.item() * B
            n_train += B

        train_loss = train_loss_acc / n_train

        # ---- Validation ----
        model.eval()
        val_loss_acc = 0.0
        n_val = 0
        with torch.no_grad():
            for forces, joints in val_loader:
                forces = forces.to(device)
                joints = joints.to(device)

                forces_norm = (forces - forces_mean) / forces_std
                joints_norm = (joints - joints_mean) / joints_std

                B = forces.shape[0]
                t = torch.randint(0, scheduler.num_steps, (B,), device=device)

                noise = torch.randn_like(joints_norm)
                q_noisy = scheduler.q_sample(joints_norm, t, noise)

                noise_pred = model(q_noisy, forces_norm, t)
                val_loss = torch.mean((noise - noise_pred) ** 2)

                val_loss_acc += val_loss.item() * B
                n_val += B

        val_loss_mean = val_loss_acc / n_val

        print(f"Epoch {epoch+1}/{num_epochs} - train_loss={train_loss:.6f} - val_loss={val_loss_mean:.6f}")

        # Sauvegarde du meilleur modèle
        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_num_steps": scheduler.num_steps,
                        "stats": {
                            "forces_mean": stats["forces_mean"].tolist(),
                            "forces_std": stats["forces_std"].tolist(),
                            "joints_mean": stats["joints_mean"].tolist(),
                            "joints_std": stats["joints_std"].tolist(),
                        },
                        "epoch": epoch + 1,
                        "val_loss": best_val_loss,
                    }, ckpt_path)
            print(f"  -> Saved new best model to {ckpt_path}")

    # ======================
    # 8. INFERENCE / ANALYSE
    # ======================

    print("\n=== Inference & plots ===")

    # Recharger le meilleur modèle (optionnel mais propre)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    import numpy as npf
    f_mean_np = np.array(checkpoint["stats"]["forces_mean"])
    f_std_np  = np.array(checkpoint["stats"]["forces_std"])
    q_mean_np = np.array(checkpoint["stats"]["joints_mean"])
    q_std_np  = np.array(checkpoint["stats"]["joints_std"])

    # Une batch de validation
    forces, joints = next(iter(val_loader))
    forces = forces.to(device)
    joints = joints.to(device)

    forces_norm = (forces - forces_mean) / forces_std
    joints_norm = (joints - joints_mean) / joints_std

    B, T, D = joints.shape

    # On prend l'exemple 0 du batch
    forces_norm_0 = forces_norm[0:1]  # (1, T, D)
    joints_norm_0 = joints_norm[0:1]

    # 1) Denoising d'une version bruitée à un t fixe (par ex. t ~ 80% de la chaîne)
    t_denoise = torch.full((1,), int(0.8 * scheduler.num_steps), device=device, dtype=torch.long)
    noise = torch.randn_like(joints_norm_0)
    q_noisy_0 = scheduler.q_sample(joints_norm_0, t_denoise, noise)  # (1, T, D)

    # Prédiction du bruit
    eps_pred = model(q_noisy_0, forces_norm_0, t_denoise)  # (1, T, D)

    # Reconstruction approx de x0 (angles normalisés)
    alpha_bar_t = scheduler.alphas_cumprod[t_denoise].view(-1, 1, 1)  # (1,1,1)
    q0_hat_norm = (q_noisy_0 - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

    # 2) Sampling complet conditionné par les forces
    angles_sample_norm = sample_angles(model, scheduler, forces_norm_0, num_steps=scheduler.num_steps)

    # 3) Dénormalisation pour visualisation
    joints_gt = joints[0].cpu().numpy()              # (T, D)
    joints_noisy = q_noisy_0[0].cpu().numpy() * q_std_np + q_mean_np
    joints_denoised = q0_hat_norm[0].detach().cpu().numpy() * q_std_np + q_mean_np
    joints_sample = angles_sample_norm[0].cpu().numpy() * q_std_np + q_mean_np

    # Choisir un DOF à tracer, par ex. le premier angle (0) = hanche gauche flexion
    dofs= [0,1,2,3,4,5,6,7,8,9,10,11]
    t_axis = np.arange(T)
    for dof in dofs:
        plt.figure(figsize=(10, 6))
        plt.plot(t_axis, joints_gt[:, dof], label="GT", linewidth=2)
        plt.plot(t_axis, joints_noisy[:, dof], label="Noisy (t haut)", alpha=0.6)
        plt.plot(t_axis, joints_denoised[:, dof], label="Denoised (1-step x0_hat)", alpha=0.8)
        plt.plot(t_axis, joints_sample[:, dof], label="Full sampled", linestyle="--")
        plt.xlabel("Frame")
        plt.ylabel(f"Joint angle DOF {dof}")
        plt.title("Comparaison GT / noisy / denoised / sampled")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{dof}_pred_vs_gt.png')

if __name__ == "__main__":
    main()

