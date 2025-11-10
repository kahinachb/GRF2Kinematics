import os, math, argparse, pickle
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils import build_loaders_shared

# =========================
# Utilities: time embeddings
# =========================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.LongTensor):  # [B]
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device).float() / (half - 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 2*half]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1), mode='constant')
        return emb  # [B, dim]

# =========================
# Model: 1D Conditional U-Net (no pooling, dilated)
# =========================
class FiLM(nn.Module):
    """Feature-wise Linear Modulation from a global conditioning vector."""
    def __init__(self, d_in:int, d_feat:int):
        super().__init__()
        self.linear = nn.Linear(d_in, 2*d_feat)  # gamma, beta
    def forward(self, cond:torch.Tensor):  # [B, d_in]
        x = self.linear(cond)
        gamma, beta = x.chunk(2, dim=-1)
        return gamma, beta

class ResBlock1D(nn.Module):
    def __init__(self, d_in:int, d_out:int, time_dim:int, cond_dim:int, dilation:int=1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_in, d_out, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(d_out, d_out, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, d_out)
        self.norm2 = nn.GroupNorm(8, d_out)
        self.act = nn.SiLU()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, d_out))
        self.film = FiLM(cond_dim, d_out)
        self.skip = nn.Conv1d(d_in, d_out, 1) if d_in != d_out else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        """
        x: [B, C, L]; t_emb: [B, time_dim]; c_emb: [B, cond_dim]
        """
        gamma, beta = self.film(c_emb)  # [B, C]
        h = self.conv1(x)
        h = self.norm1(h)
        # add time embedding
        h = h + self.time_mlp(t_emb).unsqueeze(-1)
        # FiLM
        h = h * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.skip(x))

class DenoiserUNet1D(nn.Module):
    """
    Predicts epsilon for DDPM: eps_theta(x_t, forces, t)
    Inputs:
      x_t:    [B, L, J]  (noisy joints)
      cond:   [B, L, F]  (forces)
      t:      [B]        (integer timesteps)
    """
    def __init__(self, joint_dim:int, cond_dim:int, base_dim:int=128, time_dim:int=128, n_blocks:int=6):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, base_dim), nn.SiLU(),
            nn.Linear(base_dim, base_dim)
        )  # per-timestep projection (used via global pooling for FiLM)
        self.in_proj = nn.Conv1d(joint_dim + cond_dim, base_dim, kernel_size=3, padding=1)

        blocks = []
        dims = [base_dim]*n_blocks
        dilations = [1,2,4,8,4,2] if n_blocks>=6 else [1]*n_blocks
        for i in range(n_blocks):
            blocks.append(ResBlock1D(
                d_in=dims[i-1] if i>0 else base_dim,
                d_out=dims[i],
                time_dim=time_dim,
                cond_dim=base_dim,   # FiLM from pooled cond
                dilation=dilations[i]
            ))
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Sequential(
            nn.Conv1d(base_dim, base_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(base_dim, joint_dim, 3, padding=1)
        )

    def forward(self, x_t:torch.Tensor, cond:torch.Tensor, t:torch.LongTensor):
        """
        x_t:  [B, L, J]
        cond: [B, L, F]
        """
        B, L, J = x_t.shape
        _, _, Fd = cond.shape
        # concat along channel dim for input
        x_in = torch.cat([x_t, cond], dim=-1).permute(0,2,1)  # [B, J+F, L] -> [B, C, L]
        h = self.in_proj(x_in)                                # [B, C, L]

        t_emb = self.time_emb(t)                              # [B, time_dim]
        # per-timestep cond embedding then global pool (mean over time) for FiLM
        cond_flat = cond.reshape(B*L, Fd)
        cond_per_t = self.cond_proj(cond_flat).reshape(B, L, -1)  # [B, L, C]
        c_emb = cond_per_t.mean(dim=1)                         # [B, C]

        for blk in self.blocks:
            h = blk(h, t_emb, c_emb)

        eps = self.out(h).permute(0,2,1)  # [B, L, J]
        return eps

# =========================
# Diffusion core (DDPM + DDIM sampler)
# =========================
class Diffusion:
    def __init__(self, timesteps:int=1000, beta_schedule:str="cosine"):
        self.T = timesteps
        self.register_schedule(beta_schedule)

    def register_schedule(self, schedule:str):
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.T)
        elif schedule == "cosine":
            # as in Nichol & Dhariwal 2021
            s = 0.008
            steps = self.T + 1
            x = torch.linspace(0, self.T, steps)
            alphas_cumprod = torch.cos(((x / self.T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(1e-8, 0.999)
        else:
            raise ValueError("Unknown schedule")
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x0:torch.Tensor, t:torch.LongTensor, noise:Optional[torch.Tensor]=None):
        """
        q(x_t | x_0) = sqrt(a_bar_t) x_0 + sqrt(1-a_bar_t) eps
        x0: [B, L, J]
        """
        if noise is None: noise = torch.randn_like(x0)
        a_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)                 # [B,1,1]
        sig = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return a_bar * x0 + sig * noise, noise

    @torch.no_grad()
    def sample(self, model:DenoiserUNet1D, cond:torch.Tensor, steps:int=50, eta:float=0.0, device="cuda"):
        """
        DDIM sampling: start from N(0,I), do 'steps' deterministic updates.
        cond: [B, L, F]
        returns x_0: [B, L, J]
        """
        B, L, Fd = cond.shape
        J = model.out[-1].out_channels if hasattr(model.out[-1], 'out_channels') else None
        # we can infer J from model input by running a dummy
        x = torch.randn(B, L, J if J is not None else 12, device=device)  # init noise
        # make a time schedule subset
        ts = torch.linspace(self.T-1, 0, steps, dtype=torch.long, device=device)
        alphas = self.alphas.to(device); a_bar = self.alphas_cumprod.to(device)
        for i in range(steps):
            t = ts[i].repeat(B)  # [B]
            eps = model(x, cond, t)
            a_t = a_bar[t].view(B,1,1)
            # a_prev = a_bar[ts[i+1]].view(B,1,1) if i < steps-1 else torch.ones_like(a_t)
            if i < steps - 1:
                t_next = ts[i+1].repeat(B)             # [B]
                a_prev = a_bar[t_next].view(B, 1, 1)   # [B,1,1]
            else:
                a_prev = torch.ones_like(a_t) 
            # DDIM update
            x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            dir_xt = torch.sqrt(1 - a_prev) * eps
            x = torch.sqrt(a_prev) * x0_pred + dir_xt * (1.0 if eta>0 else 0.0)
        return x

# =========================
# Training
# =========================
def train_one_epoch(model, diffusion, loader, opt, device):
    model.train()
    total = 0.0
    for forces, joints in tqdm(loader, desc="Train"):
        forces = forces.to(device)    # [B,L,F]
        joints = joints.to(device)    # [B,L,J]
        B = forces.size(0)
        t = torch.randint(0, diffusion.T, (B,), device=device).long()
        x_t, noise = diffusion.add_noise(joints, t)  # noisy joints + target noise
        pred = model(x_t, forces, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate(model, diffusion, loader, device):
    model.eval()
    total = 0.0
    for forces, joints in tqdm(loader, desc="Val"):
        forces = forces.to(device)
        joints = joints.to(device)
        B = forces.size(0)
        t = torch.randint(0, diffusion.T, (B,), device=device).long()
        x_t, noise = diffusion.add_noise(joints, t)
        pred = model(x_t, forces, t)
        loss = F.mse_loss(pred, noise)
        total += loss.item()
    return total / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_dim", type=int, default=128)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--exclude_subject", type=str, default="Jovana")
    parser.add_argument("--scaler_dir", type=str, default="./scalers_diff")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=50)

    
    args = parser.parse_args()


    # Build loaders with the same utility
    train_ds, val_ds, train_loader, val_loader, Fd, Jd, train_subs, test_subs = build_loaders_shared(
        data_dir=args.data_dir,
        seq_len=args.seq_len,
        stride_train=args.stride,
        stride_val=args.seq_len,    # disjoint val
        batch_size=args.batch_size,
        normalize=args.normalize,
        scaler_dir=args.scaler_dir,
        test_ratio=args.test_ratio,
        exclude_subjects=[args.exclude_subject],
        num_workers=args.num_workers,
        scale_forces=True,
        scale_joints=True,          # Diffusion often benefits from normalized targets
    )

    print(f"Train subjects: {train_subs}")
    print(f"Test subjects : {test_subs}")
    print(f"Train sequences: {len(train_ds)} | Val sequences: {len(val_ds)}")

    # Use Fd, Jd for model dims
    model = DenoiserUNet1D(joint_dim=Jd, cond_dim=Fd, base_dim=args.base_dim,
                        time_dim=args.time_dim, n_blocks=args.n_blocks).to(args.device)

    # Model + diffusion
    model = DenoiserUNet1D(joint_dim=Jd, cond_dim=Fd, base_dim=args.base_dim,
                           time_dim=args.time_dim, n_blocks=args.n_blocks).to(args.device)
    diffusion = Diffusion(timesteps=args.timesteps, beta_schedule="cosine")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)


    best = float("inf")
    ckpt_dir = Path("./checkpoints_diff"); ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr = train_one_epoch(model, diffusion, train_loader, opt, args.device)
        va = evaluate(model, diffusion, val_loader, args.device)
        print(f"train_loss={tr:.6f} | val_loss={va:.6f}")
        if va < best:
            best = va
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "opt": opt.state_dict(),
                "config": vars(args),
                "dims": {"F": Fd, "J": Jd}
            }, ckpt_dir / "best_diffusion.pth")
            print("✓ Saved best")

    # --- Quick sample demo on a validation batch ---
    with torch.no_grad():
        forces, joints = next(iter(val_loader))
        forces = forces.to(args.device)
        samples = diffusion.sample(model, forces, steps=args.sample_steps, eta=0.0, device=args.device)  # [B,L,J]
        # If normalized, you may want to inverse-transform for inspection:
        if args.normalize:
            with open(Path(args.scaler_dir)/"joint_scaler.pkl","rb") as f:
                jscaler:StandardScaler = pickle.load(f)
            B,L,J = samples.shape
            arr = samples.detach().cpu().numpy().reshape(-1, J)
            arr = jscaler.inverse_transform(arr).reshape(B, L, J)
            np.save("pred_joints_sample.npy", arr)
            print("Saved one sample batch to pred_joints_sample.npy")

if __name__ == "__main__":
    main()
