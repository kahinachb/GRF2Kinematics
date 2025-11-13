import os, math, argparse, pickle
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from loader_utils import build_loaders_shared, visualize_collected
import matplotlib.pyplot as plt
from inference_utils import visualize_noise_and_prediction, visualize_per_joint_timestep,JOINT_NAMES

# --------------------------
# Time embedding (sin/cos) so every step in the sequence “knows” the current diffusion time.
# --------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.LongTensor):  # [B]
        device = t.device
        half = self.dim // 2
        # guard for tiny dims
        denom = max(1, half - 1)
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device).float() / denom)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, 2*half]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1), mode='constant')
        return emb  # [B, dim]mais

# --------------------------
# FiLM (optional) Feature-wise modulation
# --------------------------
class FiLM(nn.Module):
    def __init__(self, d_in:int, d_feat:int):
        super().__init__()
        self.linear = nn.Linear(d_in, 2*d_feat)
    def forward(self, cond:torch.Tensor):  # [B, d_in]
        gamma, beta = self.linear(cond).chunk(2, dim=-1)
        return gamma, beta

# --------------------------
# LSTM denoiser (predicts noise eps)
# --------------------------
class LSTMDiffusionDenoiser(nn.Module):
    """
    eps_theta(x_t, forces, t) -> [B, L, J]
    - Concatenate per-step [x_t, forces, time_emb] -> project -> LSTM
    - (Optional) FiLM modulation from global pooled cond
    """
    def __init__(self,
                 joint_dim:int,
                 cond_dim:int,
                 hidden_size:int=128,
                 num_layers:int=2,
                 dropout:float=0.2,
                 bidirectional:bool=False,
                 time_dim:int=128,
                 use_film:bool=True):
        super().__init__()
        self.joint_dim = joint_dim
        self.cond_dim  = cond_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1
        self.use_film = use_film

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        in_feat = joint_dim + cond_dim + time_dim
        self.in_proj = nn.Linear(in_feat, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        lstm_out_dim = hidden_size * self.num_dirs

        # Build a global cond representation (for FiLM + init states)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        if use_film:
            self.film = FiLM(d_in=hidden_size, d_feat=lstm_out_dim)

        # Init hidden states from global cond
        self.init_h = nn.Linear(hidden_size, self.num_layers * self.num_dirs * hidden_size)
        self.init_c = nn.Linear(hidden_size, self.num_layers * self.num_dirs * hidden_size)

        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim), nn.SiLU(),
            nn.Linear(lstm_out_dim, joint_dim)
        )

    def forward(self, x_t:torch.Tensor, cond:torch.Tensor, t:torch.LongTensor):
        B, L, J = x_t.shape
        Fd = cond.size(-1)

        # per-step features
        t_emb = self.time_emb(t)                      # [B, time_dim]
        t_rep = t_emb.unsqueeze(1).expand(B, L, -1)   # [B, L, time_dim]
        x_in  = torch.cat([x_t, cond, t_rep], dim=-1) # [B, L, J+F+T]
        x_in  = self.in_proj(x_in)                    # [B, L, H]

        # global cond
        cond_flat = cond.reshape(B*L, Fd)
        cond_per_t = self.cond_proj(cond_flat).reshape(B, L, -1)  # [B, L, H]
        c_emb = cond_per_t.mean(dim=1)                             # [B, H]

        # init states from global cond
        H = self.hidden_size
        num_layers_total = self.num_layers * self.num_dirs
        h0 = self.init_h(c_emb).view(num_layers_total, B, H).contiguous()
        c0 = self.init_c(c_emb).view(num_layers_total, B, H).contiguous()

        h_seq, _ = self.lstm(x_in, (h0, c0))          # [B, L, H*num_dirs]

        if self.use_film:
            gamma, beta = self.film(c_emb)            # [B, H*num_dirs]
            gamma = gamma.unsqueeze(1)                 # [B, 1, D]
            beta  = beta.unsqueeze(1)                  # [B, 1, D]
            h_seq = h_seq * (1 + gamma) + beta

        eps = self.head(h_seq)                         # [B, L, J]
        return eps

# --------------------------
# Diffusion core (DDPM + DDIM sample)
# --------------------------
class Diffusion:
    def __init__(self, timesteps:int=1000, beta_schedule:str="cosine"):
        self.T = timesteps
        self.register_schedule(beta_schedule)

    def register_schedule(self, schedule:str):
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.T)
        elif schedule == "cosine":
            s = 0.008
            steps = self.T + 1
            x = torch.linspace(0, self.T, steps)
            alphas_cumprod = torch.cos(((x / self.T) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(1e-8, 0.999)
        else:
            raise ValueError("Unknown schedule")
        

        self.betas = betas.to(torch.float64)
        self.alphas = (1.0 - self.betas)   #coeff [0-1]
        a_bar = torch.cumprod(self.alphas, dim=0) 
        a_bar = a_bar.clamp(min=1e-5, max=0.999999) 
        self.alphas_cumprod = a_bar.to(torch.float32)
        self.alphas = self.alphas.to(torch.float32)

    def _buffers(self, device):
        a_bar = self.alphas_cumprod.to(device)
        return a_bar, self.alphas.to(device)

    def add_noise(self, x0:torch.Tensor, t:torch.LongTensor, noise:Optional[torch.Tensor]=None): #t + at - 
        if noise is None: noise = torch.randn_like(x0)
        a_bar, _ = self._buffers(t.device)
        a_t = a_bar[t].view(-1, 1, 1)                       # [B,1,1]
        x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * noise
        return x_t, noise
    
    @torch.no_grad()
    def sample_with_visualization(self, model, cond, y,visualise=True, steps=50, eta=0.0, device="cuda", 
                                save_path="denoising_viz.png"):
        """
        Sample with visualization showing:
        - The noisy input x fed to the model
        - The predicted clean joints (x0) that DDIM produces
        - The ground truth joints (y)
        """
        model.eval()

        B, L, Fd = cond.shape
        J = model.joint_dim
        x = torch.randn(B, L, J, device=device) #initial noise

        # Store both noisy inputs and predicted clean outputs
        noisy_trajectory = []
        clean_trajectory = []
        trajectory_steps = []
        
        # Save initial noise
        noisy_trajectory.append(x.detach().cpu().clone())

        a_bar, _ = self._buffers(device)
        eps_ = 1e-5
        min_a_bar = 1e-2

        # --- choose a noisy start ---
        valid = torch.nonzero(a_bar >= min_a_bar, as_tuple=False).flatten()

        if len(valid) == 0:
            t_start = int((self.T - 1))
        else:
            t_start = int(valid[-1].item())
        
        # cosine-spaced schedule
        s = torch.linspace(0, 1, steps, device=device)
        s = (1 - torch.cos(s * math.pi)) / 2
        t_float = t_start * (1 - s)
        ts = torch.round(t_float).long()
        ts = torch.unique_consecutive(ts)
        if ts[0] != t_start: ts = torch.cat([torch.tensor([t_start], device=device), ts])
        if ts[-1] != 0:      ts = torch.cat([ts, torch.tensor([0], device=device)])

        print("a_t[0] =", a_bar[ts[0]].item(), "a_t[-1] =", a_bar[ts[-1]].item())

        # --- Denoising loop ---
        for i in range(len(ts)): 
            t = ts[i].repeat(B)
            eps = model(x, cond, t)

            a_t = a_bar[t].view(B,1,1).clamp(eps_, 1 - eps_)
            sqrt_a_t    = torch.sqrt(a_t)
            sqrt_one_t  = torch.sqrt(1 - a_t)

            if i < len(ts) - 1:
                t_next = ts[i+1].repeat(B)
                a_prev = a_bar[t_next].view(B,1,1).clamp(eps_, 1 - eps_)
            else:
                a_prev = torch.ones_like(a_t)
            sqrt_a_prev   = torch.sqrt(a_prev)
            sqrt_one_prev = torch.sqrt(1 - a_prev)

            # Predict clean data (x0)
            x0_pred = (x - sqrt_one_t * eps) / sqrt_a_t
                
            if i % 5 == 0 or i == len(ts)-1:
                print(f"[DDIM] step {i:02d} | x.std={x.std().item():.3f} "
                    f"| eps.std={eps.std().item():.3f} | x0_pred.std={x0_pred.std().item():.3f} "
                    f"| a_t={a_t.mean().item():.6f}")
            if i == 49: 
                visualize_per_joint_timestep(
                    noisy=x,
                    predicted=x0_pred,
                    ground_truth=y,
                    predicted_noise=eps,
                    t=i,
                    joint_names=JOINT_NAMES,
                    show=True
                )

            # Save every 10 steps
            if (i + 1) % 10 == 0 or i == len(ts) - 1:
                clean_trajectory.append(x0_pred.detach().cpu().clone())
                trajectory_steps.append(f"Step {i+1}")

            # DDIM update
            if eta == 0.0:
                # x = sqrt_a_prev *x0_pred + sqrt_one_prev* eps
                x = (sqrt_a_prev / sqrt_a_t) * (x - sqrt_one_t * eps) + sqrt_one_prev * eps
            else: #ddpm 
                sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - a_t / a_prev)
                z = torch.randn_like(x)
                x = (sqrt_a_prev / sqrt_a_t) * (x - sqrt_one_t * eps)+ torch.sqrt(1 - a_prev - sigma_t**2) * eps + sigma_t * z
            
            # Save noisy x after update (input for next step)
            if (i + 1) % 10 == 0 or i == len(ts) - 1:
                noisy_trajectory.append(x.detach().cpu().clone())
        
        return x
    
def reconstruct_x0_from_pred(x_t, pred_eps, t, diffusion, device, eps=1e-5):
    # x0 = (x_t - sqrt(1 - a_t) * eps_pred) / sqrt(a_t)
    a_bar, _ = diffusion._buffers(device)
    a_t = a_bar[t].view(-1, 1, 1).clamp(eps, 1 - eps)
    sqrt_a_t = torch.sqrt(a_t)
    sqrt_one_t = torch.sqrt(1 - a_t)
    x0_pred = (x_t - sqrt_one_t * pred_eps) / sqrt_a_t
    return x0_pred
# --------------------------
# Training loops
# --------------------------
def train_one_epoch(model, diffusion, loader, opt, device):
    model.train()
    total = 0.0
    for forces, joints in tqdm(loader, desc="Train"):
        forces = forces.to(device)    # [B,L,F]
        joints = joints.to(device)    # [B,L,J]
        B = forces.size(0)
        t = torch.randint(0, diffusion.T, (B,), device=device).long()#randomly picks a timestep t between 0 and T (noise strength)
        x_t, noise = diffusion.add_noise(joints, t)  #Adds the right amount of Gaussian noise to the ground-truth joints depending on t
        pred = model(x_t, forces, t)                  # predict noise
        loss = F.mse_loss(pred, noise)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / len(loader)



@torch.no_grad()
def evaluate_simple(model, diffusion, loader, device):
    model.eval()
    total = 0.0
    for forces, joints in tqdm(loader, desc="Val"):
        forces = forces.to(device)
        joints = joints.to(device)
        B = forces.size(0)
        t = torch.randint(0, diffusion.T, (B,), device=device).long() #randomly picks a timestep t between 0 and T (noise strength)
        x_t, noise = diffusion.add_noise(joints, t) #Adds the right amount of Gaussian noise to the ground-truth joints depending on t
        pred = model(x_t, forces, t)
        loss = F.mse_loss(pred, noise)
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def evaluate(model, diffusion, loader, device, max_joints=12, samples_per_batch=1):
    model.eval()
    total = 0.0
    collected = []  # liste d'items { 'gt', 'noisy', 'true_noise', 'pred_noise', 'x0_pred', 'timestep', 'batch_idx' }

    for bidx, (forces, joints) in enumerate(tqdm(loader, desc="Val")):
        forces = forces.to(device)
        joints = joints.to(device)
        B = forces.size(0)

        # Tirage t sur tout l’espace (fidèle à l’entraînement)
        t = torch.randint(0, diffusion.T, (B,), device=device).long()

        x_t, noise = diffusion.add_noise(joints, t)
        pred = model(x_t, forces, t)
        loss = F.mse_loss(pred, noise)
        total += loss.item()

        # Reconstruit x0 pour inspection
        x0_pred = reconstruct_x0_from_pred(x_t, pred, t, diffusion, device)

        # On sélectionne quelques séquences par batch pour visualisation
        take = min(samples_per_batch, B)
        idxs = torch.arange(take, device=device)

        # En CPU + numpy (pas de graph, faible mémoire)
        gt_np         = joints[idxs].detach().cpu().numpy()
        noisy_np      = x_t[idxs].detach().cpu().numpy()
        true_noise_np = noise[idxs].detach().cpu().numpy()
        pred_noise_np = pred[idxs].detach().cpu().numpy()
        x0_pred_np    = x0_pred[idxs].detach().cpu().numpy()
        t_np          = t[idxs].detach().cpu().tolist()

        collected.append({
            "gt": gt_np, "noisy": noisy_np, "true_noise": true_noise_np,
            "pred_noise": pred_noise_np, "x0_pred": x0_pred_np,
            "timesteps": t_np, "batch_idx": bidx
        })

    return total / len(loader), collected
# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", action="store_true", default=False)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--exclude_subject", type=str, default="Jovana")
    parser.add_argument("--scaler_dir", type=str, default="./scalers_diff")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--use_film", action="store_true", default=True)
    args = parser.parse_args()

    # Build loaders (note: diffusion likes normalized targets)
    train_ds, val_ds, train_loader, val_loader, Fd, Jd, train_subs, test_subs = build_loaders_shared(
        data_dir=args.data_dir,
        seq_len=args.sequence_length,
        stride_train=args.stride,
        stride_val=args.sequence_length,   # disjoint val
        batch_size=args.batch_size,
        normalize=args.normalize,
        scaler_dir=args.scaler_dir,
        test_ratio=args.test_ratio,
        exclude_subjects=[args.exclude_subject],
        num_workers=args.num_workers,
        scale_forces=True,
        scale_joints=True,   
    )

    print(f"Training subjects: {train_subs}")
    print(f"Testing subjects:  {test_subs}")
    print(f"Training sequences: {len(train_ds)} | Validation sequences: {len(val_ds)}")

    model = LSTMDiffusionDenoiser(
        joint_dim=Jd,
        cond_dim=Fd,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        time_dim=args.time_dim,
        use_film=args.use_film
    ).to(args.device)


    diffusion = Diffusion(timesteps=args.timesteps, beta_schedule="cosine")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = float("inf")
    ckpt_dir = Path("./checkpoints_diff"); ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "best_diffusion_lstm.pth"

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr = train_one_epoch(model, diffusion, train_loader, opt, args.device)
        va = evaluate_simple(model, diffusion, val_loader, args.device)

        print(f"train_loss={tr:.6f} | val_loss={va:.6f}")
        if va < best:
            best = va
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "opt": opt.state_dict(),
                "config": vars(args),
                "dims": {"F": Fd, "J": Jd}
            }, ckpt_path)
            print("✓ Saved best")
    
    print("\nLoading best checkpoint for final evaluation & visualization...")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)

    final_val_loss, collected = evaluate(
        model, diffusion, val_loader, args.device,
        max_joints=12,        
        samples_per_batch=1   
    )
    print(f"Final best-model val_loss={final_val_loss:.6f}")

    viz_dir = "./viz_diff_best"
    Path(viz_dir).mkdir(exist_ok=True, parents=True)
    visualize_collected(collected, max_joints=12, save_dir=viz_dir, prefix="val_best", epoch=ckpt["epoch"])
    print(f"✓ Saved visualizations to {viz_dir}")


    # --- Quick sample demo on a validation batch ---
    # model.eval()
    # with torch.no_grad():
    #     forces, joints = next(iter(val_loader))
    #     forces = forces.to(args.device)
    #     joints = joints.to(args.device)

    #     print("forces mean/std:", forces.mean().item(), forces.std().item())
    #     print("joints mean/std:", joints.mean().item(), joints.std().item())

    #     samples = diffusion.sample_with_visualization(
    #         model, forces,joints,visualise=True, steps=args.sample_steps, eta=0.0, device=args.device
    #     )  # [B,L,J]

    #     # If normalized, inverse-transform for inspection
    #     if args.normalize:
    #         with open(Path(args.scaler_dir) / "joint_scaler.pkl", "rb") as f:
    #             jscaler: StandardScaler = pickle.load(f)

    #         B, L, J = samples.shape
    #         preds = samples.detach().cpu().numpy().reshape(-1, J)
    #         preds = jscaler.inverse_transform(preds).reshape(B, L, J)

    #         gt = joints.detach().cpu().numpy().reshape(-1, J)
    #         gt = jscaler.inverse_transform(gt).reshape(B, L, J)
    #     else:
    #         preds = samples.detach().cpu().numpy()
    #         gt = joints.detach().cpu().numpy()

    #     # --- Plot ground truth vs prediction---
    # b = 0  # pick first sample
    # num_joints_to_plot = min(12, preds.shape[-1])  # up to 12 joints
    # fig, axes = plt.subplots(num_joints_to_plot // 3, 3, figsize=(14, 10))
    # axes = axes.flatten()

    # for j in range(num_joints_to_plot):
    #     ax = axes[j]
    #     ax.plot(gt[b, :, j], label="True", lw=0.8)
    #     ax.plot(preds[b, :, j], label="Pred", lw=0.8, alpha=0.8)
    #     ax.set_title(f"Joint {j}")
    #     ax.legend(fontsize=7)
    #     ax.grid(True)

    # for ax in axes[num_joints_to_plot:]:
    #     ax.axis("off")

    # plt.suptitle(f"Diffusion | Val sample #{b} | steps={args.sample_steps}")
    # plt.tight_layout()
    # plt.savefig('pred_vs_gt.png')

if __name__ == "__main__":
    main()





    # @torch.no_grad()
    # def sample_test(self, model:LSTMDiffusionDenoiser, cond:torch.Tensor, steps:int=50, eta:float=0.0, device="cuda"):
    #     """
    #     DDIM sampling (deterministic if eta=0).
    #     cond: [B, L, F]  -> returns x_0: [B, L, J]
    #     """
    #     B, L, Fd = cond.shape
    #     J = model.joint_dim
    #     x = torch.randn(B, L, J, device=device) #initial noise

    #     start = int(self.T * 0.999) 
    #     ts = torch.linspace(start, 0, steps, dtype=torch.long, device=device) #timesteps
    #     a_bar, alphas = self._buffers(device)

    #     eps_ = 1e-5 ####

    #     for i in range(steps):
    #         t = ts[i].repeat(B)
    #         eps = model(x, cond, t) # predict noise
    #         # a_t = a_bar[t].view(B, 1, 1)
    #         a_t = a_bar[t].view(B,1,1).clamp(eps_, 1 - eps_)  
    #         x0_pred = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)

    #         if i % 5 == 0 or i == steps-1:
    #             print(f"[DDIM] step {i:02d} | x.std={x.std().item():.3f} "
    #                 f"| eps.std={eps.std().item():.3f} | x0_pred.std={x0_pred.std().item():.3f} "
    #                 f"| a_t={a_t.mean().item():.6f}")

    #         if i < steps - 1:
    #             t_next = ts[i+1].repeat(B)
    #             # a_prev = a_bar[t_next].view(B, 1, 1)
    #             a_prev = a_bar[t_next].view(B,1,1).clamp(eps_, 1 - eps_)
    #         else:
    #             a_prev = torch.ones_like(a_t)

    #         if eta == 0.0:
    #             # deterministic DDIM
    #             x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev) * eps
    #         else:
    #             # stochastic DDIM (proper variance)
    #             sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - a_t / a_prev)
    #             z = torch.randn_like(x)
    #             x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev - sigma_t**2) * eps + sigma_t * z

    #     return x