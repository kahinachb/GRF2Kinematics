"""
train.py
========
Training loop for the motion diffusion model.

Usage:
    python train.py --npy-root /path/to/npy
    python train.py --npy-root /path/to/npy --epochs 200 --batch-size 64
    python train.py --npy-root /path/to/npy --resume checkpoints/last.pt

Checkpoints and normalizers are saved in --out-dir (default: ./checkpoints/).
"""
from dataset import Normalizer

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.diffuser_utils   import DDPM, DiffusionTransformer
from dataset import build_datasets
import matplotlib.pyplot as plt
import numpy as np
# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: dict, path: Path):
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")



# ─────────────────────────────────────────────────────────────────────────────
# OVERFIT TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train_overfit(model, ddpm, fixed_batch, optimizer, device):
    """one fixed batch."""
    model.train()
    joints, kinetics = fixed_batch
    joints   = joints.to(device)    
    kinetics = kinetics.to(device)  # (B, T, 18)
    B        = joints.size(0)
    # print(kinetics.min())
    # print(kinetics.max())
    # print(joints.min())
    # print(joints.max())
    # input()
    t = torch.randint(0, ddpm.n_steps, (B,), device=device)
    # t = torch.full((B,), 500, device=device) # Hardcode to middle-range noise
    noise = torch.randn_like(joints)
    x_t = ddpm.sample_forward(joints, t, noise)
    t_norm = t.float() / ddpm.n_steps
    eps_pred = model(x_t, t_norm, kinetics)
    
    loss = F.mse_loss(eps_pred, noise)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    # scheduler.step()

    return loss.item()

def plot_results_comparison(gt_real, pred_real, out_dir, epoch):
    """
    gt_real/pred_real : (B, T, len(dofs)) 
    """
    plot_path = out_dir / f"plots_real_scale_epoch_{epoch+1}"
    plot_path.mkdir(parents=True, exist_ok=True)

    B, T, D = gt_real.shape
    time_axis = np.arange(T) # frames

    idx_to_plot = 0
    
    dof_names = [f"DOF_{i}" for i in range(D)]
    # dof_names[0:6] = ["delta_x","delta_y","delta_z","delta_rx","delta_ry","delta_rz"]
    dof_names = [
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",
]
    
#     dof_names[18:] =[
#     "Lumbar_flex_ext", "Lumbar_lateral_flex",
#     "Lcalvicule_x",
#     "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
#     "Lelbow_flex_ext", "Lelbow_pron_supi",
#     "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
#     "rcalvicule_x",
#     "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot",
#     "Relbow_flex_ext", "Relbow_pron_supi",
# ]

    for d in range(D):
        plt.figure(figsize=(12, 6))
        
        plt.plot(time_axis, gt_real[idx_to_plot, :, d], 
                 color='black', label='GT', linewidth=2, alpha=0.8)
        plt.plot(time_axis, pred_real[idx_to_plot, :, d], 
                 color='red', label='predicetd', linewidth=1.5)

        plt.title(f"{dof_names[d]} (Epoch {epoch+1})", fontsize=14)
        plt.xlabel("Frames")
        plt.grid(True, which='both', linestyle='--', alpha=0.5)

        file_name = f"dof_{d:02d}_{dof_names[d]}.png"
        # plt.savefig(plot_path / file_name)
        plt.show()
        # plt.close() 
def plot_loss_curve(loss_history, out_dir):

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, color='tab:blue', label='Train Loss (MSE)')
    
    plt.title("train loss", fontsize=14)
    plt.xlabel("epochs")
    plt.ylabel("MSE Loss (sur le bruit prédit)")
    plt.yscale('log') 
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # loss_path = out_dir / "loss_curve.png"
    # plt.savefig(loss_path)
    # plt.close()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy-root", required=True)
    parser.add_argument("--epochs", type=int, default=1000) 
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=256) 
    parser.add_argument("--nhead", type=int, default=8)       
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--out-dir", default="checkpoints_overfit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds, _, _,joint_norm, kinetics_norm = build_datasets(npy_root=args.npy_root, seq_len=128)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────────
    model = DiffusionTransformer(
        joint_dim=12, force_dim=18, 
        embed_dim=args.embed_dim, nhead=args.nhead, 
        num_layers=4, seq_len=128, dropout=0.0 
    ).to(device)

    n_params = count_parameters(model)
    print(f"\n  Model parameters : {n_params:,}")
    print(f"  Train windows    : {len(train_ds):,}")

    ddpm = DDPM(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


    data_iter = iter(train_loader)
    fixed_batch = next(data_iter) # first batch

   
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #                         optimizer, 
    #                         max_lr=5e-4,             
    #                         epochs=args.epochs, 
    #                         steps_per_epoch=1,
    #                         pct_start=0.1,           # On monte vite (pendant 10% du temps)
    #                         anneal_strategy='cos',   # Descente en cosinus (très fluide)
    #                         final_div_factor=100,    # Le LR final sera 100x plus petit que le max
    #                     )

    print(f"\n🚀 Starting OVERFIT test...")
    loss_history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        
        loss = train_overfit(model, ddpm, fixed_batch, optimizer, device)
        loss_history.append(loss)
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"Epoch {epoch+1:4d}/{args.epochs} | Loss: {loss:.6f} | lr: {optimizer.param_groups[0]['lr']:.2e}")
            

        # if (epoch + 1) == args.epochs:
        #     torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args), "best_val_loss": loss}, 
        #                out_dir / "overfit_model.pt")
    plot_loss_curve(loss_history, out_dir)
    print(f"\n inference...")
    
    model.eval()
    joints_gt_norm, kinetics_norm_batch = fixed_batch
    joints_gt_norm = joints_gt_norm.to(device)
    kinetics_norm_batch = kinetics_norm_batch.to(device)

    with torch.no_grad():
        joints_gen_norm = ddpm.generate(model, kinetics_norm_batch, joint_dim=12)

    joints_gen_norm_np = joints_gen_norm.cpu().numpy()
    joints_gt_norm_np  = joints_gt_norm.cpu().numpy()

    joints_gen_real = joint_norm.inverse_transform(joints_gen_norm_np)
    joints_gt_real  = joint_norm.inverse_transform(joints_gt_norm_np)

    kinetics_real= kinetics_norm.inverse_transform(kinetics_norm_batch)
    print(kinetics_real)

    print(joints_gt_real)
    plot_results_comparison(joints_gt_real, joints_gen_real, out_dir, epoch)
   

if __name__ == "__main__":
    main()

