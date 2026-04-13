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

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.diffuser_utils   import DDPM, DiffusionTransformer
from dataset import build_datasets


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state: dict, path: Path):
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(path: Path, model, optimizer, scheduler, device):
    ckpt      = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt["epoch"] + 1
    best_val    = ckpt.get("best_val_loss", float("inf"))
    print(f"  [Checkpoint] Resumed from epoch {ckpt['epoch']}  (best val={best_val:.6f})")
    return start_epoch, best_val


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, ddpm, loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0.0

    for step, (joints, kinetics) in enumerate(loader):
        joints   = joints.to(device)    # (B, T, len(dof))
        kinetics = kinetics.to(device)  # (B, T, 12)
        B        = joints.size(0)

        # 1. Sample a random diffusion timestep per sample
        t = torch.randint(0, ddpm.n_steps, (B,), device=device)

        # 2. Sample Gaussian noise
        noise = torch.randn_like(joints)   # (B, T, len(dof))

        # 3. Forward diffusion: corrupt the clean joints
        x_t = ddpm.sample_forward(joints, t, noise)   # (B, T, len(dof))

        # 4. Normalize t to [0, 1] for the model
        t_norm = t.float() / ddpm.n_steps

        # 5. Predict the noise
        eps_pred = model(x_t, t_norm, kinetics)        # (B, T, len(dof))

        # 6. Simple MSE loss on the noise prediction (DDPM objective)
        loss = F.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
       #

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, ddpm, loader, device):
    model.eval()
    total_loss = 0.0

    for joints, kinetics in loader:
        joints   = joints.to(device)
        kinetics = kinetics.to(device)
        B        = joints.size(0)

        t     = torch.randint(0, ddpm.n_steps, (B,), device=device)
        noise = torch.randn_like(joints)
        x_t   = ddpm.sample_forward(joints, t, noise)
        t_norm= t.float() / ddpm.n_steps

        eps_pred   = model(x_t, t_norm, kinetics)
        total_loss += F.mse_loss(eps_pred, noise).item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train motion diffusion model")
    # Data
    parser.add_argument("--npy-root",   required=True, help="Path to the npy/ folder")
    parser.add_argument("--seq-len",    type=int,   default=128,  help="Window size in frames (default 128 = 1.28 s)")
    parser.add_argument("--stride",     type=int,   default=None, help="Stride between windows (default seq_len//2)")
    parser.add_argument("--val-ratio",    type=float, default=0.15,
                        help="Fraction of subjects per dataset held out for val (default 0.15)")
    parser.add_argument("--test-ratio",   type=float, default=0.15,
                        help="Fraction of subjects per dataset held out for test (default 0.15)")
    parser.add_argument("--val-subjects", nargs="+",  default=None,
                        help="Explicit val subject names, e.g. --val-subjects Anais/S03 Vinc/S02. "
                             "Overrides --val-ratio if provided.")
    parser.add_argument("--test-subjects",nargs="+",  default=None,
                        help="Explicit test subject names. Overrides --test-ratio if provided.")
    # Model
    parser.add_argument("--embed-dim",  type=int,   default=128,  help="Transformer embedding dimension (default 128)")
    parser.add_argument("--nhead",      type=int,   default=4,    help="Number of attention heads (default 4, must divide embed-dim)")
    parser.add_argument("--num-layers", type=int,   default=4,    help="Number of Transformer encoder layers (default 4)")
    parser.add_argument("--dropout",    type=float, default=0.1,  help="Dropout probability")
    # DDPM
    parser.add_argument("--n-steps",    type=int,   default=1000, help="Number of diffusion steps")
    parser.add_argument("--min-beta",   type=float, default=1e-4, help="Beta schedule minimum")
    parser.add_argument("--max-beta",   type=float, default=0.02, help="Beta schedule maximum")
    # Training
    parser.add_argument("--epochs",     type=int,   default=300,  help="Total training epochs")
    parser.add_argument("--batch-size", type=int,   default=64,   help="Batch size")
    parser.add_argument("--lr",         type=float, default=3e-4, help="Peak learning rate (AdamW)")
    parser.add_argument("--weight-decay",type=float,default=1e-4, help="AdamW weight decay")
    parser.add_argument("--num-workers",type=int,   default=4,    help="DataLoader worker count")
    # I/O
    parser.add_argument("--out-dir",    default="checkpoints",    help="Directory for checkpoints and normalizers")
    parser.add_argument("--resume",     default=None,             help="Path to checkpoint to resume from")
    parser.add_argument("--log-every",  type=int,   default=10,   help="Print training loss every N steps")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*64}")
    print("  MOTION DIFFUSION TRAINING")
    print(f"  Device  : {device}")
    print(f"  npy root: {args.npy_root}")
    print(f"{'═'*64}\n")

    # ── Datasets & DataLoaders ───────────────────────────────────────────────
    norm_prefix = str(out_dir / "normalizer")
    train_ds, val_ds, test_ds, joint_norm, kinetics_norm = build_datasets(
        npy_root       = args.npy_root,
        seq_len        = args.seq_len,
        stride         = args.stride,
        val_ratio      = args.val_ratio,
        test_ratio     = args.test_ratio,
        norm_save_path = norm_prefix,
        val_subjects   = args.val_subjects,
        test_subjects  = args.test_subjects,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = DiffusionTransformer(
        joint_dim  = 35,
        force_dim  = 12,
        embed_dim  = args.embed_dim,
        nhead      = args.nhead,
        num_layers = args.num_layers,
        seq_len    = args.seq_len,
        dropout    = args.dropout,
    ).to(device)

    ddpm = DDPM(device     = device,
                n_steps    = args.n_steps,
                min_beta   = args.min_beta,
                max_beta   = args.max_beta)

    n_params = count_parameters(model)
    print(f"\n  Model parameters : {n_params:,}")
    print(f"  Train windows    : {len(train_ds):,}")
    print(f"  Val   windows    : {len(val_ds):,}")
    print(f"  Test  windows    : {len(test_ds):,}\n")

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    # Cosine annealing with linear warm-up via OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr        = args.lr,
        epochs        = args.epochs,
        steps_per_epoch = len(train_loader),
        pct_start     = 0.05,    # 5% of training for warm-up
        anneal_strategy = "cos",
    )

    # ── Optional resume ──────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float("inf")
    if args.resume:
        start_epoch, best_val = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device
        )

    # Save training config
    cfg_path = out_dir / "train_config.json"
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Training loop ────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, ddpm, train_loader,
                                     optimizer, device, scheduler)
        val_loss   = evaluate(model, ddpm, val_loader, device)
        

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"  Epoch {epoch+1:4d}/{args.epochs}  |  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  |  "
              f"{elapsed:.1f}s  |  lr={optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val,
                "args":          vars(args),
            }, out_dir / "best.pt")

        # Save latest checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val,
                "args":          vars(args),
            }, out_dir / "last.pt")

    # Save loss history
    history_path = out_dir / "loss_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"\n  Training complete.  Best val loss: {best_val:.6f}")

    # ── Final evaluation on the test set ─────────────────────────────────────
    # Load the best checkpoint before evaluating
    print(f"\n  Loading best checkpoint for test evaluation...")
    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_loss = evaluate(model, ddpm, test_loader, device)
    print(f"\n{'═'*52}")
    print(f"  TEST SET EVALUATION  (unseen subjects)")
    print(f"{'═'*52}")
    print(f"  Best val loss  : {best_val:.6f}")
    print(f"  Test loss      : {test_loss:.6f}")
    gap = test_loss - best_val
    print(f"  Gap (test-val) : {gap:+.6f}"
          f"  {'⚠ possible overfit' if gap > 0.01 else '✓ consistent'}")
    print(f"{'═'*52}\n")

    # Append test loss to history
    history["test_loss"] = test_loss
    with open(history_path, "w") as f:
        json.dump(history, f)

    print(f"  Checkpoints saved in: {out_dir}/\n")


if __name__ == "__main__":
    main()