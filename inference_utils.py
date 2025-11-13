import numpy as np
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import torch
import math


JOINT_NAMES = [
    'Left Hip Z', 'Left Hip X', 'Left Hip Y',
    'Left Knee Z', 'Left Ankle Z', 'Left Ankle X',
    'Right Hip Z', 'Right Hip X', 'Right Hip Y',
    'Right Knee Z', 'Right Ankle Z', 'Right Ankle X'
]
FORCE_NAMES = [
    'Left Fx', 'Left Fy', 'Left Fz',
    'Left Mx', 'Left My', 'Left Mz',
    'Right Fx', 'Right Fy', 'Right Fz',
    'Right Mx', 'Right My', 'Right Mz'
]
joint_names =JOINT_NAMES
force_names = FORCE_NAMES
    # --------- evaluation ----------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Shapes:
        y_true: [T, J], y_pred: [T, J]
    Returns dict with overall MSE/RMSE/MAE and per-joint metrics + correlation.
    """
    mse_per_joint = ((y_true - y_pred) ** 2).mean(axis=0)
    rmse_per_joint = np.sqrt(mse_per_joint)
    mae_per_joint = np.abs(y_true - y_pred).mean(axis=0)

    overall_mse = mse_per_joint.mean()
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = mae_per_joint.mean()

    corrs = []
    for j in range(y_true.shape[1]):
        if np.std(y_true[:, j]) < 1e-8 or np.std(y_pred[:, j]) < 1e-8:
            corrs.append(0.0)
        else:
            c = np.corrcoef(y_true[:, j], y_pred[:, j])[0, 1]
            corrs.append(float(c) if np.isfinite(c) else 0.0)

    return dict(
        overall_mse=float(overall_mse),
        overall_rmse=float(overall_rmse),
        overall_mae=float(overall_mae),
        mse_per_joint=mse_per_joint,
        rmse_per_joint=rmse_per_joint,
        mae_per_joint=mae_per_joint,
        correlations=np.array(corrs, dtype=np.float32)
    )

def print_metrics(metrics: Dict, degrees: bool = True):
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Overall (rad): MSE={metrics['overall_mse']:.6f} | RMSE={metrics['overall_rmse']:.6f} | MAE={metrics['overall_mae']:.6f}")

    if degrees:
        deg = 180/np.pi
        print(f"Overall (deg): RMSE={metrics['overall_rmse']*deg:.3f} | MAE={metrics['overall_mae']*deg:.3f}")

    print(f"\nPer-Joint Metrics:")
    print(f"{'Joint':<20} {'RMSE(deg)':>12} {'MAE(deg)':>12} {'Corr':>10}")
    print("-"*56)
    deg = 180/np.pi
    for i, name in enumerate(joint_names):
        rmse_d = metrics['rmse_per_joint'][i] * deg
        mae_d  = metrics['mae_per_joint'][i]  * deg
        corr   = metrics['correlations'][i]
        print(f"{name:<20} {rmse_d:>12.4f} {mae_d:>12.4f} {corr:>10.3f}")
    print("="*60)

# --------- viz ----------
def plot_sequence(y_true: np.ndarray, y_pred: np.ndarray,
                    title: str = "Force-to-Angle Prediction", in_degrees: bool = True):
    """
    Plot per-joint curves for one aligned sequence.
    """
    if in_degrees:
        y_true = y_true * (180/np.pi)
        y_pred = y_pred * (180/np.pi)
        ylab = "Angle (deg)"
    else:
        ylab = "Angle (rad)"

    T, J = y_true.shape
    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)

    t = np.arange(T)
    for j, (ax, name) in enumerate(zip(axes.flat, joint_names)):
        ax.plot(t, y_true[:, j], label="True", alpha=0.7)
        ax.plot(t, y_pred[:, j], label="Pred", alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    return fig



def visualize_noise_and_prediction(noisy_trajectory, clean_trajectory, trajectory_steps, 
                                   ground_truth, num_samples, num_joints, save_path=None):
    """
    Visualize:
    - Top row: Noisy input x fed to the model
    - Middle row: Predicted clean joints (x0)
    - Bottom row: Ground truth joints (y)
    
    Args:
        noisy_trajectory: List of noisy inputs [(B, L, J), ...]
        clean_trajectory: List of predicted clean outputs [(B, L, J), ...]
        trajectory_steps: List of step labels
        ground_truth: Ground truth tensor (B, L, J)
        num_samples: Number of samples (B)
        num_joints: Number of joints (J)
        save_path: Where to save the figure
    """
    num_snapshots = len(clean_trajectory)
    
    # Convert ground truth to CPU
    ground_truth = ground_truth.detach().cpu()
    
    # Color map for different joints
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_joints, 10)))
    if num_joints > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, num_joints))
    
    # Create separate figure for each sample
    for sample_idx in range(num_samples):
        # 3 rows: noisy input (top), clean prediction (middle), ground truth (bottom)
        fig, axes = plt.subplots(3, num_snapshots, 
                                figsize=(4*num_snapshots, 8))
        
        if num_snapshots == 1:
            axes = axes.reshape(3, 1)
        
        for snap_idx in range(num_snapshots):
            # --- Top row: Noisy input x ---
            ax_noisy = axes[0, snap_idx]
            noisy_data = noisy_trajectory[snap_idx][sample_idx].numpy()
            L = noisy_data.shape[0]
            
            for j in range(num_joints):
                ax_noisy.plot(range(L), noisy_data[:, j], 
                            color=colors[j % len(colors)], 
                            alpha=0.6, 
                            linewidth=1.5,
                            label=f'Joint {j+1}' if snap_idx == 0 else '')
            
            if snap_idx == 0:
                ax_noisy.set_ylabel('Noisy Input x', fontsize=11, fontweight='bold')
                if num_joints <= 10:
                    ax_noisy.legend(fontsize=8, loc='upper right', framealpha=0.9)
            
            ax_noisy.set_title(trajectory_steps[snap_idx], fontsize=11, fontweight='bold')
            ax_noisy.grid(alpha=0.3, linestyle='--')
            ax_noisy.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
            ax_noisy.set_xticklabels([])
            
            # --- Middle row: Predicted clean joints ---
            ax_clean = axes[1, snap_idx]
            clean_data = clean_trajectory[snap_idx][sample_idx].numpy()
            
            for j in range(num_joints):
                ax_clean.plot(range(L), clean_data[:, j], 
                            color=colors[j % len(colors)], 
                            alpha=0.8, 
                            linewidth=2)
            
            if snap_idx == 0:
                ax_clean.set_ylabel('Predicted Joints', fontsize=11, fontweight='bold')
            
            ax_clean.grid(alpha=0.3, linestyle='--')
            ax_clean.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
            ax_clean.set_xticklabels([])
            
            # --- Bottom row: Ground truth ---
            ax_gt = axes[2, snap_idx]
            gt_data = ground_truth[sample_idx].numpy()
            
            for j in range(num_joints):
                ax_gt.plot(range(L), gt_data[:, j], 
                          color=colors[j % len(colors)], 
                          alpha=0.8, 
                          linewidth=2,
                          linestyle='--')  # Dashed line to distinguish from prediction
            
            if snap_idx == 0:
                ax_gt.set_ylabel('Ground Truth', fontsize=11, fontweight='bold')
            
            ax_gt.set_xlabel('Time Steps', fontsize=10)
            ax_gt.grid(alpha=0.3, linestyle='--')
            ax_gt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
        
        plt.suptitle(f'Sample {sample_idx+1}: Denoising Process\n' + 
                    '(Top: Noisy Input | Middle: Model Prediction | Bottom: Ground Truth)', 
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Save with sample index
        # save_path_sample = save_path.replace('.png', f'_sample{sample_idx+1}.png')
        # plt.savefig(save_path_sample, dpi=150, bbox_inches='tight')
        # print(f"✓ Saved {save_path_sample}")
        # plt.close()

def visualize_per_joint_timestep(noisy, predicted, ground_truth, t,
                                 predicted_noise=None, joint_names=JOINT_NAMES, sample_idx=0, 
                                 save_dir=None, show=True):
    """
    Visualize all joints for a given timestep during denoising.

    Columns:
      1 - Ground Truth
      2 - Noisy Input
      3 - Predicted Clean (x0)
      4 - Predicted Noise (eps_pred)
    """
    # --- Handle list inputs ---
    if isinstance(noisy, list):
        noisy = noisy[-1]
    if isinstance(predicted, list):
        predicted = predicted[-1]
    if isinstance(predicted_noise, list):
        predicted_noise = predicted_noise[-1]

    # --- Move to numpy ---
    noisy_np = noisy[sample_idx].detach().cpu().numpy()
    pred_np = predicted[sample_idx].detach().cpu().numpy()
    gt_np = ground_truth[sample_idx].detach().cpu().numpy()
    noise_np = predicted_noise[sample_idx].detach().cpu().numpy() if predicted_noise is not None else None

    L, num_joints = gt_np.shape
    # if joint_names is None:
    #     joint_names = [f'Joint {j+1}' for j in range(num_joints)]

    # --- Number of columns (3 or 4 depending on noise) ---
    num_cols = 4 if noise_np is not None else 3

    fig, axes = plt.subplots(num_joints, num_cols, figsize=(4*num_cols, 2.2*num_joints), sharex=True)
    if num_joints == 1:
        axes = axes.reshape(1, num_cols)

    for j in range(num_joints):
        # 1️⃣ Ground truth
        ax_gt = axes[j, 0]
        ax_gt.plot(range(L), gt_np[:, j], color='green', lw=2)
        if j == 0:
            ax_gt.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax_gt.set_ylabel(joint_names[j], fontsize=9)
        ax_gt.grid(alpha=0.3)

        # 2️⃣ Noisy input
        ax_noisy = axes[j, 1]
        ax_noisy.plot(range(L), noisy_np[:, j], color='orange', lw=1.8)
        if j == 0:
            ax_noisy.set_title('Noisy Input', fontsize=12, fontweight='bold')
        ax_noisy.grid(alpha=0.3)

        # 3️⃣ Predicted clean
        ax_pred = axes[j, 2]
        ax_pred.plot(range(L), pred_np[:, j], color='blue', lw=2)
        if j == 0:
            ax_pred.set_title('data constructed', fontsize=12, fontweight='bold')
        ax_pred.grid(alpha=0.3)

        # 4️⃣ Predicted noise (optional)
        if noise_np is not None:
            ax_noise = axes[j, 3]
            ax_noise.plot(range(L), noise_np[:, j], color='red', lw=2)
            if j == 0:
                ax_noise.set_title('Predicted Noise (ε)', fontsize=12, fontweight='bold')
            ax_noise.grid(alpha=0.3)

    plt.suptitle(f"Timestep t = {t}", fontsize=14, fontweight='bold')
    plt.xlabel("Sequence frame index")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    
    plt.show()
 
def sample_from_gt_with_visualization(
    model, diffusion, cond, y,
    t_start:int=None, steps:int=1000, eta:float=0.0, device="cpu",
    save_path="denoise_from_gt.png", visualise=True
):
    """
    Démarre le DDIM à partir d'un x_t construit depuis la ground truth (y).
    - Si t_start est None: on prend le t_start auto (a_bar >= 1e-2).
    - Sinon, on utilise exactement le t_start demandé.
    """
    model.eval()
    cond = cond.to(device)  # [B,L,F]
    y    = y.to(device)     # [B,L,J]
    B, L, Fd = cond.shape
    J = y.size(-1)

    a_bar, _ = diffusion._buffers(device)
    eps_ = 1e-5
    min_a_bar = 1e-2

    # --- choix du t_start ---
    if t_start is None:
        valid = torch.nonzero(a_bar >= min_a_bar, as_tuple=False).flatten()
        t0 = int(valid[-1].item()) if len(valid) > 0 else int((diffusion.T - 1) * 0.98)
    else:
        t0 = int(t_start)

    # --- init depuis la GT bruitée au pas t0 ---
    true_noise0 = torch.randn(B, L, J, device=device)
    t0_vec = torch.full((B,), t0, dtype=torch.long, device=device)
    x, _ = diffusion.add_noise(y, t0_vec, noise=true_noise0)  # <= différence clé

    # --- planning ts (cosine spacing décroissant t0 -> 0) ---
    s = torch.linspace(0, 1, steps, device=device)
    s = (1 - torch.cos(math.pi * s)) / 2
    t_float = t0 * (1 - s)
    ts = torch.round(t_float).long().unique_consecutive()
    if ts[0] != t0: ts = torch.cat([torch.tensor([t0], device=device), ts])
    if ts[-1] != 0: ts = torch.cat([ts, torch.tensor([0], device=device)])

    noisy_trajectory, clean_trajectory, trajectory_steps = [], [], []
    noisy_trajectory.append(x.detach().cpu().clone())

    for i in range(len(ts)):
        t = ts[i].repeat(B)
        eps = model(x, cond, t)

        a_t = a_bar[t].view(B,1,1).clamp(eps_, 1-eps_)
        sqrt_a_t, sqrt_one_t = torch.sqrt(a_t), torch.sqrt(1 - a_t)

        # reconstruit x0 à ce pas
        x0_pred = (x - sqrt_one_t * eps) / sqrt_a_t
        
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
            
        # a_prev
        if i < len(ts) - 1:
            t_next = ts[i+1].repeat(B)
            a_prev = a_bar[t_next].view(B,1,1).clamp(eps_, 1-eps_)
        else:
            a_prev = torch.ones_like(a_t)
        sqrt_a_prev, sqrt_one_prev = torch.sqrt(a_prev), torch.sqrt(1 - a_prev)

        # update DDIM
        if eta == 0.0:
            x = (sqrt_a_prev / sqrt_a_t) * (x - sqrt_one_t * eps) + sqrt_one_prev * eps
        
        else:
            sigma_t = eta * torch.sqrt((1 - a_prev) / (1 - a_t)) * torch.sqrt(1 - a_t / a_prev)
            z = torch.randn_like(x)
            x = (sqrt_a_prev / sqrt_a_t) * (x - sqrt_one_t * eps) \
                + torch.sqrt(1 - a_prev - sigma_t**2) * eps + sigma_t * z

    return x

def analyze_single_t(model, diffusion, cond, y, t:int, device="cuda"):
    """
    Ajoute un bruit contrôlé à la GT au pas t, fait prédire eps,
    reconstruit x0, et renvoie des métriques utiles.
    """
    model.eval()
    cond = cond.to(device)      # [B,L,F]
    y    = y.to(device)         # [B,L,J]
    B    = y.size(0)

    # On force le même t pour tout le batch (tu peux aussi passer un vecteur de t)
    t_vec = torch.full((B,), int(t), dtype=torch.long, device=device)

    # Fabrique x_t avec bruit connu
    true_noise = torch.randn_like(y)
    x_t, _ = diffusion.add_noise(y, t_vec, noise=true_noise)

    # Prédit le bruit puis reconstruit x0
    pred_noise = model(x_t, cond, t_vec)
    a_bar, _ = diffusion._buffers(device)
    a_t = a_bar[t_vec].view(B,1,1).clamp(1e-5, 1-1e-5)
    x0_pred = (x_t - torch.sqrt(1 - a_t) * pred_noise) / torch.sqrt(a_t)

    # Métriques
    mse_eps  = torch.mean((pred_noise - true_noise)**2).item()
    mse_x0   = torch.mean((x0_pred - y)**2).item()

    return {
        "x_t": x_t, "true_noise": true_noise, "pred_noise": pred_noise,
        "x0_pred": x0_pred, "mse_eps": mse_eps, "mse_x0": mse_x0, "t": int(t)
    }
