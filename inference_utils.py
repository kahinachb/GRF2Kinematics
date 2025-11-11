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
                                   ground_truth, num_samples, num_joints, save_path):
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
        # plt.show()
        
        # Save with sample index
        save_path_sample = save_path.replace('.png', f'_sample{sample_idx+1}.png')
        plt.savefig(save_path_sample, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {save_path_sample}")
        plt.close()