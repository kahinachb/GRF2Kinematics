import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


# ============================================================
# 1. Config avec dataclass
# ============================================================
@dataclass
class DiffusionConfig:
    """diffusion hyperparam"""
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingConfig:
    """hyperparam config"""
    batch_size: int = 128
    num_epochs: int = 150
    learning_rate: float = 1e-3
    num_workers: int = 0


@dataclass
class DataConfig:
    """data"""
    data_dir: str = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique"
    output_dir: str = "minimal_model_local"
    
    @property
    def joints_train_file(self) -> str:
        return os.path.join(self.data_dir, "joints_train.npy")
    
    @property
    def wrench_train_file(self) -> str:
        return os.path.join(self.data_dir, "wrench_train.npy")
    
    @property
    def joints_val_file(self) -> str:
        return os.path.join(self.data_dir, "joints_val.npy")
    
    @property
    def wrench_val_file(self) -> str:
        return os.path.join(self.data_dir, "wrench_val.npy")
    
    @property
    def joints_test_file(self) -> str:
        return os.path.join(self.data_dir, "joints_test.npy")
    
    @property
    def wrench_test_file(self) -> str:
        return os.path.join(self.data_dir, "wrench_test.npy")


# ============================================================
# 2. Diffusion Process Class
# ============================================================

class DiffusionProcess:
    """
    - Forward process (add noise)
    - Reverse process (sampling)
    """
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.betas = self._make_beta_schedule().to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_betas = torch.sqrt(self.betas)
    
    def _make_beta_schedule(self) -> torch.Tensor:
        """linear schedule for betas"""
        return torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.num_timesteps
        )
    
    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process : q(x_t | x_0)
        
        Args:
            x0: reference data (B, D)
            t: timesteps (B,)
            noise: noise
            
        Returns:
            x_t: reference data noised using ddpm
            noise: noise used
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise
    
    @torch.no_grad()
    def reverse_diffusion(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: torch.Tensor,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Reverse process : from pur noise generate data
        
        Args:
            model: NN to predict noise
            shape: forme de l'échantillon à générer (B, D)
            condition: condition (B, C)
            return_trajectory: if True, retourne tous les x_t
            
        Returns:
            x0: generated sample at T=0 (ou liste si return_trajectory=True)
        """
        batch_size = shape[0]
        x_t = torch.randn(shape, device=self.device)

        trajectory = [x_t] if return_trajectory else None
        
        for t_step in reversed(range(self.config.num_timesteps)):
            t_tensor = torch.full(
                (batch_size,),
                t_step,
                device=self.device,
                dtype=torch.long
            )
            
            eps_pred = model(x_t, t_tensor, condition)
            
            x_t = self._reverse_step(x_t, eps_pred, t_step)
            
            if return_trajectory:
                trajectory.append(x_t)
        
        return trajectory if return_trajectory else x_t
    
    def _reverse_step(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t: int
    ) -> torch.Tensor:
        """denoise step: x_t -> x_{t-1}"""
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]
        
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
        
        mean = coef1 * (x_t - coef2 * eps_pred)
        
        # add noise if t !=0
        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            x_t_minus_1 = mean + sigma_t * noise
        else:
            x_t_minus_1 = mean
        
        return x_t_minus_1
    
    def _compute_x0_from_noise(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Estimate x_0 from x_t and predicted noise"""
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        return x0_pred
    
    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        condition: torch.Tensor,
        joint_limits: Optional[torch.Tensor] = None,
        limit_penalty_weight: float = 0.1

    ) -> torch.Tensor:
        """
 
        Args:
            model: réseau de débruitage
            x0: données réelles (B, D)
            condition: condition (B, C)
            
        Returns:
            loss: MSE entre bruit prédit et réel
        """
        batch_size = x0.shape[0]
        
        # random timestep for each batch
        t = torch.randint(
            0,
            self.config.num_timesteps,
            (batch_size,),
            device=self.device
        )
        
        # Forward diffusion
        x_t, noise = self.forward_diffusion(x0, t)
        
        noise_pred = model(x_t, t, condition)
        
        # Loss MSE
        loss = torch.mean((noise_pred - noise) ** 2)

        
        
        return loss
    
    def to_dict(self) -> Dict:

        return {
            "betas": self.betas.cpu().numpy(),
            "alphas": self.alphas.cpu().numpy(),
            "alpha_bars": self.alpha_bars.cpu().numpy(),
            "config": {
                "num_timesteps": self.config.num_timesteps,
                "beta_start": self.config.beta_start,
                "beta_end": self.config.beta_end,
            }
        }


# ============================================================
# 3. Dataset (inchangé mais amélioré)
# ============================================================

class JointsWrenchDataset(Dataset):
    
    def __init__(self, joints: np.ndarray, wrench: np.ndarray):
        assert joints.shape[0] == wrench.shape[0]
        self.joints = joints.astype(np.float32)
        self.wrench = wrench.reshape(-1, 1).astype(np.float32)
    
    def __len__(self) -> int:
        return self.joints.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.joints[idx], self.wrench[idx]


# ============================================================
# 4. Normalizer Class
# ============================================================

class DataNormalizer:
    
    def __init__(self, joints_train: np.ndarray, wrench_train: np.ndarray):
        self.joints_mean = joints_train.mean(axis=0, keepdims=True)
        self.joints_std = joints_train.std(axis=0, keepdims=True) + 1e-8
        self.wrench_mean = wrench_train.mean(axis=0, keepdims=True)
        self.wrench_std = wrench_train.std(axis=0, keepdims=True) + 1e-8
    
    def normalize_joints(self, joints: np.ndarray) -> np.ndarray:
        return (joints - self.joints_mean) / self.joints_std
    
    def denormalize_joints(self, joints_norm: np.ndarray) -> np.ndarray:
        return joints_norm * self.joints_std + self.joints_mean
    
    def normalize_wrench(self, wrench: np.ndarray) -> np.ndarray:
        return (wrench - self.wrench_mean) / self.wrench_std
    
    def denormalize_wrench(self, wrench_norm: np.ndarray) -> np.ndarray:
        return wrench_norm * self.wrench_std + self.wrench_mean
    
    def collate_fn(self, batch):

        joints = np.stack([item[0] for item in batch], axis=0)
        wrench = np.stack([item[1] for item in batch], axis=0)
        
        joints_norm = self.normalize_joints(joints)
        wrench_norm = self.normalize_wrench(wrench)
        
        return (
            torch.from_numpy(joints_norm).float(),
            torch.from_numpy(wrench_norm).float()
        )
    
    def to_dict(self) -> Dict:

        return {
            "joints_mean": self.joints_mean,
            "joints_std": self.joints_std,
            "wrench_mean": self.wrench_mean,
            "wrench_std": self.wrench_std,
        }


# ============================================================
# 5. Time Embedding (inchangé, déjà propre)
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Embedding sinusoïdal to encode timestep"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        
        return emb


# ============================================================
# 6. Model (inchangé)
# ============================================================

class CondDiffusionMLP(nn.Module):
    """MLP conditionnal"""
    
    def __init__(
        self,
        x_dim: int = 2,
        cond_dim: int = 1,
        time_dim: int = 32,
        hidden_dim: int = 128
    ):
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
    
    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = torch.cat([x_noisy, cond, t_emb], dim=-1)
        return self.net(h)


# ============================================================
# 7. Trainer Class
# ============================================================

class DiffusionTrainer:
    
    def __init__(
        self,
        model: nn.Module,
        diffusion: DiffusionProcess,
        normalizer: DataNormalizer,
        config: TrainingConfig,
        output_dir: Path,
        joint_limits: Optional[np.ndarray] = None,
        limit_penalty_weight: float = 0.1
    ):
        self.model = model
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.config = config
        self.output_dir = output_dir
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        self.train_losses = []
        self.val_losses = []

        if joint_limits is not None:
            # Normalize the limits
            limits_norm = normalizer.normalize_joints(joint_limits)
            self.joint_limits = torch.from_numpy(limits_norm.astype(np.float32))
        else:
            self.joint_limits = None
        
        self.limit_penalty_weight = limit_penalty_weight
    
    def train_epoch(self, train_loader: DataLoader) -> float:

        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for joints_norm, wrench_norm in train_loader:
            joints_norm = joints_norm.to(self.diffusion.device)
            wrench_norm = wrench_norm.to(self.diffusion.device)

            joint_limits_device = None
            if self.joint_limits is not None:
                joint_limits_device = self.joint_limits.to(self.diffusion.device)
            
            
            # loss
            loss = self.diffusion.training_loss(
                self.model,
                joints_norm,
                wrench_norm,
                joint_limits=joint_limits_device,
                limit_penalty_weight=self.limit_penalty_weight
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(1, n_batches)
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Valide sur le set de validation"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for joints_norm, wrench_norm in val_loader:
            joints_norm = joints_norm.to(self.diffusion.device)
            wrench_norm = wrench_norm.to(self.diffusion.device)

            joint_limits_device = None
            if self.joint_limits is not None:
                joint_limits_device = self.joint_limits.to(self.diffusion.device)
            
            loss = self.diffusion.training_loss(
                self.model,
                joints_norm,
                wrench_norm,
                joint_limits=joint_limits_device,
                limit_penalty_weight=self.limit_penalty_weight
                
            )
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(1, n_batches)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """Boucle d'entraînement complète"""
        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(
                f"[Epoch {epoch:03d}/{self.config.num_epochs}] "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )
        
        self.save_model()
        self.plot_losses()
    
    def save_model(self):

        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "normalizer": self.normalizer.to_dict(),
            "diffusion": self.diffusion.to_dict(),
        }
        
        save_path = self.output_dir / "cond_diffusion_model.pt"
        torch.save(save_dict, save_path)
        print(f"save model {save_path}")
    
    def plot_losses(self):

        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label="train_loss")
        plt.plot(self.val_losses, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        save_path = self.output_dir / "loss_curves.png"
        plt.savefig(save_path, dpi=150)
        print(f"save plots {save_path}")


# ============================================================
# 8. Evaluator Class
# ============================================================

class DiffusionEvaluator:
    
    def __init__(
        self,
        model: nn.Module,
        diffusion: DiffusionProcess,
        normalizer: DataNormalizer,
        output_dir: Path
    ):
        self.model = model
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.output_dir = output_dir
        self.model.eval()
    
    @torch.no_grad()
    def generate_samples(
        self,
        wrench: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        
        Args:
            wrench: (1,1) valeur de My
            n_samples: nombre d'échantillons à générer
            
        Returns:
            samples: (n_samples, 2) en coordonnées originales
        """
        # Normalize condition
        wrench_norm = self.normalizer.normalize_wrench(wrench)
        wrench_norm = torch.from_numpy(wrench_norm.astype(np.float32))
        wrench_norm = wrench_norm.to(self.diffusion.device)
        wrench_norm = wrench_norm.repeat(n_samples, 1)
        
        #reverse diffusion
        samples_norm = self.diffusion.reverse_diffusion(
            self.model,
            shape=(n_samples, 2),
            condition=wrench_norm
        )
        
        # Denormalize
        samples = samples_norm.cpu().numpy()
        samples = self.normalizer.denormalize_joints(samples)
        
        return samples
    
    def evaluate_multisamples(
        self,
        joints_test: np.ndarray,
        wrench_test: np.ndarray,
        n_examples: int = 6,
        n_samples_per_cond: int = 200
    ):
        """Evaluate with sampling several noises"""
        print("=== Multi-sample evaluation ===")
        
        N_test = joints_test.shape[0]

        #choose frame indices to do inference
        idxs = np.random.choice(N_test, size=min(n_examples, N_test), replace=False)
        
        fig, axes = plt.subplots(
            n_examples, 2,
            figsize=(10, 3 * n_examples),
            squeeze=False
        )
        
        q1_min, q1_max = -np.pi / 4.0, +np.pi / 4.0
        q2_min, q2_max = -np.pi / 2.0, +np.pi / 2.0
        
        for i, idx in enumerate(idxs):
            q_gt = joints_test[idx]
            My = wrench_test[idx:idx+1]
            
            samples = self.generate_samples(My, n_samples_per_cond)
            
            err_q1 = np.abs(samples[:, 0] - q_gt[0])
            err_q2 = np.abs(samples[:, 1] - q_gt[1])
            
            best_idx_q1 = np.argmin(err_q1)
            best_idx_q2 = np.argmin(err_q2)
            
            # Plot q1
            ax_q1 = axes[i, 0]
            sample_idx = np.arange(n_samples_per_cond)
            ax_q1.scatter(sample_idx, samples[:, 0], s=8, alpha=0.3, label="samples")
            ax_q1.scatter(best_idx_q1, samples[best_idx_q1, 0], s=30, color="green", label="best")
            ax_q1.scatter(sample_idx.mean(), samples[:, 0].mean(), s=30, color="darkblue", label="mean")
            ax_q1.axhline(q_gt[0], color="red", linestyle="--", linewidth=2, label="GT")
            ax_q1.axhline(q1_min, color="black", linestyle="--", linewidth=1)
            ax_q1.axhline(q1_max, color="black", linestyle="--", linewidth=1)
            ax_q1.set_ylabel("q1 (rad)")
            ax_q1.grid(True)
            if i == 0:
                ax_q1.legend()
            
            # Plot q2
            ax_q2 = axes[i, 1]
            ax_q2.scatter(sample_idx, samples[:, 1], s=8, alpha=0.3, label="samples")
            ax_q2.scatter(best_idx_q2, samples[best_idx_q2, 1], s=30, color="green", label="best")
            ax_q2.scatter(sample_idx.mean(), samples[:, 1].mean(), s=30, color="darkblue", label="mean")
            ax_q2.axhline(q_gt[1], color="red", linestyle="--", linewidth=2, label="GT")
            ax_q2.axhline(q2_min, color="black", linestyle="--", linewidth=1)
            ax_q2.axhline(q2_max, color="black", linestyle="--", linewidth=1)
            ax_q2.set_ylabel("q2 (rad)")
            ax_q2.set_xlabel("sample index")
            ax_q2.grid(True)
            if i == 0:
                ax_q2.legend()
            
            ax_q1.set_title(
                f"My={My[0,0]:.3f}, "
                f"err_q1={err_q1[best_idx_q1]:.4f}, "
                f"err_q2={err_q2[best_idx_q2]:.4f}"
            )
        
        plt.tight_layout()
        save_path = self.output_dir / "multisamples_evaluation.png"
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardée dans {save_path}")
    
    def evaluate_best_over_dataset(
        self,
        joints_test: np.ndarray,
        wrench_test: np.ndarray,
        n_samples_per_cond: int = 200,
        max_test_points: Optional[int] = None
    ):
        """Évalue en sélectionnant le meilleur échantillon"""
        print("=== Best-over-dataset evaluation ===")
        
        N_test = joints_test.shape[0]
        
        if max_test_points is not None and max_test_points < N_test:
            idxs = np.random.choice(N_test, size=max_test_points, replace=False)
            idxs = np.sort(idxs)
        else:
            idxs = np.arange(N_test)
        
        N_eval = len(idxs)
        
        best_q1 = np.zeros(N_eval)
        best_q2 = np.zeros(N_eval)
        gt_q1 = np.zeros(N_eval)
        gt_q2 = np.zeros(N_eval)
        
        for k, idx in enumerate(idxs):
            if (k + 1) % 50 == 0:
                print(f"  Processing {k+1}/{N_eval}")
            
            q_gt = joints_test[idx]
            My = wrench_test[idx:idx+1]
            
            gt_q1[k] = q_gt[0]
            gt_q2[k] = q_gt[1]
            
            # Générer échantillons
            samples = self.generate_samples(My, n_samples_per_cond)
            
            # Trouver les meilleurs
            err_q1 = np.abs(samples[:, 0] - q_gt[0])
            err_q2 = np.abs(samples[:, 1] - q_gt[1])
            
            best_q1[k] = samples[np.argmin(err_q1), 0]
            best_q2[k] = samples[np.argmin(err_q2), 1]
        
        # Sauvegarder les prédictions
        best_q = np.stack([best_q1, best_q2], axis=1)
        np.save(self.output_dir / "best_q_test.npy", best_q)
        np.savetxt(
            self.output_dir / "best_q_test.csv",
            best_q,
            delimiter=",",
            header="q1_best_rad,q2_best_rad",
            comments=""
        )
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        sample_idx = np.arange(N_eval)
        
        axes[0].scatter(sample_idx, gt_q1, label="GT", color="red", s=10)
        axes[0].scatter(sample_idx, best_q1, label="best pred", s=8, alpha=0.7)
        axes[0].set_ylabel("q1 (rad)")
        axes[0].set_title("Best q1 predictions vs GT")
        axes[0].grid(True)
        axes[0].legend()
        
        axes[1].scatter(sample_idx, gt_q2, label="GT", color="red", s=10)
        axes[1].scatter(sample_idx, best_q2, label="best pred", s=8, alpha=0.7)
        axes[1].set_ylabel("q2 (rad)")
        axes[1].set_xlabel("test sample index")
        axes[1].set_title("Best q2 predictions vs GT")
        axes[1].grid(True)
        axes[1].legend()
        
        plt.tight_layout()
        save_path = self.output_dir / "best_over_dataset.png"
        plt.savefig(save_path, dpi=150)
        print(f"Figure sauvegardée dans {save_path}")


# ============================================================
# 9. Main Pipeline
# ============================================================

def load_data(data_config: DataConfig):
    """Charge les données"""
    joints_train = np.load(data_config.joints_train_file)
    wrench_train = np.load(data_config.wrench_train_file)
    joints_val = np.load(data_config.joints_val_file)
    wrench_val = np.load(data_config.wrench_val_file)
    joints_test = np.load(data_config.joints_test_file)
    wrench_test = np.load(data_config.wrench_test_file)
    
    train_dataset = JointsWrenchDataset(joints_train, wrench_train)
    val_dataset = JointsWrenchDataset(joints_val, wrench_val)
    test_dataset = JointsWrenchDataset(joints_test, wrench_test)
    
    normalizer = DataNormalizer(joints_train, wrench_train)
    
    return train_dataset, val_dataset, test_dataset, normalizer


def main():
    # Config
    data_config = DataConfig()
    diffusion_config = DiffusionConfig()
    training_config = TrainingConfig()
    
    output_dir = Path(data_config.output_dir)
    output_dir.mkdir(exist_ok=True)
    

    train_dataset, val_dataset, test_dataset, normalizer = load_data(data_config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=normalizer.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        collate_fn=normalizer.collate_fn
    )

    joint_limits = np.array([
        [-np.pi/4, +np.pi/4],  # q1 limits
        [-np.pi/2, +np.pi/2]   # q2 limits
    ], dtype=np.float32)

    diffusion = DiffusionProcess(diffusion_config)
    
    model = CondDiffusionMLP(
        x_dim=2,
        cond_dim=1,
        time_dim=32,
        hidden_dim=128
    ).to(diffusion.device)
    
    trainer = DiffusionTrainer(
        model=model,
        diffusion=diffusion,
        normalizer=normalizer,
        config=training_config,
        output_dir=output_dir,
        joint_limits=joint_limits,
        limit_penalty_weight=1
    )
    
    trainer.train(train_loader, val_loader)
    
    # evaluate
    joints_test = np.load(data_config.joints_test_file).astype(np.float32)
    wrench_test = np.load(data_config.wrench_test_file).astype(np.float32).reshape(-1, 1)
    
    evaluator = DiffusionEvaluator(
        model=model,
        diffusion=diffusion,
        normalizer=normalizer,
        output_dir=output_dir
    )
    
    evaluator.evaluate_multisamples(
        joints_test=joints_test,
        wrench_test=wrench_test,
        n_examples=4,
        n_samples_per_cond=20
    )
    
    evaluator.evaluate_best_over_dataset(
        joints_test=joints_test,
        wrench_test=wrench_test,
        n_samples_per_cond=20,
        max_test_points=100
    )


if __name__ == "__main__":
    main()