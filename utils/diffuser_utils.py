import torch
import torch.nn as nn
import math
from dataset import Normalizer
import torch.nn.functional as F

class DDPM:
    def __init__(self, device, n_steps=1000, min_beta=1e-4, max_beta=0.02):
        self.n_steps = n_steps
        self.device = device
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample_forward(self, x_0, t, noise):
        """Forward process q(x_t | x_0) — adds noise at step t.

                Args:
                    x_0   : (B, T, D)  clean signal
                    t     : (B,)       integer diffusion steps
                    noise : (B, T, D)  standard Gaussian noise
                Returns:
                    x_t   : (B, T, D)  noisy signal
        """
        sqrt_ab  = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_1ab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        return sqrt_ab * x_0 + sqrt_1ab * noise

    def sample_reverse_norm(self, model, x_t, t, cond):
        """One reverse step p(x_{t-1} | x_t, cond).

        Args:
            model : DiffusionTransformer
            x_t   : (B, T, D)  current noisy signal
            t     : int         current diffusion step
            cond  : (B, T, len(kinetics)) kinetics condition
        Returns:
            x_{t-1} : (B, T, D)
        """
        z          = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        # z = torch.zeros_like(x_t)
        beta_t     = self.betas[t]
        alpha_t    = self.alphas[t]
        alpha_bar  = self.alphas_cumprod[t]
        if t > 0:
            alpha_bar_prev = self.alphas_cumprod[t-1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=self.device)

        beta_tilde = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar)


        t_norm     = torch.full((x_t.shape[0],), t / self.n_steps,
                                device=self.device, dtype=torch.float32)
        
        eps_theta  = model(x_t, t_norm, cond)

        mean       = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar)) * eps_theta
        )
        return mean + torch.sqrt(beta_tilde) * z
    
    def sample_reverse(self, model, x_t, t, cond):
        """One reverse step p(x_{t-1} | x_t, cond).

        Args:
            model : DiffusionTransformer
            x_t   : (B, T, D)  current noisy signal
            t     : int         current diffusion step
            cond  : (B, T, len(kinetics)) kinetics condition
        Returns:
            x_{t-1} : (B, T, D)
        """
        z          = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        # z = torch.zeros_like(x_t)
        beta_t     = self.betas[t]
        alpha_t    = self.alphas[t]
        alpha_bar  = self.alphas_cumprod[t]
        if t > 0:
            alpha_bar_prev = self.alphas_cumprod[t-1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=self.device)

        beta_tilde = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar)


        # t_norm     = torch.full((x_t.shape[0],), t / self.n_steps,
        #                         device=self.device, dtype=torch.float32)
        
        t_tensor   = torch.full((x_t.shape[0],), t,
                                device=self.device, dtype=torch.float32)
        eps_theta  = model(x_t, t_tensor, cond)

        mean       = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar)) * eps_theta
        )
        return mean + torch.sqrt(beta_tilde) * z
    
    def generate(self, model, cond, joint_dim=29):
        """Full reverse chain: pure noise → joint angles.

        Args:
            model     : DiffusionTransformer (eval mode)
            cond      : (B, T, len(kinetics))  kinetics condition (normalized)
            joint_dim : int          number of joint DOFs
        Returns:
            x_0 : (B, T, joint_dim)  predicted joint angles (normalized)
        """
        B, T, _ = cond.shape
        x_t = torch.randn(B, T, joint_dim, device=self.device)
        for step in reversed(range(self.n_steps)):
            x_t = self.sample_reverse(model, x_t, step, cond)
        return x_t
    
    def _predict_x_t_minus_1(self, x_t, eps_theta, t_idx):
        """p(x_{t-1} | x_t, eps_theta)"""
        z          = torch.randn_like(x_t) if t_idx > 0 else torch.zeros_like(x_t)
        beta_t     = self.betas[t_idx]
        alpha_t    = self.alphas[t_idx]
        alpha_bar  = self.alphas_cumprod[t_idx]

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar)) * eps_theta
        )
        return mean + torch.sqrt(beta_t) * z
    
    @torch.enable_grad()
    def _get_guidance_grad(self, x_t, joint_norm, q_min, q_max):
        x_t_grad = x_t.detach().clone().requires_grad_(True)
        x_real = joint_norm.inverse_transform_torch(x_t_grad)
        
        loss_max = torch.sum(torch.relu(x_real - q_max) ** 2)
        loss_min = torch.sum(torch.relu(q_min - x_real) ** 2)
        loss = loss_max + loss_min
        
        loss.backward()
        return x_t_grad.grad

    def generate_with_guidance(self, model, cond, joint_dim=35, joint_norm=None, joint_limits=None, guidance_scale=0.1):
        """Version alternative avec guidance physique."""
        model.eval()
        B, T, _ = cond.shape
        x_t = torch.randn(B, T, joint_dim, device=self.device)

        # Init des limites
        q_min, q_max = None, None
        if joint_limits is not None and joint_norm is not None:
            joint_names = [
                "delta_x","delta_y","delta_z","delta_rx","delta_ry","delta_rz",
                "right_hip_Z", "right_hip_X", "right_hip_Y", "right_knee_Z", "right_ankle_Z", "right_ankle_X",
                "left_hip_Z", "left_hip_X", "left_hip_Y", "left_knee_Z", "left_ankle_Z", "left_ankle_X",
                "middle_lumbar_Z", "middle_lumbar_X", "left_clavicle_joint_X",
                "left_shoulder_Z", "left_shoulder_X", "left_shoulder_Y", "left_elbow_Z", "left_elbow_Y",
                "middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y", "right_clavicle_joint_X",
                "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y", "right_elbow_Z", "right_elbow_Y"
            ]
            q_min = torch.tensor([-1e6]*6 + [joint_limits[n][0] for n in joint_names[6:]], device=self.device).float()
            q_max = torch.tensor([1e6]*6 + [joint_limits[n][1] for n in joint_names[6:]], device=self.device).float()

        for step in reversed(range(self.n_steps)):
            # 1. Appliquer la guidance sur x_t
            if q_min is not None and step > 50:
                grad = self._get_guidance_grad(x_t, joint_norm, q_min, q_max)
                x_t = x_t - guidance_scale * grad

            # 2. Prédire le bruit et faire le pas inverse
            t_norm = torch.full((B,), step / self.n_steps, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                eps_theta = model(x_t, t_norm, cond)
                x_t = self._predict_x_t_minus_1(x_t, eps_theta, step)

        return x_t
    
    
class DiffusionTransformer(nn.Module):
    def __init__(self, joint_dim=12, force_dim=12, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
        self.joint_embed = nn.Linear(joint_dim, embed_dim) #input embeddings
        self.force_embed = nn.Linear(force_dim, embed_dim)
        self.time_embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)) #time embedding, Encodes the diffusion timestep 
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, joint_dim)

    def forward(self, x, t, cond):
        t_emb = self.time_embed(t.view(-1, 1)).unsqueeze(1)
        x_emb = self.joint_embed(x) + self.force_embed(cond) + t_emb #All information is blended into the same 256-dimensional space
        return self.output_layer(self.transformer(x_emb))

