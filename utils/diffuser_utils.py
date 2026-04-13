import torch
import torch.nn as nn
import math
from dataset import Normalizer
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
        beta_t     = self.betas[t]
        alpha_t    = self.alphas[t]
        alpha_bar  = self.alphas_cumprod[t]

        t_norm     = torch.full((x_t.shape[0],), t / self.n_steps,
                                device=self.device, dtype=torch.float32)
        eps_theta  = model(x_t, t_norm, cond)

        mean       = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar)) * eps_theta
        )
        return mean + torch.sqrt(beta_t) * z
    
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
            if q_min is not None :#and step > 50:
                grad = self._get_guidance_grad(x_t, joint_norm, q_min, q_max)
                x_t = x_t - guidance_scale * grad

            # 2. Prédire le bruit et faire le pas inverse
            t_norm = torch.full((B,), step / self.n_steps, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                eps_theta = model(x_t, t_norm, cond)
                x_t = self._predict_x_t_minus_1(x_t, eps_theta, step)

        return x_t
    
class DiffusionTransformer(torch.nn.Module):
    def __init__(self, joint_dim=12, force_dim=12, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
        self.joint_embed = torch.nn.Linear(joint_dim, embed_dim)#input embeddings
        self.force_embed = torch.nn.Linear(force_dim, embed_dim)
        self.time_embed = torch.nn.Sequential(  
            torch.nn.Linear(1, embed_dim), 
            torch.nn.SiLU(), 
            torch.nn.Linear(embed_dim, embed_dim)
        ) #time embedding, Encodes the diffusion timestep 
        layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True, norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_layer = torch.nn.Linear(embed_dim, joint_dim)

    def forward(self, x, t, cond):
        t_emb = self.time_embed(t.view(-1, 1)).unsqueeze(1)
        x_emb = self.joint_embed(x) + self.force_embed(cond) + t_emb
        return self.output_layer(self.transformer(x_emb))
    
class DiffusionTransformerConcat(nn.Module):
    def __init__(self, joint_dim=12, force_dim=12, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
        
        # Separate embeddings (same as before)
        self.joint_embed = nn.Linear(joint_dim, embed_dim)
        self.force_embed = nn.Linear(force_dim, embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim), 
            nn.SiLU(), 
            nn.Linear(embed_dim, embed_dim)
        )
        
        # NEW: Projection layer to combine concatenated embeddings
        # 256 (joints) + 256 (forces) + 256 (time) = 768 total
        self.combine_proj = nn.Linear(embed_dim * 3, embed_dim)
        
        # Transformer (same as before)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # Output layer (same as before)
        self.output_layer = nn.Linear(embed_dim, joint_dim)

    def forward(self, x, t, cond):
        # x: [batch, 128, 12] - noisy joints
        # t: [batch] - timestep
        # cond: [batch, 128, 12] - force conditions
        
        batch_size, seq_len, _ = x.shape
        
        # 1. Create embeddings
        x_emb = self.joint_embed(x)              # [batch, 128, 256]
        f_emb = self.force_embed(cond)           # [batch, 128, 256]
        t_emb = self.time_embed(t.view(-1, 1))   # [batch, 256]
        
        # 2. Expand time embedding to match sequence length
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, 128, 256]
        
        # 3. CONCATENATE along the feature dimension
        combined = torch.cat([x_emb, f_emb, t_emb], dim=-1)  # [batch, 128, 768]
        
        # 4. Project back down to embed_dim
        combined = self.combine_proj(combined)    # [batch, 128, 256]
        
        # 5. Process through transformer
        transformed = self.transformer(combined)  # [batch, 128, 256]
        
        # 6. Predict noise
        return self.output_layer(transformed)     # [batch, 128, 12]
    


# ─────────────────────────────────────────────────────────────────────────────
# SINUSOIDAL TIME EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for the diffusion timestep,
    followed by a small MLP (same spirit as Nichol & Dhariwal 2021).
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
 
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : (B,) float in [0, 1]
        Returns:
            emb : (B, embed_dim)
        """
        device = t.device
        half   = self.embed_dim // 2
        # Frequencies spaced on a log scale
        freqs  = torch.exp(
            -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=device)
            / (half - 1)
        )
        args   = t[:, None] * freqs[None] * 1_000   # rescale [0,1] → [0,1000]
        emb    = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)                         # (B, embed_dim)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# DIFFUSION TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────
 
class DiffusionTransformer(nn.Module):
    """
    Transformer-based denoising network for the DDPM reverse process.
 
    Conditioning strategy: at each position the force/moment embedding
    is added to the joint embedding (element-wise), so the model sees
    the kinetics context at every frame.  The time embedding is broadcast
    over the whole sequence.
 
    Architecture choices vs. your original:
      • joint_dim 12 → 35   (all joints,  with freeflyer)
      • sinusoidal time embedding instead of a plain linear
      • learnable positional encoding over the time axis
      • LayerNorm + linear output head (more stable at the start)
      • dim_feedforward = 4 × embed_dim  (standard Transformer ratio)
      • num_layers 4 → 6  (slightly deeper; easy to tune down)
    """
 
    def __init__(
        self,
        joint_dim:  int = 35,     # all_joints DOFs (no freeflyer)
        force_dim:  int = 12,     # kinetics channels (R+L plates)
        embed_dim:  int = 128,    # 
        nhead:      int = 4,      
        num_layers: int = 4,      
        seq_len:    int = 128,    # window size in frames (128 frames = 1.28 s @ 100 Hz)
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.joint_dim = joint_dim
        self.seq_len   = seq_len
 
        # ── Input projections ─────────────────────────────────────────────
        self.joint_embed = nn.Linear(joint_dim, embed_dim)
        self.force_embed = nn.Linear(force_dim, embed_dim)
 
        # ── Time embedding ────────────────────────────────────────────────
        self.time_embed = SinusoidalTimeEmbedding(embed_dim)
 
        # ── Learnable positional encoding ─────────────────────────────────
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
 
        # ── Transformer encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,     # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
 
        # ── Output head ───────────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, joint_dim),
        )
 
        self._init_weights()
 
    def _init_weights(self):
        """Small-scale initialization for stable early training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
 
    def forward(
        self,
        x:    torch.Tensor,   # (B, T, joint_dim)  noisy joint angles
        t:    torch.Tensor,   # (B,)               normalized timestep ∈ [0, 1]
        cond: torch.Tensor,   # (B, T, force_dim)  kinetics condition
    ) -> torch.Tensor:        # (B, T, joint_dim)  predicted noise
        """
        Predict the noise eps_theta(x_t, t, cond).
        """
        B, T, _ = x.shape
 
        # Time embedding: (B, embed_dim) → broadcast over sequence → (B, 1, embed_dim)
        t_emb = self.time_embed(t).unsqueeze(1)          # (B, 1, D)
 
        # Combine all signals at the token level
        tokens = (
            self.joint_embed(x)          # (B, T, D)  noisy joints
            + self.force_embed(cond)     # (B, T, D)  kinetics condition
            + t_emb                      # (B, 1, D)  timestep (broadcast)
            + self.pos_embed[:, :T, :]   # (1, T, D)  position
        )
 
        out = self.transformer(tokens)   # (B, T, D)
        return self.output_head(out)     # (B, T, joint_dim)
    

class MotionPredictor(nn.Module):
    def __init__(self, joint_dim=29, force_dim=12,
                 embed_dim=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()

        # Project kinetics into embedding space
        self.input_proj = nn.Linear(force_dim, embed_dim)

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, 512, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project back to joint space
        self.output_proj = nn.Linear(embed_dim, joint_dim)

    def forward(self, kinetics):  # (B, T, 12)
        x = self.input_proj(kinetics)               # (B, T, embed_dim)
        x = x + self.pos_enc[:, :x.size(1), :]     # add positional info
        x = self.encoder(x)                         # (B, T, embed_dim)
        return self.output_proj(x)                  # (B, T, 29)