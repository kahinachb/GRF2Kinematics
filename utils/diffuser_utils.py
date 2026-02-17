import torch
import torch.nn as nn

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
        """Forward Diffusion: q(x_t | x_0)"""
        return (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x_0 +
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise
        )

    def sample_reverse(self, model, x_t, t, cond):
        """Reverse Diffusion Step: p(x_{t-1} | x_t, cond)"""
        if t == 0:
            z = 0
        else:
            z = torch.randn_like(x_t)
            
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        
        t_tensor = torch.full((x_t.shape[0],), t / self.n_steps, device=self.device)
        eps_theta = model(x_t, t_tensor, cond)
        
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
        )
        sigma = torch.sqrt(beta_t)
        
        return mean + sigma * z

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