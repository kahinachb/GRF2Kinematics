import torch

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
