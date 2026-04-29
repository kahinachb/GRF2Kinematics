import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import json
from utils.diffuser_utils import DDPM 

# ==========================================
# 1. DATASET 
# ==========================================
class BiomechDiffusionDataset(Dataset):
    def __init__(self, file_list, window_size=128, stats=None):
        self.samples = []
        for f_path, j_path in file_list:
            f_data, j_data = np.load(f_path).astype(np.float32), np.load(j_path).astype(np.float32)
            j_data = j_data[:, 6:18]
            
            for i in range(0, len(f_data) - window_size, window_size // 2):
                self.samples.append((f_data[i:i+window_size], j_data[i:i+window_size]))
        self.stats = stats

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        f, j = self.samples[idx]
        f, j = torch.from_numpy(f), torch.from_numpy(j)
        if self.stats:
            f = (f - self.stats['f_m']) / (self.stats['f_s'] + 1e-6)
            j = (j - self.stats['j_m']) / (self.stats['j_s'] + 1e-6)
        return f, j

def compute_and_save_stats(file_list, save_path):
    print("\n[INFO] Calcul des scalers sur le Train Set...")
    all_f, all_j = [], []
    for f_p, j_p in file_list:
        all_f.append(np.load(f_p)); all_j.append(np.load(j_p))
    f_cat, j_cat = np.vstack(all_f), np.vstack(all_j)
    j_cat= j_cat[:, 6:18]
    
    stats = {
        'f_m': f_cat.mean(axis=0), 'f_s': f_cat.std(axis=0),
        'j_m': j_cat.mean(axis=0), 'j_s': j_cat.std(axis=0)
    }
    serializable_stats = {k: v.tolist() for k, v in stats.items()}
    with open(save_path, 'w') as f:
        json.dump(serializable_stats, f)
    return {k: torch.tensor(v).float() for k, v in stats.items()}

# ==========================================
# 2. ARCHITECTURE
# ==========================================
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ---> ADDED: Sinusoidal Timestep Embedding Class <---
class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: 1D tensor of shape (batch_size,)
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionTransformer(nn.Module):
    def __init__(self, joint_dim=12, force_dim=18, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
        self.joint_embed = nn.Linear(joint_dim, embed_dim) 
        self.force_embed = nn.Linear(force_dim, embed_dim)
        
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        ) 
        
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=1000) 
        
        # ---> CHANGED: Encoder to Decoder for Cross-Attention <---
        layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(embed_dim, joint_dim)

    def forward(self, x, t, cond):
        # 1. Safeguard: Ensure t is a 1D tensor (handles inference loop integers)
        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=torch.long).expand(x.shape[0])
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])

        # 2. Process Time Embedding
        t_emb = self.time_mlp(t).unsqueeze(1) 
        
        # 3. Process Target Sequence (Noisy Joints + Time + Position)
        x_emb = self.joint_embed(x) + t_emb 
        x_emb = self.pos_encoder(x_emb) 
        
        # 4. Process Memory Sequence (Forces + Position)
        # Note: Time-series forces also need positional awareness!
        cond_emb = self.force_embed(cond)
        cond_emb = self.pos_encoder(cond_emb)
        
        # 5. Transformer Decoder (tgt=Joints, memory=Forces)
        out = self.transformer(tgt=x_emb, memory=cond_emb)
        
        return self.output_layer(out)
    
# ==========================================
# 3. FONCTION INFERENCE COMPLETE (SLIDING WINDOW)
# ==========================================
def predict_full_trial(model, ddpm, f_path, j_path, stats, device, window_size=128, stride=64):
    model.eval()
    f_raw = np.load(f_path).astype(np.float32)
    j_raw = np.load(j_path).astype(np.float32)
    j_raw = j_raw[:, 6:18]
  
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)
    
    T = f_norm.shape[0]
    full_pred = torch.zeros((T, 12)).to(device)
    count_map = torch.zeros((T, 12)).to(device)

    print(f"  [INF] Sampling trial complet ({T} frames)...")
    
    for start in range(0, T - window_size, stride):
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        
        curr_j = torch.randn((1, window_size, 12)).to(device)
        
        # Reverse Diffusion avec DDPM (sur 50 steps pour plus de vitesse en inférence)
        inference_steps = 1000 
        step_size = ddpm.n_steps // inference_steps
        
        for t_idx in reversed(range(0, ddpm.n_steps, step_size)):
            with torch.no_grad():
                curr_j = ddpm.sample_reverse(model, curr_j, t_idx, f_win)
        
        full_pred[start:end] += curr_j.squeeze(0)
        count_map[start:end] += 1.0

    final_pred = full_pred / torch.clamp(count_map, min=1.0)
    final_pred = (final_pred.cpu() * stats['j_s']) + stats['j_m']
    return j_raw, final_pred.numpy()

# ==========================================
# 4. RUN EXPERIMENT
# ==========================================
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("/lustre/fsn1/projects/rech/vsi/ulm94jm/dataset_grf2kine/processed_data_feet")

    results_dir = Path("results_PE_sin_cross_real")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # subjects = [d for d in data_root.iterdir() if d.is_dir()]
    # if len(subjects) > 1:
    #     print("⚠️ Tu as plus d'un sujet")
    #     print (subjects)
    #     subject = subjects[0]
    #     print(subject)

    # trials = sorted([t for t in subject.iterdir() if t.is_dir()])
    # print(f"Total trials: {len(trials)}")
    # random.seed(42)
    # random.shuffle(trials)
    # n = len(trials)
    # train_trials = trials[:int(0.7*n)]
    # val_trials   = trials[int(0.7*n):int(0.85*n)]
    # test_trials  = trials[int(0.85*n):]

    # print(f"\n[SPLIT SUMMARY]")
    # print(f"TRAIN ({len(train_trials)} trials): {[t.name for t in train_trials]}")
    # print(f"VAL   ({len(val_trials)} trials): {[t.name for t in val_trials]}")
    # print(f"TEST  ({len(test_trials)} trials): {[t.name for t in test_trials]}")

    subjects = sorted([d for d in data_root.iterdir() if d.is_dir()])
    task_subs = [s for s in subjects if "subject" in s.name.lower()]
    squat_subs = [s for s in subjects if "subject" not in s.name.lower()]
    random.seed(42); random.shuffle(task_subs); random.shuffle(squat_subs)

    train_subs = task_subs[:11] + squat_subs[:4]
    val_subs = task_subs[11:14] + squat_subs[4:5]
    test_subs = task_subs[14:] + squat_subs[5:]

    print(f"\n[SPLIT SUMMARY]")
    print(f"TRAIN ({len(train_subs)} sujets): {[s.name for s in train_subs]}")
    print(f"VAL   ({len(val_subs)} sujets): {[s.name for s in val_subs]}")
    print(f"TEST  ({len(test_subs)} sujets): {[s.name for s in test_subs]}")
    print(f"{'='*30}")

    def get_pairs(subs):
        p = []
        for s in subs:
            # On parcourt chaque essai (task) dans le dossier du sujet
            for t in s.iterdir():
                if t.is_dir():
                    f = t / "kinetics_feet.npy"  
                    j = t / "all_joints.npy"
                    # On vérifie que les deux fichiers existent bien
                    if f.exists() and j.exists():
                        p.append((f, j))
        return p


    train_pairs = get_pairs(train_subs)
    print("train_pairs", train_pairs)
    stats = compute_and_save_stats(train_pairs, results_dir /"scalers_concat.json")
    
    train_loader = DataLoader(BiomechDiffusionDataset(train_pairs, stats=stats), batch_size=64, shuffle=True)
    val_loader = DataLoader(BiomechDiffusionDataset(get_pairs(val_subs), stats=stats), batch_size=64)

    # Initialisation DDPM
    ddpm = DDPM(device, n_steps=1000)
    model = DiffusionTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    train_losses, val_losses = [], []

    epochs = 500
    print(f"\n[START] Entraînement DDPM...")
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for f, j in train_loader:
            f, j = f.to(device), j.to(device)
            
            # DDPM Step sampling
            t = torch.randint(0, ddpm.n_steps, (j.shape[0],)).to(device)
            noise = torch.randn_like(j)
            j_noisy = ddpm.sample_forward(j, t, noise)
            
            optimizer.zero_grad()
            # On normalise t pour le modèle (0 à 1)
            pred_noise = model(j_noisy, t, f)
            loss = nn.MSELoss()(pred_noise, noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #gradient need to be clipped(to avoid explosion gradient) while using cross attention
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for f, j in val_loader:
                f, j = f.to(device), j.to(device)
                t = torch.randint(0, ddpm.n_steps, (j.shape[0],)).to(device)
                noise = torch.randn_like(j)
                j_noisy = ddpm.sample_forward(j, t, noise)
                v_loss += nn.MSELoss()(model(j_noisy, t, f), noise).item()
        
        train_losses.append(t_loss/len(train_loader))
        val_losses.append(v_loss/len(val_loader))
        print(f"Epoch {epoch:02d} | Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}")

    avg_train_loss = t_loss / len(train_loader)
    avg_val_loss = v_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        print(f"--> Validation loss improved from {best_val_loss:.6f} to {avg_val_loss:.6f}. Saving model...")
        best_val_loss = avg_val_loss
        
        # Save the model state dict specifically as the "best" model
        torch.save(model.state_dict(), results_dir / "diffusion_biomech_model_best.pth")

    torch.save(model.state_dict(), results_dir /"diffusion_biomech_model_concat.pth")

    # --- INFERENCE SUR TEST (FENÊTRE) ---
    print("[INF] Génération d'un exemple de test...")
    test_ds = BiomechDiffusionDataset(get_pairs(test_subs), stats=stats)
    f_in, j_ref = test_ds[random.randint(0, len(test_ds)-1)]
    f_in = f_in.unsqueeze(0).to(device)
    
    curr_j = torch.randn((1, 128, 12)).to(device)
    for t_idx in reversed(range(ddpm.n_steps)):
        with torch.no_grad():
            curr_j = ddpm.sample_reverse(model, curr_j, t_idx, f_in)

    pred = (curr_j.cpu().squeeze(0) * stats['j_s']) + stats['j_m']
    ref = (j_ref * stats['j_s']) + stats['j_m']

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(ref[:, i], 'k--', label='Ref'); ax.plot(pred[:, i], 'r', label='Pred')
        ax.set_title(f"Joint {i+1}"); ax.legend()
    plt.tight_layout(); plt.savefig(results_dir /"inference_test_concat.png"); plt.close()

    # --- INFERENCE COMPLÈTE ---
    print("\n[INF] Inférence sur un essai COMPLET...")
    test_pairs = get_pairs(test_subs)
    random_trial = random.choice(test_pairs)
    print("random_trial for test", random_trial)
    ref_full, pred_full = predict_full_trial(model, ddpm, random_trial[0], random_trial[1], stats, device)

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(ref_full[:, i], 'k--', alpha=0.6, label='Reference')
        ax.plot(pred_full[:, i], 'r', label='DDPM Pred')
        ax.set_title(f"Joint {i+1}")
        if i == 0: ax.legend()
    plt.suptitle(f"Full Trial DDPM Inference: {random_trial[0].parent.name}", fontsize=16)
    plt.tight_layout(); plt.savefig(results_dir /"full_trial_inference_concat.png"); plt.close()
    
    plt.figure(); plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
    plt.title("Loss History"); plt.legend(); plt.savefig(results_dir /"loss_curve_concat.png"); plt.close()
    print(f"\n[FINISH] Résultats sauvegardés.")

if __name__ == "__main__":
    run_experiment()

