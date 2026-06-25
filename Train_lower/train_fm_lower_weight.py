import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import json
from utils.utils import is_squat_task,is_squat_task_only
import re
import yaml

squat = False
# ==========================================
# 1. DATASET 
# ==========================================
class BiomechDiffusionDataset(Dataset):
    def __init__(self, file_list, window_size=128, stats=None):
        self.samples = []
        for f_path, j_path, weight in file_list:
            f_data, j_data = np.load(f_path).astype(np.float32), np.load(j_path).astype(np.float32)
            j_data = j_data[:, 6:18]
            
            for i in range(0, len(f_data) - window_size, window_size // 2):
                self.samples.append((f_data[i:i+window_size], j_data[i:i+window_size], weight))
        self.stats = stats

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        f, j, w = self.samples[idx]
        f, j = torch.from_numpy(f), torch.from_numpy(j)
        w = torch.tensor([w], dtype=torch.float32)
        if self.stats:
            f = (f - self.stats['f_m']) / (self.stats['f_s'] + 1e-6)
            j = (j - self.stats['j_m']) / (self.stats['j_s'] + 1e-6)
            w = (w - self.stats['w_m']) / (self.stats['w_s'] + 1e-6)

        return f, j, w

def compute_and_save_stats(file_list, save_path):
    print("\n[INFO] Calcul des scalers sur le Train Set...")
    all_f, all_j, all_w = [], [], []

    for f_p, j_p, w in file_list:
        all_f.append(np.load(f_p)); all_j.append(np.load(j_p))
        all_w.append(w)
    w_arr = np.array(all_w)

    f_cat, j_cat = np.vstack(all_f), np.vstack(all_j)
    j_cat= j_cat[:, 6:18]
    
    stats = {
        'f_m': f_cat.mean(axis=0), 'f_s': f_cat.std(axis=0),
        'j_m': j_cat.mean(axis=0), 'j_s': j_cat.std(axis=0),
        'w_m': w_arr.mean(),       'w_s': w_arr.std()
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
        
        self.weight_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

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

    def forward(self, x, t, cond,weight):

        if isinstance(t, (int, float)):
            t = torch.tensor([t], device=x.device, dtype=torch.float32).expand(x.shape[0])
        elif t.dim() == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
            
        # Optionnel mais recommandé : On "étire" t pour l'embedding sinusoidal
        t_input = t * 1000.0
        # Process Time Embedding
        t_emb = self.time_mlp(t_input).unsqueeze(1) 
        
        # 3. Process Target Sequence (Noisy Joints + Time + Position)
        x_emb = self.joint_embed(x) + t_emb 
        x_emb = self.pos_encoder(x_emb) 
        
        # Process Memory Sequence (Forces + Position)
        # Note: Time-series forces also need positional awareness!
        #Process weight and add it to forces <---
        cond_emb = self.force_embed(cond)          # Shape: [Batch, SeqLen, EmbedDim]
        w_emb = self.weight_embed(weight)          # Shape: [Batch, EmbedDim]
        w_emb = w_emb.unsqueeze(1)                 # Shape: [Batch, 1, EmbedDim]
        
        # Broadcasting adds the subject's weight embedding to every frame
        cond_emb = cond_emb + w_emb 
        cond_emb = self.pos_encoder(cond_emb)

        # Transformer Decoder (tgt=Joints, memory=Forces)
        out = self.transformer(tgt=x_emb, memory=cond_emb)
        
        return self.output_layer(out)
    
# ==========================================
# 3. FONCTION INFERENCE COMPLETE (SLIDING WINDOW)
# ==========================================
def predict_full_trial(model, f_path, j_path,weight, stats, device, window_size=128, stride=64, steps=25):
    model.eval()
    f_raw = np.load(f_path).astype(np.float32)
    j_raw = np.load(j_path).astype(np.float32)
    j_raw = j_raw[:, 6:18] # Focus bas du corps

    # ---> ADDED: Format weight for inference <---
    w_norm = (weight - stats['w_m'].item()) / (stats['w_s'].item() + 1e-6)
    w_tensor = torch.tensor([[w_norm]], dtype=torch.float32).to(device)

    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)
    
    T = f_norm.shape[0]
    full_pred = torch.zeros((T, 12)).to(device)
    count_map = torch.zeros((T, 12)).to(device)

    print(f"  [INF] Sampling CFM (Euler) sur essai complet ({T} frames)...")
    
    dt = 1.0 / steps

    for start in range(0, T - window_size, stride):
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        
        # 1. Départ du bruit pur (x_0) à t=0
        curr_x = torch.randn((1, window_size, 12)).to(device)
        
        # 2. Intégration d'Euler (Flow Matching Inference)
        for i in range(steps):
            t_val = i / steps
            t = torch.ones((1,), device=device) * t_val
            
            with torch.no_grad():
                # Le modèle prédit la vitesse (v)
                v = model(curr_x, t, f_win, w_tensor)
                
            # x_{t+dt} = x_t + v * dt
            curr_x = curr_x + v * dt
        
        full_pred[start:end] += curr_x.squeeze(0)
        count_map[start:end] += 1.0

    final_pred = full_pred / torch.clamp(count_map, min=1.0)
    final_pred = (final_pred.cpu() * stats['j_s']) + stats['j_m']
    return j_raw, final_pred.numpy()

# ==========================================
# 4. RUN EXPERIMENT
# ==========================================
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if squat == True :
        dataset = "processed_data_feet"  #dataset vinc + humanoids bu only normal squat.
        res = "results_FM_S_weight"
    else : 
        dataset = "processed_data_feet_HUM" #else humanoids squat normal and paired
        res = "results_FM_HUM_weight"

    data_root = Path(f"/datasets/GRF2Kine/{dataset}")

    print("squat", squat)
    print(dataset)

    results_dir = Path(f"{res}")
    results_dir.mkdir(parents=True, exist_ok=True)

    
    all_samples = []

    subjects = sorted([d for d in data_root.iterdir() if d.is_dir()])

    for subject_dir in subjects:
        subject_name = subject_dir.name.lower()
        canonical_subject = re.sub(r'\d+$', '', subject_name)
        print(canonical_subject)

        # ---> ADDED: Extract weight from YAML <---
        weight = 70.0 # Fallback in case a file is missing
        
        # Look for any .yaml or .yml file in the subject directory
        yaml_files = list(subject_dir.glob("*.yaml")) + list(subject_dir.glob("*.yml"))
        
        if yaml_files:
            try:
                with open(yaml_files[0], 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    if 'weight_kg' in yaml_data:
                        weight = float(yaml_data['weight_kg'])
            except Exception as e:
                print(f"  [WARNING] Could not read weight for {subject_name}: {e}")
        else:
            print(f"  [WARNING] No YAML found for {subject_name}, using default 70.0 kg")
            
        task_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]

        for task_dir in task_dirs:
            task_name = task_dir.name.lower()

            if squat == True:
                if not is_squat_task_only(subject_name, task_name):
                    continue
            else : 
                if not is_squat_task(subject_name, task_name):
                    continue

            kinetics_file = task_dir / "kinetics_feet.npy"
            joints_file = task_dir / "all_joints.npy"

            if kinetics_file.exists() and joints_file.exists():
                all_samples.append({
                    "subject": subject_name,
                    "canonical_subject": canonical_subject,
                    "task": task_name,
                    "kinetics": kinetics_file,
                    "joints": joints_file,
                    "weight": weight  # <--- Passes the dynamically loaded weight
                })

    print(f"Nombre total de samples squat : {len(all_samples)}")
    print(all_samples)
    unique_subjects = sorted(list(set(s["canonical_subject"] for s in all_samples)))

    random.seed(42)
    random.shuffle(unique_subjects)


    def split_list(lst, train_ratio=0.7, val_ratio=0.15):

        n = len(lst)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = lst[:n_train]
        val = lst[n_train:n_train + n_val]
        test = lst[n_train + n_val:]

        return train, val, test


    train_subjects, val_subjects, test_subjects = split_list(
        unique_subjects,
        train_ratio=0.7,
        val_ratio=0.15
    )


    train_subs = [s for s in all_samples if s["canonical_subject"] in train_subjects]
    val_subs = [s for s in all_samples if s["canonical_subject"] in val_subjects]
    test_subs = [s for s in all_samples if s["canonical_subject"] in test_subjects]

    print(f"\n[SPLIT SUMMARY]")
    print(f"Train subjects : {len(train_subjects)}")
    print(f"Train subjects   : {train_subjects}")
    print(f"Val subjects   : {len(val_subjects)}")
    print(f"Val subjects   : {val_subjects}")
    print(f"Test subjects  : {len(test_subjects)}")
    print(f"Test subjects   : {test_subjects}")
    print(f"{'='*30}")

    def get_pairs(samples):

        pairs = []

        for s in samples:

            f = s["kinetics"]
            j = s["joints"]
            w = s["weight"]
            if f.exists() and j.exists():
                pairs.append((f, j,w))

        return pairs


    train_pairs = get_pairs(train_subs)
    print("train_pairs", train_pairs)

    stats = compute_and_save_stats(train_pairs, results_dir /"scalers_concat.json")
    
    train_loader = DataLoader(BiomechDiffusionDataset(train_pairs, stats=stats), batch_size=64, shuffle=True)
    val_loader = DataLoader(BiomechDiffusionDataset(get_pairs(val_subs), stats=stats), batch_size=64)

    # --- Initialisation Modèle (Ton Transformer est compatible !) ---
    model = DiffusionTransformer(joint_dim=12).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    train_losses, val_losses = [], []

    epochs = 1 # Le CFM peut nécessiter plus d'epochs mais converge vers un meilleur résultat
    best_val_loss = float('inf')

    print(f"\n[START] Entraînement Conditional Flow Matching...")
    for epoch in range(epochs):
        model.train()
        t_epoch_loss = 0
        for f, j,w in train_loader:
            f, j,w  = f.to(device), j.to(device), w.to(device) # j est x_1 (target)
            
            # 1. CFM Sampling: t ~ U(0, 1)
            t = torch.rand((j.shape[0],), device=device)
            t_view = t.view(-1, 1, 1)
            
            # 2. Draw noise x_0 and create interpolation x_t
            x_0 = torch.randn_like(j)
            x_t = (1 - t_view) * x_0 + t_view * j 
            
            # 3. Target velocity: v_t = x_1 - x_0
            target_v = j - x_0
            
            optimizer.zero_grad()
            pred_v = model(x_t, t, f, w)
            
            loss = nn.MSELoss()(pred_v, target_v)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_epoch_loss += loss.item()
        
        # --- Validation ---
        model.eval()
        v_epoch_loss = 0
        with torch.no_grad():
            for f, j, w in val_loader:
                f, j, w = f.to(device), j.to(device), w.to(device)
                t = torch.rand((j.shape[0],), device=device)
                t_view = t.view(-1, 1, 1)
                x_0 = torch.randn_like(j)
                x_t = (1 - t_view) * x_0 + t_view * j
                target_v = j - x_0
                
                pred_v = model(x_t, t, f, w)
                v_epoch_loss += nn.MSELoss()(pred_v, target_v).item()
        
        avg_train = t_epoch_loss / len(train_loader)
        avg_val = v_epoch_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(f"Epoch {epoch:02d} | Loss: {avg_train:.6f} | Val: {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), results_dir / "cfm_best_model.pth")

   # --- INFERENCE TEST ---
    # Utilise la nouvelle fonction sample_cfm (Euler)
    print("\n[INF] Inférence finale sur un essai complet...")
    test_pairs = get_pairs(test_subs)
    random_trial = random.choice(test_pairs)
    print("random_trial for test", random_trial)
    
    ref_full, pred_full = predict_full_trial(model, random_trial[0], random_trial[1], random_trial[2], stats, device, steps=25)

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(ref_full[:, i], 'k--', alpha=0.6, label='Reference')
        ax.plot(pred_full[:, i], 'r', label='CFM Pred')
        ax.set_title(f"Joint {i+1}")
        if i == 0: ax.legend()
    plt.suptitle(f"Full Trial FM Inference: {random_trial[0].parent.name}", fontsize=16)
    plt.tight_layout(); plt.savefig(results_dir /"full_trial_inference_concat.png"); plt.close()
    
    plt.figure(); plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
    plt.title("Loss History"); plt.legend(); plt.savefig(results_dir /"loss_curve_concat.png"); plt.close()
    print(f"\n[FINISH] Résultats sauvegardés. {results_dir}")

if __name__ == "__main__":
    run_experiment()

