import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import json
import yaml
from utils.utils import is_squat_task, is_squat_task_only
import re

squat = False

# ==========================================
# 1. DATASET 
# ==========================================
class BiomechDiffusionDataset(Dataset):
    def __init__(self, file_list, window_size=40, stats=None):
        self.samples = []
        self.window_size = window_size # Sauvegarde pour getitem
        for f_path, j_path, weight in file_list:
            f_data, j_data = np.load(f_path).astype(np.float32), np.load(j_path).astype(np.float32)
            j_data = j_data[:, 6:18]
            f_data = f_data[:, [0,1,2,3,4,5,9,10,11,12,13,14]]
            
            # CORRECTION 1 : Many-to-one et stride de 1
            for i in range(len(f_data) - window_size):
                self.samples.append((f_data[i : i+window_size], j_data[i + window_size - 1], weight))
        self.stats = stats

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        f, j, a = self.samples[idx]
        f, j = torch.from_numpy(f), torch.from_numpy(j)
        a = torch.tensor(a, dtype=torch.float32)
        
        if self.stats:
            f = (f - self.stats['f_m']) / (self.stats['f_s'] + 1e-6)
            j = (j - self.stats['j_m']) / (self.stats['j_s'] + 1e-6)
            a = (a - self.stats['a_m']) / (self.stats['a_s'] + 1e-6)

        # CORRECTION 2 : Dupliquer les 5 données anthropométriques sur les 40 frames
        # a passe de shape (5,) à (40, 5)
        a_repeated = a.unsqueeze(0).repeat(self.window_size, 1)
        
        # Concaténer les 12 forces avec les 5 valeurs d'anthro -> shape (40, 17)
        f_combined = torch.cat((f, a_repeated), dim=1)

        # On renvoie 2 variables pour que la boucle d'entrainement fonctionne
        return f_combined, j

def compute_and_save_stats(file_list, save_path):
    print("\n[INFO] Calcul des scalers sur le Train Set...")
    all_f, all_j, all_a = [], [], []

    for f_p, j_p, a in file_list:
        all_f.append(np.load(f_p)); all_j.append(np.load(j_p))
        all_a.append(a)
    

    f_cat, j_cat = np.vstack(all_f), np.vstack(all_j)
    j_cat= j_cat[:, 6:18]
    f_cat = f_cat[:, [0,1,2,3,4,5,9,10,11,12,13,14]]
            
    a_cat = np.vstack(all_a)
    
    stats = {
        'f_m': f_cat.mean(axis=0), 'f_s': f_cat.std(axis=0),
        'j_m': j_cat.mean(axis=0), 'j_s': j_cat.std(axis=0),
        'a_m': a_cat.mean(axis=0), 'a_s': a_cat.std(axis=0)
    }
    serializable_stats = {k: v.tolist() for k, v in stats.items()}
    with open(save_path, 'w') as f:
        json.dump(serializable_stats, f)
    return {k: torch.tensor(v).float() for k, v in stats.items()}

# ==========================================
# 2. ARCHITECTURE
# ==========================================
class BiLSTM_MLP(nn.Module):
    def __init__(self, input_dim=17, hidden_lstm=32, output_dim=12):
        super(BiLSTM_MLP, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_lstm, num_layers=3, batch_first=True, bidirectional=True)
        mlp_input_dim = hidden_lstm * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_final = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) 
        return self.mlp(h_final) 
    
# ==========================================
# 3. FONCTION INFERENCE COMPLETE (SLIDING WINDOW)
# ==========================================
def predict_full_trial(model, f_path, j_path, anchor, stats, device, window_size=40):
    model.eval()
    f_raw = np.load(f_path).astype(np.float32)
    j_raw = np.load(j_path).astype(np.float32)[:, 6:18]
    f_raw = f_raw[:, [0,1,2,3,4,5,9,10,11,12,13,14]]
    T = len(f_raw)
    
    print(f"  [INF] Inférence directe Bi-LSTM-MLP sur essai complet ({T} frames)...")
    
    # Normalisation
    f_norm = (f_raw - stats['f_m']) / (stats['f_s'] + 1e-6)
    a_norm = (np.array(anchor, dtype=np.float32) - stats['a_m'].numpy()) / (stats['a_s'].numpy() + 1e-6)
    
    # Concaténation de l'anthropométrie statique sur toute la série temporelle
    # f_norm passe de 12 à 17 dimensions
    a_repeated_full = np.tile(a_norm, (T, 1))
    f_combined = np.concatenate((f_norm, a_repeated_full), axis=1)
    
    # Padding sur les 17 dimensions
    f_padded = np.vstack([np.tile(f_combined[0], (window_size - 1, 1)), f_combined])
    windows = [f_padded[i : i + window_size] for i in range(T)]
    windows_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred_j_norm = model(windows_tensor)
        
    pred_j_norm = pred_j_norm.cpu().numpy()
    final_pred = (pred_j_norm * stats['j_s'].numpy()) + stats['j_m'].numpy()
    
    return j_raw, final_pred

# ==========================================
# 4. RUN EXPERIMENT
# ==========================================
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if squat == True :
        dataset = "processed_data_feet"  #dataset vinc + humanoids bu only normal squat.
        res = "results_FM_S_weight_seg"
    else : 
        dataset = "/lustre/fsn1/projects/rech/vsi/ulm94jm/dataset_grf2kine/processed_data_feet_HUM" #else humanoids squat normal and paired
        res = "results_lstm_HUM_weight_seg"
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
        
        anthro_data = [70.0, 450.0, 400.0, 450.0, 400.0] 
        
        if yaml_files:
            try:
                with open(yaml_files[0], 'r') as f:
                    yd = yaml.safe_load(f)
                    if 'weight_kg' in yd:
                        anthro_data = [
                            float(yd['weight_kg']),
                            float(yd['left_femur_mm']),
                            float(yd['left_tibia_mm']),
                            float(yd['right_femur_mm']),
                            float(yd['right_tibia_mm']),

                        ]
            except Exception as e:
                print(f"  [WARNING] Could not read anthro data for {subject_name}: {e}")
            
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
                    "anchor": anthro_data  # <--- Passes the dynamically loaded weight
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
            w = s["anchor"]
            if f.exists() and j.exists():
                pairs.append((f, j,w))

        return pairs


    train_pairs = get_pairs(train_subs)
    print("train_pairs", train_pairs)

    stats = compute_and_save_stats(train_pairs, results_dir /"scalers_concat.json")
    
    train_loader = DataLoader(BiomechDiffusionDataset(train_pairs, stats=stats), batch_size=64, shuffle=True)
    val_loader = DataLoader(BiomechDiffusionDataset(get_pairs(val_subs), stats=stats), batch_size=64)

    # --- Initialisation Modèle (Ton Transformer est compatible !) ---
    model = BiLSTM_MLP(input_dim=17, output_dim=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=1e-3)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    epochs = 2800
    best_val_loss = float('inf')

    print("\n[START] Entraînement Bi-LSTM-MLP (Direct Prediction)...")
    for epoch in range(epochs):
        model.train()
        t_epoch_loss = 0
        
        for f, j in train_loader:
            f, j = f.to(device), j.to(device)
            optimizer.zero_grad()
            pred_j = model(f)
            loss = criterion(pred_j, j) 
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_epoch_loss += loss.item()
        
        # --- Validation ---
        model.eval()
        v_epoch_loss = 0
        val_rmse_deg = 0.0 # Correction : Initialisation de la variable
        
        with torch.no_grad():
            for f, j in val_loader:
                f, j = f.to(device), j.to(device)
            
                pred_j = model(f)
                v_epoch_loss += nn.MSELoss()(pred_j, j).item()

                j_s_tensor = stats['j_s'].to(device)
                j_m_tensor = stats['j_m'].to(device)
                
                pred_j_deg = (pred_j * j_s_tensor) + j_m_tensor
                true_j_deg = (j * j_s_tensor) + j_m_tensor
                
                mse_deg = nn.MSELoss()(pred_j_deg, true_j_deg)
                val_rmse_deg += torch.sqrt(mse_deg).item()
        
        avg_train = t_epoch_loss / len(train_loader)
        avg_val = v_epoch_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        avg_rmse = val_rmse_deg / len(val_loader)
        print(f"Epoch {epoch:02d} | Train Loss (norm): {avg_train:.4f} | Val RMSE (degrés): {avg_rmse:.2f}°")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), results_dir / "bilstm_best_model.pth")


   # --- INFERENCE TEST ---
    print("\n[INF] Inférence finale sur un essai complet...")
    test_pairs = get_pairs(test_subs)
    random_trial = random.choice(test_pairs)
    print(random_trial)
    
    # Correction : Suppression de l'argument steps=25
    ref_full, pred_full = predict_full_trial(model, random_trial[0], random_trial[1], random_trial[2], stats, device)
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(ref_full[:, i], 'k--', alpha=0.6, label='Reference')
        ax.plot(pred_full[:, i], 'r', label='BiLSTM Pred') # Correction label
        ax.set_title(f"Joint {i+1}")
        if i == 0: ax.legend()
    plt.suptitle(f"Full Trial Inference: {random_trial[0].parent.name}", fontsize=16)
    plt.tight_layout(); plt.savefig(results_dir /"full_trial_inference_concat.png"); plt.close()
    
    plt.figure(); plt.plot(train_losses, label="Train"); plt.plot(val_losses, label="Val")
    plt.title("Loss History"); plt.legend(); plt.savefig(results_dir /"loss_curve_concat.png"); plt.close()
    print(f"\n[FINISH] Résultats sauvegardés. {results_dir}")

if __name__ == "__main__":
    run_experiment()

