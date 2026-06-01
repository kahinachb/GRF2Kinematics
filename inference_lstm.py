import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import torch.nn as nn
import math
import yaml
# ==========================================
# 1. CFM INFERENCE FUNCTION (SLIDING WINDOW)
# ==========================================
def predict_full_trial(model, f_path, j_path, anchor, stats, device, window_size=40):
    """
    Génère les prédictions en un seul passage (Many-to-One) pour tout un essai.
    """
    model.eval()
    
    # 1. Load data
    f_raw = np.load(f_path).astype(np.float32)
    j_raw = np.load(j_path).astype(np.float32)[:, 6:18] # Focus bas du corps
    f_raw = f_raw[:, [0,1,2,3,4,5,9,10,11,12,13,14]]    # Filtrage des 12 forces/moments
    T = len(f_raw)
    
    print(f"  [INFO] Predicting trial with {T} frames using Bi-LSTM-MLP...")
    
    # 2. Normalisation avec les stats converties en numpy
    f_norm = (f_raw - stats['f_m'].numpy()) / (stats['f_s'].numpy() + 1e-6)
    a_norm = (np.array(anchor, dtype=np.float32) - stats['a_m'].numpy()) / (stats['a_s'].numpy() + 1e-6)
    
    # 3. Concaténation des 17 features
    a_repeated_full = np.tile(a_norm, (T, 1))
    f_combined = np.concatenate((f_norm, a_repeated_full), axis=1)
    
    # 4. Padding et découpage (stride de 1)
    f_padded = np.vstack([np.tile(f_combined[0], (window_size - 1, 1)), f_combined])
    windows = [f_padded[i : i + window_size] for i in range(T)]
    windows_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
    
    # 5. Prédiction Batchée (très rapide)
    with torch.no_grad():
        pred_j_norm = model(windows_tensor)
        
    # 6. Dénormalisation
    pred_j_norm = pred_j_norm.cpu().numpy()
    final_pred = (pred_j_norm * stats['j_s'].numpy()) + stats['j_m'].numpy()
    
    return j_raw, final_pred


# ==========================================
# 2. ARCHITECTURE (Compatible Flow Matching)
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
# 3. MAIN INFERENCE SCRIPT
# ==========================================
def run_inference(subject_name, trial_name, model_path, scalers_path, 
                  data_root="./processed_data", output_dir="./inference_results"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Bi-LSTM INFERENCE ON: {subject_name} / {trial_name}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # ===== 1. LOAD SCALERS =====
    print("[1/4] Loading scalers...")
    with open(scalers_path, 'r') as f:
        scalers_dict = json.load(f)
    stats = {k: torch.tensor(v).float() for k, v in scalers_dict.items()}
    print(f"  ✓ Loaded normalization statistics (including anthropometry)")
    
    # ===== 2. LOAD MODEL =====
    print("\n[2/4] Loading Bi-LSTM-MLP model...")
    model = BiLSTM_MLP(input_dim=17, output_dim=12).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  ✓ Loaded model from {model_path}")
    
    # ===== 3. LOCATE DATA & ANTHROPOMETRY =====
    print("\n[3/4] Locating data files...")
    subject_dir = Path(data_root) / subject_name
    data_path = subject_dir / trial_name
    f_path = data_path / "kinetics_feet.npy"
    j_path = data_path / "all_joints.npy"
    
    if not f_path.exists(): raise FileNotFoundError(f"Forces file not found: {f_path}")
    if not j_path.exists(): raise FileNotFoundError(f"Joints file not found: {j_path}")
    
    # Extraction des données anthropométriques du sujet
    anthro_data = [70.0, 450.0, 400.0, 450.0, 400.0] # Fallback
    yaml_files = list(subject_dir.glob("*.yaml")) + list(subject_dir.glob("*.yml"))
    
    if yaml_files:
        try:
            with open(yaml_files[0], 'r') as f:
                yd = yaml.safe_load(f)
                if 'weight_kg' in yd:
                    anthro_data = [
                        float(yd['weight_kg']), float(yd['left_femur_mm']),
                        float(yd['left_tibia_mm']), float(yd['right_femur_mm']),
                        float(yd['right_tibia_mm'])
                    ]
        except Exception as e:
            print(f"  [WARNING] Could not read anthro data: {e}")
            
    print(f"  ✓ Found forces: {f_path}\n  ✓ Found joints: {j_path}")
    print(f"  ✓ Anthropometry extracted: {anthro_data}")
    
    # ===== 4. RUN INFERENCE =====
    print("\n[4/4] Running inference...")
    j_ref, j_pred = predict_full_trial(
        model, f_path, j_path, anthro_data, stats, device, window_size=40
    )
    
    # ===== 5. SAVE RESULTS =====
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_name = f"{subject_name}_{trial_name}_bilstm_prediction.npy"
    np.save(output_path / save_name, j_pred)
    print(f"\n  ✓ Saved predictions to: {output_path / save_name}")
    
    # ===== 6. VISUALIZE =====
    print("\nCreating visualization...")
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    
    for i, ax in enumerate(axes.flatten()):
        ax.plot(j_ref[:, i], 'k--', alpha=0.6, linewidth=1.5, label='Ground Truth')
        ax.plot(j_pred[:, i], 'r', linewidth=1.5, label='BiLSTM Prediction')
        ax.set_title(f"Joint {i+1}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle")
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='upper right')
    
    plt.suptitle(f"BiLSTM Direct Prediction: {subject_name} / {trial_name}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_name = f"{subject_name}_{trial_name}_bilstm_comparison.png"
    plt.savefig(output_path / plot_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot to: {output_path / plot_name}")
    
    # ===== 7. COMPUTE METRICS =====
    print("\n" + "="*60)
    print("RESULTS SUMMARY (Bi-LSTM-MLP)")
    print("="*60)
    
    mse = np.mean((j_ref - j_pred)**2)
    mae = np.mean(np.abs(j_ref - j_pred))
    rmse = np.sqrt(mse)
    
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f} degrés")
    print(f"RMSE: {rmse:.6f} degrés")
    
    print("\nPer-joint RMSE (degrés):")
    for i in range(12):
        joint_rmse = np.sqrt(np.mean((j_ref[:, i] - j_pred[:, i])**2))
        print(f"  Joint {i+1:2d}: {joint_rmse:.6f}°")
    print("\n" + "="*60)


# ==========================================
# 4. RUN EXEMPLE
# ==========================================

if __name__ == "__main__":
    
    SUBJECT_NAME = "Mohamed"  # Assure-toi que c'est le bon nom (souvent en minuscules dans tes dossiers)
    TRIAL_NAME = "squat"     

    # Mets à jour ces chemins selon où ton entraînement a sauvegardé les fichiers !
    MODEL_PATH = "./results_lstm_HUM_weight_seg/bilstm_best_model.pth"
    SCALERS_PATH = "./results_lstm_HUM_weight_seg/scalers_concat.json"
    DATA_ROOT = "processed_data_feet_HUM"
    OUTPUT_DIR = "./results_lstm_HUM_weight_seg"
    
    run_inference(
        subject_name=SUBJECT_NAME,
        trial_name=TRIAL_NAME,
        model_path=MODEL_PATH,
        scalers_path=SCALERS_PATH,
        data_root=DATA_ROOT,
        output_dir=OUTPUT_DIR
    )