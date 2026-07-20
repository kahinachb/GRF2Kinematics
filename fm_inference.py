import math
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
#  BUT DE CE SCRIPT
#  Un seul harnais d'inférence/évaluation pour les DEUX modèles :
#    - "baseline" : l'archi d'origine (sans segment embeddings, PE après concat)
#    - "improved" : l'archi travaillée (segment embeddings, PE par bloc, EMA)
#
#  Pour une comparaison HONNÊTE, on évalue les deux avec le MÊME sampler et le
#  MÊME nombre de pas -> même budget d'évaluations réseau (NFE). Le seul facteur
#  qui change est alors le modèle lui-même.
#      run_inference(..., variant="baseline", solver="heun", n_steps=20)
#      run_inference(..., variant="improved", solver="heun", n_steps=20)
#
#  Le temps t :
#    - baseline : entraîné avec t*1000 passé de l'extérieur (embedding non scalé)
#    - improved : entraîné avec t in [0,1] (scaling dans l'embedding)
#  -> géré par le champ "time_mul" de chaque variante. Côté appelant : t in [0,1].
# =====================================================================


# =====================================================================
#  BRIQUES PARTAGÉES
# =====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SinusoidalTimeEmbeddings(nn.Module):
    """time_scale=1.0  -> embedding "brut" (baseline : le ×1000 est fait dehors)
       time_scale=1000 -> scaling interne (improved : l'appelant passe t in [0,1])"""
    def __init__(self, dim, time_scale=1.0):
        super().__init__()
        self.dim = dim
        self.time_scale = time_scale

    def forward(self, time):
        time = time.float() * self.time_scale
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# =====================================================================
#  MODÈLE 1 — BASELINE (reproduit EXACTEMENT l'archi entraînée d'origine,
#  sinon le checkpoint ne se charge pas : PE après concat, pas de seg emb,
#  2 couches en dur, embedding de temps non scalé)
# =====================================================================
class FlowTransformerBaseline(nn.Module):
    def __init__(self, embed_dim=256, nhead=8):
        super().__init__()
        self.embed_Rleg = nn.Linear(6, embed_dim)
        self.embed_Lleg = nn.Linear(6, embed_dim)
        self.embed_Upper = nn.Linear(17, embed_dim)
        self.embed_F_R = nn.Linear(3, embed_dim)
        self.embed_M_R = nn.Linear(3, embed_dim)
        self.embed_CoP_R = nn.Linear(3, embed_dim)
        self.embed_F_L = nn.Linear(3, embed_dim)
        self.embed_M_L = nn.Linear(3, embed_dim)
        self.embed_CoP_L = nn.Linear(3, embed_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(embed_dim, time_scale=1.0),
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim),
        )
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=2000)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=512,
            activation="gelu", batch_first=True, norm_first=True, dropout=0.25,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=2)  # 2 couches (comme à l'entraînement)

        self.out_Rleg = nn.Linear(embed_dim, 6)
        self.out_Lleg = nn.Linear(embed_dim, 6)
        self.out_Upper = nn.Linear(embed_dim, 17)

    def forward(self, x, t, cond):
        B, W, _ = x.shape
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        t_emb = self.time_mlp(t).unsqueeze(1)

        emb_R = self.embed_Rleg(x[:, :, 0:6])
        emb_L = self.embed_Lleg(x[:, :, 6:12])
        emb_U = self.embed_Upper(x[:, :, 12:29])
        x_emb = torch.cat([emb_R, emb_L, emb_U], dim=1) + t_emb
        x_emb = self.pos_encoder(x_emb)

        emb_FR = self.embed_F_R(cond[:, :, 0:3]);  emb_MR = self.embed_M_R(cond[:, :, 3:6])
        emb_CoPR = self.embed_CoP_R(cond[:, :, 6:9])
        emb_FL = self.embed_F_L(cond[:, :, 9:12]); emb_ML = self.embed_M_L(cond[:, :, 12:15])
        emb_CoPL = self.embed_CoP_L(cond[:, :, 15:18])
        cond_emb = torch.cat([emb_FR, emb_MR, emb_CoPR, emb_FL, emb_ML, emb_CoPL], dim=1) + t_emb
        cond_emb = self.pos_encoder(cond_emb)

        out = self.transformer(tgt=x_emb, memory=cond_emb)
        out_R = self.out_Rleg(out[:, 0:W, :])
        out_L = self.out_Lleg(out[:, W:2 * W, :])
        out_U = self.out_Upper(out[:, 2 * W:3 * W, :])
        return torch.cat([out_R, out_L, out_U], dim=2)


# =====================================================================
#  MODÈLE 2 — IMPROVED (segment embeddings + PE par bloc + temps continu scalé)
# =====================================================================
class FlowTransformerImproved(nn.Module):
    def __init__(self, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
        self.embed_Rleg = nn.Linear(6, embed_dim)
        self.embed_Lleg = nn.Linear(6, embed_dim)
        self.embed_Upper = nn.Linear(17, embed_dim)
        self.embed_F_R = nn.Linear(3, embed_dim)
        self.embed_M_R = nn.Linear(3, embed_dim)
        self.embed_CoP_R = nn.Linear(3, embed_dim)
        self.embed_F_L = nn.Linear(3, embed_dim)
        self.embed_M_L = nn.Linear(3, embed_dim)
        self.embed_CoP_L = nn.Linear(3, embed_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(embed_dim, time_scale=1000.0),
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim),
        )
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=2000)
        self.seg_target = nn.Parameter(torch.randn(3, embed_dim) * 0.02)
        self.seg_cond = nn.Parameter(torch.randn(6, embed_dim) * 0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=512,
            activation="gelu", batch_first=True, norm_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=num_layers)

        self.out_Rleg = nn.Linear(embed_dim, 6)
        self.out_Lleg = nn.Linear(embed_dim, 6)
        self.out_Upper = nn.Linear(embed_dim, 17)

    def forward(self, x, t, cond):
        B, W, _ = x.shape
        if t.dim() == 0:
            t = t.float().unsqueeze(0).expand(B)
        t_emb = self.time_mlp(t).unsqueeze(1)

        emb_R = self.pos_encoder(self.embed_Rleg(x[:, :, 0:6]))   + self.seg_target[0]
        emb_L = self.pos_encoder(self.embed_Lleg(x[:, :, 6:12]))  + self.seg_target[1]
        emb_U = self.pos_encoder(self.embed_Upper(x[:, :, 12:29])) + self.seg_target[2]
        x_emb = torch.cat([emb_R, emb_L, emb_U], dim=1) + t_emb

        cond_blocks = [
            self.embed_F_R(cond[:, :, 0:3]), self.embed_M_R(cond[:, :, 3:6]), self.embed_CoP_R(cond[:, :, 6:9]),
            self.embed_F_L(cond[:, :, 9:12]), self.embed_M_L(cond[:, :, 12:15]), self.embed_CoP_L(cond[:, :, 15:18]),
        ]
        cond_blocks = [self.pos_encoder(b) + self.seg_cond[i] for i, b in enumerate(cond_blocks)]
        cond_emb = torch.cat(cond_blocks, dim=1) + t_emb

        out = self.transformer(tgt=x_emb, memory=cond_emb)
        out_R = self.out_Rleg(out[:, 0:W, :])
        out_L = self.out_Lleg(out[:, W:2 * W, :])
        out_U = self.out_Upper(out[:, 2 * W:3 * W, :])
        return torch.cat([out_R, out_L, out_U], dim=2)


# =====================================================================
#  CONFIG DES VARIANTES + CHARGEMENT
# =====================================================================
VARIANTS = {
    "baseline": {"cls": FlowTransformerBaseline, "ckpt_key": None,    "time_mul": 1000.0},
    "improved": {"cls": FlowTransformerImproved, "ckpt_key": "model", "time_mul": 1.0},
}


def build_model(variant, model_path, device):
    cfg = VARIANTS[variant]
    model = cfg["cls"]().to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and cfg["ckpt_key"] is not None and cfg["ckpt_key"] in ckpt:
        state = ckpt[cfg["ckpt_key"]]       # improved : poids EMA
    else:
        state = ckpt                        # baseline : state_dict brut
    model.load_state_dict(state)
    model.eval()
    return model, cfg["time_mul"]


# =====================================================================
#  SAMPLER UNIFIÉ (Euler ou Heun) + inpainting — identique pour les 2 modèles
# =====================================================================
def _v(model, x, t_scalar, f, time_mul):
    tt = torch.full((x.shape[0],), float(t_scalar) * time_mul, device=x.device)
    return model(x, tt, f)


@torch.no_grad()
def predict_full_trial(model, f_path, j_path, stats, device, time_mul,
                       window_size=128, stride=64, n_steps=20, solver="heun", seed=0):
    torch.manual_seed(seed)               # bruit reproductible (par graine)
    model.eval()
    f_raw = np.load(f_path).astype(np.float32)
    # f_raw = np.concatenate(
    #     [f_raw[:,-9:], f_raw[:, :9]],
    #     axis=1
    # ) 

    j_raw = np.load(j_path).astype(np.float32)[:, 6:]
    f_norm = (torch.from_numpy(f_raw) - stats['f_m']) / (stats['f_s'] + 1e-6)

    T = f_norm.shape[0]
    full_pred = torch.zeros((T, 29), device=device)
    overlap = window_size - stride
    dt = 1.0 / n_steps

    if T < window_size:
        raise ValueError(
            f"Trial has {T} frames, shorter than window_size={window_size}."
        )

    # Regular windows may not land exactly on T - window_size.  Append one
    # final window ending at T so every frame receives a prediction.
    starts = list(range(0, T - window_size + 1, stride))
    last_start = T - window_size
    if starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        end = start + window_size
        f_win = f_norm[start:end].unsqueeze(0).to(device)
        has_ctx = start > 0

        x0 = torch.randn((1, window_size, 29), device=device)
        x = x0.clone()
        if has_ctx:
            known_x1 = full_pred[start:start + overlap].unsqueeze(0)
            x0_known = x0[:, :overlap, :]
            interp = lambda tv: (1.0 - tv) * x0_known + tv * known_x1
            x[:, :overlap, :] = interp(0.0)

        for i in range(n_steps):
            t0 = i / n_steps
            t1 = (i + 1) / n_steps
            if has_ctx:
                x[:, :overlap, :] = interp(t0)
            v1 = _v(model, x, t0, f_win, time_mul)
            if solver == "euler":
                x = x + v1 * dt
            else:  # heun (2e ordre, 2 NFE/pas)
                x_pred = x + v1 * dt
                if has_ctx:
                    x_pred[:, :overlap, :] = interp(t1)
                v2 = _v(model, x_pred, t1, f_win, time_mul)
                x = x + 0.5 * (v1 + v2) * dt

        if has_ctx:
            x[:, :overlap, :] = known_x1
        full_pred[start:end] = x.squeeze(0)

    final_pred = (full_pred.cpu() * stats['j_s']) + stats['j_m']
    return j_raw, final_pred.numpy()


# =====================================================================
#  MÉTRIQUES
# =====================================================================
def per_joint_metrics(j_ref, j_pred):
    err = j_ref - j_pred
    rmse = np.sqrt(np.mean(err ** 2, axis=0))   # (29,)
    mae = np.mean(np.abs(err), axis=0)          # (29,)
    return rmse, mae


JOINT_NAMES = [
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",
    "Lumbar_flex_ext", "Lumbar_lateral_flex", "Lcalvicule_x",
    "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
    "Lelbow_flex_ext", "Lelbow_pron_supi",
    "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
    "Rcalvicule_x", "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot",
    "Relbow_flex_ext", "Relbow_pron_supi",
]


# =====================================================================
#  PLOTS — repris à l'identique de ton code (3 figures)
# =====================================================================
def make_plots(j_ref, j_preds, output_dir, subject_name):
    lower_names = JOINT_NAMES[:12]
    upper_names = JOINT_NAMES[12:]

    # FIG 1 : Lower body (Right vs Left)
    fig, axes = plt.subplots(6, 2, figsize=(15, 18))
    for row in range(6):
        for col, idx in ((0, row), (1, row + 6)):
            rmse = np.sqrt(np.mean((j_ref[:, idx] - j_preds[:, idx]) ** 2))
            axes[row, col].plot(j_ref[:, idx], 'k--', alpha=0.6, label="Reference")
            axes[row, col].plot(j_preds[:, idx], 'r', linewidth=2, label="Prediction")
            axes[row, col].set_title(f"{lower_names[idx]} (RMSE: {rmse:.4f})", fontsize=10)
            axes[row, col].grid(True)
    axes[0, 0].legend()
    plt.suptitle("Lower body joints (Right vs Left)", fontsize=16)
    plt.tight_layout(); plt.savefig(output_dir / f"{subject_name}_lower_body_joints.png", dpi=300); plt.close()

    # FIG 2 : Lumbar & Left side
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    lumbar_idx, left_c1, left_c2 = [12, 13], [14, 15, 16], [17, 18, 19]
    for row in range(3):
        if row < len(lumbar_idx):
            i = lumbar_idx[row]
            rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i]) ** 2))
            axes[row, 0].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")
            axes[row, 0].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
            axes[row, 0].set_title(f"{upper_names[i - 12]} (RMSE: {rmse:.4f})", fontsize=10)
            axes[row, 0].grid(True)
            if row == 0: axes[row, 0].legend()
        else:
            axes[row, 0].axis('off')
        for col, i in ((1, left_c1[row]), (2, left_c2[row])):
            rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i]) ** 2))
            axes[row, col].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")
            axes[row, col].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
            axes[row, col].set_title(f"{upper_names[i - 12]} (RMSE: {rmse:.4f})", fontsize=10)
            axes[row, col].grid(True)
    plt.suptitle("Upper body joints - Lumbar & Left side", fontsize=16)
    plt.tight_layout(); plt.savefig(output_dir / f"{subject_name}_upper_body_lumbar_left.png", dpi=300); plt.close()

    # FIG 3 : Cervical & Right side
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    cervical_idx, right_c1, right_c2 = [20, 21, 22], [23, 24, 25], [26, 27, 28]
    for row in range(3):
        for col, i in ((0, cervical_idx[row]), (1, right_c1[row]), (2, right_c2[row])):
            rmse = np.sqrt(np.mean((j_ref[:, i] - j_preds[:, i]) ** 2))
            axes[row, col].plot(j_ref[:, i], 'k--', alpha=0.6, label="Reference")
            axes[row, col].plot(j_preds[:, i], 'r', linewidth=2, label="Prediction")
            axes[row, col].set_title(f"{upper_names[i - 12]} (RMSE: {rmse:.4f})", fontsize=10)
            axes[row, col].grid(True)
        if row == 0: axes[row, 0].legend()
    plt.suptitle("Upper body joints - Cervical & Right side", fontsize=16)
    plt.tight_layout(); plt.savefig(output_dir / f"{subject_name}_upper_body_cervical_right.png", dpi=300); plt.close()


# =====================================================================
#  SCRIPT PRINCIPAL — garde ta logique de chargement, CSV et plots
# =====================================================================
def run_inference(subject_name, trial_name, model_path, scalers_path,
                  variant="improved", solver="heun", n_steps=20, n_seeds=1,
                  data_root="./processed_data", output_dir="./inference_results"):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nfe = n_steps * (2 if solver == "heun" else 1)
    print(f"\n{'='*60}")
    print(f"INFERENCE  variant={variant} | solver={solver} | n_steps={n_steps} | NFE/window={nfe}")
    print(f"  {subject_name} / {trial_name}  | device={device} | n_seeds={n_seeds}")
    print(f"{'='*60}\n")

    # ===== 1. SCALERS =====
    print("[1/5] Loading scalers...")
    with open(scalers_path, 'r') as f:
        stats = {k: torch.tensor(v).float() for k, v in json.load(f).items()}

    # ===== 2. MODEL =====
    print("[2/5] Loading model...")
    model, time_mul = build_model(variant, model_path, device)
    print(f"  ✓ {variant} loaded from {model_path}")

    # ===== 3. DATA FILES (ta logique) =====
    print("[3/5] Locating data files...")
    data_path = Path(data_root) / subject_name / f"{trial_name}"
    f_path = data_path / "kinetics_deltaf.npy"
    j_path = data_path / "all_joints_deltaf.npy"

    # f_path = data_path / "kinetics_glob.npy"
    # j_path = data_path / "all_joints.npy"

    if not f_path.exists():
        raise FileNotFoundError(f"Forces file not found: {f_path}")
    if not j_path.exists():
        raise FileNotFoundError(f"Joints file not found: {j_path}")
    print(f"  ✓ forces: {f_path}\n  ✓ joints: {j_path}")

    # ===== 4. INFERENCE (multi-graines pour des métriques robustes) =====
    print("[4/5] Running inference...")
    rmse_list, mae_list = [], []
    j_ref = None
    j_preds_seed0 = None
    for s in range(n_seeds):
        j_ref, j_preds = predict_full_trial(
            model, f_path, j_path, stats, device, time_mul,
            window_size=128, stride=64, n_steps=n_steps, solver=solver, seed=s,
        )
        rmse_s, mae_s = per_joint_metrics(j_ref, j_preds)
        rmse_list.append(rmse_s); mae_list.append(mae_s)
        if s == 0:
            j_preds_seed0 = j_preds

    rmse_arr = np.stack(rmse_list)   # (n_seeds, 29)
    mae_arr = np.stack(mae_list)
    rmse_mean, rmse_std = rmse_arr.mean(0), rmse_arr.std(0)
    mae_mean, mae_std = mae_arr.mean(0), mae_arr.std(0)

    # ===== SAVE PREDICTION CSV (graine 0, ta logique) =====
    df_pred = pd.DataFrame(j_preds_seed0, columns=JOINT_NAMES)
    csv_path = output_dir / f"{subject_name}_{trial_name}_{variant}_prediction.csv"
    df_pred.to_csv(csv_path, index=False)
    print(f"  ✓ Prediction saved: {csv_path}")

    # ===== 5. EVALUATION =====
    print("[5/5] Evaluating and plotting...")
    global_rmse = float(np.sqrt(np.mean(rmse_arr ** 2)))   # agrégé sur joints + graines
    global_mae = float(mae_arr.mean())
    print(f"  ✓ Global RMSE: {global_rmse:.6f}  | Global MAE: {global_mae:.6f}  (moyenne sur {n_seeds} graine(s))")

    # tableau par articulation -> CSV de métriques
    df_metrics = pd.DataFrame({
        "joint": JOINT_NAMES,
        "rmse_mean": rmse_mean, "rmse_std": rmse_std,
        "mae_mean": mae_mean, "mae_std": mae_std,
    })
    metrics_csv = output_dir / f"{subject_name}_{trial_name}_{variant}_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"  ✓ Per-joint metrics saved: {metrics_csv}")

    # plots (graine 0) — tes 3 figures
    make_plots(j_ref, j_preds_seed0, output_dir, subject_name)
    print(f"\n[FINISH] Plots saved to {output_dir}")

    return {
        "variant": variant, "solver": solver, "n_steps": n_steps, "nfe": nfe,
        "global_rmse": global_rmse, "global_mae": global_mae,
        "per_joint_rmse_mean": rmse_mean, "per_joint_mae_mean": mae_mean,
    }


# =====================================================================
#  EXEMPLE D'APPEL — comparaison à budget ÉGAL (même solver, même n_steps)
# =====================================================================
if __name__ == "__main__":
    SUBJECT, TRIAL = "subject_102", "variant_194_dz-0.200_dx-0.020_dy+0.012"
    DATA_ROOT = "DATA/synth_npy_102"


    # SUBJECT, TRIAL = "Jeremy", "Trial111"
    # DATA_ROOT = "processed_data_feet"


    res = {}
    # même solver + même n_steps -> même NFE : seul le modèle change
    res["baseline"] = run_inference(
        SUBJECT, TRIAL,
        model_path="results_full_102/fm_biomech_model_best.pth",
        scalers_path="results_full_102/scalers_concat.json",
        variant="baseline", solver="euler", n_steps=20, n_seeds=3,
        data_root=DATA_ROOT, output_dir="./results_full_102",
    )
    res["improved"] = run_inference(
        SUBJECT, TRIAL,
        model_path="results_full_102_improved/fm_biomech_model_best.pth",
        scalers_path="results_full_102_improved/scalers_concat.json",
        variant="improved", solver="euler", n_steps=20, n_seeds=3,
        data_root=DATA_ROOT, output_dir="results_full_102_improved",
    )

    print("\n================  RÉCAP  ================")
    for k, r in res.items():
        print(f"{k:10s} | NFE={r['nfe']:3d} | RMSE={r['global_rmse']:.5f} | MAE={r['global_mae']:.5f}")
