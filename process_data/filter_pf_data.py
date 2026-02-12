import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def lowpass_filter(data, fs, fc, order=4):
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, data, axis=0)


def compute_cop_single_fp(forces, moments, fz_threshold=30, eps=1e-8):
    """
    forces  : (N, 3) -> Fx, Fy, Fz
    moments : (N, 3) -> Mx, My, Mz
    """
    N = forces.shape[0]
    cop = np.full((N, 3), np.nan)

    for i in range(N):
        Fx, Fy, Fz = forces[i]
        if Fz < fz_threshold:
            continue

        F = forces[i]
        M = moments[i]

        force_norm_sq = np.linalg.norm(F)**2 + eps

        CoP_i = np.cross(F, M) / force_norm_sq
        CoP_i -= (CoP_i[2] / (F[2] + eps)) * F

        cop[i] = CoP_i

    return cop


def filter_two_forceplates_csv(
    input_csv,
    output_csv="kinetics_filtered.csv",
    fs=1000,
    fc=20,
    fz_threshold=30
):
    df = pd.read_csv(input_csv)

    # --- FP1 ---
    force_cols_1 = ["Fx1", "Fy1", "Fz1"]
    moment_cols_1 = ["Mx1_glob", "My1_glob", "Mz1_glob"]

    forces1 = df[force_cols_1].values
    moments1 = df[moment_cols_1].values

    forces1_filt = lowpass_filter(forces1, fs, fc)
    moments1_filt = lowpass_filter(moments1, fs, fc)

    cop1 = compute_cop_single_fp(
        forces1_filt, moments1_filt, fz_threshold
    )

    # --- FP2 ---
    force_cols_2 = ["Fx2", "Fy2", "Fz2"]
    moment_cols_2 = ["Mx2_glob", "My2_glob", "Mz2_glob"]

    forces2 = df[force_cols_2].values
    moments2 = df[moment_cols_2].values

    forces2_filt = lowpass_filter(forces2, fs, fc)
    moments2_filt = lowpass_filter(moments2, fs, fc)

    cop2 = compute_cop_single_fp(
        forces2_filt, moments2_filt, fz_threshold
    )

    # --- Sauvegarde ---
    df_out = df.copy()

    df_out[["Fx1_filt", "Fy1_filt", "Fz1_filt"]] = forces1_filt
    df_out[["Mx1_filt", "My1_filt", "Mz1_filt"]] = moments1_filt
    df_out[["CoP1_x", "CoP1_y", "CoP1_z"]] = cop1

    df_out[["Fx2_filt", "Fy2_filt", "Fz2_filt"]] = forces2_filt
    df_out[["Mx2_filt", "My2_filt", "Mz2_filt"]] = moments2_filt
    df_out[["CoP2_x", "CoP2_y", "CoP2_z"]] = cop2

    df_out.to_csv(output_csv, index=False)
    print(f"✅ Fichier sauvegardé : {output_csv}")

# subjects = ["subject01","subject02","subject03","subject04","subject05","subject06","subject07","subject08","subject09","subject10","subject11","subject12",
#             "subject13","subject14","subject15","subject16", ]
# tasks = ["bend","cmjs", "cmjs_2", "dyna","luyo", "lufe", "walk", "stsk", "stsf", "static", "static2", "static3"]
# data_root = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais"

# import os
# for subject in subjects:
#     for task in tasks:
#         input_csv = os.path.join(data_root, subject, task, "kinetics.csv")
#         output_csv = os.path.join(data_root, subject, task, "kinetics_filtered.csv")
        
#         if not os.path.exists(input_csv):
#             print(f"Fichier manquant : {input_csv} → je passe au suivant")
#             continue

#         try:
#             filter_two_forceplates_csv(
#                 input_csv=input_csv,
#                 output_csv=output_csv,
#                 fs=300,
#                 fc=20,
#                 fz_threshold=30
#             )
#         except Exception as e:
#             print(f"Erreur sur {input_csv} : {e}")
#             # continue

import os
from pathlib import Path

# --- Configuration ---
subjects = [f"subject{i:02d}" for i in range(1, 17)]  # Génère de subject01 à subject16
data_root = Path("/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais")

# --- Boucle de traitement ---
for subject in subjects:
    subject_path = data_root / subject
    
    if not subject_path.exists():
        print(f"⚠️ Dossier sujet introuvable : {subject_path}")
        continue

    print(f"\n--- Traitement du {subject} ---")
    
    # On cherche récursivement tous les fichiers "kinetics.csv" dans le dossier du sujet
    # Cela trouvera toutes les tâches (bend, walk, etc.) automatiquement
    kinetics_files = list(subject_path.rglob("kinetics.csv"))

    if not kinetics_files:
        print(f"  Aucun fichier kinetics.csv trouvé pour {subject}")
        continue

    for input_csv in kinetics_files:
        # On définit le nom du fichier de sortie dans le même dossier
        output_csv = input_csv.parent / "kinetics_filtered.csv"
        
        # Récupération du nom de la tâche (nom du dossier parent) pour le log
        task_name = input_csv.parent.name
        
        try:
            print(f"  ➡️ Filtrage de la tâche : {task_name}")
            
            filter_two_forceplates_csv(
                input_csv=str(input_csv),
                output_csv=str(output_csv),
                fs=300,
                fc=20,
                fz_threshold=30
            )
            
        except Exception as e:
            print(f"  ❌ Erreur sur {subject}/{task_name} : {e}")

print("\n✅ Terminé ! Tous les fichiers trouvés ont été traités.")