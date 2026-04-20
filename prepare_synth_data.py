import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DES COLONNES
# ─────────────────────────────────────────────────────────────────────────────

# Colonnes attendues dans tes CSV de sortie "feet_frame" (Right=1, Left=2)
KINETICS_COLS = [
    'Fx1', 'Fy1', 'Fz1', 'Mx1', 'My1', 'Mz1', 'COPx1', 'COPy1', 'COPz1',
    'Fx2', 'Fy2', 'Fz2', 'Mx2', 'My2', 'Mz2', 'COPx2', 'COPy2', 'COPz2'
]

# Réorganisation demandée des Joints (Freeflyer -> Right Leg -> Left Leg -> Upper Body)
JOINTS_REORDER = [
    "FF_X", "FF_Y", "FF_Z", "FF_quatx", "FF_quaty", "FF_quatz", "FF_quatw", 
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot", "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot", "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add", 
    "Lumbar_flex_ext", "Lumbar_lateral_flex", "Lcalvicule_x",
    "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot", "Lelbow_flex_ext", "Lelbow_pron_supi",
    "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot", "rcalvicule_x",
    "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot", "Relbow_flex_ext", "Relbow_pron_supi"
]

# Mapping pour le Freeflyer (si tes CSV utilisent delta_x ou q_0, q_1...)
# On suppose que les 7 premières colonnes du CSV sont le FF dans l'ordre X,Y,Z, qx,qy,qz,qw


def process_folder(input_folder, output_base):
    input_path = Path(input_folder)
    output_path = Path(output_base)
    
    # 1. Lister tous les fichiers "joint_filtered" pour identifier les trials
    joint_files = glob.glob(str(input_path / "joint_filtered_*.csv"))
    
    print(f"Nombre de fichiers joints trouvés : {len(joint_files)}")

    for j_file in joint_files:
        # Extraire le nom du trial (tout ce qui est après 'joint_filtered_')
        trial_id = os.path.basename(j_file).replace("joint_filtered_", "").replace(".csv", "")
        
        # Chercher le fichier feet_frame correspondant
        k_file = input_path / f"feet_frame_{trial_id}.csv"
        
        if not k_file.exists():
            print(f"  [SKIP] Pas de fichier kinetics pour le trial : {trial_id}")
            continue

        # Créer le dossier du trial
        trial_dir = output_path / trial_id
        
        # --- ANTI-DOUBLON ---
        if (trial_dir / "kinetics.npy").exists():
            print(f"  [SKIP] Déjà converti : {trial_id}")
            continue

        print(f"  [PROC] Trial : {trial_id}")
        trial_dir.mkdir(parents=True, exist_ok=True)

        try:
            # --- TRAITEMENT KINETICS ---
            df_k = pd.read_csv(k_file)
            # On vérifie si les colonnes utilisent fR ou footR (selon ta version de code)
            rename_map = {col: col.replace('footR', 'fR').replace('footL', 'fL') for col in df_k.columns}
            df_k = df_k.rename(columns=rename_map)
            
            # Sélection et conversion (Right puis Left)
            arr_k = df_k[KINETICS_COLS].values.astype(np.float32)
            np.save(trial_dir / "kinetics.npy", arr_k)

            # --- TRAITEMENT JOINTS ---
            df_j = pd.read_csv(j_file)
            
            # On gère le renommage du Freeflyer pour correspondre à ta liste JOINTS_REORDER
            # On suppose que les 7 premières colonnes sont le FF
            old_ff_cols = df_j.columns[:7]
            ff_rename = {old: new for old, new in zip(old_ff_cols, JOINTS_REORDER[:7])}
            df_j = df_j.rename(columns=ff_rename)

            # Réorganisation selon ta liste précise
            arr_j = df_j[JOINTS_REORDER].values.astype(np.float32)
            np.save(trial_dir / "all_joints.npy", arr_j)

        except Exception as e:
            print(f"  [ERROR] Erreur sur {trial_id} : {e}")

if __name__ == "__main__":
    # Dossier où se trouvent tes fichiers en vrac
    IN_FOLDER = "DATA/generated_human_like_motions_csv/generated_human_like_motions_csv"
    # Dossier où tu veux créer tes dossiers de trials
    OUT_FOLDER = "DATA/npy_synth"
    
    process_folder(IN_FOLDER, OUT_FOLDER)
    print("\n--- Opération terminée ---")