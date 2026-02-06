import ezc3d
import numpy as np
import pandas as pd
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
input_dir = "/home/kchalabi/Documents/THESE/datasets_kinetics/Anais"
base_output_dir = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais"

DOWNSAMPLE_FACTOR = 10  # Rapport 3000 Hz / 300 Hz

# =============================================================================
# FONCTION DE TRAITEMENT D'UN FICHIER C3D
# =============================================================================
def process_c3d_file(file_path, output_dir):
    """
    Traite un fichier C3D et sauvegarde les marqueurs et cinétique
    """
    try:
        print(f"\n{'='*80}")
        print(f"Traitement de : {os.path.basename(file_path)}")
        print(f"{'='*80}")
        
        # Création du dossier de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Chargement du C3D avec extraction des plateformes
        c3d = ezc3d.c3d(file_path, extract_forceplat_data=True)
        
        marker_names = c3d['parameters']['POINT']['LABELS']['value']
        points = c3d['data']['points']
        n_frames_markers = points.shape[2]
        
        # =====================================================================
        # 1. MARQUEURS (300 Hz)
        # =====================================================================
        marker_dict = {"frame": np.arange(n_frames_markers)}
        for i, name in enumerate(marker_names):
            marker_dict[f"{name}_x"] = points[0, i, :]
            marker_dict[f"{name}_y"] = points[1, i, :]
            marker_dict[f"{name}_z"] = points[2, i, :]
        
        df_markers = pd.DataFrame(marker_dict)
        
        # =====================================================================
        # 2. FORCES, MOMENTS ET COP
        # =====================================================================
        num_platforms = len(c3d["data"]["platform"])
        print(f"Nombre de plateformes détectées : {num_platforms}")
        
        kinetics_data = []
        platforms_data = []
        
        for num_PF in range(num_platforms):
            pf = c3d["data"]["platform"][num_PF]
            
            # Extraction des données brutes
            PF_force = pf['force']
            PF_moment = pf['moment'] 
            PF_origin = np.mean(pf['corners'], axis=1) 
            
            n_frames_analog = PF_force.shape[1]
            
            # Calcul du moment à l'origine du repère global
            PF_moment0 = np.zeros_like(PF_moment)
            for i in range(PF_moment0.shape[1]):
                PF_moment0[:, i] = PF_moment[:, i] + np.cross(PF_origin, PF_force[:, i])
            
            # Calcul du CoP frame par frame
            CoP = np.zeros_like(PF_force)
            for i in range(n_frames_analog):
                force_norm_sq = np.linalg.norm(PF_force[:, i])**2
                
                if force_norm_sq > 1e-6:
                    CoP_i = np.cross(PF_force[:, i], PF_moment0[:, i]) / force_norm_sq
                    CoP_i -= (CoP_i[2] / PF_force[2, i]) * PF_force[:, i]
                    CoP[:, i] = CoP_i
                else:
                    CoP[:, i] = np.array([0, 0, 0])
            
            platforms_data.append({
                'force': PF_force,
                'moment0': PF_moment0,
                'cop': CoP
            })
        
        # Création du DataFrame complet (3000 Hz)
        for i in range(platforms_data[0]['force'].shape[1]):
            row = {}
            
            for num_PF in range(num_platforms):
                pf_data = platforms_data[num_PF]
                
                row[f"Fx{num_PF+1}"] = pf_data['force'][0, i]
                row[f"Fy{num_PF+1}"] = pf_data['force'][1, i]
                row[f"Fz{num_PF+1}"] = pf_data['force'][2, i]
                
                row[f"Mx{num_PF+1}_glob"] = pf_data['moment0'][0, i]
                row[f"My{num_PF+1}_glob"] = pf_data['moment0'][1, i]
                row[f"Mz{num_PF+1}_glob"] = pf_data['moment0'][2, i]
                
                row[f"CoP{num_PF+1}_x"] = pf_data['cop'][0, i]
                row[f"CoP{num_PF+1}_y"] = pf_data['cop'][1, i]
                row[f"CoP{num_PF+1}_z"] = pf_data['cop'][2, i]
            
            kinetics_data.append(row)
        
        df_kinetics_full = pd.DataFrame(kinetics_data)
        
        # =====================================================================
        # 3. DOWNSAMPLING ET SYNCHRONISATION
        # =====================================================================
        idx = np.arange(0, n_frames_markers * DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR)
        idx = idx[idx < len(df_kinetics_full)]
        
        df_kinetics_sync = df_kinetics_full.iloc[idx].reset_index(drop=True)
        df_kinetics_sync.insert(0, "frame", np.arange(len(df_kinetics_sync)))
        
        # =====================================================================
        # SAUVEGARDE
        # =====================================================================
        df_markers.to_csv(os.path.join(output_dir, "markers.csv"), index=False)
        df_kinetics_sync.to_csv(os.path.join(output_dir, "kinetics.csv"), index=False)
        
        print(f"✓ Fichiers sauvegardés dans : {output_dir}")
        print(f"  - Markers : {df_markers.shape[0]} frames")
        print(f"  - Kinetics : {df_kinetics_sync.shape[0]} frames (synchronisées)")
        
        # Vérification
        if len(df_kinetics_sync) > 0:
            mid = len(df_kinetics_sync) // 2
            print(f"\n  Check Frame {mid}:")
            for num_PF in range(num_platforms):
                cop_vals = df_kinetics_sync.loc[mid, [f'CoP{num_PF+1}_x', f'CoP{num_PF+1}_y', f'CoP{num_PF+1}_z']].values
                print(f"    CoP{num_PF+1} (x,y,z): {cop_vals}")
        
        return True
        
    except Exception as e:
        print(f"✗ ERREUR lors du traitement de {os.path.basename(file_path)}")
        print(f"  {type(e).__name__}: {e}")
        return False

# =============================================================================
# TRAITEMENT DE TOUS LES FICHIERS C3D
# =============================================================================
# Recherche de tous les fichiers .c3d dans le dossier
c3d_files = list(Path(input_dir).glob("*.c3d"))

if not c3d_files:
    print(f"Aucun fichier C3D trouvé dans : {input_dir}")
else:
    print(f"\n{'#'*80}")
    print(f"# {len(c3d_files)} fichiers C3D trouvés")
    print(f"{'#'*80}")
    
    success_count = 0
    fail_count = 0
    
    for c3d_file in sorted(c3d_files):
        # Extraction du nom de base (ex: "subject02_static2")
        base_name = c3d_file.stem
        
        # Détermination du sujet et du trial
        # Exemple: subject02_static2 -> subject02/static2
        parts = base_name.split('_', 1)
        if len(parts) == 2:
            subject, trial = parts
            output_dir = os.path.join(base_output_dir, subject, trial)
        else:
            # Si pas de '_', utiliser directement le nom
            output_dir = os.path.join(base_output_dir, base_name)
        
        # Traitement du fichier
        if process_c3d_file(str(c3d_file), output_dir):
            success_count += 1
        else:
            fail_count += 1
    
    # Résumé final
    print(f"\n{'#'*80}")
    print(f"# TRAITEMENT TERMINÉ")
    print(f"{'#'*80}")
    print(f"Succès : {success_count}/{len(c3d_files)}")
    if fail_count > 0:
        print(f"Échecs : {fail_count}/{len(c3d_files)}")
    print(f"\nFichiers sauvegardés dans : {base_output_dir}")
