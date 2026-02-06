####################################################draft
import ezc3d
import numpy as np
import pandas as pd
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
file_path = "/home/kchalabi/Documents/THESE/datasets_kinetics/Anais/subject02_static2.c3d"
output_dir = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais/subject02/static2"
os.makedirs(output_dir, exist_ok=True)

DOWNSAMPLE_FACTOR = 10  # Rapport 3000 Hz / 300 Hz
EPS = 1e-6

# =============================================================================
# CHARGEMENT DU C3D
# =============================================================================
c = ezc3d.c3d(file_path)

marker_names = c['parameters']['POINT']['LABELS']['value']
analog_names = c['parameters']['ANALOG']['LABELS']['value']
corners = c['parameters']['FORCE_PLATFORM']['CORNERS']['value']
corners_fp1 = corners[:, :, 0]
corners_fp2 = corners[:, :, 1]

analogs = np.squeeze(c['data']['analogs'])

def get_rotation_matrix():
    """
    Retourne la matrice de passage du premier repère au deuxième
    selon les axes identifiés :
    x2 = -y1
    y2 = -x1
    z2 = -z1
    """
    R = np.array([
        [ 0, -1,  0],
        [-1,  0,  0],
        [ 0,  0, -1]
    ])
    
    return R

# --- Exemple d'utilisation ---
matrice = get_rotation_matrix()

print("Matrice de rotation / passage :")
print(matrice)

# Vérification du déterminant
det = np.linalg.det(matrice)
print(f"\nDéterminant : {det}")

# Calcul des centres des plateformes dans le repère Global (mm)
center_fp1 = np.mean(corners[:, :, 0], axis=1) 
center_fp2 = np.mean(corners[:, :, 1], axis=1)

# =============================================================================
# 1. MARQUEURS (300 Hz)
# =============================================================================
points = c['data']['points'] 
n_frames_markers = points.shape[2]

marker_dict = {"frame": np.arange(n_frames_markers)}
for i, name in enumerate(marker_names):
    marker_dict[f"{name}_x"] = points[0, i, :]
    marker_dict[f"{name}_y"] = points[1, i, :]
    marker_dict[f"{name}_z"] = points[2, i, :]

df_markers = pd.DataFrame(marker_dict)

# =============================================================================
# 2. FORCES, MOMENTS ET COP (Boucle frame par frame)
# =============================================================================
analogs = np.squeeze(c['data']['analogs'])

def get_channel(name):
    return analogs[analog_names.index(name), :]

analog_names = c['parameters']['ANALOG']['LABELS']['value']
print("Noms des canaux analogiques :", analog_names)

# Extraction des signaux bruts 3000 Hz
F1_raw = np.array([get_channel("Fx1"), get_channel("Fy1"), get_channel("Fz1")])
M1_raw = np.array([get_channel("Mx1"), get_channel("My1"), get_channel("Mz1")])
F2_raw = np.array([get_channel("Fx2"), get_channel("Fy2"), get_channel("Fz2")])
M2_raw = np.array([get_channel("Mx2"), get_channel("My2"), get_channel("Mz2")])

n_frames_analog = F1_raw.shape[1]

kinetics_data = []

for i in range(n_frames_analog):
    # --- Plateforme 1 ---
    f1 = F1_raw[:, i]

    m1_loc = M1_raw[:, i]

    # Calcul du moment global (PF_moment0)
    m1_glob = matrice@m1_loc + np.cross(center_fp1, f1)
    # Calcul CoP Global
    cop1 = np.cross(f1, m1_glob) / (np.linalg.norm(f1)**2 )
    cop1 -= (cop1[2] / f1[2]) * f1 #projeter sur le plan z=0 

    
    # --- Plateforme 2 ---
    f2 = F2_raw[:, i]
    # f2 = matrice@f2
    m2_loc = M2_raw[:, i]
    # Calcul du moment global (PF_moment0)
    m2_glob =  matrice@m2_loc + np.cross(center_fp2, f2)
    # Calcul CoP Global
    cop2 = np.cross(f2, m2_glob) / (np.linalg.norm(f2)**2 ) 
    cop2 -= (cop2[2] / f2[2] ) * f2

    # Stockage de toutes les composantes
    kinetics_data.append({
        "Fx1": f1[0], "Fy1": f1[1], "Fz1": f1[2],
        "Mx1_glob": m1_glob[0], "My1_glob": m1_glob[1], "Mz1_glob": m1_glob[2],
        "CoP1_x": cop1[0], "CoP1_y": cop1[1], "CoP1_z": cop1[2],
        
        "Fx2": f2[0], "Fy2": f2[1], "Fz2": f2[2],
        "Mx2_glob": m2_glob[0], "My2_glob": m2_glob[1], "Mz2_glob": m2_glob[2],
        "CoP2_x": cop2[0], "CoP2_y": cop2[1], "CoP2_z": cop2[2]
    })

# Création du DataFrame complet (3000 Hz)
df_kinetics_full = pd.DataFrame(kinetics_data)

# =============================================================================
# 3. DOWNSAMPLING ET SYNCHRONISATION
# =============================================================================
# Sélection des indices (0, 10, 20...) pour correspondre aux frames markers
idx = np.arange(0, n_frames_markers * DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR)
idx = idx[idx < n_frames_analog] # Sécurité limite de fichier

df_kinetics_sync = df_kinetics_full.iloc[idx].reset_index(drop=True)
df_kinetics_sync.insert(0, "frame", np.arange(len(df_kinetics_sync)))

# =============================================================================
# SAUVEGARDE
# =============================================================================
base_name = os.path.basename(file_path).replace(".c3d", "")
df_markers.to_csv(os.path.join(output_dir, f"markers.csv"), index=False)
df_kinetics_sync.to_csv(os.path.join(output_dir, f"kinetics.csv"), index=False)

print(f"--- Fichiers sauvegardés pour {base_name} ---")
print(f"Markers : {df_markers.shape[0]} frames")
print(f"Kinetics : {df_kinetics_sync.shape[0]} frames (synchronisées)")

# Petit check console
mid = len(df_kinetics_sync) // 2
print(f"\nCheck Frame {mid}:")
print(f"CoP1 (x,y,z): {df_kinetics_sync.loc[mid, ['CoP1_x', 'CoP1_y', 'CoP1_z']].values}")