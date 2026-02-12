import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import time
import meshcat
from utils.utils import read_mks_data
from utils.viz_utils import add_sphere, place,set_tf
from utils.model_utils import *
import meshcat_shapes
import ezc3d
import meshcat.geometry as g

trial = "subject03"
task = "static2"
# === Paths ===
c3d_path = f"/home/kchalabi/Documents/THESE/datasets_kinetics/Anais/{trial}_{task}.c3d" # os.path.join(input_dir, file)
c3d = ezc3d.c3d(c3d_path)
corners_all = c3d['parameters']['FORCE_PLATFORM']['CORNERS']['value']
corners_fp1 = corners_all[:, :, 0]
corners_fp2 = corners_all[:, :, 1]



# /home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais/subject01/beam_markers.csv
mks_csv = f"DATA/Anais/{trial}/{task}/markers.csv"
cop_csv = f"DATA/Anais/{trial}/{task}/kinetics_filtered.csv"  # <-- X1 Y1 Z1 X2 Y2 Z2

# === Units ===
# Ton code mks utilise converter=1000.0 => mks en mm -> sortie en m.
# On applique la même conversion aux COP si le fichier force est en mm.
CONVERTER = 1000.0

# === Load data ===
df_mks = pd.read_csv(mks_csv)
mks_dict, start_sample_dict = read_mks_data(df_mks, start_sample=0, converter=CONVERTER)
mks_names = start_sample_dict.keys()

# === Load COP ===
df_cop = pd.read_csv(cop_csv)

# Sélection & conversion (mm -> m si nécessaire)
def to_m(vals):
    arr = vals.values.astype(float)
    return arr / CONVERTER

# Tente de trouver les colonnes (permissif si casse/espace)
def find_col(df, name):
    cols_lower = {c.lower(): c for c in df.columns}
    key = name.lower()
    if key not in cols_lower:
        raise KeyError(f"Column '{name}' not found in {df.columns.tolist()}")
    return cols_lower[key]

X1 = to_m(df_cop[find_col(df_cop, "CoP1_x")])
Y1 = to_m(df_cop[find_col(df_cop, "CoP1_y")])
Z1 = to_m(df_cop[find_col(df_cop, "CoP1_z")])
X2 = to_m(df_cop[find_col(df_cop, "CoP2_x")])
Y2 = to_m(df_cop[find_col(df_cop, "CoP2_y")])
Z2 = to_m(df_cop[find_col(df_cop, "CoP2_z")])

def to_val(vals):
    return vals.values.astype(float)

Fx1 = to_val(df_cop[find_col(df_cop, "Fx1")])
Fy1 = to_val(df_cop[find_col(df_cop, "Fy1")])
Fz1 = to_val(df_cop[find_col(df_cop, "Fz1")])

Fx2 = to_val(df_cop[find_col(df_cop, "Fx2")])
Fy2 = to_val(df_cop[find_col(df_cop, "Fy2")])
Fz2 = to_val(df_cop[find_col(df_cop, "Fz2")])

forces1 = np.stack([Fx1, Fy1, Fz1], axis=1)
forces2 = np.stack([Fx2, Fy2, Fz2], axis=1)

# Empile en (N,3)
cop1 = np.stack([X1, Y1, Z1], axis=1)
cop2 = np.stack([X2, Y2, Z2], axis=1)

# === Sync lengths ===
n_frames_mks = len(mks_dict)
n_frames_cop = min(len(cop1), len(cop2))
n_frames = min(n_frames_mks, n_frames_cop)

# === MeshCat ===
vis = meshcat.Visualizer().open()
# Markers
for name in mks_names:
    add_sphere(vis, f"world/{name}", radius=0.01, color= 0xff0000)

def safe_place(node_name, pos3):
    """Place si non-NaN; sinon ignore la frame pour ce point."""
    if np.any(np.isnan(pos3)):
        return
    p = pos3.copy()
    p[2] = p[2] + Z_EPS
    place(vis, node_name, p)
# COP nodes (couleurs distinctes)
add_sphere(vis, "world/COP_left",  radius=0.015, color=0x00aaFF)  # bleu clair
add_sphere(vis, "world/COP_right", radius=0.015, color=0xFF8800)  # orange

FORCE_SCALE = 0.001  # Ajuste selon la taille voulue (0.001 signifie 1000N = 1m)
F_THRESHOLD = 20     # Seuil en Newtons pour afficher la flèche

# Initialisation des segments (vides au départ)
vis["world/GRF_left"].set_object(g.LineSegments(
    g.PointsGeometry(np.zeros((3, 2))), 
    g.LineBasicMaterial(color=0xFF8800, linewidth=3)
))
vis["world/GRF_right"].set_object(g.LineSegments(
    g.PointsGeometry(np.zeros((3, 2))), 
    g.LineBasicMaterial(color=0x00aaFF, linewidth=3)
))



all_corners = [corners_fp1, corners_fp2]
# Boucle pour créer et placer toutes les sphères
for i_fp, fp in enumerate(all_corners, start=1):
    for i_corner in range(fp.shape[1]):
        # Conversion mm -> m
        x, y, z = fp[0, i_corner]/1000, fp[1, i_corner]/1000, fp[2,0]/1000
        name = f"corner{i_fp}{i_corner+1}"
        add_sphere(vis, name, radius=0.025, color=0xFFaaFF)
        place(vis, name, (x, y, z))

meshcat_shapes.frame(
        vis["R_world"],
        axis_length=0.4,
        axis_thickness=0.009,
        opacity=1,
        origin_radius=0.02,
    )


# Option: petit offset pour éviter Z=0 confondu avec le plan
Z_EPS = 0.005  # mets 0.002 pour surélever de 2 mm si tu veux



# === Animate ===
for i in range(n_frames):
    # Markers
    frame = mks_dict[i]
    for name in mks_names:
        pos = frame[name].reshape(3,)
        place(vis,name, pos)

    # COPs
    safe_place("COP_left",  cop1[i])
    safe_place("COP_right", cop2[i])

    p_start1 = cop1[i]
    f1 = forces1[i]
    if not np.any(np.isnan(p_start1)) and abs(f1[2]) > F_THRESHOLD:
        safe_place("COP_left", p_start1)
        # Calcul du point d'arrivée : départ + (vecteur force * scale)
        # Note : on utilise "-" car Fz est souvent négatif dans le C3D (Action)
        # Si la flèche va vers le bas, change le "-" en "+"
        p_end1 = p_start1 + (f1 * FORCE_SCALE) 
        
        verts1 = np.array([p_start1, p_end1]).T.astype(np.float32)
        vis["world/GRF_left"].set_object(g.LineSegments(
            g.PointsGeometry(verts1), g.LineBasicMaterial(color=0x00aaFF)
        ))
    else:
        place(vis, "COP_left", [0, 0, -10]) # On cache sous le sol
        vis["world/GRF_left"].set_object(g.LineSegments(g.PointsGeometry(np.zeros((3, 2)))))

    # --- Gestion COP et GRF 2 (Left dans ton code) ---
    p_start2 = cop2[i]
    f2 = forces2[i]
    if not np.any(np.isnan(p_start2)) and abs(f2[2]) > F_THRESHOLD:
        safe_place("COP_right", p_start2)
        p_end2 = p_start2 + (f2 * FORCE_SCALE)
        
        verts2 = np.array([p_start2, p_end2]).T.astype(np.float32)
        vis["world/GRF_right"].set_object(g.LineSegments(
            g.PointsGeometry(verts2), g.LineBasicMaterial(color=0xFF8800)
        ))
    else:
        place(vis, "COP_right", [0, 0, -10])
        vis["world/GRF_right"].set_object(g.LineSegments(g.PointsGeometry(np.zeros((3, 2)))))
        

    for i_fp, fp in enumerate(all_corners, start=1):
        for i_corner in range(fp.shape[1]):
            # Conversion mm -> m
            x, y, z = fp[0, i_corner]/1000, fp[1, i_corner]/1000, fp[2,0]/1000
            name = f"world/corner{i_fp}{i_corner+1}"
            add_sphere(vis, name, radius=0.025, color=0xFFaaFF)
            place(vis, name, (x, y, z))
        
    time.sleep(0.01)
