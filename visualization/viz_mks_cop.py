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
# === Paths ===
trial = "squat"
subject = "Kahina"

which = "HUMANOIDS"
mks_csv = f"DATA/{which}/{subject}/{trial}_markers.csv"
cop_csv = f"DATA/{which}/{subject}/{trial}_kinetics_global.csv"  
path_joint = f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/{which}/{subject}/{trial}_joints.csv"
q_ref_df = pd.read_csv(path_joint).iloc[:, 1:]

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

# X1 = to_m(df_cop[find_col(df_cop, "X1")])
# Y1 = to_m(df_cop[find_col(df_cop, "Y1")])
# Z1 = to_m(df_cop[find_col(df_cop, "Z1")])
# X2 = to_m(df_cop[find_col(df_cop, "X2")])
# Y2 = to_m(df_cop[find_col(df_cop, "Y2")])
# Z2 = to_m(df_cop[find_col(df_cop, "Z2")])

X1 = to_m(df_cop[find_col(df_cop, "COPx1_glob")])
Y1 = to_m(df_cop[find_col(df_cop, "COPy1_glob")])
X2 = to_m(df_cop[find_col(df_cop, "COPx2_glob")])
Y2 = to_m(df_cop[find_col(df_cop, "COPy2_glob")])
Z1 = np.zeros_like(X1)
Z2 = np.zeros_like(X2)

# X1 = to_m(df_cop[find_col(df_cop, "Cx1")])
# Y1 = to_m(df_cop[find_col(df_cop, "Cy1")])
# Z1 = to_m(df_cop[find_col(df_cop, "Cz1")])
# X2 = to_m(df_cop[find_col(df_cop, "Cx2")])
# Y2 = to_m(df_cop[find_col(df_cop, "Cy2")])
# Z2 = to_m(df_cop[find_col(df_cop, "Cz2")])

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
# add_sphere(vis, f"world/rkne_jcp", radius=0.01, color= 0x00ff00)
# add_sphere(vis, f"world/lkne_jcp", radius=0.01, color= 0x00ff00)
# add_sphere(vis, f"world/rank_jcp", radius=0.01, color= 0x00ff00)
# add_sphere(vis, f"world/lank_jcp", radius=0.01, color= 0x00ff00)
# add_sphere(vis, f"world/RHJC", radius=0.01, color= 0x00ff00)
# add_sphere(vis, f"world/LHJC", radius=0.01, color= 0x00ff00)
# add_sphere(vis, f"world/pelvis_jcp", radius=0.01, color= 0x00ff00)

# add_sphere(vis, f"world/rmkne", radius=0.01, color= 0x00ff00)
# add_sphere(vis, f"world/lmkne", radius=0.01, color= 0x00ff00)

# add_sphere(vis, f"world/rmank", radius=0.01, color= 0x00ff00)
# add_sphere(vis, f"world/lmank", radius=0.01, color= 0x00ff00)

text_name = f"text"
meshcat_shapes.textarea(vis[text_name],"text_name", font_size=32)
T_name = np.eye(4)
T_name[0, 3] += 0.15
T_name[1, 3] += 0.15
T_name[2, 3] += 0.15


# COP nodes (couleurs distinctes)
add_sphere(vis, "world/COP_right",  radius=0.015, color=0x00aaFF)  # bleu clair
add_sphere(vis, "world/COP_left", radius=0.015, color=0xFF8800)  # orange

add_sphere(vis, "world/pos_bassin_RNEA", radius=0.03, color=0xFFFF00) # Jaune

# Repère local pour vérifier l'orientation (Quaternion)
meshcat_shapes.frame(vis["world/pos_bassin_RNEA/frame"], axis_length=0.2)

meshcat_shapes.frame(
        vis["R_world"],
        axis_length=0.4,
        axis_thickness=0.009,
        opacity=1,
        origin_radius=0.02,
    )

# Option: petit offset pour éviter Z=0 confondu avec le plan
Z_EPS = 0.005  # mets 0.002 pour surélever de 2 mm si tu veux

def safe_place(node_name, pos3):
    """Place si non-NaN; sinon ignore la frame pour ce point."""
    if np.any(np.isnan(pos3)):
        return
    p = pos3.copy()
    p[2] = p[2] + Z_EPS
    place(vis, node_name, p)

# === Animate ===
for i in range(n_frames):

    q_current = q_ref_df.iloc[i].to_numpy()
    pos_bassin_rnea = q_current[0:3]
    quat_bassin = q_current[3:7] # qx, qy, qz, qw

    T_bassin = np.eye(4)
    T_bassin[:3, 3] = pos_bassin_rnea

    place(vis, "pos_bassin_RNEA", pos_bassin_rnea)
    set_tf(vis, "world/frame", T_bassin[:3, 3] )


    # Markers
    frame = mks_dict[i]

    for name in mks_names:
        pos = frame[name].reshape(3,)
        place(vis,name, pos)
  
    # COPs
    center_fp1 = np.array([0.300, 0.250, 0])
    center_fp2 = np.array([0.300, -0.340, 0])

    safe_place("COP_right",  cop1[i])
    safe_place("COP_left",cop2[i])

    # safe_place("COP_right",  center_fp1)
    # safe_place("COP_left",center_fp2)

    # pelvis_pose = get_pelvis_pose(frame)
    # pelvis = as_col(pelvis_pose[:3, 3])
    # place(vis,'pelvis_jcp',pelvis.T)

    

    ##################################################""right knee
#     RHJC=get_thighR_pose(frame, 0.05, gender='male')
#     RHJC = as_col(RHJC[:3, 3])
#     place(vis,'RHJC',RHJC.T)
#     RKNE_lat = frame['RKNE']
#     RTHI = frame ['RTHI']
#     side = 'right'
#     RKJC = knee_joint_center(
#         RHJC, RKNE_lat, RTHI, 0.05, side,
#         knee_rotation_offset_deg=0.0, MKNE_med=None, eps=1e-12
#     )

#     med_dir = RKJC - RKNE_lat
#     med_dir /= np.linalg.norm(med_dir)

#     knee_width = 0.10  
#     MKNE_est = medial_from_ajc(RKJC, RKNE_lat, knee_width)

#     place(vis, "rkne_jcp", RKJC)
#     place(vis, "rmkne", MKNE_est)

# ######################################################### left kneee
#     LHJC=get_thighL_pose(frame, 0.05, gender='male')
#     LHJC = as_col(LHJC[:3, 3])
#     place(vis,'LHJC',LHJC.T)

#     LKNE_lat = frame['LKNE']
#     LTHI = frame ['LTHI']
#     side = 'left'
#     LKJC = knee_joint_center(
#         LHJC, LKNE_lat, LTHI, 0.05, side,
#         knee_rotation_offset_deg=0.0, MKNE_med=None, eps=1e-12
#     )

#     MKNE_est = medial_from_ajc(LKJC, LKNE_lat, knee_width)
    
#     place(vis, "lkne_jcp", LKJC)
#     place(vis, "lmkne", MKNE_est)

# ########################################## right ankle
#     RANK_lat = frame['RANK']
#     RTIB =frame ['RTIB']
#     side = 'right'
#     RANKJC = ankle_joint_center(
#         RKJC, RANK_lat, RTIB, 0.04, side,
#         ankle_rotation_offset_deg=0.0, MANK_med=None, eps=1e-12
#     )

#     # Define knee width if known
#     ankle_width = 0.07  
#     MANK_est = medial_from_ajc(RANKJC, RANK_lat, ankle_width)

#     place(vis, "rank_jcp", RANKJC)
#     place(vis, "rmank", MANK_est)

# ####################################### left ankle
#     LANK_lat = frame['LANK']
#     LTIB =frame ['LTIB']
#     side = 'left'
#     LANKJC = ankle_joint_center(
#         LKJC, LANK_lat, LTIB, 0.04, side,
#         ankle_rotation_offset_deg=0.0, MANK_med=None, eps=1e-12
#     )

#     MANK_est = medial_from_ajc(LANKJC, LANK_lat, ankle_width)


#     place(vis, "lank_jcp", LANKJC)
#     place(vis, "lmank", MANK_est)

    
#     # timing (ajuste selon ta fréquence; ici ~100 Hz)
#     time.sleep(0.01)