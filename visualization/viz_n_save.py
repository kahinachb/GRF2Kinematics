import sys
import os
import time
import numpy as np
import pandas as pd
import meshcat
import meshcat_shapes

# === Import local utils ===
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.utils import read_mks_data
from utils.viz_utils import add_sphere, place, set_tf
from utils.model_utils import (
    get_pelvis_pose, get_thighR_pose, get_thighL_pose,
    knee_joint_center, ankle_joint_center, medial_from_ajc, as_col, create_virtual_foot_marker,create_virtual_MTOE
)

# === Parameters ===
TRIAL = "Trial111"
SUBJECT = "Christine"
CONVERTER = 1000.0  # mm -> m
Z_EPS = 0.005
FPS = 100  # for animation delay
KNEE_WIDTH = 0.10
ANKLE_WIDTH = 0.08
THIGH_LEN = 0.05 #(KNEE_WIDTH/2)
TIBIA_LEN = 0.04 #(AnkleWidth)/2.
FOOT_WIDTH = 0.05  #toe to 5eme toe

# === File paths ===
mks_csv = f"DATA/{SUBJECT}/{TRIAL}.csv"
cop_csv = f"DATA/{SUBJECT}/{TRIAL}_forces.csv"


# === Utils ===
def to_m(vals):
    """Convert from mm to meters."""
    return vals.values.astype(float) / CONVERTER


def find_col(df, name):
    """Find column case-insensitively."""
    cols_lower = {c.lower(): c for c in df.columns}
    key = name.lower()
    if key not in cols_lower:
        raise KeyError(f"Column '{name}' not found in {list(df.columns)}")
    return cols_lower[key]


def safe_place(vis, node_name, pos3, z_offset=Z_EPS):
    """Place node if valid."""
    if np.any(np.isnan(pos3)):
        return
    pos = pos3.copy()
    pos[2] += z_offset
    place(vis, node_name, pos)


# === Load Data ===
df_mks = pd.read_csv(mks_csv)
mks_dict, start_sample_dict = read_mks_data(df_mks, start_sample=0, converter=CONVERTER)
mks_names = list(start_sample_dict.keys())

# COP
df_cop = pd.read_csv(cop_csv)
cop1 = np.stack([to_m(df_cop[find_col(df_cop, f"{ax}1")]) for ax in ("X", "Y", "Z")], axis=1)
cop2 = np.stack([to_m(df_cop[find_col(df_cop, f"{ax}2")]) for ax in ("X", "Y", "Z")], axis=1)

n_frames = min(len(mks_dict), len(cop1), len(cop2))

# === Meshcat setup ===
vis = meshcat.Visualizer().open()

# Markers
for name in mks_names:
    add_sphere(vis, f"world/{name}", radius=0.01, color=0xFF0000)

# Joints
joint_names = [
    "rkne_jcp", "lkne_jcp", "rank_jcp", "lank_jcp",
    "RHJC", "LHJC", "pelvis_jcp", "rmkne", "lmkne", "rmank", "lmank","rMTOE","lMTOE"
]
for j in joint_names:
    add_sphere(vis, f"world/{j}", radius=0.01, color=0x00FF00)

# COPs
add_sphere(vis, "world/COP_left", radius=0.015, color=0x00AAFF)
add_sphere(vis, "world/COP_right", radius=0.015, color=0xFF8800)

# Reference frame
meshcat_shapes.frame(
    vis["R_world"], axis_length=0.4, axis_thickness=0.009,
    opacity=1, origin_radius=0.02
)




# === Joint Computations ===
def compute_joint_centers(frame, side):
    """Compute all joint centers (hip, knee, ankle) for given side."""
    if side == "right":
        get_thigh_pose = get_thighR_pose
        hip_label, knee_label, thigh_label, ankle_label, tib_label,toe_label, heel_label = (
            "RHJC", "RKNE", "RTHI", "RANK", "RTIB", "RTOE", "RHEE"
        )
    else:
        get_thigh_pose = get_thighL_pose
        hip_label, knee_label, thigh_label, ankle_label, tib_label,toe_label, heel_label = (
            "LHJC", "LKNE", "LTHI", "LANK", "LTIB", "LTOE", "LHEE"
        )

    # Hip
    HJC_pose = get_thigh_pose(frame, THIGH_LEN, gender="male")
    HJC = as_col(HJC_pose[:3, 3])

    # Knee
    K_lat = frame[knee_label]
    THI = frame[thigh_label]
    KJC = knee_joint_center(
        HJC, K_lat, THI, THIGH_LEN, side,
        knee_rotation_offset_deg=0.0, MKNE_med=None, eps=1e-12
    )
    MKNE = medial_from_ajc(KJC, K_lat, KNEE_WIDTH)

    # Ankle
    A_lat = frame[ankle_label]
    TIB = frame[tib_label]
    AJC = ankle_joint_center(
        KJC, A_lat, TIB, TIBIA_LEN, side,
        ankle_rotation_offset_deg=0.0, MANK_med=None, eps=1e-12
    )
    MANK = medial_from_ajc(AJC, A_lat, ANKLE_WIDTH)

    #heel, toe, ankle are (T,3) arrays in meters (e.g., from Vicon CSV)
    HEE = frame[heel_label]
    toe = frame[toe_label]

    MTOE = create_virtual_MTOE(HEE, toe, A_lat,MANK,side, foot_width=FOOT_WIDTH)


    return {
        "HJC": HJC, "KJC": KJC, "A_lat": A_lat,
        "MKNE": MKNE, "AJC": AJC, "MANK": MANK, "MTOE": MTOE
    }

RMKNE_buf = np.full((n_frames, 3), np.nan)
LMKNE_buf = np.full((n_frames, 3), np.nan)
RMANK_buf = np.full((n_frames, 3), np.nan)
LMANK_buf = np.full((n_frames, 3), np.nan)
RMTOE_buf = np.full((n_frames, 3), np.nan)
LMTOE_buf = np.full((n_frames, 3), np.nan)
# === Animation ===
for i in range(n_frames):
    frame = mks_dict[i]

    # Markers
    for name in mks_names:
        place(vis, name, frame[name].reshape(3,))

    # COPs
    safe_place(vis, "COP_left", cop1[i])
    safe_place(vis, "COP_right", cop2[i])

    # Pelvis
    pelvis_pose = get_pelvis_pose(frame)
    pelvis_pos = as_col(pelvis_pose[:3, 3])
    place(vis, "pelvis_jcp", pelvis_pos.T)

    # Joints
    for side in ("right", "left"):
        joints = compute_joint_centers(frame, side)
        prefix = "r" if side == "right" else "l"

        place(vis, f"{prefix}kne_jcp", joints["KJC"])
        place(vis, f"{prefix}mkne", joints["MKNE"])
        place(vis, f"{prefix}ank_jcp", joints["AJC"])
        place(vis, f"{prefix}mank", joints["MANK"])
        place(vis, f"{prefix}MTOE", joints["MTOE"])
        place(vis, f"{prefix.upper()}HJC",joints["HJC"].T)
    
        if side == "right":
                RMKNE_buf[i, :] = joints["MKNE"].reshape(3,)
                RMANK_buf[i, :] = joints["MANK"].reshape(3,)
                RMTOE_buf[i, :] = joints["MTOE"].reshape(3,)
        else:
            LMKNE_buf[i, :] = joints["MKNE"].reshape(3,)
            LMANK_buf[i, :] = joints["MANK"].reshape(3,)
            LMTOE_buf[i, :] = joints["MTOE"].reshape(3,)

    # Delay (simulate real-time)
    # time.sleep(1 / FPS)

import shutil

# Backup de sécurité
backup_path = mks_csv + ".bak"
try:
    shutil.copyfile(mks_csv, backup_path)
except Exception as e:
    print(f"[WARN] Backup non créé ({e}). Poursuite de l'écriture directe…")

# Reconvertion en mm pour rester cohérent avec le fichier MKS d'origine
def write_vec(df, base_name, vec_m):
    vec_mm = vec_m * CONVERTER
    df[f"{base_name}_X"] = vec_mm[:, 0]
    df[f"{base_name}_Y"] = vec_mm[:, 1]
    df[f"{base_name}_Z"] = vec_mm[:, 2]

# Allonge éventuellement le DataFrame si besoin (au cas où n_frames < len(df_mks) ou inversement)
if len(df_mks) < n_frames:
    # On pad avec NaN pour aligner
    extra = n_frames - len(df_mks)
    df_mks = pd.concat([df_mks, pd.DataFrame(index=range(extra))], ignore_index=True)
elif len(df_mks) > n_frames:
    # On tronque proprement
    df_mks = df_mks.iloc[:n_frames].copy()

# Écritures des nouvelles colonnes
write_vec(df_mks, "RMKNE", RMKNE_buf)
write_vec(df_mks, "LMKNE", LMKNE_buf)
write_vec(df_mks, "RMANK", RMANK_buf)
write_vec(df_mks, "LMANK", LMANK_buf)
write_vec(df_mks, "RMTOE", RMTOE_buf)
write_vec(df_mks, "LMTOE", LMTOE_buf)
# Sauvegarde finale (même fichier)
df_mks.to_csv(mks_csv, index=False)
print(f"[INFO] MKNE/MANK ajoutés au CSV : {mks_csv}\n       Backup : {backup_path}")