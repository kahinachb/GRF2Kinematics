import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.signal import butter, filtfilt

fs = 100.0       # Fréquence d'échantillonnage en Hz
cutoff = 10.0    # Fréquence de coupure en Hz
order = 4        # Ordre du filtre (4 est un standard courant)

# Fonction pour créer et appliquer le filtre
def apply_butterworth_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # filtfilt applique le filtre sur chaque colonne (axis=0) sans créer de déphasage
    # padlen est ajusté automatiquement, mais on vérifie que la donnée est assez longue
    if data.shape[0] > 27: # filtfilt a besoin de suffisamment de points
        y = filtfilt(b, a, data, axis=0)
        return y
    else:
        print("Attention : données trop courtes pour être filtrées.")
        return data
    
path_joint_pred = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/results_lstm_HUMpf_weight_seg/Thomas_squat_bilstm_prediction.csv"
path_joint = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/HUMANOIDS/Thomas/squat_joints.csv"

dofs = [
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",
]
dofs_h =  ["right_hip_Z", "right_hip_X", "right_hip_Y",
    "right_knee_Z", "right_ankle_Z", "right_ankle_X",
    "left_hip_Z",  "left_hip_X",  "left_hip_Y",
    "left_knee_Z", "left_ankle_Z", "left_ankle_X", ]

mapping = dict(zip(dofs_h, dofs))
new_dofs = [mapping[d] for d in dofs_h]

# Load data
q_ref_df = pd.read_csv(path_joint, usecols=dofs_h) #.iloc[1:, :]
q_ref_df = q_ref_df.rename(columns=mapping)

q_pred_df = pd.read_csv(path_joint_pred)[dofs]

q_ref_df[:] = apply_butterworth_filter(
    q_ref_df.values,
    cutoff=2,
    fs=fs,
    order=order
)

q_pred_df[:] = apply_butterworth_filter(
    q_pred_df.values,
    cutoff=2,
    fs=fs,
    order=order
)

# Match length
min_len = min(len(q_ref_df), len(q_pred_df))
q_ref_df = q_ref_df.iloc[:min_len]
q_pred_df = q_pred_df.iloc[:min_len]

# Subplots
n_dofs = len(dofs)
n_cols = 3
n_rows = math.ceil(n_dofs / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten()

# Plot
for i, dof in enumerate(dofs):
    ax = axes[i]
    
    ref = q_ref_df[dof].values
    pred = q_pred_df[dof].values
    
    # Metrics
    mse = np.mean((ref - pred) ** 2)
    rmse = np.sqrt(mse)
    print(mse)
    
    # Plot
    ax.plot(ref, label="Ref", color = 'black')
    ax.plot(pred, label="Pred", color= "red")
    
    ax.set_title(f"{dof}\nMSE={mse:.4f} | RMSE={rmse:.4f}", fontsize=8)
    ax.grid()
    

    if i == 0:
        ax.legend()

# Remove empty axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.show()