#this code allows to compare between inverse dynamic (using pin.rnea) vs grfm data (from pf)
import pandas as pd
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt 
from utils.model_utils import build_human_model

def lowpass_filter(data, cutoff=7, fs=300, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

subject = "subject13"
task = "cmjs"
fps = 300 
dt = 1.0 / fps

path_joint = f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais/{subject}/{task}/joints.csv"
urdf_path = f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/urdf_scaled/{subject}_scaled.urdf"
path_grf = f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais/{subject}/{task}/kinetics.csv"
urdf_meshes_path = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/motif/model/human_urdf/meshes"

grf_df_raw = pd.read_csv(path_grf)
cols_to_filter = [
    'Fx1', 'Fy1', 'Fz1', 'Mx1_glob', 'My1_glob', 'Mz1_glob',
    'Fx2', 'Fy2', 'Fz2', 'Mx2_glob', 'My2_glob', 'Mz2_glob'
]
grf_data_filtered = lowpass_filter(grf_df_raw[cols_to_filter].to_numpy(), cutoff=7, fs=fps)
grf_df = grf_df_raw.copy()
grf_df[cols_to_filter] = grf_data_filtered

q_ref_df = pd.read_csv(path_joint).iloc[:, 1:]
q_ref_raw = q_ref_df.to_numpy(dtype=float)
q_ref = lowpass_filter(q_ref_raw, cutoff=7, fs=fps)

model_h, _, _, _ = build_human_model(urdf_path, urdf_meshes_path)

# Renormalisation des quaternions
for i in range(len(q_ref)):
    q_quat = q_ref[i, 3:7]
    q_ref[i, 3:7] = q_quat / np.linalg.norm(q_quat)

# On calcule le poids réel mesuré pour ajuster l'URDF (supprime l'offset Fz)
pf_total_z_init = grf_df['Fz1'][0] + grf_df['Fz2'][0]
masse_reelle = pf_total_z_init / 9.81
masse_urdf = pin.computeTotalMass(model_h)
ratio_masse = masse_reelle / masse_urdf

for inertia in model_h.inertias:
    inertia.mass *= ratio_masse
    inertia.inertia *= ratio_masse

data_h = model_h.createData()
n_samples = len(q_ref)
nv = model_h.nv

print(f"--- ANALYSE MODÈLE ---")
# print(f"Masse URDF initiale: {masse_urdf:.2f} kg")
print(f"Masse ajustée (PF): {pin.computeTotalMass(model_h):.2f} kg")
print(f"Degrés de liberté: {nv}")

v_ref = np.zeros((n_samples, nv))
a_ref = np.zeros((n_samples, nv))
for i in range(n_samples - 1):
    v_ref[i, :] = pin.difference(model_h, q_ref[i, :], q_ref[i+1, :]) / dt
for i in range(n_samples - 1):
    a_ref[i, :] = (v_ref[i+1, :] - v_ref[i, :]) / dt

rnea_forces_world = np.zeros((n_samples, 3))
rnea_moments_world = np.zeros((n_samples, 3))
pf_moments_au_bassin = np.zeros((n_samples, 3))

print("Exécution de la RNEA...")
for i in range(n_samples):
    # RNEA
    tau = pin.rnea(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])
    
    # Repère Pelvis (depuis q_ref)
    quat = pin.Quaternion(q_ref[i, 3:7])
    rot_base = quat.matrix()
    pos_bassin = q_ref[i, 0:3]

    # Projection Force/Moment RNEA dans World
    rnea_forces_world[i, :] = rot_base @ tau[0:3]
    rnea_moments_world[i, :] = rot_base @ tau[3:6]

    # Données Plateformes (Conversion mm -> m)
    F1 = grf_df.loc[i, ['Fx1', 'Fy1', 'Fz1']].values
    M1 = grf_df.loc[i, ['Mx1_glob', 'My1_glob', 'Mz1_glob']].values / 1000.0 #moment calculé a lorigin du labo

    F2 = grf_df.loc[i, ['Fx2', 'Fy2', 'Fz2']].values
    M2= grf_df.loc[i, ['Mx2_glob', 'My2_glob', 'Mz2_glob']].values / 1000.0

    # F1 = grf_df.loc[i, ['FX1', 'FY1', 'FZ1']].values
    # CoP1 = grf_df.loc[i, ['X1', 'Y1', 'Z1']].values / 1000.0
    # M1 = grf_df.loc[i, ['MX1', 'MY1', 'MZ1']].values / 1000.0 #moment calculé a lorigin du labo

    # F2 = grf_df.loc[i, ['FX2', 'FY2', 'FZ2']].values
    # CoP2 = grf_df.loc[i, ['X2', 'Y2', 'Z2']].values / 1000.0
    # M2= grf_df.loc[i, ['MX2', 'MY2', 'MZ2']].values / 1000.0

    
    # Transfert au bassin : M_total = M_pur + (CoP - Bassin) x F
    M1_transfer = M1 - np.cross((pos_bassin), F1)
    M2_transfer = M2 - np.cross((pos_bassin), F2)
    pf_moments_au_bassin[i, :] = M1_transfer + M2_transfer




pf_forces_total = np.array([grf_df['Fx1']+grf_df['Fx2'], 
                            grf_df['Fy1']+grf_df['Fy2'], 
                            grf_df['Fz1']+grf_df['Fz2']]).T

# pf_forces_total = np.array([grf_df['FX1']+grf_df['FX2'], 
#                             grf_df['FY1']+grf_df['FY2'], 
#                             grf_df['FZ1']+grf_df['FZ2']]).T

err_f = np.mean(np.abs(rnea_forces_world - pf_forces_total[:n_samples]), axis=0)
err_m = np.mean(np.abs(rnea_moments_world - pf_moments_au_bassin), axis=0)

print(f"\n--- BILAN DES ERREURS MOYENNES (MAE) ---")
print(f"FORCES (N)  -> X: {err_f[0]:.2f}, Y: {err_f[1]:.2f}, Z: {err_f[2]:.2f}")
print(f"MOMENTS(Nm) -> X: {err_m[0]:.2f}, Y: {err_m[1]:.2f}, Z: {err_m[2]:.2f}")

fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
time = np.linspace(0, n_samples * dt, n_samples)

titles_f = ['Fx (Antéro-post)', 'Fy (Médiolat)', 'Fz (Vertical)']
titles_m = ['Mx (Sagittal)', 'My (Frontal)', 'Mz (Vertical)']

for j in range(3):
    axs[0, j].plot(time, pf_forces_total[:n_samples, j], label="PF", color='blue', alpha=0.6)
    axs[0, j].plot(time, rnea_forces_world[:, j], label="RNEA", color='red', linestyle='--')
    axs[0, j].set_title(titles_f[j])
    axs[0, j].set_ylabel("Force (N)")
    axs[0, j].legend()
    axs[0, j].grid(alpha=0.3)

    axs[1, j].plot(time, pf_moments_au_bassin[:, j], label="PF (au pelvis)", color='green', alpha=0.6)
    axs[1, j].plot(time, rnea_moments_world[:, j], label="RNEA", color='orange', linestyle='--')
    axs[1, j].set_title(titles_m[j])
    axs[1, j].set_ylabel("Moment (Nm)")
    axs[1, j].legend()
    axs[1, j].grid(alpha=0.3)

plt.xlabel("Temps (s)")
plt.suptitle(f"{subject} {task}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()