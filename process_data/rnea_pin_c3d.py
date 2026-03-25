import os
import pandas as pd
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from utils.model_utils import build_human_model
from utils.linear_algebra_utils import lowpass_filter
import ezc3d

# =====================================================================
# CONFIG
# =====================================================================
BASE_DATA_DIR = "DATA/Anais"
URDF_DIR      = "DATA/urdf_scaled"
C3D_BASE_DIR  = "/home/kchalabi/Documents/THESE/datasets_kinetics/Anais"
MESHES_PATH   = "motif/model/human_urdf/meshes"
OUTPUT_DIR    = "FIGURES"

TASK_KEYWORDS = ["static", "cmjs", "dyna", "lufe", "luyo", "walk", "bend"]
fps = 300
dt  = 1.0 / fps

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# HELPERS
# =====================================================================
def task_is_valid(task_name):
    t = task_name.lower()
    return any(kw in t for kw in TASK_KEYWORDS)

def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=0))

def compute_mae(a, b):
    return np.mean(np.abs(a - b), axis=0)

def process_subject_task(subject, task):
    print(f"\n{'='*60}")
    print(f"  {subject}  |  {task}")
    print(f"{'='*60}")

    # --- Paths ---
    path_to_c3d  = os.path.join(C3D_BASE_DIR,  f"{subject}_{task}.c3d")
    path_joint   = os.path.join(BASE_DATA_DIR,  subject, task, "joints_whole_body.csv")
    path_grf     = os.path.join(BASE_DATA_DIR,  subject, task, "kinetics.csv")
    urdf_path    = os.path.join(URDF_DIR,        f"{subject}_scaled.urdf")

    # vérification existence des fichiers nécessaires
    for p in [path_to_c3d, path_joint, path_grf, urdf_path]:
        if not os.path.exists(p):
            print(f"  [SKIP] Fichier manquant : {p}")
            return

    c3d  = ezc3d.c3d(path_to_c3d, extract_forceplat_data=True)
    pf1  = c3d["data"]["platform"][0]
    pf2  = c3d["data"]["platform"][1]
    p_pf_center1 = np.mean(pf1['corners'], axis=1) / 1000
    p_pf_center2 = np.mean(pf2['corners'], axis=1) / 1000

    # --- GRF ---
    fx1,fy1,fz1 = 'Fx1','Fy1','Fz1'
    mx1,my1,mz1 = 'Mx1_glob','My1_glob','Mz1_glob'
    fx2,fy2,fz2 = 'Fx2','Fy2','Fz2'
    mx2,my2,mz2 = 'Mx2_glob','My2_glob','Mz2_glob'

    grf_df_raw      = pd.read_csv(path_grf)
    cols_to_filter  = [fx1,fy1,fz1,mx1,my1,mz1,fx2,fy2,fz2,mx2,my2,mz2]
    grf_df          = grf_df_raw.copy()
    grf_df[cols_to_filter] = lowpass_filter(
        grf_df_raw[cols_to_filter].to_numpy(), cutoff=2, fs=fps)

    off_fx1 = grf_df[fx1].mean();  off_fy1 = grf_df[fy1].mean()
    off_fx2 = grf_df[fx2].mean();  off_fy2 = grf_df[fy2].mean()

    # --- Joints ---
    q_ref_df   = pd.read_csv(path_joint).iloc[:, 1:]
    joint_names = list(q_ref_df.columns)
    q_ref_raw  = q_ref_df.to_numpy(dtype=float)
    q_ref      = lowpass_filter(q_ref_raw, cutoff=2, fs=fps)

    for i in range(len(q_ref)):
        q_q = q_ref[i, 3:7]
        q_ref[i, 3:7] = q_q / np.linalg.norm(q_q)

    # --- human model ---
    model_h, _, _, _ = build_human_model(urdf_path, MESHES_PATH)
    data_h  = model_h.createData()
    n_samples = len(q_ref)
    nv        = model_h.nv

    print(f"  Masse: {pin.computeTotalMass(model_h):.2f} kg | nv={nv} | n={n_samples}")

    # --- velocity and acc ---
    v_ref = np.zeros((n_samples, nv))
    a_ref = np.zeros((n_samples, nv))
    for i in range(n_samples - 1):
        v_ref[i] = pin.difference(model_h, q_ref[i], q_ref[i+1]) / dt
    for i in range(n_samples - 1):
        a_ref[i] = (v_ref[i+1] - v_ref[i]) / dt

    # --- RNEA + PF ---
    rnea_forces_world  = np.zeros((n_samples, 3))
    rnea_moments_world = np.zeros((n_samples, 3))
    pf_forces_clean    = np.zeros((n_samples, 3))
    pf_moments_clean   = np.zeros((n_samples, 3))

    for i in range(n_samples):
        tau       = pin.rnea(model_h, data_h, q_ref[i], v_ref[i], a_ref[i])
        rot_base  = pin.Quaternion(q_ref[i, 3:7]).matrix()
        pos_bassin = q_ref[i, 0:3]

        rnea_forces_world[i]  = rot_base @ tau[0:3]
        rnea_moments_world[i] = (rot_base @ tau[3:6]) + \
                                 np.cross(pos_bassin, rnea_forces_world[i])

        F1 = grf_df.loc[i, [fx1,fy1,fz1]].values
        M1 = grf_df.loc[i, [mx1,my1,mz1]].values / 1000.0
        F2 = grf_df.loc[i, [fx2,fy2,fz2]].values
        M2 = grf_df.loc[i, [mx2,my2,mz2]].values / 1000.0

        F1c = F1.copy(); F1c[0] -= off_fx1; F1c[1] -= off_fy1
        M1c = M1 + np.cross(p_pf_center1, np.array([off_fx1, off_fy1, 0]))

        F2c = F2.copy(); F2c[0] -= off_fx2; F2c[1] -= off_fy2
        M2c = M2 + np.cross(p_pf_center2, np.array([off_fx2, off_fy2, 0]))

        pf_forces_clean[i]  = F1c + F2c
        pf_moments_clean[i] = M1c + M2c

    # --- metrics ---
    rmse_f = compute_rmse(rnea_forces_world,  pf_forces_clean)
    rmse_m = compute_rmse(rnea_moments_world, pf_moments_clean)
    mae_f  = compute_mae(rnea_forces_world,   pf_forces_clean)
    mae_m  = compute_mae(rnea_moments_world,  pf_moments_clean)

    print(f"  MAE  Forces (N) : X={mae_f[0]:.2f} Y={mae_f[1]:.2f} Z={mae_f[2]:.2f}")
    print(f"  MAE  Moments(Nm): X={mae_m[0]:.2f} Y={mae_m[1]:.2f} Z={mae_m[2]:.2f}")
    print(f"  RMSE Forces (N) : X={rmse_f[0]:.2f} Y={rmse_f[1]:.2f} Z={rmse_f[2]:.2f}")
    print(f"  RMSE Moments(Nm): X={rmse_m[0]:.2f} Y={rmse_m[1]:.2f} Z={rmse_m[2]:.2f}")

    time = np.linspace(0, n_samples * dt, n_samples)

    # =====================================================================
    # plots
    # =====================================================================
    fig1, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    titles_f = ['Fx (Antéro-post)', 'Fy (Médiolat)', 'Fz (Vertical)']
    titles_m = ['Mx (Sagittal)',     'My (Frontal)',  'Mz (Vertical)']

    for j in range(3):
        axs[0,j].plot(time, pf_forces_clean[:,j],    color='blue',   alpha=0.6, label='PF')
        axs[0,j].plot(time, rnea_forces_world[:,j],  color='red',              label='RNEA')
        axs[0,j].set_title(f"{titles_f[j]}\nRMSE={rmse_f[j]:.2f} N  |  MAE={mae_f[j]:.2f} N")
        axs[0,j].set_ylabel("Force (N)"); axs[0,j].legend(); axs[0,j].grid(alpha=0.3)

        axs[1,j].plot(time, pf_moments_clean[:,j],   color='blue',  alpha=0.6, label='PF')
        axs[1,j].plot(time, rnea_moments_world[:,j], color='red',            label='RNEA')
        axs[1,j].set_title(f"{titles_m[j]}\nRMSE={rmse_m[j]:.2f} Nm  |  MAE={mae_m[j]:.2f} Nm")
        axs[1,j].set_ylabel("Moment (Nm)"); axs[1,j].set_xlabel("Temps (s)")
        axs[1,j].legend(); axs[1,j].grid(alpha=0.3)

    fig1.suptitle(f"{subject} — {task}", fontsize=16)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    out1 = os.path.join(OUTPUT_DIR, f"{subject}_{task}_forces_moments.png")
    # fig1.savefig(out1, dpi=150, bbox_inches='tight')
    # plt.close(fig1)
    # print(f"  Saved: {out1}")
    plt.show()


    n_dofs      = 12
    dof_raw     = np.degrees(q_ref_raw[:, 7:7+n_dofs])
    dof_filt    = np.degrees(q_ref[:,    7:7+n_dofs])
    dof_names   = joint_names[7:7+n_dofs]

    n_cols_fig  = 4
    n_rows_fig  = int(np.ceil(n_dofs / n_cols_fig))

    fig2, axs2 = plt.subplots(n_rows_fig, n_cols_fig,
                               figsize=(18, 4*n_rows_fig), sharex=True)
    axs2 = axs2.flatten()

    for k in range(n_dofs):
        axs2[k].plot(time, dof_raw[:,k],  color='red',label='raw')
        axs2[k].plot(time, dof_filt[:,k], color='black',label='filtered')
        axs2[k].set_title(dof_names[k], fontsize=10)
        axs2[k].set_ylabel("Angle rad)")
        axs2[k].grid(alpha=0.3)
        if k >= (n_rows_fig - 1) * n_cols_fig:
            axs2[k].set_xlabel("Temps (s)")

    for k in range(n_dofs, len(axs2)):
        axs2[k].set_visible(False)

    axs2[0].legend(loc='upper right', fontsize=8)
    fig2.suptitle(f"{subject} — {task} ", fontsize=16)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    out2 = os.path.join(OUTPUT_DIR, f"{subject}_{task}_angles.png")
    # fig2.savefig(out2, dpi=150, bbox_inches='tight')
    # plt.close(fig2)
    # print(f"  Saved: {out2}")
    plt.show()



subjects = sorted([
    d for d in os.listdir(BASE_DATA_DIR)
    if os.path.isdir(os.path.join(BASE_DATA_DIR, d))
])

print(f"Sujets trouvés : {subjects}")

for subject in subjects:
    subject_dir = os.path.join(BASE_DATA_DIR, subject)
    tasks = sorted([
        d for d in os.listdir(subject_dir)
        if os.path.isdir(os.path.join(subject_dir, d)) and task_is_valid(d)
    ])
    print(f"\n{subject} — tasks : {tasks}")
    for task in tasks:
        try:
            process_subject_task(subject, task)
        except Exception as e:
            print(f"  [ERROR] {subject}/{task} : {e}")