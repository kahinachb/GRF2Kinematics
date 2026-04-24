import os
import re
import pandas as pd
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from utils.model_utils import build_human_model
from utils.linear_algebra_utils import lowpass_filter

# =====================================================================
# CONFIG
# =====================================================================
BASE_DATA_DIR = "DATA/HUMANOIDS"
URDF_DIR      = "DATA/urdf_scaled/HUMANOIDS"
MESHES_PATH   = "motif/model/human_urdf/meshes"
OUTPUT_DIR    = "FIGURES"

fps = 100
dt  = 1.0 / fps

fx1, fy1, fz1 = 'Fx1_glob', 'Fy1_glob', 'Fz1_glob'
mx1,my1,mz1 =  'Mx1_glob', 'My1_glob', 'Mz1_glob'
fx2, fy2, fz2 = 'Fx2_glob', 'Fy2_glob', 'Fz2_glob'
mx2,my2,mz2 =  'Mx2_glob', 'My2_glob', 'Mz2_glob'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# HELPERS
# =====================================================================
def discover_trials(subject_dir):
    """
    all trial that contains 'squat' 
    """
    trial_names = set()

    for fname in os.listdir(subject_dir):
        if fname.lower().endswith("_joints.csv") and "squat" in fname.lower():
            
          
            trial_name = fname[:-11]  
            
            kinetics_file = os.path.join(subject_dir, f"{trial_name}_kinetics_global.csv")
            
            if os.path.exists(kinetics_file):
                trial_names.add(trial_name)

    return sorted(trial_names)

def compute_rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=0))

def compute_mae(a, b):
    return np.mean(np.abs(a - b), axis=0)


def process_subject_trial(subject, trial_num):
    trial_id = f"{trial_num}"
    print(f"\n{'='*60}")
    print(f"  {subject}  |  {trial_id}")
    print(f"{'='*60}")

    subject_dir = os.path.join(BASE_DATA_DIR, subject)
    path_joint  = os.path.join(subject_dir, f"{trial_id}_joints.csv")
    path_grf    = os.path.join(subject_dir, f"{trial_id}_kinetics_global.csv")

    base_subject = re.sub(r'\d+$', '', subject)

    if "attelle_poids" in trial_id:
        suffix = "_attelle_poids"
    elif "attelle" in trial_id:
        suffix = "_attelle"
    else:
        suffix = ""

    urdf_name = f"{base_subject}{suffix}.urdf"
    urdf_path = os.path.join(URDF_DIR, urdf_name)
    print(urdf_path)


    for p in [path_joint, path_grf, urdf_path]:
        if not os.path.exists(p):
            print(f"  [SKIP] : {p}")
            return

    p_pf_center1 = (0.3, 0.250, 0.048)
    p_pf_center2 = (0.3, -0.34, 0.048)


    # --- GRF ---
    cols_to_filter = [fx1,fy1,fz1,mx1,my1,mz1,fx2,fy2,fz2,mx2,my2,mz2]
    grf_df_raw     = pd.read_csv(path_grf)
    grf_df         = grf_df_raw.copy()
    grf_df[cols_to_filter] = lowpass_filter(
        grf_df_raw[cols_to_filter].to_numpy(), cutoff=2, fs=fps)

    off_fx1 = grf_df[fx1].mean();  off_fy1 = grf_df[fy1].mean()
    off_fx2 = grf_df[fx2].mean();  off_fy2 = grf_df[fy2].mean()

    # --- Joints ---
    q_ref_df    = pd.read_csv(path_joint).iloc[:, 1:]
    joint_names = list(q_ref_df.columns)
    q_ref_raw   = q_ref_df.to_numpy(dtype=float)
    q_ref       = lowpass_filter(q_ref_raw, cutoff=2, fs=fps)

    for i in range(len(q_ref)):
        q_q = q_ref[i, 3:7]
        q_ref[i, 3:7] = q_q / np.linalg.norm(q_q)

    model_h, _, _, _ = build_human_model(urdf_path, MESHES_PATH)
    data_h    = model_h.createData()
    n_samples = len(q_ref)
    nv        = model_h.nv

    print(f"  Weight: {pin.computeTotalMass(model_h):.2f} kg | nv={nv} | n={n_samples}")
    fz_mean = np.mean(grf_df[fz1].iloc[:].values + grf_df[fz2].iloc[:].values)

    g = abs(model_h.gravity.linear[2]) 
    target_mass = fz_mean / g
    print(f"weight from grf : {target_mass:.2f} kg")

    print(f"urdf weight : {pin.computeTotalMass(model_h):.2f} kg")
    print(f"dofs: {nv}")
    # print(f"model gravity : {model_h.gravity.linear}")

    current_mass = pin.computeTotalMass(model_h)
    ratio = target_mass / current_mass
    for i in range(len(model_h.inertias)):
        model_h.inertias[i].mass *= ratio
        model_h.inertias[i].inertia *= ratio
    data_h = model_h.createData()

    print(f"new urdf weight : {pin.computeTotalMass(model_h):.2f} kg")

    v_ref = np.zeros((n_samples, nv))
    a_ref = np.zeros((n_samples, nv))
    for i in range(n_samples - 1):
        v_ref[i] = pin.difference(model_h, q_ref[i], q_ref[i+1]) / dt
    for i in range(n_samples - 1):
        a_ref[i] = (v_ref[i+1] - v_ref[i]) / dt

    rnea_forces_world  = np.zeros((n_samples, 3))
    rnea_moments_world = np.zeros((n_samples, 3))
    pf_forces_clean    = np.zeros((n_samples, 3))
    pf_moments_clean   = np.zeros((n_samples, 3))

    pf1_forces_clean  = np.zeros((n_samples, 3))
    pf1_moments_clean = np.zeros((n_samples, 3))
    pf2_forces_clean  = np.zeros((n_samples, 3))
    pf2_moments_clean = np.zeros((n_samples, 3))

    pf1_cop_clean = np.zeros((n_samples, 3))
    pf2_cop_clean = np.zeros((n_samples, 3))

    pf1_cop_pelvis = np.zeros((n_samples, 3))
    pf2_cop_pelvis = np.zeros((n_samples, 3))

    for i in range(n_samples):
        tau        = pin.rnea(model_h, data_h, q_ref[i], v_ref[i], a_ref[i])
        rot_base   = pin.Quaternion(q_ref[i, 3:7]).matrix()
        pos_bassin = q_ref[i, 0:3]

        rnea_forces_world[i]  = rot_base @ tau[0:3]
        rnea_moments_world[i] = (rot_base @ tau[3:6]) + \
                                  np.cross(pos_bassin, rnea_forces_world[i])

        F1 = grf_df.loc[i, [fx1,fy1,fz1]].values
        F2 = grf_df.loc[i, [fx2,fy2,fz2]].values
        M1 = grf_df.loc[i, [mx1,my1,mz1]].values / 1000.0
        M2 = grf_df.loc[i, [mx2,my2,mz2]].values / 1000.0

        F1c = F1.copy(); F1c[0] -= off_fx1; F1c[1] -= off_fy1
        F2c = F2.copy(); F2c[0] -= off_fx2; F2c[1] -= off_fy2

        M1c = M1 +np.cross(p_pf_center1, np.array([off_fx1, off_fy1, 0]))
        M2c = M2 +np.cross(p_pf_center2, np.array([off_fx2, off_fy2, 0]))


        pf_forces_clean[i]  = F1c + F2c
        pf_moments_clean[i] = M1c + M2c

        pf1_forces_clean[i]  = F1c
        pf1_moments_clean[i] = M1c

        pf2_forces_clean[i]  = F2c
        pf2_moments_clean[i] = M2c

    mean_mrnea = np.mean(rnea_moments_world, axis=0)
    mean_mpf = np.mean(pf_moments_clean, axis=0)

    offset_Mx = mean_mrnea[0] - mean_mpf[0]
    offset_My = mean_mrnea[1] - mean_mpf[1]
    offset_Mz = -np.mean(pf_moments_clean[:, 2]) # Pour centrer Mz à 0
   
    for i in range(n_samples):
        #au prorata
        fz_total = pf1_forces_clean[i, 2] + pf2_forces_clean[i, 2]
        
        if fz_total > 10.0: 
            alpha1 = pf1_forces_clean[i, 2] / fz_total
            alpha2 = pf2_forces_clean[i, 2] / fz_total
        else:
            alpha1 = 0.5
            alpha2 = 0.5

        pf1_moments_clean[i, 0] += alpha1 * offset_Mx
        pf1_moments_clean[i, 1] += alpha1 * offset_My
        pf1_moments_clean[i, 2] +=alpha1 *offset_Mz

        pf2_moments_clean[i, 0] += alpha2 * offset_Mx
        pf2_moments_clean[i, 1] += alpha2 * offset_My
        pf2_moments_clean[i, 2] +=  alpha2 * offset_Mz

        pf_moments_clean[i] = pf1_moments_clean[i] + pf2_moments_clean[i]

        if pf1_forces_clean[i, 2] > 20.0:
            pf1_cop_clean[i] = np.array([-pf1_moments_clean[i, 1] / pf1_forces_clean[i, 2],
                                      pf1_moments_clean[i, 0] / pf1_forces_clean[i, 2], 
                                      0.0])
        if pf2_forces_clean[i, 2] > 20.0:
            pf2_cop_clean[i] = np.array([-pf2_moments_clean[i, 1] / pf2_forces_clean[i, 2],
                                      pf2_moments_clean[i, 0] / pf2_forces_clean[i, 2], 
                                      0.0])
    
    rnea_forces_pelvis  = np.zeros((n_samples, 3))
    rnea_moments_pelvis = np.zeros((n_samples, 3))
    pf_forces_pelvis    = np.zeros((n_samples, 3))
    pf_moments_pelvis   = np.zeros((n_samples, 3))

    pf1_forces_pelvis  = np.zeros((n_samples, 3))
    pf1_moments_pelvis = np.zeros((n_samples, 3))
    pf2_forces_pelvis  = np.zeros((n_samples, 3))
    pf2_moments_pelvis = np.zeros((n_samples, 3))

    for i in range(n_samples):
        rot_base   = pin.Quaternion(q_ref[i, 3:7]).matrix() # R_base
        pos_bassin = q_ref[i, 0:3]                         # 
        
        # F_pelvis = R^T * F_world
        rnea_forces_pelvis[i] = rot_base.T @ rnea_forces_world[i]
        
        #M_pelvis = R^T * (M_world - OP ^ F_world)
        m_transport_rnea = rnea_moments_world[i] - np.cross(pos_bassin, rnea_forces_world[i])
        rnea_moments_pelvis[i] = rot_base.T @ m_transport_rnea

        # pf from world to pelvis
        pf1_forces_pelvis[i] = rot_base.T @ pf1_forces_clean[i]
        pf2_forces_pelvis[i] = rot_base.T @ pf2_forces_clean[i]
        pf_forces_pelvis[i] = pf1_forces_pelvis[i] + pf2_forces_pelvis[i]
        

        m_world_to_bassin1 = pf1_moments_clean[i] - np.cross(pos_bassin, pf1_forces_clean[i])
        pf1_moments_pelvis[i] = rot_base.T @ m_world_to_bassin1
        m_world_to_bassin2 = pf2_moments_clean[i] - np.cross(pos_bassin, pf2_forces_clean[i])
        pf2_moments_pelvis[i] = rot_base.T @ m_world_to_bassin2

        pf_moments_pelvis[i] = pf1_moments_pelvis[i] + pf2_moments_pelvis[i]
        
        pf1_cop_pelvis[i] = rot_base.T @ (pf1_cop_clean[i] - pos_bassin)
        pf2_cop_pelvis[i] = rot_base.T @ (pf2_cop_clean[i] - pos_bassin)

    delta_pos_local = np.zeros((n_samples - 1, 3))
    delta_rot_local = np.zeros((n_samples - 1, 3)) # Vecteur de rotation (so3)

    print("Original shape :", q_ref.shape)

    # for i in range(n_samples-1):
    #     R_t = pin.Quaternion(q_ref[i, 3:7]).matrix()
    #     p_t = q_ref[i, 0:3]
        
    #     # --- Frame suivante (t+1) ---
    #     R_next = pin.Quaternion(q_ref[i+1, 3:7]).matrix()
    #     p_next = q_ref[i+1, 0:3]
        
    #     # 1. Translation locale : "De combien j'ai avancé/glissé par rapport à ma position actuelle ?"
    #     # On projette le déplacement global dans le repère de la frame actuelle
    #     delta_pos_local[i] = R_t.T @ (p_next - p_t)
        
    #     # 2. Rotation locale : "De combien j'ai tourné sur moi-même ?"
    #     # On calcule la rotation relative : R_rel = R_t^T * R_next
    #     R_rel = R_t.T @ R_next
    #     # On transforme cette matrice en vecteur de rotation (plus facile pour l'IA que 9 chiffres)
    #     delta_rot_local[i] = pin.log3(R_rel)

    q_ref = q_ref[:, :]

    

    # reconstruire nouveau q_ref
    # q_ref = np.concatenate([
    #     delta_pos_local,
    #     delta_rot_local,
    #     q_ref
    # ], axis=1)

    
    print("New shape      :", q_ref.shape)

    cut = -20
    
    pf_forces_clean = pf_forces_clean[:cut]
    rnea_forces_world = rnea_forces_world[:cut]
    pf_moments_clean = pf_moments_clean[:cut]
    rnea_moments_world = rnea_moments_world[:cut]

    pf_forces_pelvis = pf_forces_pelvis[:cut]
    rnea_forces_pelvis = rnea_forces_pelvis[:cut]
    pf_moments_pelvis = pf_moments_pelvis[:cut]
    rnea_moments_pelvis = rnea_moments_pelvis[:cut]

    rmse_f = compute_rmse(rnea_forces_world,  pf_forces_clean)
    rmse_m = compute_rmse(rnea_moments_world, pf_moments_clean)
    mae_f  = compute_mae(rnea_forces_world,   pf_forces_clean)
    mae_m  = compute_mae(rnea_moments_world,  pf_moments_clean)

    print(f"  MAE  Forces (N) : X={mae_f[0]:.2f} Y={mae_f[1]:.2f} Z={mae_f[2]:.2f}")
    print(f"  MAE  Moments(Nm): X={mae_m[0]:.2f} Y={mae_m[1]:.2f} Z={mae_m[2]:.2f}")
    print(f"  RMSE Forces (N) : X={rmse_f[0]:.2f} Y={rmse_f[1]:.2f} Z={rmse_f[2]:.2f}")
    print(f"  RMSE Moments(Nm): X={rmse_m[0]:.2f} Y={rmse_m[1]:.2f} Z={rmse_m[2]:.2f}")

    time = np.linspace(0, n_samples * dt, n_samples)
    time = time[:cut]

    
    fig1, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    titles_f = ['Fx (Antéro-post)', 'Fy (Médiolat)',  'Fz (Vertical)']
    titles_m = ['Mx (Sagittal)',     'My (Frontal)',   'Mz (Vertical)']

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

    fig1.suptitle(f"{subject} — {trial_id}", fontsize=16)
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    out1 = os.path.join(OUTPUT_DIR, f"{subject}_{trial_id}_forces_moments.png")
    # fig1.savefig(out1, dpi=150, bbox_inches='tight')
    # plt.close(fig1)
    # print(f"  Saved: {out1}")
    # plt.show()

    rmse_f = compute_rmse(rnea_forces_pelvis,  pf_forces_pelvis)
    rmse_m = compute_rmse(rnea_moments_pelvis, pf_moments_pelvis)
    mae_f  = compute_mae(rnea_forces_pelvis,   pf_forces_pelvis)
    mae_m  = compute_mae(rnea_moments_pelvis,  pf_moments_pelvis)


    fig3, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    titles_f = ['Fx (Antéro-post)', 'Fy (Médiolat)',  'Fz (Vertical)']
    titles_m = ['Mx (Sagittal)',     'My (Frontal)',   'Mz (Vertical)']
    for j in range(3):
        axs[0,j].plot(time, pf_forces_pelvis[:,j],    color='blue',   alpha=0.6, label='PF')
        axs[0,j].plot(time, rnea_forces_pelvis[:,j],  color='red',              label='RNEA')
        axs[0,j].set_title(f"{titles_f[j]}\nRMSE={rmse_f[j]:.2f} N  |  MAE={mae_f[j]:.2f} N")
        axs[0,j].set_ylabel("Force (N)"); axs[0,j].legend(); axs[0,j].grid(alpha=0.3)

        axs[1,j].plot(time, pf_moments_pelvis[:,j],   color='blue',  alpha=0.6, label='PF')
        axs[1,j].plot(time, rnea_moments_pelvis[:,j], color='red',            label='RNEA')
        axs[1,j].set_title(f"{titles_m[j]}\nRMSE={rmse_m[j]:.2f} Nm  |  MAE={mae_m[j]:.2f} Nm")
        axs[1,j].set_ylabel("Moment (Nm)"); axs[1,j].set_xlabel("Temps (s)")
        axs[1,j].legend(); axs[1,j].grid(alpha=0.3)

    fig3.suptitle(f"{subject} — {trial_id}", fontsize=16)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    out1 = os.path.join(OUTPUT_DIR, f"{subject}_{trial_id}_forces_moments.png")
    # fig1.savefig(out1, dpi=150, bbox_inches='tight')
    # plt.close(fig1)
    # print(f"  Saved: {out1}")
    # plt.show()


    n_dofs     = 12
    dof_raw    = np.degrees(q_ref_raw[:, 7:7+6])
    last_6 = q_ref_raw[:, -6:]
    dof_raw = np.concatenate((dof_raw,last_6),axis=1)

    dof_filt   = np.degrees(q_ref[:,    6:6+6])
    last_6_filt = q_ref[:, -6:]
    dof_filt = np.concatenate((dof_filt,last_6_filt),axis=1)

    dof_names  =  joint_names[7:7+6] + joint_names[-6:]

    n_cols_fig = 4
    n_rows_fig = int(np.ceil(n_dofs / n_cols_fig))

    fig2, axs2 = plt.subplots(n_rows_fig, n_cols_fig,
                               figsize=(18, 4*n_rows_fig), sharex=True)
    axs2 = axs2.flatten()

    dof_raw = dof_raw[:cut]
    dof_filt = dof_filt[:cut]
    time_q = time[:]

    for k in range(n_dofs):
        axs2[k].plot(time_q, dof_raw[:,k],  color='red', label='raw')
        axs2[k].plot(time_q, dof_filt[:,k], color='black',label='filtered')
        axs2[k].set_title(dof_names[k], fontsize=10)
        axs2[k].set_ylabel("Angle (°)"); axs2[k].grid(alpha=0.3)
        if k >= (n_rows_fig - 1) * n_cols_fig:
            axs2[k].set_xlabel("Temps (s)")

    for k in range(n_dofs, len(axs2)):
        axs2[k].set_visible(False)

    axs2[0].legend(loc='upper right', fontsize=8)
    fig2.suptitle(f"{subject} — {trial_id}", fontsize=16)
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    out2 = os.path.join(OUTPUT_DIR, f"{subject}_{trial_id}_angles.png")
    # fig2.savefig(out2, dpi=150, bbox_inches='tight')
    # plt.close(fig2)
    # print(f"  Saved: {out2}")
    # plt.show()

    save_dir = os.path.join(BASE_DATA_DIR, subject, trial_id)
    os.makedirs(save_dir, exist_ok=True)

    kinetics_filtered_df = pd.DataFrame({
        'Fx1_glob': pf1_forces_clean[1:cut,0],
        'Fy1_glob': pf1_forces_clean[1:cut,1],
        'Fz1_glob': pf1_forces_clean[1:cut,2],
        'Mx1_glob': pf1_moments_clean[1:cut,0],
        'My1_glob': pf1_moments_clean[1:cut,1],
        'Mz1_glob': pf1_moments_clean[1:cut,2],
        'Fx2_glob': pf2_forces_clean[1:cut,0],
        'Fy2_glob': pf2_forces_clean[1:cut,1],
        'Fz2_glob': pf2_forces_clean[1:cut,2],
        'Mx2_glob': pf2_moments_clean[1:cut,0],
        'My2_glob': pf2_moments_clean[1:cut,1],
        'Mz2_glob': pf2_moments_clean[1:cut,2],

        'COPx1_glob':pf1_cop_clean[1:cut,0],
        'COPy1_glob':pf1_cop_clean[1:cut,1],
        'COPz1_glob':pf1_cop_clean[1:cut,2],
        'COPx2_glob':pf2_cop_clean[1:cut,0],
        'COPy2_glob':pf2_cop_clean[1:cut,1],
        'COPz2_glob':pf2_cop_clean[1:cut,2],
    })

    kinetics_filtered_df_pelvis = pd.DataFrame({
        'Fx1': pf1_forces_pelvis[1:cut,0],
        'Fy1': pf1_forces_pelvis[1:cut,1],
        'Fz1': pf1_forces_pelvis[1:cut,2],
        'Mx1': pf1_moments_pelvis[1:cut,0],
        'My1': pf1_moments_pelvis[1:cut,1],
        'Mz1': pf1_moments_pelvis[1:cut,2],
        'Fx2': pf2_forces_pelvis[1:cut,0],
        'Fy2': pf2_forces_pelvis[1:cut,1],
        'Fz2': pf2_forces_pelvis[1:cut,2],
        'Mx2': pf2_moments_pelvis[1:cut,0],
        'My2': pf2_moments_pelvis[1:cut,1],
        'Mz2': pf2_moments_pelvis[1:cut,2],

        'COPx1':pf1_cop_pelvis[1:cut,0],
        'COPy1':pf1_cop_pelvis[1:cut,1],
        'COPz1':pf1_cop_pelvis[1:cut,2],
        'COPx2':pf2_cop_pelvis[1:cut,0],
        'COPy2':pf2_cop_pelvis[1:cut,1],
        'COPz2':pf2_cop_pelvis[1:cut,2],
    })

    # kinetics_file = os.path.join(save_dir, "kinetics_glob_filtered.csv")
    # kinetics_filtered_df.to_csv(kinetics_file, index=False)
    # print(f"  Saved filtered kinetics (PF1 + PF2): {kinetics_file}")

    # kinetics_file_pelvis = os.path.join(save_dir, "kinetics_pelvis_filtered.csv")
    # kinetics_filtered_df_pelvis.to_csv(kinetics_file_pelvis, index=False)
    # print(f"  Saved filtered kinetics (PF1 + PF2): {kinetics_file_pelvis}")


    # # --- Sauvegarde joints filtrés ---
    # new_joint_names = [
    # "delta_x",
    # "delta_y",
    # "delta_z",
    # "delta_rx",
    # "delta_ry",
    # "delta_rz"
    # ] + joint_names[7:]
    joints_filtered_df = pd.DataFrame(q_ref[:cut], columns=joint_names)
    joints_file = os.path.join(save_dir, "joints_filtered_FF.csv")
    joints_filtered_df.to_csv(joints_file, index=False)
    print(f"  Saved filtered joints: {joints_file}")



# =====================================================================
# BOUCLE PRINCIPALE
# =====================================================================
subjects = sorted([
    d for d in os.listdir(BASE_DATA_DIR)
    if os.path.isdir(os.path.join(BASE_DATA_DIR, d))
])

print(f"Sujets trouvés : {subjects}")

for subject in subjects:
    subject_dir  = os.path.join(BASE_DATA_DIR, subject)
    trial_names   = discover_trials(subject_dir)
    print(f"\n{subject} — trials trouvés : {trial_names}")

    for trial_name in trial_names:
        try:
            process_subject_trial(subject, trial_name)
        except Exception as e:
            print(f"  [ERROR] {subject}/{trial_name} : {e}")