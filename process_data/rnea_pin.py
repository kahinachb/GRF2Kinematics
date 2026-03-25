#this code allows to compare between inverse dynamic (using pin.rnea) vs grfm data (from pf)
import pandas as pd
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt 
from utils.model_utils import build_human_model
from utils.linear_algebra_utils import lowpass_filter
import ezc3d

subject = "subject01"
task = "dyna"
fps = 300 
dt = 1.0 / fps
which = 'Anais'
path_to_c3d = f"/home/kchalabi/Documents/THESE/datasets_kinetics/Anais/{subject}_{task}.c3d"
c3d = ezc3d.c3d(path_to_c3d, extract_forceplat_data=True)
pf1 = c3d["data"]["platform"][0]
pf2 = c3d["data"]["platform"][1]
center_fp1 = np.mean(pf1['corners'], axis=1)/1000
center_fp2 = np.mean(pf2['corners'], axis=1)/1000
# center_fp1 = (0.3, 0.250, 0.048)
# center_fp2 = (0.3, -0.34, 0.048)

path_joint = f"DATA/{which}/{subject}/{task}/joints_whole_body.csv"
urdf_path = f"DATA/urdf_scaled/{subject}_scaled_wholebody.urdf"
path_grf = f"DATA/{which}/{subject}/{task}/kinetics.csv"
urdf_meshes_path = "motif/model/human_urdf/meshes"

grf_df_raw = pd.read_csv(path_grf)

if fps ==300 : 
    #Anais data
    fx1, fy1, fz1 = 'Fx1', 'Fy1', 'Fz1'
    mx1,my1,mz1 =  'Mx1_glob', 'My1_glob', 'Mz1_glob'

    fx2, fy2, fz2 = 'Fx2', 'Fy2', 'Fz2'
    mx2,my2,mz2 =  'Mx2_glob', 'My2_glob', 'Mz2_glob'


else: 
     #vinc data
    # fx1, fy1, fz1 = 'FX1', 'FY1', 'FZ1'
    # mx1,my1,mz1 =  'MX1', 'MY1', 'MZ1'
    # fx2, fy2, fz2 = 'FX2', 'FY2', 'FZ2'
    # mx2,my2,mz2 =  'MX2', 'MY2', 'MZ2'


    #humanoids data
    fx1, fy1, fz1 = 'Fx1_glob', 'Fy1_glob', 'Fz1_glob'
    mx1,my1,mz1 =  'Mx1_glob', 'My1_glob', 'Mz1_glob'
    fx2, fy2, fz2 = 'Fx2_glob', 'Fy2_glob', 'Fz2_glob'
    mx2,my2,mz2 =  'Mx2_glob', 'My2_glob', 'Mz2_glob'


cols_to_filter = [fx1, fy1, fz1,mx1,my1,mz1,fx2, fy2, fz2,mx2,my2,mz2]

grf_data_filtered = lowpass_filter(grf_df_raw[cols_to_filter].to_numpy(), cutoff=2, fs=fps)
grf_df = grf_df_raw.copy()
grf_df[cols_to_filter] = grf_data_filtered
print(len(grf_data_filtered))

pf_forces_total = np.array([grf_df[fx1]+grf_df[fx2], 
                            grf_df[fy1]+grf_df[fy2], 
                            grf_df[fz1]+grf_df[fz2]]).T

off_fx_total = pf_forces_total[:, 0].mean()
off_fy_total = pf_forces_total[:, 1].mean()

off_fx1 = grf_df[fx1].mean()
off_fy1 = grf_df[fy1].mean()
off_fx2 = grf_df[fx2].mean()
off_fy2 = grf_df[fy2].mean()

########################################################################
q_ref_df = pd.read_csv(path_joint) #.iloc[:, 1:]
joint_names = list(q_ref_df.columns)  
q_ref_raw = q_ref_df.to_numpy(dtype=float)
q_ref = lowpass_filter(q_ref_raw, cutoff=2, fs=fps)
print(len(q_ref))

for i in range(len(q_ref)):
    q_quat = q_ref[i, 3:7]
    q_ref[i, 3:7] = q_quat / np.linalg.norm(q_quat)

model_h, _, _, _ = build_human_model(urdf_path, urdf_meshes_path)
data_h = model_h.createData()
n_samples = len(q_ref)
nv = model_h.nv

print(f"Masse urdf : {pin.computeTotalMass(model_h):.2f} kg")
print(f"Degrés de liberté: {nv}")
print(f"Vecteur gravité du modèle : {model_h.gravity.linear}")

v_ref = np.zeros((n_samples, nv))
a_ref = np.zeros((n_samples, nv))
for i in range(n_samples - 1):
    v_ref[i, :] = pin.difference(model_h, q_ref[i, :], q_ref[i+1, :]) / dt
for i in range(n_samples - 1):
    a_ref[i, :] = (v_ref[i+1, :] - v_ref[i, :]) / dt

rnea_forces_world = np.zeros((n_samples, 3))
rnea_moments_world = np.zeros((n_samples, 3))
pf_forces_total_clean = np.zeros((n_samples, 3))
pf_moments_world_clean = np.zeros((n_samples, 3))

angle = np.pi / 2 
R_corr = np.array([[1, 0,           0          ],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle),  np.cos(angle)]])


print("RNEA...")
for i in range(n_samples):

    #tourner le modele avc le nouveau urdf et ik faite avc nouveau urdf (robex)
    # q_current = q_ref_df.iloc[i].to_numpy()
    # pos_bassin_rnea = q_current[0:3]
    # quat_bassin = q_current[3:7] # qx, qy, qz, qw

    # quat_original = pin.Quaternion(q_current[6], q_current[3], q_current[4], q_current[5]) #(w,x,y,z)
    # R_original = quat_original.toRotationMatrix()

    # R_final = R_corr @ R_original
    # quat_final = pin.Quaternion(R_final) 

    # q_ref[i][3:7] = [quat_final.x, quat_final.y, quat_final.z, quat_final.w]
    # q_ref[i][0:3] = R_corr @ q_current[0:3]

    tau = pin.rnea(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])
    tau = pin.rnea(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])
    # pin.forwardKinematics(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])
    # wrench_base = data_h.f[1]         
    # oM1 = data_h.oMi[1]              
    # f_world = oM1.act(wrench_base)          
    tau = pin.rnea(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])       
    # pin.forwardKinematics(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])
    # wrench_base = data_h.f[1]         
    # oM1 = data_h.oMi[1]              
    # f_world = oM1.act(wrench_base)          

    quat = pin.Quaternion(q_ref[i, 3:7])
    rot_base = quat.matrix()
    pos_bassin = q_ref[i, 0:3]

    rnea_forces_world[i, :] = rot_base @ tau[0:3]
    rnea_moments_world[i, :] = (rot_base @ tau[3:6]) + np.cross(pos_bassin, rnea_forces_world[i, :])

    F1 = grf_df.loc[i, [fx1, fy1, fz1]].values
    M1 = grf_df.loc[i, [mx1, my1, mz1]].values /1000.0

    F2 = grf_df.loc[i, [fx2, fy2, fz2]].values
    M2 = grf_df.loc[i, [mx2, my2, mz2]].values / 1000.0

    F1_clean =F1
    F1_clean[0] -= off_fx1
    F1_clean[1] -= off_fy1
    bias_force1 = np.array([off_fx1, off_fy1, 0])
    M1_clean = M1 - np.cross(center_fp1, bias_force1)

    F2_clean =F2
    F2_clean[0] -= off_fx2
    F2_clean[1] -= off_fy2
    bias_force2 = np.array([off_fx2, off_fy2, 0])
    M2_clean = M2 - np.cross(center_fp2, bias_force2)

    pf_forces_total_clean[i, :] = F1_clean + F2_clean
    pf_moments_world_clean[i, :] = M1_clean + M2_clean


err_f = np.mean(np.abs(rnea_forces_world - pf_forces_total_clean[:n_samples]), axis=0)
err_m = np.mean(np.abs(rnea_moments_world - pf_moments_world_clean), axis=0)

rmse_f = np.sqrt(np.mean((rnea_forces_world - pf_forces_total_clean[:n_samples])**2, axis=0))
rmse_m = np.sqrt(np.mean((rnea_moments_world - pf_moments_world_clean)**2, axis=0))

print(f"MAE  FORCES (N)   -> X: {err_f[0]:.2f}, Y: {err_f[1]:.2f}, Z: {err_f[2]:.2f}")
print(f"MAE  MOMENTS(Nm)  -> X: {err_m[0]:.2f}, Y: {err_m[1]:.2f}, Z: {err_m[2]:.2f}")
print(f"RMSE FORCES (N)   -> X: {rmse_f[0]:.2f}, Y: {rmse_f[1]:.2f}, Z: {rmse_f[2]:.2f}")
print(f"RMSE MOMENTS(Nm)  -> X: {rmse_m[0]:.2f}, Y: {rmse_m[1]:.2f}, Z: {rmse_m[2]:.2f}")


fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
time = np.linspace(0, n_samples * dt, n_samples)

titles_f = ['Fx (Antéro-post)', 'Fy (Médiolat)', 'Fz (Vertical)']
titles_m = ['Mx (Sagittal)', 'My (Frontal)', 'Mz (Vertical)']

for j in range(3):
    axs[0, j].plot(time, pf_forces_total_clean[:n_samples, j], label="PF", color='blue', alpha=0.6)
    axs[0, j].plot(time, rnea_forces_world[:, j], label="RNEA", color='red')
    axs[0, j].set_title(f"{titles_f[j]}\nRMSE = {rmse_f[j]:.2f} N")
    axs[0, j].set_ylabel("Force (N)")
    axs[0, j].legend()
    axs[0, j].grid(alpha=0.3)

    axs[1, j].plot(time, pf_moments_world_clean[:, j], label="PF", color='blue', alpha=0.6)
    axs[1, j].plot(time, rnea_moments_world[:, j], label="RNEA", color='red')
    axs[1, j].set_title(f"{titles_m[j]}\nRMSE = {rmse_m[j]:.2f} Nm")
    axs[1, j].set_ylabel("Moment (Nm)")
    axs[1, j].set_xlabel("Temps (s)")
    axs[1, j].legend()
    axs[1, j].grid(alpha=0.3)

plt.suptitle(f"{subject} {task}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


n_dofs = 12
dof_cols_ = q_ref_raw[:, 7:7+6]         
last_6 = q_ref_raw[:, -6:]
dof_cols = np.concatenate((dof_cols_,last_6),axis=1)

dof_cols_filt_ = q_ref[:, 7:7+6]          
last_6_filt = q_ref[:, -6:]
dof_cols_filt = np.concatenate((dof_cols_filt_,last_6_filt),axis=1)

dof_names = joint_names[7:7+6] + joint_names[-6:]        
dof_cols_deg      = np.degrees(dof_cols)
dof_cols_filt_deg = np.degrees(dof_cols_filt)

n_cols_fig = 4
n_rows_fig = int(np.ceil(n_dofs / n_cols_fig))  

fig2, axs2 = plt.subplots(n_rows_fig, n_cols_fig,
                           figsize=(18, 4 * n_rows_fig), sharex=True)
axs2 = axs2.flatten()

for k in range(n_dofs):
    axs2[k].plot(time, dof_cols_deg[:, k],
                 color='red', label='raw')
    axs2[k].plot(time, dof_cols_filt_deg[:, k],
                 color='black',label='filtered')
    axs2[k].set_title(dof_names[k], fontsize=10)
    axs2[k].set_ylabel("Angle (deg)")
    axs2[k].grid(alpha=0.3)
    if k >= (n_rows_fig - 1) * n_cols_fig:   
        axs2[k].set_xlabel("Temps (s)")

for k in range(n_dofs, len(axs2)):
    axs2[k].set_visible(False)

axs2[0].legend(loc='upper right', fontsize=8)
fig2.suptitle(f"{subject} {task}", fontsize=16)
fig2.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()