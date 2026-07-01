
#visualize joint and cop from pf and rnea
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.model_utils import build_human_model
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(script_directory))
import meshcat
import meshcat_shapes
from pinocchio.visualize import MeshcatVisualizer
import pandas as pd
from utils.utils import find_col
from utils.linear_algebra_utils import lowpass_filter 
import pinocchio as pin
import numpy as np
import imageio.v2 as imageio


subject = 'Jeremy'

task = 'Trial111'
which = 'Vinc'
fps = 100  #kinetics_glob_filtered are all 100hz
dt = 1.0 / fps

path_grf = f"DATA/{which}/{subject}/{task}/kinetics_glob_filtered.csv"
# path_grf ="DATA/generated_data/subject_11_squat_variant_082_dz+0.006_dx+0.032_dy+0.013_grfm.csv"

grf_df = pd.read_csv(path_grf)
fx1, fy1, fz1 = 'Fx1_glob', 'Fy1_glob', 'Fz1_glob'
mx1,my1,mz1 =  'Mx1_glob', 'My1_glob', 'Mz1_glob'
fx2, fy2, fz2 = 'Fx2_glob', 'Fy2_glob', 'Fz2_glob'
mx2,my2,mz2 =  'Mx2_glob', 'My2_glob', 'Mz2_glob'


# urdf_path = "DATA/urdf/human_subject_08.urdf"
urdf_path = f"DATA/urdf_scaled/{which}/{subject}_scaled.urdf"# Human base
# urdf_path ='/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/rt-cosmik/urdf/human.urdf'

path_joint_pred = f"inference_results/Jeremy_Trial111_improved_prediction.csv"
path_joint_pred = f"results_full_step2motion_fm_improved_ff/Jeremy_Trial111_prediction_absolute_euler.csv"
path_joint = f"DATA/{which}/{subject}/{task}/joints_filtered_FF.csv"

# path_joint = f"DATA/generated_data/subject_11_squat_variant_082_dz+0.006_dx+0.032_dy+0.013_q.csv"
# path_joint_pred= 'results_full_step2motion_fm_ff/subject_11_variant_082_dz+0.006_dx+0.032_dy+0.013_prediction_absolute.csv'

q_ref_df_pred = pd.read_csv(path_joint_pred)

# dofs = ["root_joint","root_joint.1","root_joint.2","root_joint.3","root_joint.4","root_joint.5","root_joint.6",
#         "middle_lumbar_Z", "middle_lumbar_X",
#     "left_clavicle_joint_X",
#     "left_shoulder_Z", "left_shoulder_X", "left_shoulder_Y",
#     "left_elbow_Z", "left_elbow_Y",
#     "middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y",
#     "right_clavicle_joint_X",
#     "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",
#     "right_elbow_Z", "right_elbow_Y",

# "left_hip_Z",  "left_hip_X",  "left_hip_Y",
# "left_knee_Z", "left_ankle_Z", "left_ankle_X",
# "right_hip_Z", "right_hip_X", "right_hip_Y",
# "right_knee_Z", "right_ankle_Z", "right_ankle_X",]

dofs = [
    "FF_X","FF_Y","FF_Z","FF_quatx","FF_quaty","FF_quatz","FF_quatw",
    "Lhip_flex_ext","Lhip_abd_add","Lhip_int_ext_rot",
    "Lknee_flex_ext",
    "Lankle_flex_ext","Lankle_abd_add",
    "Lumbar_flex_ext","Lumbar_lateral_flex",
    "Lcalvicule_x",
    "Lshoulder_flex_ext","Lshoulder_abd_add","Lshoulder_int_ext_rot",
    "Lelbow_flex_ext","Lelbow_pron_supi",
    "Cervical_flex_ext","Cervical_lat_bend","Cervical_int_ext_rot",
    "rcalvicule_x",
    "Rshoulder_flex_ext","Rshoulder_abd_add","Rshoulder_int_ext_rot",
    "Relbow_flex_ext","Relbow_pron_supi",
    "Rhip_flex_ext","Rhip_abd_add","Rhip_int_ext_rot",
    "Rknee_flex_ext",
    "Rankle_flex_ext","Rankle_abd_add"
]
q_ref_df = pd.read_csv(path_joint, usecols=dofs).iloc[1:,:]

print(q_ref_df_pred.shape)
print(q_ref_df.shape)
desired_order_ref = [
    "FF_X","FF_Y","FF_Z","FF_quatx","FF_quaty","FF_quatz","FF_quatw",
    "Lhip_flex_ext","Lhip_abd_add","Lhip_int_ext_rot",
    "Lknee_flex_ext",
    "Lankle_flex_ext","Lankle_abd_add",
    "Lumbar_flex_ext","Lumbar_lateral_flex",
    "Lcalvicule_x",
    "Lshoulder_flex_ext","Lshoulder_abd_add","Lshoulder_int_ext_rot",
    "Lelbow_flex_ext","Lelbow_pron_supi",
    "Cervical_flex_ext","Cervical_lat_bend","Cervical_int_ext_rot",
    "rcalvicule_x",
    "Rshoulder_flex_ext","Rshoulder_abd_add","Rshoulder_int_ext_rot",
    "Relbow_flex_ext","Relbow_pron_supi",
    "Rhip_flex_ext","Rhip_abd_add","Rhip_int_ext_rot",
    "Rknee_flex_ext",
    "Rankle_flex_ext","Rankle_abd_add"
]



# desired_order_ref = ["root_joint","root_joint.1","root_joint.2","root_joint.3","root_joint.4","root_joint.5","root_joint.6",
                    
#                     "left_hip_Z",  "left_hip_X",  "left_hip_Y",
#                     "left_knee_Z", "left_ankle_Z", "left_ankle_X",

#                         "middle_lumbar_Z", "middle_lumbar_X","left_clavicle_joint_X",
#                         "left_shoulder_Z", "left_shoulder_X", "left_shoulder_Y",
#                         "left_elbow_Z", "left_elbow_Y",
#                         "middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y",
#                         "right_clavicle_joint_X",
#                         "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",
#                         "right_elbow_Z", "right_elbow_Y",

#                     "right_hip_Z", "right_hip_X", "right_hip_Y",
#                     "right_knee_Z", "right_ankle_Z", "right_ankle_X",
#                         ]

desired_order_pred = [
   "FF_X","FF_Y","FF_Z","FF_quatx","FF_quaty","FF_quatz","FF_quatw",
    "Lhip_flex_ext","Lhip_abd_add","Lhip_int_ext_rot",
    "Lknee_flex_ext",
    "Lankle_flex_ext","Lankle_abd_add",

    "Lumbar_flex_ext","Lumbar_lateral_flex",
    "Lcalvicule_x",
    "Lshoulder_flex_ext","Lshoulder_abd_add","Lshoulder_int_ext_rot",
    "Lelbow_flex_ext","Lelbow_pron_supi",
    "Cervical_flex_ext","Cervical_lat_bend","Cervical_int_ext_rot",
    "Rcalvicule_x",
    "Rshoulder_flex_ext","Rshoulder_abd_add","Rshoulder_int_ext_rot",
    "Relbow_flex_ext","Relbow_pron_supi",

    "Rhip_flex_ext","Rhip_abd_add","Rhip_int_ext_rot",
    "Rknee_flex_ext",
    "Rankle_flex_ext","Rankle_abd_add"
]

# Vérifier les colonnes manquantes (très important)
missing = [col for col in desired_order_ref if col not in q_ref_df.columns]
if missing:
    print("Colonnes manquantes :", missing)

# Réordonner
q_ref_df = q_ref_df[desired_order_ref]
q_ref_df_pred = q_ref_df_pred[desired_order_pred]


# Convertir en numpy si besoin
q_ref = q_ref_df.to_numpy(dtype=float)
q_ref_pred = q_ref_df_pred.to_numpy(dtype=float)
q_ref_pred = lowpass_filter(q_ref_pred, cutoff=2, fs=100, order=4)

urdf_name = "human.urdf"
urdf_meshes_path = "motif/model/human_urdf"
model_h, coll_h, vis_h, _ = build_human_model(urdf_path, urdf_meshes_path)

model_h_pred, coll_h_pred, vis_h_pred, _ = build_human_model(urdf_path, urdf_meshes_path)


# ################################################################################LOCK JOINTS
# all_joint_ids = set(range(1, model_h.njoints))
# joints_to_lock = [#"root_joint",
#                   "middle_lumbar_Z", "middle_lumbar_X",
#     "left_clavicle_joint_X",
#     "left_shoulder_Z", "left_shoulder_X", "left_shoulder_Y",
#     "left_elbow_Z", "left_elbow_Y",
#     "middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y",
#     "right_clavicle_joint_X",
#     "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",
#     "right_elbow_Z", "right_elbow_Y" ]

# joint_ids_to_lock = []
# for jn in joints_to_lock:
#     if model_h.existJointName(jn):
#         joint_ids_to_lock.append(model_h.getJointId(jn))
#     else:
#         print('Warning: joint ' + str(jn) + ' does not belong to the model!')

# q0 = pin.neutral(model_h)
# # Build reduced model
# model_h, vis_h = pin.buildReducedModel(
#     model_h, vis_h, joint_ids_to_lock, q0)

# print(model_h.nq)
# data_h = pin.Data(model_h)
############################################################""
# joint_ids_to_lock = []
# for jn in joints_to_lock:
#     if model_h_pred.existJointName(jn):
#         joint_ids_to_lock.append(model_h_pred.getJointId(jn))
#     else:
#         print('Warning: joint ' + str(jn) + ' does not belong to the model!')

# q0 = pin.neutral(model_h_pred)
# # Build reduced model
# model_h_pred, vis_h_pred = pin.buildReducedModel(
#     model_h_pred, vis_h_pred, joint_ids_to_lock, q0)

# print(model_h_pred.nq)
# data_h_pred = pin.Data(model_h_pred)
##############################################################################################################

data_h = model_h.createData()
data_h_pred = model_h_pred.createData()

##################################################



#Meshcat
viewer = meshcat.Visualizer()

# Human ref
viz_human = MeshcatVisualizer(model_h, coll_h, vis_h)
viz_human.initViewer(viewer, open=True)
viz_human.viewer.delete()  # clear if relaunch
viz_human.loadViewerModel("ref", color=[0.5, 0.5, 0.5, 0.5])

# Human pred
viz_human_pred = MeshcatVisualizer(model_h_pred, coll_h_pred, vis_h_pred)
viz_human_pred.initViewer(viewer)   # <-- IMPORTANT
viz_human_pred.loadViewerModel("pred", color=[0.0, 1.0, 0.0, 0.8])

###########################################################################
q0 = pin.neutral(model_h)
# q0[3:7]=quat

# Background/grid
bg_top = (1,1,1)
bg_bottom = (1,1,1)
grid_height = -0.0
native_viz = viz_human.viewer
native_viz["/Background"].set_property("top_color", list(bg_top))
native_viz["/Background"].set_property("bottom_color", list(bg_bottom))
native_viz["/Grid"].set_transform(
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, grid_height], [0, 0, 0, 1]])
)

n_samples = len(q_ref)
nv = model_h.nv
print("q shape:", q_ref.shape)
print("nv:", nv)


v_ref = np.zeros((n_samples, nv))
a_ref = np.zeros((n_samples, nv))
for i in range(n_samples - 1):
    v_ref[i, :] = pin.difference(model_h, q_ref[i, :], q_ref[i+1, :]) / dt
for i in range(n_samples - 1):
    a_ref[i, :] = (v_ref[i+1, :] - v_ref[i, :]) / dt
##############################################################pred
nq = model_h_pred.nq
q_pred_full = np.zeros((n_samples, nq))
for i in range(n_samples):

    # q_pred_full[i, :7] = q_ref[i][:7]             
    # q_pred_full[i,7:] = q_ref_pred[i][:]
    q_pred_full[i,:] = q_ref_pred[i][:]

nv = model_h_pred.nv
n_samples = len(q_pred_full)
v_pred = np.zeros((n_samples, nv))
a_pred = np.zeros((n_samples, nv))
for i in range(n_samples - 1):
    v_pred[i, :] = pin.difference(model_h_pred, q_pred_full[i, :], q_pred_full[i+1, :]) / dt
for i in range(n_samples - 1):
    a_pred[i, :] = (v_pred[i+1, :] - v_pred[i, :]) / dt


q0 = pin.neutral(model_h)
viz_human.display(q0)
images = []

f_ref_world = []
m_ref_world = []
f_pred_world = []
m_pred_world = []

tau_ref_all = []
tau_pred_all = []

F_world = []
M_world = []

for i in range(len(q_ref)):


    q_pred_full = np.zeros_like(q_ref[i])
    # q_pred_full[:7] = q_ref[i][:7]      # freeflyer
    # q_pred_full[7:] = q_ref_pred[i][:]
    q_pred_full[:] = q_ref_pred[i][:]

    tau_ref = pin.rnea(model_h,data_h,q_ref[i],v_ref[i],a_ref[i]) #N et N.m 
    tau_ref_all.append(tau_ref)

    pin.forwardKinematics(model_h, data_h, q_ref[i], v_ref[i], a_ref[i])
    wrench_base = data_h.f[1]
    oM1 = data_h.oMi[1]
    f_world = oM1.act(wrench_base)

    f_ref_world.append(f_world.linear)
    m_ref_world.append(f_world.angular)

    tau_pred = pin.rnea(model_h_pred,data_h_pred, q_pred_full,v_pred[i],a_pred[i])
    tau_pred_all.append(tau_pred)

    pin.forwardKinematics(model_h_pred, data_h_pred, q_pred_full, v_pred[i], a_pred[i])
    wrench_base_pred = data_h_pred.f[1]
    oM1_pred = data_h_pred.oMi[1]
    f_world_pred = oM1_pred.act(wrench_base_pred)

    f_pred_world.append(f_world_pred.linear)
    m_pred_world.append(f_world_pred.angular)


    F1 = grf_df.loc[i, [fx1,fy1,fz1]].values
    F2 = grf_df.loc[i, [fx2,fy2,fz2]].values
    M1 = grf_df.loc[i, [mx1,my1,mz1]].values 
    M2 = grf_df.loc[i, [mx2,my2,mz2]].values 
    F = F1 + F2
    M = M1 + M2

    F_world.append(F)
    M_world.append(M)

    

  
    viz_human.display(q_ref[i])
    viz_human_pred.display(q_pred_full)
#     images.append(viewer.get_image())


f_ref_world = np.array(f_ref_world)
f_pred_world = np.array(f_pred_world)
m_ref_world = np.array(m_ref_world)



F_world = np.array(F_world)
M_world = np.array(M_world)

import matplotlib.pyplot as plt
t = np.arange(len(F_world))
t = np.arange(4000)
fig, axs = plt.subplots(2, 3, figsize=(15, 7))

labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

data_ref = np.hstack([f_ref_world, m_ref_world])
data_pred = np.hstack([f_pred_world, m_pred_world])
data_FM = np.hstack([F_world, M_world])


for j, ax in enumerate(axs.flatten()):

    ax.plot(t, data_ref[:4000, j], label="Reference joint wrench")
    ax.plot(t, data_pred[:4000, j], label="Predicted joint wrench")
    ax.plot(t, data_FM[:4000, j], label="Force plate")

    ax.set_title(labels[j])
    ax.set_xlabel("Frame")
    ax.grid(True)

    if j == 0:
        ax.set_ylabel("Force (N)")
    elif j == 3:
        ax.set_ylabel("Moment (N.m)")


axs[0,0].legend()

plt.tight_layout()
plt.show()
t = np.arange(len(F_world))
tau_ref_all = np.array(tau_ref_all)
tau_pred_all = np.array(tau_pred_all)

tau_ref_j = tau_ref_all[:, :]
tau_pred_j = tau_pred_all[:, :]

n_joints = tau_ref_j.shape[1]

joint_names = model_h_pred.names[:]  

for j in range(n_joints):

    plt.figure(figsize=(10, 3))

    plt.plot(t, tau_ref_j[:, j], label="Ref")
    plt.plot(t, tau_pred_j[:, j], label="Pred")

    plt.title(joint_names[j])  
    plt.xlabel("Frame")
    plt.ylabel("Torque (N·m)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

# video_path = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_CFM_real/out.mp4"
# imageio.mimsave(video_path, images, fps=fps, codec='libx264')
# print(f"[MeshCat] Video saved to: {video_path}")



