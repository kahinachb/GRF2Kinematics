
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
import pinocchio as pin
import numpy as np
import imageio.v2 as imageio


subject = 'Kahina'

task = 'squat'
which = 'HUMANOIDS'
fps = 100  #kinetics_glob_filtered are all 100hz
dt = 1.0 / fps

path_grf = f"DATA/{which}/{subject}/{task}/kinetics_glob_filtered.csv"
grf_df = pd.read_csv(path_grf)
fx1, fy1, fz1 = 'Fx1_glob', 'Fy1_glob', 'Fz1_glob'
mx1,my1,mz1 =  'Mx1_glob', 'My1_glob', 'Mz1_glob'
fx2, fy2, fz2 = 'Fx2_glob', 'Fy2_glob', 'Fz2_glob'
mx2,my2,mz2 =  'Mx2_glob', 'My2_glob', 'Mz2_glob'



urdf_path = f"DATA/urdf_scaled/{which}/{subject}.urdf"# Human base
# urdf_path ='/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/rt-cosmik/urdf/human.urdf'

path_joint_pred = f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_CFM_HUM/Kahina_squat_cfm_prediction.csv"
path_joint = f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/{which}/{subject}/{task}/joints_filtered_FF.csv"


# path_joint_pred= '/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_PE_sin_cross_synth_guided11/s1_squat_variant_980_dz-0.080_dx+0.023_dy-0.017_prediction.csv'
# path_joint= "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/generated_human_like_motions_csv_new/generated_human_like_motions_csv/joint_filtered_squat_variant_980_dz-0.080_dx+0.023_dy-0.017.csv"
q_ref_df_pred = pd.read_csv(path_joint_pred)

dofs = ["root_joint","root_joint.1","root_joint.2","root_joint.3","root_joint.4","root_joint.5","root_joint.6",
        "middle_lumbar_Z", "middle_lumbar_X",
    "left_clavicle_joint_X",
    "left_shoulder_Z", "left_shoulder_X", "left_shoulder_Y",
    "left_elbow_Z", "left_elbow_Y",
    "middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y",
    "right_clavicle_joint_X",
    "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",
    "right_elbow_Z", "right_elbow_Y",

"left_hip_Z",  "left_hip_X",  "left_hip_Y",
"left_knee_Z", "left_ankle_Z", "left_ankle_X",
"right_hip_Z", "right_hip_X", "right_hip_Y",
"right_knee_Z", "right_ankle_Z", "right_ankle_X",]

q_ref_df = pd.read_csv(path_joint, usecols=dofs).iloc[1:,:]

print(q_ref_df_pred.shape)
print(q_ref_df.shape)
desired_order_ref = [
    "FF_X","FF_Y","FF_Z","FF_quatx","FF_quaty","FF_quatz","FF_quatw",
    "Lhip_flex_ext","Lhip_abd_add","Lhip_int_ext_rot",
    "Lknee_flex_ext",
    "Lankle_flex_ext","Lankle_abd_add",
    # "Lumbar_flex_ext","Lumbar_lateral_flex",
    # "Lcalvicule_x",
    # "Lshoulder_flex_ext","Lshoulder_abd_add","Lshoulder_int_ext_rot",
    # "Lelbow_flex_ext","Lelbow_pron_supi",
    # "Cervical_flex_ext","Cervical_lat_bend","Cervical_int_ext_rot",
    # "rcalvicule_x",
    # "Rshoulder_flex_ext","Rshoulder_abd_add","Rshoulder_int_ext_rot",
    # "Relbow_flex_ext","Relbow_pron_supi",
    "Rhip_flex_ext","Rhip_abd_add","Rhip_int_ext_rot",
    "Rknee_flex_ext",
    "Rankle_flex_ext","Rankle_abd_add"
]



desired_order_ref = ["root_joint","root_joint.1","root_joint.2","root_joint.3","root_joint.4","root_joint.5","root_joint.6",
                    

"left_hip_Z",  "left_hip_X",  "left_hip_Y",
"left_knee_Z", "left_ankle_Z", "left_ankle_X",

    "middle_lumbar_Z", "middle_lumbar_X","left_clavicle_joint_X",
    "left_shoulder_Z", "left_shoulder_X", "left_shoulder_Y",
    "left_elbow_Z", "left_elbow_Y",
    "middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y",
    "right_clavicle_joint_X",
    "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",
    "right_elbow_Z", "right_elbow_Y",

"right_hip_Z", "right_hip_X", "right_hip_Y",
"right_knee_Z", "right_ankle_Z", "right_ankle_X",
]

desired_order_pred = [
    "Lhip_flex_ext","Lhip_abd_add","Lhip_int_ext_rot",
    "Lknee_flex_ext",
    "Lankle_flex_ext","Lankle_abd_add",
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

    q_pred_full[i, :7] = q_ref[i][:7]              # freeflyer
    q_pred_full[i, 7:13] = q_ref_pred[i][:6]       # joints prédits
    q_pred_full[i, 13:30] = q_ref[i][13:30]
    q_pred_full[i, 30:] = q_ref_pred[i][6:]

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

for i in range(len(q_ref)):


    q_pred_full = np.zeros_like(q_ref[i])
    q_pred_full[:7] = q_ref[i][:7]      # freeflyer
    q_pred_full[7:13] = q_ref_pred[i][:6]    # joints prédits
    q_pred_full[13:30] = q_ref[i][13:30]
    q_pred_full[30:] = q_ref_pred[i][6:]

    tau_ref = pin.rnea(model_h,data_h,q_ref[i],v_ref[i],a_ref[i])
    pin.forwardKinematics(model_h, data_h, q_ref[i], v_ref[i], a_ref[i])
    wrench_base = data_h.f[1]
    oM1 = data_h.oMi[1]
    f_world = oM1.act(wrench_base)

    f_ref_world.append(f_world.linear)
    m_ref_world.append(f_world.angular)

    tau_pred = pin.rnea(model_h_pred,data_h_pred, q_pred_full,v_pred[i],a_pred[i])
    pin.forwardKinematics(model_h_pred, data_h_pred, q_pred_full, v_pred[i], a_pred[i])
    wrench_base_pred = data_h_pred.f[1]
    oM1_pred = data_h_pred.oMi[1]
    f_world_pred = oM1_pred.act(wrench_base)

    f_pred_world.append(f_world_pred.linear)
    m_pred_world.append(f_world_pred.angular)


    F1 = grf_df.loc[i, [fx1,fy1,fz1]].values
    F2 = grf_df.loc[i, [fx2,fy2,fz2]].values
    M1 = grf_df.loc[i, [mx1,my1,mz1]].values / 1000.0
    M2 = grf_df.loc[i, [mx2,my2,mz2]].values / 1000.0
    F = F1 + F2
    M = M1 + M2

  
    viz_human.display(q_ref[i])
    viz_human_pred.display(q_pred_full)
#     images.append(viewer.get_image())

    
# video_path = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_CFM_real/out.mp4"
# imageio.mimsave(video_path, images, fps=fps, codec='libx264')
# print(f"[MeshCat] Video saved to: {video_path}")



