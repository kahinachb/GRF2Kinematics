
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


subject = 'Jeremy'

task = 'Trial111'
which = 'Vinc'
fps = 100  #kinetics_glob_filtered are all 100hz
dt = 1.0 / fps

urdf_path = f"DATA/urdf_scaled/{which}/{subject}_scaled.urdf"# Human base
# urdf_path ='/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/rt-cosmik/urdf/human.urdf'

path_joint_pred = f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_feet_cop/Jeremy_Trial111_prediction.csv"
path_joint = f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_PE_sin_aug/Jeremy_Trial111.csv"
# path_joint_pred="/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_PE_sin_aug/s1_squat_variant_980_dz-0.080_dx+0.023_dy-0.017_prediction.csv"
q_ref_df_pred = pd.read_csv(path_joint_pred).iloc[:,:19]
# path_joint ="/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_PE_sin_aug/s1_squat_variant_980_dz-0.080_dx+0.023_dy-0.017.csv"
q_ref_df = pd.read_csv(path_joint).iloc[:,:19]

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
all_joint_ids = set(range(1, model_h.njoints))
joints_to_lock = [#"root_joint",
                  "middle_lumbar_Z", "middle_lumbar_X",
    "left_clavicle_joint_X",
    "left_shoulder_Z", "left_shoulder_X", "left_shoulder_Y",
    "left_elbow_Z", "left_elbow_Y",
    "middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y",
    "right_clavicle_joint_X",
    "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",
    "right_elbow_Z", "right_elbow_Y" ]

joint_ids_to_lock = []
for jn in joints_to_lock:
    if model_h.existJointName(jn):
        joint_ids_to_lock.append(model_h.getJointId(jn))
    else:
        print('Warning: joint ' + str(jn) + ' does not belong to the model!')

q0 = pin.neutral(model_h)
# Build reduced model
model_h, vis_h = pin.buildReducedModel(
    model_h, vis_h, joint_ids_to_lock, q0)

print(model_h.nq)
data_h = pin.Data(model_h)
############################################################""
joint_ids_to_lock = []
for jn in joints_to_lock:
    if model_h_pred.existJointName(jn):
        joint_ids_to_lock.append(model_h_pred.getJointId(jn))
    else:
        print('Warning: joint ' + str(jn) + ' does not belong to the model!')

q0 = pin.neutral(model_h_pred)
# Build reduced model
model_h_pred, vis_h_pred = pin.buildReducedModel(
    model_h_pred, vis_h_pred, joint_ids_to_lock, q0)

print(model_h_pred.nq)
data_h_pred = pin.Data(model_h_pred)
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


# import pinocchio as pin

# # Exemple : rotation de 90° autour de X
# R = pin.utils.rotate('x', np.pi/2)

# # translation (optionnelle)
# t = np.array([0, 0, 0])

# placement = pin.SE3(R, t)

# viz_human.viewer["ref"].set_transform(placement.homogeneous)
# viz_human_pred.viewer["pred"].set_transform(placement.homogeneous)


q0 = pin.neutral(model_h)
viz_human.display(q0)
images = []
for i in range(len(q_ref)):


    q_pred_full = np.zeros_like(q_ref[i])
    q_pred_full[:7] = q_ref[i][:7]      # freeflyer
    q_pred_full[7:] = q_ref_pred[i]     # joints prédits

    viz_human.display(q_ref[i])
    viz_human_pred.display(q_pred_full)
    images.append(viewer.get_image())

    
video_path = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/real.mp4"
imageio.mimsave(video_path, images, fps=fps, codec='libx264')
print(f"[MeshCat] Video saved to: {video_path}")



