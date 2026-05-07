
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
from utils.utils import read_mks_data, find_col, to_m
from utils.linear_algebra_utils import lowpass_filter
import pinocchio as pin
import numpy as np
from utils.viz_utils import add_sphere, place,set_tf, safe_place, place
from pinocchio import Quaternion
import example_robot_data as robex

which = 'Vinc'
subject = 'Jeremy'
task = 'Trial111'


urdf_path = f"DATA/urdf_scaled/{which}/{subject}_scaled.urdf"# Human base
# urdf_path = f'/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/rt-cosmik/urdf/human.urdf'


path_joint = f"DATA/{which}/{subject}/{task}/joints_filtered_FF.csv"
q_ref_df = pd.read_csv(path_joint)#.iloc[:,1:]
q_ref = q_ref_df.to_numpy(dtype=float)
######################################################""


if which =="Vinc":
            mks ={
            "RHEE": "r_calc_study", "RMTOE": "r_5meta_study","RTOE": "r_toe_study", 
            "RANK": "r_ankle_study",  "RMANK": "r_mankle_study", "RKNE": "r_knee_study", "RMKNE": "r_mknee_study",  
            "LHEE": "L_calc_study",  "LMTOE": "L_5meta_study", "LTOE": "L_toe_study", 
            "LANK": "L_ankle_study", "LMANK": "L_mankle_study", 
            "LKNE": "L_knee_study", "LMKNE": "L_mknee_study", 
            "RASI": "r.ASIS_study", "LASI": "L.ASIS_study",  "RPSI": "r.PSIS_study", "LPSI": "L.PSIS_study", 
            "RBHD": "RBHD","LBHD": "LBHD","RFHD": "RFHD","LFHD": "LFHD",
            "RSHO": "r_shoulder_study",   "LSHO": "L_shoulder_study", "C7": "C7_study", 
            "RWRA": "r_lwrist_study", "RWRB": "r_mwrist_study", "LWRA": "L_lwrist_study",  "LWRB": "L_mwrist_study",  
            "LELB": "L_lelbow_study", "LMELB": "L_melbow_study","RELB": "r_lelbow_study",  "RMELB": "r_melbow_study",
            }
            marekrs_path = f'/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/{which}/{subject}/{task}_filled.csv'


elif which =="Anais":
       mks = {
                "RFCC": "r_calc_study", "RFM5": "r_5meta_study","RFM1": "r_toe_study",  
                "RFAL": "r_ankle_study", "RTAM": "r_mankle_study", "RFLE": "r_knee_study",  "RFME": "r_mknee_study",  
                "LFCC": "L_calc_study", "LFM5": "L_5meta_study","LFM1": "L_toe_study",  
                "LFAL": "L_ankle_study","LTAM": "L_mankle_study","LFLE": "L_knee_study","LFME": "L_mknee_study",  
                "RIAS": "r.ASIS_study","LIAS": "L.ASIS_study","RIPS": "r.PSIS_study", "LIPS": "L.PSIS_study",    
                "HeadR": "REar", "HeadL": "LEar", "SV": "Head",
                "RSAT": "r_shoulder_study", "LSAT": "L_shoulder_study", "CV7": "C7_study",   
                "RRSP": "r_lwrist_study","RUSP": "r_mwrist_study", "LUSP": "L_mwrist_study", "LRSP": "L_lwrist_study", 
                "LHLE": "L_lelbow_study","LHME": "L_melbow_study","RHLE": "r_lelbow_study", "RHME": "r_melbow_study",

                }
       marekrs_path = f'/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/{which}/{subject}/{task}/markers_filled.csv'

else : 
      mks = {
            "RFCC": "RHEE", "RFM5": "R5MHD","RFM1": "RTOE", "RFAL": "RANK","RTAM": "RMANK","RFLE": "RKNE",  "RFME": "RMKNE",  #
            "LFCC": "LHEE", "LFM5": "L5MHD","LFM1": "LTOE","LFAL": "LANK", "LTAM": "LMANK","LFLE": "LKNE",  
            "LFME": "LMKNE",  
            "RIAS": "RASI",   
            "LIAS": "LASI",  
            "RIPS": "RPSI",  
            "LIPS": "LPSI", 

      }
      marekrs_path = f'/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/{which}/{subject}/{task}_markers_filled.csv'

df = pd.read_csv(marekrs_path)
result_markers_old, start_sample_dict_old = read_mks_data(df, start_sample=0, converter=1000.0)
    #mapping names to match cosmik library
current_names = list(start_sample_dict_old.keys())
result_markers = [
    {mks.get(old_name, old_name): data for old_name, data in sample_dict.items()}
    for sample_dict in result_markers_old
]
start_sample_dict = {
    mks[old_name]: start_sample_dict_old[old_name]
    for old_name in mks.keys()
    if old_name in current_names
}
mks_names = start_sample_dict.keys()

###############################################""

urdf_name = "human.urdf"
urdf_meshes_path = "motif/model/human_urdf"
model_h, coll_h, vis_h, _ = build_human_model(urdf_path, urdf_meshes_path)
# human = robex.human.HumanLoader(height=1.55, weight=68.78, gender='male').robot
# model_h = human.model
# data_h = human.data
# coll_h = human.collision_model
# vis_h = human.visual_model


################################################################################LOCK JOINTS
all_joint_ids = set(range(1, model_h.njoints))
joints_to_lock = ["middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z", "left_wrist_X", "left_wrist_Z", "right_wrist_X","right_wrist_Z"]
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
# ###############################################################################################################

pin_body_names = [frame.name for frame in model_h.frames if frame.type == pin.FrameType.BODY]

# Afficher le résultat
print(f"Nombre de frames de type BODY dans Pinocchio : {len(pin_body_names)}")
print("Noms des BODY Pinocchio :")
for name in pin_body_names:
    print(f"- {name}")

data_h = model_h.createData()
#Meshcat
viewer = meshcat.Visualizer()
# Visualizers
viz_human = MeshcatVisualizer(model_h, coll_h, vis_h)
viz_human.initViewer(viewer, open=True)
viz_human.viewer.delete()  # clear if relaunch
viz_human.loadViewerModel("ref",color=[0.0, 1.0, 0.0, 0.8])

# for name in mks_names:
#     add_sphere(viewer, f"world/{name}", radius=0.01, color=0xff0000)

for frame in model_h.frames:
    name = frame.name
    add_sphere(viz_human.viewer, f"world/{name}", radius=0.01, color=0x00ff00)

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




q0 = pin.neutral(model_h)
viz_human.display(q0)

pin.forwardKinematics(model_h, data_h, q0)
markers_fk = {}

angle = -np.pi / 2 
R_corr = np.array([[1, 0,           0          ],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle),  np.cos(angle)]])


data_list = []
for i in range(len(q_ref)):
    frame_data = {}

    q_current = q_ref_df.iloc[i].to_numpy()
    pos_bassin_rnea = q_current[0:3]
    quat_bassin = q_current[3:7] # qx, qy, qz, qw

    quat_original = pin.Quaternion(q_current[6], q_current[3], q_current[4], q_current[5]) #(w,x,y,z) or : q_current[3:7]
    R_original = quat_original.toRotationMatrix()

    R_final = R_corr @ R_original 
    quat_final = pin.Quaternion( R_final)
    
    # q_ref[i][3:7] = [quat_final.x, quat_final.y, quat_final.z, quat_final.w]
    # q_ref[i][0:3] = R_corr @ q_current[0:3]

    pos = pin.forwardKinematics(model_h, data_h, q_ref[i])
    pin.updateFramePlacements(model_h, data_h)
   
    # for name in mks_names:
    #     print(name)
    #     frame_id = model_h.getFrameId(name)
    #     print(frame_id)
    #     marker_pos = data_h.oMf[frame_id].translation 
    #     print(marker_pos)
    #     place(viewer, name, marker_pos)

    for frame in model_h.frames:
        name = frame.name
        frame_id = model_h.getFrameId(name)
        marker_pos = data_h.oMf[frame_id].translation 
        add_sphere(viz_human.viewer, f"world/{name}", radius=0.01, color=0xff0000)
        place(viewer, name, marker_pos)

        frame_data[f"{name}_X"] = marker_pos[0]
        frame_data[f"{name}_Y"] = marker_pos[1]
        frame_data[f"{name}_Z"] = marker_pos[2]
    
    data_list.append(frame_data)

    
    viz_human.display(q_ref[i])


# --- CRÉATION DU CSV ---
df_output = pd.DataFrame(data_list)

# Définir le nom du fichier de sortie (par exemple dans le même dossier que le script)
output_csv_path = f"pin_fk_results_{subject}_{task}_urdf.csv"
df_output.to_csv(output_csv_path, index=False)

print(f"✅ Sauvegarde terminée : {output_csv_path}")
print(f"Dimensions du fichier : {df_output.shape}") # (frames, joints * 3)
