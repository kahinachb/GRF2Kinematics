
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
from utils.viz_utils import add_sphere, place,set_tf, safe_place
from pinocchio import Quaternion
import example_robot_data as robex


subject = 'subject10'

task = 'luyo'
which = 'Anais'
fps = 100  #kinetics_glob_filtered are all 100hz
dt = 1.0 / fps

urdf_path = f"DATA/urdf_scaled/{which}/{subject}_scaled.urdf"# Human base
# urdf_path ='/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/rt-cosmik/urdf/human.urdf'

meshes= ['middle_pelvis_0','left_upperleg_0','right_upperleg_0','left_lowerleg_0','right_lowerleg_0','left_lowerleg_1','right_lowerleg_1',
         'right_foot_0','left_foot_0']

# df_mks= pd.read_csv(f"DATA/Vinc/{subject}/{task}_filled.csv")
# mks_dict, start_sample_dict = read_mks_data(df_mks, start_sample=0, converter=1000.0)
# mks_names = start_sample_dict.keys()


if which =='HUMANOIDS' or which=='Anais' or which=='Vinc':
    path_joint = f"DATA/{which}/{subject}/{task}/joints_filtered_FF.csv"
    q_ref_df = pd.read_csv(path_joint).iloc[1:,:]

    cop_csv = f"DATA/{which}/{subject}/{task}/kinetics_glob_filtered.csv"
    df_cop = pd.read_csv(cop_csv)#.iloc[:,1:]
    print(len(df_cop))
    print(len(q_ref_df))

    X1 = to_m(df_cop[find_col(df_cop, "COPx1_glob")])
    Y1 = to_m(df_cop[find_col(df_cop, "COPy1_glob")])
    X2 = to_m(df_cop[find_col(df_cop, "COPx2_glob")])
    Y2 = to_m(df_cop[find_col(df_cop, "COPy2_glob")])
    Z1 = np.zeros_like(X1)
    Z2 = np.zeros_like(X1)

    Fx1 = df_cop[find_col(df_cop, "Fx1_glob")]
    Fy1 = df_cop[find_col(df_cop, "Fy1_glob")]
    Fz1 = df_cop[find_col(df_cop, "Fz1_glob")]

    Fx2 = df_cop[find_col(df_cop, "Fx2_glob")]
    Fy2 = df_cop[find_col(df_cop, "Fy2_glob")]
    Fz2 = df_cop[find_col(df_cop, "Fz2_glob")]

    cop1 = np.stack([X1, Y1, Z1], axis=1)

    cop2 = np.stack([X2, Y2, Z2], axis=1)

    X1, Y1, Fz1,  X2, Y2,Fz2 = "COPx1_glob","COPy1_glob","Fz1_glob","COPx2_glob","COPy2_glob","Fz2_glob"
    cols_to_filter = [
                        X1]

# elif which =='Vinc':
#     path_joint = f"DATA/{which}/{subject}/{task}/joints_filtered.csv"
#     q_ref_df = pd.read_csv(path_joint)
#     cop_csv = f"DATA/{which}/{subject}/{task}/kinetics_glob_filtered.csv"
#     df_cop = pd.read_csv(cop_csv)#.iloc[:,1:]
#     X1 = to_m(df_cop[find_col(df_cop, "X1")])
#     Y1 = to_m(df_cop[find_col(df_cop, "Y1")])
#     Z1 = to_m(df_cop[find_col(df_cop, "Z1")])

#     X2 = to_m(df_cop[find_col(df_cop, "X2")])
#     Y2 = to_m(df_cop[find_col(df_cop, "Y2")])
#     Z2 = to_m(df_cop[find_col(df_cop, "Z2")])
#     cop1 = np.stack([X1, Y1, Z1], axis=1)
#     cop2 = np.stack([X2, Y2, Z2], axis=1)
#     cols_to_filter = [
#     'X1', 'Y1', 'Z1', 'FZ1',  'X2', 'Y2', 'Z2','FZ2']
    
# else : 
#     path_joint = f"DATA/{which}/{subject}/{task}/joints_filtered.csv"
#     q_ref_df = pd.read_csv(path_joint)
#     cop_csv = f"DATA/{which}/{subject}/{task}/kinetics_glob_filtered.csv"
#     df_cop = pd.read_csv(cop_csv)#.iloc[:,1:]

#     X1 = to_m(df_cop[find_col(df_cop, "CoP1_x")])
#     Y1 = to_m(df_cop[find_col(df_cop, "CoP1_y")])
#     Z1 = to_m(df_cop[find_col(df_cop, "CoP1_z")])
#     X2 = to_m(df_cop[find_col(df_cop, "CoP2_x")])
#     Y2 = to_m(df_cop[find_col(df_cop, "CoP2_y")])
#     Z2 = to_m(df_cop[find_col(df_cop, "CoP2_z")])

#     cop1 = np.stack([X1, Y1, Z1], axis=1)
#     cop2 = np.stack([X2, Y2, Z2], axis=1)

#     cols_to_filter = [
#     'CoP1_x', 'CoP1_y', 'CoP1_z', 'Fz1',  'CoP2_x', 'CoP2_y', 'CoP2_z','Fz2']



q_ref = q_ref_df.to_numpy(dtype=float)
q_ref = lowpass_filter(q_ref, cutoff=2, fs=fps)
for i in range(len(q_ref)):
    q_quat = q_ref[i, 3:7]
    q_ref[i, 3:7] = q_quat / np.linalg.norm(q_quat)

grf_data_filtered = lowpass_filter(df_cop[cols_to_filter].to_numpy(), cutoff=2, fs=fps)
df_cop = df_cop.copy()
df_cop[cols_to_filter] = grf_data_filtered

urdf_name = "human.urdf"
urdf_meshes_path = "motif/model/human_urdf"
model_h, coll_h, vis_h, _ = build_human_model(urdf_path, urdf_meshes_path)
print(model_h.nq)
print(model_h.gravity)
input()
# human = robex.human.HumanLoader(height=1.70, weight=60, gender='male').robot
# model_h = human.model
# data_h = human.data
# coll_h = human.collision_model
# vis_h = human.visual_model

# quat = pin.Quaternion(pin.rpy.rpyToMatrix(np.deg2rad(90), 0, 0)).coeffs()#set the human model uprigth


# ################################################################################LOCK JOINTS
# all_joint_ids = set(range(1, model_h.njoints))
 

# joints_to_lock = ["middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z", "left_wrist_X", "left_wrist_Z", "right_wrist_X","right_wrist_Z",
#                   "middle_lumbar_X" ,
#     "middle_lumbar_Z",
#     "left_clavicle_joint_X",
#     "left_shoulder_Z"     ,
#     "left_shoulder_X"     ,
#     "left_shoulder_Y"     ,
#     "left_elbow_Z"       ,
#     "left_elbow_Y"        ,
#     "middle_cervical_Z"   ,
#     "middle_cervical_X"   ,
#     "middle_cervical_Y"   ,
#     "right_clavicle_joint_X",
#     "right_shoulder_Z"    ,
#     "right_shoulder_X"    ,
#     "right_shoulder_Y"    ,
#     "right_elbow_Z"     ,
#     "right_elbow_Y"     ]
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
###############################################################################################################


data_h = model_h.createData()


import meshcat.geometry as g

def draw_force_arrow(viewer, name, cop, force, color=0xff0000, scale=0.001):
    """
    Affiche une ligne partant du COP pour représenter la force.
    color: format hexadécimal (ex: 0xff0000 pour rouge)
    """
    # Calcul du point d'arrivée
    end_point = cop + (force * scale)
    
    # Création des points de la ligne (doit être un array 3x2)
    points = np.array([cop, end_point]).astype(np.float32).T
    
    # Envoi au viewer Meshcat
    viewer[name].set_object(g.Line(g.PointsGeometry(points), 
                                   g.LineBasicMaterial(color=color, linewidth=3)))

#Meshcat
viewer = meshcat.Visualizer()
# Visualizers
viz_human = MeshcatVisualizer(model_h, coll_h, vis_h)
viz_human.initViewer(viewer, open=True)
viz_human.viewer.delete()  # clear if relaunch
viz_human.loadViewerModel("ref",color=[0.0, 1.0, 0.0, 0.8])

q0 = pin.neutral(model_h)
# q0[3:7]=quat
viz_human.display(q0)

add_sphere(viewer, "world/pos_bassin_RNEA", radius=0.05, color=0xFF0000) 
meshcat_shapes.frame(viewer["world/pos_bassin_RNEA/frame"], axis_length=0.2)

# COP
add_sphere(viewer, "world/COP_right",  radius=0.015, color=0x0000FF)  # bleu 
add_sphere(viewer, "world/COP_left", radius=0.015, color=0xFF8800)  # orange
add_sphere(viewer, "world/COP_RNEA",  radius=0.015, color=0xFF0000)  # red 
add_sphere(viewer, "world/COP_platform_global", radius=0.015, color=0x00FF00)  # green

# for name in mks_names:

#     add_sphere(viewer, f"world/{name}", radius=0.01, color=0xff0000)


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
T1 = data_h.oMf[model_h.getJointId('root_joint')].translation
R1 = data_h.oMf[model_h.getJointId('root_joint')].rotation

n_samples = len(q_ref)
nv = model_h.nv
v_ref = np.zeros((n_samples, nv))
a_ref = np.zeros((n_samples, nv))
for i in range(n_samples - 1):
    v_ref[i, :] = pin.difference(model_h, q_ref[i, :], q_ref[i+1, :]) / dt
for i in range(n_samples - 1):
    a_ref[i, :] = (v_ref[i+1, :] - v_ref[i, :]) / dt

angle = -np.pi / 2 
R_corr = np.array([[1, 0,           0          ],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle),  np.cos(angle)]])


for i in range(len(q_ref)):

    # q_current = q_ref_df.iloc[i].to_numpy()
    # pos_bassin_rnea = q_current[0:3]
    # quat_bassin = q_current[3:7] # qx, qy, qz, qw

    # quat_original = pin.Quaternion(q_current[6], q_current[3], q_current[4], q_current[5]) #(w,x,y,z) or : q_current[3:7]
    # R_original = quat_original.toRotationMatrix()

    # T_bassin = np.eye(4)

    # R_final = R_original 
    # quat_final = pin.Quaternion(R_final)

    # T_bassin[:3, :3] = R_final
    # T_bassin[:3, 3] = pos_bassin_rnea
    # place(viewer, "pos_bassin_RNEA", T_bassin[:3, 3])
    # set_tf(viewer, "pos_bassin_RNEA/frame", T_bassin[:3, :3])
    
    # q_ref[i][3:7] = [quat_final.x, quat_final.y, quat_final.z, quat_final.w]
    # q_ref[i][0:3] =q_current[0:3]

    # quat = pin.Quaternion(q_ref[i, 3:7])
    # rot_base= quat.matrix()
    # pos_bassin = q_ref[i, 0:3]

    # tau = pin.rnea(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])
    # pin.forwardKinematics(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])
    # wrench_base = data_h.f[1]         
    # oM1 = data_h.oMi[1]              
    # f_world = oM1.act(wrench_base) 

    
    # F = f_world.linear
    # M = f_world.angular
    # M = (rot_base @ tau[3:6] ) + np.cross(pos_bassin , F)



    # Fx, Fy, Fz = F
    # Mx, My, Mz = M

    # cop_x = -My / Fz
    # cop_y =  Mx / Fz
    # cop_z = 0.0
    # cop_rnea = np.array([cop_x, cop_y, cop_z])
   
    cop_r = cop1[i]  # (x,y,z)
    cop_l = cop2[i]

    if which == 'HUMANOIDS' or which=='Anais' or which=='Vinc': 
        Fz_r = df_cop[find_col(df_cop, "Fz1_glob")].values.astype(float)
        Fz_l = df_cop[find_col(df_cop, "Fz2_glob")].values.astype(float)

        Fx_r = df_cop[find_col(df_cop, "Fx1_glob")].values.astype(float)
        Fx_l = df_cop[find_col(df_cop, "Fx2_glob")].values.astype(float)

        Fy_r = df_cop[find_col(df_cop, "Fy1_glob")].values.astype(float)
        Fy_l = df_cop[find_col(df_cop, "Fy2_glob")].values.astype(float)

    # elif which =='Vinc':
    #     Fz_r = df_cop[find_col(df_cop, "FZ1")].values.astype(float)
    #     Fz_l = df_cop[find_col(df_cop, "FZ2")].values.astype(float)

    #     Fx_r = df_cop[find_col(df_cop, "FX1")].values.astype(float)
    #     Fx_l = df_cop[find_col(df_cop, "FX2")].values.astype(float)

    #     Fy_r = df_cop[find_col(df_cop, "FY1")].values.astype(float)
    #     Fy_l = df_cop[find_col(df_cop, "FY2")].values.astype(float)

        
    # else : 
    #     Fz_r = df_cop[find_col(df_cop, "Fz2")].values.astype(float)
    #     Fz_l = df_cop[find_col(df_cop, "Fz1")].values.astype(float)

    #     Fx_r = df_cop[find_col(df_cop, "Fx1")].values.astype(float)
    #     Fx_l = df_cop[find_col(df_cop, "Fx2")].values.astype(float)

    #     Fy_r = df_cop[find_col(df_cop, "Fy1")].values.astype(float)
    #     Fy_l = df_cop[find_col(df_cop, "Fy2")].values.astype(float)

    Fz_total = Fz_r + Fz_l

    cop_global = (Fz_r[i] * cop_r + Fz_l[i] * cop_l) / Fz_total[i]

    #     # 1. Préparation des vecteurs (vérifie bien que ce sont des np.array de taille 3)
    force_r = np.array([Fx_r[i], Fy_r[i], Fz_r[i]])
    force_l = np.array([Fx_l[i], Fy_l[i], Fz_l[i]])

    # # 2. Échelle de la flèche (ex: 1000N = 1m)
    f_scale = 0.001 

    if abs(Fz_r[i]) > 10.0:
        draw_force_arrow(viewer, "force_R", cop1[i], force_r, color=0x0000ff, scale=f_scale)
    else:
        viewer["force_R"].delete() 


    if abs(Fz_l[i]) > 10.0:
        draw_force_arrow(viewer, "force_L", cop2[i], force_l, color=0xFF8800, scale=f_scale)
    else:
        viewer["force_L"].delete()

    # # Affichage de la force totale (Vert)
    force_total = force_r + force_l
    if abs(Fz_total[i]) > 10.0:
        draw_force_arrow(viewer, "force_Total", cop_global, force_total, color=0x00ff00, scale=f_scale)
            

    # safe_place(viewer,"COP_RNEA", cop_rnea)
    # safe_place(viewer, "COP_platform_global", cop_global)

    safe_place(viewer,"COP_right",  cop1[i])
    safe_place(viewer, "COP_left", cop2[i])
    # frame = mks_dict[i]
    # for name in mks_names:
    #     pos= frame[name].reshape(3,)
    #     place(viewer, name, pos)


    viz_human.display(q_ref[i])
