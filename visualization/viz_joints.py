
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


subject = 'subject13'
task = 'cmjs'

fps = 300 
dt = 1.0 / fps

meshes= ['middle_pelvis_0','left_upperleg_0','right_upperleg_0','left_lowerleg_0','right_lowerleg_0','left_lowerleg_1','right_lowerleg_1',
         'right_foot_0','left_foot_0']

path_joint = f"DATA/Anais/{subject}/{task}/joints.csv"
q_ref_df = pd.read_csv(path_joint).iloc[:,1:]
q_ref = q_ref_df.to_numpy(dtype=float)
q_ref = lowpass_filter(q_ref, cutoff=2, fs=fps)
for i in range(len(q_ref)):
    q_quat = q_ref[i, 3:7]
    q_ref[i, 3:7] = q_quat / np.linalg.norm(q_quat)


cop_csv = f"DATA/Anais/{subject}/{task}/kinetics.csv"
df_cop = pd.read_csv(cop_csv).iloc[:,1:]

if fps == 100:
    X1 = to_m(df_cop[find_col(df_cop, "X1")])
    Y1 = to_m(df_cop[find_col(df_cop, "Y1")])
    Z1 = to_m(df_cop[find_col(df_cop, "Z1")])
    X2 = to_m(df_cop[find_col(df_cop, "X2")])
    Y2 = to_m(df_cop[find_col(df_cop, "Y2")])
    Z2 = to_m(df_cop[find_col(df_cop, "Z2")])
    cop1 = np.stack([X1, Y1, Z1], axis=1)
    cop2 = np.stack([X2, Y2, Z2], axis=1)
    cols_to_filter = [
    'X1', 'Y1', 'Z1', 'FZ1',  'X2', 'Y2', 'Z2','FZ2']

else : 
    X1 = to_m(df_cop[find_col(df_cop, "CoP1_x")])
    Y1 = to_m(df_cop[find_col(df_cop, "CoP1_y")])
    Z1 = to_m(df_cop[find_col(df_cop, "CoP1_z")])
    X2 = to_m(df_cop[find_col(df_cop, "CoP2_x")])
    Y2 = to_m(df_cop[find_col(df_cop, "CoP2_y")])
    Z2 = to_m(df_cop[find_col(df_cop, "CoP2_z")])
    cop1 = np.stack([X1, Y1, Z1], axis=1)
    cop2 = np.stack([X2, Y2, Z2], axis=1)
    cols_to_filter = [
    'CoP1_x', 'CoP1_y', 'CoP1_z', 'Fz1',  'CoP2_x', 'CoP2_y', 'CoP2_z','Fz2']

grf_data_filtered = lowpass_filter(df_cop[cols_to_filter].to_numpy(), cutoff=2, fs=fps)
df_cop = df_cop.copy()
df_cop[cols_to_filter] = grf_data_filtered

urdf_name = "human.urdf"
urdf_path = f"DATA/urdf_scaled/{subject}_scaled.urdf"# Human base
urdf_meshes_path = "motif/model/human_urdf/meshes"
model_h, coll_h, vis_h, _ = build_human_model(urdf_path, urdf_meshes_path)
data_h = model_h.createData()

all_names = model_h.names
print("Tous les noms de joints + segments :")
print(all_names)

#Meshcat
viewer = meshcat.Visualizer()
# Visualizers
viz_human = MeshcatVisualizer(model_h, coll_h, vis_h)
viz_human.initViewer(viewer, open=True)
viz_human.viewer.delete()  # clear if relaunch
viz_human.loadViewerModel("ref",color=[0.0, 1.0, 0.0, 0.8])




add_sphere(viewer, "world/pos_bassin_RNEA", radius=0.05, color=0xFF0000) 
meshcat_shapes.frame(viewer["world/pos_bassin_RNEA/frame"], axis_length=0.2)

# COP
add_sphere(viewer, "world/COP_right",  radius=0.015, color=0x0000FF)  # bleu 
add_sphere(viewer, "world/COP_left", radius=0.015, color=0xFF8800)  # orange
add_sphere(viewer, "world/COP_RNEA",  radius=0.015, color=0xFF0000)  # bleu 
add_sphere(viewer, "world/COP_platform_global", radius=0.015, color=0x00FF00)  # orange



for geom in vis_h.geometryObjects:
    node_name = viz_human.getViewerNodeName(geom, pin.GeometryType.VISUAL)
    viz_human.viewer[node_name].set_property("visible", False)
for geom in vis_h.geometryObjects:
    for mesh in meshes:
        if mesh in geom.name:   
            node_name = viz_human.getViewerNodeName(geom, pin.GeometryType.VISUAL)
            viz_human.viewer[node_name].set_property("visible", True)


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

angle = np.pi / 2 
R_corr = np.array([[1, 0,           0          ],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle),  np.cos(angle)]])

for i in range(len(q_ref)):

    q_current = q_ref_df.iloc[i].to_numpy()
    pos_bassin_rnea = q_current[0:3]
    quat_bassin = q_current[3:7] # qx, qy, qz, qw

    quat_original = pin.Quaternion(q_current[6], q_current[3], q_current[4], q_current[5]) #(w,x,y,z) or : q_current[3:7]
    R_original = quat_original.toRotationMatrix()

    T_bassin = np.eye(4)

    R_final = R_corr @ R_original 
    quat_final = pin.Quaternion(R_final)

    T_bassin[:3, :3] = R_final
    T_bassin[:3, 3] = R_corr@ pos_bassin_rnea
    place(viewer, "pos_bassin_RNEA", T_bassin[:3, 3])
    set_tf(viewer, "pos_bassin_RNEA/frame", T_bassin[:3, :3])
    
    q_ref[i][3:7] = [quat_final.x, quat_final.y, quat_final.z, quat_final.w]
    q_ref[i][0:3] = R_corr @ q_current[0:3]

    quat = pin.Quaternion(q_ref[i, 3:7])
    rot_base= quat.matrix()
    pos_bassin = q_ref[i, 0:3]

    tau = pin.rnea(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])
    pin.forwardKinematics(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])
    wrench_base = data_h.f[1]         
    oM1 = data_h.oMi[1]              
    f_world = oM1.act(wrench_base) 

    
    F = f_world.linear
    M = f_world.angular
    M = (rot_base @ tau[3:6] ) + np.cross(pos_bassin , F)



    Fx, Fy, Fz = F
    Mx, My, Mz = M

    cop_x = -My / Fz
    cop_y =  Mx / Fz
    cop_z = 0.0
    cop_rnea = np.array([cop_x, cop_y, cop_z])
   
    cop_r = cop1[i]  # (x,y,z)
    cop_l = cop2[i]

    if fps == 100 : 
        Fz_r = df_cop[find_col(df_cop, "FZ1")].values.astype(float)
        Fz_l = df_cop[find_col(df_cop, "FZ2")].values.astype(float)
    else : 
        Fz_r = df_cop[find_col(df_cop, "Fz2")].values.astype(float)
        Fz_l = df_cop[find_col(df_cop, "Fz1")].values.astype(float)

    Fz_total = Fz_r + Fz_l

    cop_global = (Fz_r[i] * cop_r + Fz_l[i] * cop_l) / Fz_total[i]


    safe_place(viewer,"COP_RNEA", cop_rnea)
    safe_place(viewer, "COP_platform_global", cop_global)

    safe_place(viewer,"COP_right",  cop1[i])
    safe_place(viewer, "COP_left", cop2[i])


    viz_human.display(q_ref[i])
