
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
from utils.viz_utils import add_sphere, place,set_tf, safe_place
from utils.model_utils import get_foot_pose
import matplotlib.pyplot as plt

subject = 'Jeremy'
task = 'Trial111'
which = 'Vinc'
fps = 100  #kinetics_glob_filtered are all 100hz
dt = 1.0 / fps

# urdf_path = f"DATA/urdf_scaled/{which}/{subject}_scaled.urdf"# Human base
# path_joint = f"DATA/{which}/{subject}/{task}/joints_filtered_FF.csv"
# cop_csv = f"DATA/{which}/{subject}/{task}/kinetics_glob_filtered.csv"


urdf_path ="DATA/10_urdf/human_subject_06.urdf"
path_joint = f"DATA/generated_data/subject_06_squat_variant_000_dz+0.025_dx+0.070_dy+0.020_q.csv"
cop_csv = f"DATA/generated_data/subject_06_squat_variant_000_dz+0.025_dx+0.070_dy+0.020_grfm.csv"


# urdf_path ="DATA/urdf_scaled/URDFS/subject_001_77.4kg.urdf"
# path_joint = f"DATA/generated_102/subject_102_squat_variant_000_dz+0.075_dx-0.055_dy+0.050_q.csv"
# cop_csv = f"DATA/generated_102/subject_102_squat_variant_000_dz+0.075_dx-0.055_dy+0.050_grfm.csv"


meshes= ['middle_pelvis_0','left_upperleg_0','right_upperleg_0','left_lowerleg_0','right_lowerleg_0','left_lowerleg_1','right_lowerleg_1',
         'right_foot_0','left_foot_0']

q_ref_df = pd.read_csv(path_joint).iloc[1:,:]
q_ref = q_ref_df.to_numpy(dtype=float)

df_cop = pd.read_csv(cop_csv)#.iloc[:,1:]
pelvis =False

path_vrai = "DATA/Vinc/Jeremy/Trial111/kinetics_glob_filtered.csv"
df_vrai = pd.read_csv(path_vrai)
Fx1 = df_cop[find_col(df_cop, "Fx1_glob")]
Fy1 = df_cop[find_col(df_cop, "Fy1_glob")]
Fz1 = df_cop[find_col(df_cop, "Fz1_glob")]
Mx1 = df_cop[find_col(df_cop, "Mx1_glob")]
My1 = df_cop[find_col(df_cop, "My1_glob")]
Mz1 = df_cop[find_col(df_cop, "Mz1_glob")]
copx1 = df_cop[find_col(df_cop, "COPx1_glob")]
copy1 = df_cop[find_col(df_cop, "COPy1_glob")]
copz1 = df_cop[find_col(df_cop, "COPz1_glob")]

Fx2 = df_cop[find_col(df_cop, "Fx2_glob")]
Fy2 = df_cop[find_col(df_cop, "Fy2_glob")]
Fz2 = df_cop[find_col(df_cop, "Fz2_glob")]
Mx2 = df_cop[find_col(df_cop, "Mx2_glob")]
My2 = df_cop[find_col(df_cop, "My2_glob")]
Mz2 = df_cop[find_col(df_cop, "Mz2_glob")]
copx2 = df_cop[find_col(df_cop, "COPx2_glob")]
copy2 = df_cop[find_col(df_cop, "COPy2_glob")]
copz2 = df_cop[find_col(df_cop, "COPz2_glob")]



urdf_name = "human.urdf"
urdf_meshes_path = "motif/model/human_urdf"
model_h, coll_h, vis_h, _ = build_human_model(urdf_path, urdf_meshes_path)
print(model_h.nq)

data_h = model_h.createData()
##################################################


needed_markers = [
    'r_mankle_study', 'r_ankle_study', 'r_toe_study', 'r_5meta_study', 'r_calc_study',
    'L_mankle_study', 'L_ankle_study', 'L_toe_study', 'L_5meta_study', 'L_calc_study'
]


############################################################""
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

for side in ["R", "L"]:
    name = f"Foot_{side}"
    # On crée la forme des axes
    meshcat_shapes.frame(
        viewer[name],
        axis_length=0.2,   
        axis_thickness=0.01,
        opacity=1
    )
name ="pelvis"
meshcat_shapes.frame(
        viewer[name],
        axis_length=0.2,   
        axis_thickness=0.01,
        opacity=1
    )
###########################################################################
q0 = pin.neutral(model_h)
# q0[3:7]=quat

# COP
add_sphere(viewer, "world/COP_left",  radius=0.015, color=0x0000FF)  # bleu 
add_sphere(viewer, "world/COP_right", radius=0.015, color=0xFF8800)  # orange
add_sphere(viewer, "world/COP_RNEA",  radius=0.015, color=0xFF0000)  # red 
add_sphere(viewer, "world/COP_platform_global", radius=0.015, color=0x00FF00)  # green


add_sphere(viewer, "world/pelvis_sol", radius=0.020, color=0x00FFFF) 



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

results = []
results_feet = []
data_plot = {
    'R': {'F': [], 'M': [], 'COP': []},
    'L': {'F': [], 'M': [], 'COP': []}
}

cop_fp =[]
cop_rnea_list=[]

force_fp = []
force_rnea =[]

moment_fp=[]
moment_rnea=[]

force_vrai =[]
moment_vrai=[]
for i in range(len(q_ref)):
    # viz_human.display(q_ref[i])

    q_current = q_ref[i, :]
    pos_bassin = q_current[0:3]
    quat_bassin = q_current[3:7] # qx, qy, qz, qw

    quat_original = pin.Quaternion(q_current[6], q_current[3], q_current[4], q_current[5]) #(w,x,y,z) or : q_current[3:7]
    R = quat_original.toRotationMatrix() #r world to pelvis
    R_pelvis_world = R.T

    T_bassin = np.eye(4)
    R_final = R 
    quat_final = pin.Quaternion(R_final)
    T_bassin[:3, :3] = R_final
    T_bassin[:3, 3] = pos_bassin

    pos_bassin_proj = pos_bassin.copy()
    pos_bassin_proj[2] = 0.0
    safe_place(viewer,"pelvis_sol", pos_bassin_proj)

    set_tf(viewer, "pelvis",T_bassin)
    
##################"rnea"
    tau = pin.rnea(model_h, data_h, q_current, v_ref[i, :], a_ref[i, :])
    pin.forwardKinematics(model_h, data_h, q_current, v_ref[i, :], a_ref[i, :])
    pin.updateFramePlacements(model_h, data_h)

    wrench_base = data_h.f[1]         
    oM1 = data_h.oMi[1]              
    f_world = oM1.act(wrench_base) 
    
    F = f_world.linear
    M = f_world.angular
    # M = (R @ tau[3:6] ) + np.cross(pos_bassin , F)
    Fx, Fy, Fz = F
    Mx, My, Mz = M

    cop_x = -My / Fz
    cop_y =  Mx / Fz
    cop_z = 0.0
    cop_rnea = np.array([cop_x, cop_y, cop_z])


    F_world1 = np.array([Fx1[i], Fy1[i], Fz1[i]])
    F_world2 = np.array([Fx2[i], Fy2[i], Fz2[i]])

    M_world1= np.array([Mx1[i], My1[i], Mz1[i]])
    M_world2= np.array([Mx2[i], My2[i], Mz2[i]])

    F_1 = np.array([df_vrai[find_col(df_vrai, "Fx1_glob")][i], df_vrai[find_col(df_vrai, "Fy1_glob")][i], df_vrai[find_col(df_vrai, "Fz1_glob")][i]])
    F_2 = np.array([df_vrai[find_col(df_vrai, "Fx2_glob")][i], df_vrai[find_col(df_vrai, "Fy2_glob")][i], df_vrai[find_col(df_vrai, "Fz2_glob")][i]])

    M_1=  np.array([df_vrai[find_col(df_vrai, "Mx1_glob")][i], df_vrai[find_col(df_vrai, "My1_glob")][i], df_vrai[find_col(df_vrai, "Mz1_glob")][i]])
    M_2= np.array([df_vrai[find_col(df_vrai, "Mx2_glob")][i], df_vrai[find_col(df_vrai, "My2_glob")][i], df_vrai[find_col(df_vrai, "Mz2_glob")][i]])

    F_total_vrai = F_1 +F_2
    M_total_vrai = M_1 + M_2
    force_vrai.append(F_total_vrai)
    moment_vrai.append(M_total_vrai)

    ###cop world 
    cop_x1 = -M_world1[1] / F_world1[2]
    cop_y1 = M_world1[0] / F_world1[2]
    cop_z1 = 0.0
    cop_x2 = -M_world2[1] / F_world2[2]
    cop_y2 =  M_world2[0] / F_world2[2]
    cop_z2 = 0.0
    cop_world1 = np.array([cop_x1, cop_y1, cop_z1])
    cop_world2 = np.array([cop_x2, cop_y2, cop_z2])
    #from file 
    # cop_world1 = np.array([copx1[i], copy1[i], copz1[i]])
    # cop_world2 = np.array([copx2[i], copy2[i], copz2[i]])


    F_total = F_world1  + F_world2
    Fz_total = F_world1[2]+ F_world2[2]
    gravity = abs(model_h.gravity.linear[2]) 
    print("Fz_total en kg",Fz_total/gravity)
    print(f"urdf weight : {pin.computeTotalMass(model_h):.2f} kg")
    # input()
    force_fp.append(F_total)
    force_rnea.append(F)

    M_total = M_world1+ M_world2
    moment_fp.append(M_total)
    moment_rnea.append(M)



    cop_global = (F_world1[2] * cop_world1 + F_world2[2] * cop_world2) / Fz_total

    safe_place(viewer, "COP_right", cop_world1)
    draw_force_arrow(viewer, "force_right", cop_world1, F_world1,color=0x00ff00)

    safe_place(viewer, "COP_left", cop_world2)
    draw_force_arrow(viewer, "force_left", cop_world2, F_world2,color=0x0000ff)

    safe_place(viewer, "COP_platform_global", cop_global)
    safe_place(viewer,"COP_RNEA", cop_rnea)
    cop_fp.append(cop_global)
    
    cop_rnea_list.append(cop_rnea)


    mks_positions = {}
    for name in needed_markers:
        if model_h.existFrame(name):
            f_id = model_h.getFrameId(name)
            mks_positions[name] = data_h.oMf[f_id].translation
            add_sphere(viz_human.viewer, f"world/{name}", radius=0.01, color=0xff0000)
            place(viewer, name, data_h.oMf[f_id].translation)
        else:
            print(f"Warning: Frame {name} not found in model")

    T_w_footR = get_foot_pose(mks_positions, side='right')
    T_w_footL = get_foot_pose(mks_positions, side='left')
    set_tf(viewer, "Foot_R", T_w_footR)
    set_tf(viewer, "Foot_L", T_w_footL)


    R_footR = T_w_footR[:3, :3]
    P_footR = T_w_footR[:3, 3] 
    F_localR = R_footR.T @ F_world1
    M_localR = R_footR.T @ (M_world1 - np.cross(P_footR, F_world1))
    cop_localR = R_footR.T @ (cop_world1 - P_footR)

    R_footL = T_w_footL[:3, :3]
    P_footL = T_w_footL[:3, 3] 
    F_localL = R_footL.T @ F_world2
    M_localL = R_footR.T @ (M_world2 - np.cross(P_footL, F_world2))
    cop_localL = R_footL.T @ (cop_world2 - P_footL)

    data_plot['R']['F'].append(F_localR)
    data_plot['R']['M'].append(M_localR)
    data_plot['R']['COP'].append(cop_localR)
    
    data_plot['L']['F'].append(F_localL)
    data_plot['L']['M'].append(M_localL)
    data_plot['L']['COP'].append(cop_localL)





    viz_human.display(q_ref[i])
    # input()
    # time.sleep(0.001)

cop_fp = np.array(cop_fp)
cop_rnea_list = np.array(cop_rnea_list)
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

labels = ["X", "Y", "Z"]

for i in range(3):
    axs[i].plot(cop_fp[:, i], label="FP")
    axs[i].plot(cop_rnea_list[:, i], label="RNEA")
    axs[i].set_title(f"COP {labels[i]}")
    axs[i].legend()

plt.tight_layout()
plt.show()

force_vrai = np.array(force_vrai)
force_fp = np.array(force_fp)
force_rnea = np.array(force_rnea)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
for i in range(3):
    axs[i].plot(force_fp[:, i], label="FP")
    axs[i].plot(force_rnea[:, i], label="RNEA")
    axs[i].plot(force_vrai[:, i], label="True", color="k")
    axs[i].set_title(f"F {labels[i]}")
    axs[i].legend()

# Force vraie
# for i in range(3):
#     axs[i + 3].plot(force_vrai[:, i], label="True", color="k")
#     axs[i + 3].set_title(f"Force {labels[i]} (True)")
#     axs[i + 3].legend()

plt.tight_layout()
plt.show()

moment_vrai = np.array(moment_vrai)
moment_fp = np.array(moment_fp)
moment_rnea = np.array(moment_rnea)
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
for i in range(3):
    axs[i].plot(moment_fp[:, i], label="FP",linewidth=3)
    axs[i].plot(moment_rnea[:, i], label="RNEA")
    axs[i].plot(moment_vrai[:, i], label="True", color="k")
    axs[i].set_title(f"M {labels[i]}")
    axs[i].legend()

# Force vraie
# for i in range(3):
#     axs[i + 3].plot(moment_vrai[:, i], label="True", color="k")
#     axs[i + 3].set_title(f"Moment {labels[i]} (True)")
#     axs[i + 3].legend()

plt.tight_layout()
plt.show()

def plot_side_data(side_key, full_name):
    # Conversion en numpy arrays pour faciliter le slicing [:, 0]
    F = np.array(data_plot[side_key]['F'])
    M = np.array(data_plot[side_key]['M'])
    COP = np.array(data_plot[side_key]['COP'])
    time_axis = np.arange(len(F)) * dt

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'{full_name}', fontsize=16)

    # Subplot 1 : Forces
    axs[0].plot(time_axis, F[:, 0], label='Fx (Ant-Post)', color='red')
    axs[0].plot(time_axis, F[:, 1], label='Fy (Vertical)',color='green')
    axs[0].plot(time_axis, F[:, 2], label='Fz (Med-Lat)',color='blue')
    axs[0].set_ylabel('Force [N]')
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2 : Moments
    axs[1].plot(time_axis, M[:, 0], label='Mx',color='red')
    axs[1].plot(time_axis, M[:, 1], label='My',color='green')
    axs[1].plot(time_axis, M[:, 2], label='Mz',color='blue')
    axs[1].set_ylabel('Moment [Nm]')
    axs[1].legend()
    axs[1].grid(True)

    # Subplot 3 : COP
    axs[2].plot(time_axis, COP[:, 0], label='COPx', color='red')
    axs[2].plot(time_axis, COP[:, 1], label='COPy (dist cheville-sol)',color='green')
    axs[2].plot(time_axis, COP[:, 2], label='COPz',color='blue')
    axs[2].set_ylabel('COP [m]')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Appel pour les deux côtés
plot_side_data('R', 'Right')
plot_side_data('L', 'Left')

# df_results = pd.DataFrame(results_feet)
# output_path = cop_csv.replace(".csv", "_feet_frame.csv")
# df_results.to_csv(output_path, index=False)
# print(f"Données sauvegardées dans : {output_path}")