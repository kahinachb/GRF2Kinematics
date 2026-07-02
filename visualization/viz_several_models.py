
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

fx1, fy1, fz1 = 'Fx1_glob', 'Fy1_glob', 'Fz1_glob'
mx1,my1,mz1 =  'Mx1_glob', 'My1_glob', 'Mz1_glob'
fx2, fy2, fz2 = 'Fx2_glob', 'Fy2_glob', 'Fz2_glob'
mx2,my2,mz2 =  'Mx2_glob', 'My2_glob', 'Mz2_glob'


subject = 'Jeremy'

task = 'Trial111'
which = 'Vinc'
fps = 100  #kinetics_glob_filtered are all 100hz
dt = 1.0 / fps

urdf_path = f"DATA/urdf_scaled/{which}/{subject}_scaled.urdf"
path_grf = f"DATA/{which}/{subject}/{task}/kinetics_glob_filtered.csv"
path_joint = f"DATA/{which}/{subject}/{task}/joints_filtered_FF.csv"

grf_df = pd.read_csv(path_grf)
q_df = pd.read_csv(path_joint)
q1 = q_df.to_numpy(dtype=float)


urdf_2 = "DATA/urdf/human_subject_08.urdf"
urdf_2 =f"DATA/urdf_scaled/{which}/{subject}_scaled.urdf"
path_grf2 = "DATA/generated_data/subject_11_squat_variant_082_dz+0.006_dx+0.032_dy+0.013_grfm.csv"
path_joint2 = "DATA/generated_data/subject_11_squat_variant_082_dz+0.006_dx+0.032_dy+0.013_q.csv"

grf_df2 = pd.read_csv(path_grf2)
q_df2 = pd.read_csv(path_joint2)
q2 = q_df2.to_numpy(dtype=float)

urdf_meshes_path = "motif/model/human_urdf"
model_h, coll_h, vis_h, _ = build_human_model(urdf_path, urdf_meshes_path)

model_h2, coll_h2, vis_h2, _ = build_human_model(urdf_2, urdf_meshes_path)

data_h = model_h.createData()
data_h2 = model_h2.createData()

#Meshcat
viewer = meshcat.Visualizer()

# Human 1
viz_human = MeshcatVisualizer(model_h, coll_h, vis_h)
viz_human.initViewer(viewer, open=True)
viz_human.viewer.delete()  # clear if relaunch
viz_human.loadViewerModel("h1", color=[0.5, 0.5, 0.5, 0.5])

# Human 2
viz_human2 = MeshcatVisualizer(model_h2, coll_h2, vis_h2)
viz_human2.initViewer(viewer, open=True)
viz_human2.loadViewerModel("h2", color=[0.0, 1.0, 0.0, 0.8])
###########################################################################
q0 = pin.neutral(model_h)


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

n_samples = len(q1)
nv = model_h.nv
print("q shape:", q1.shape)
print("nv:", nv)



images = []

F1_world = []
M1_world = []

F2_world = []
M2_world = []

size = min (len(q1),len(q2))
for i in range(size):

    Fg = grf_df.loc[i, [fx1,fy1,fz1]].values
    Fd = grf_df.loc[i, [fx2,fy2,fz2]].values
    Mg = grf_df.loc[i, [mx1,my1,mz1]].values 
    Md = grf_df.loc[i, [mx2,my2,mz2]].values 
    F1 = Fg + Fd
    M1 = Mg + Md

    F1_world.append(F1)
    M1_world.append(M1)

    Fg = grf_df2.loc[i, [fx1,fy1,fz1]].values
    Fd = grf_df2.loc[i, [fx2,fy2,fz2]].values
    Mg = grf_df2.loc[i, [mx1,my1,mz1]].values 
    Md = grf_df2.loc[i, [mx2,my2,mz2]].values 
    F2 = Fg + Fd
    M2 = Mg + Md

    F2_world.append(F2)
    M2_world.append(M2)

    

  
    viz_human.display(q1[i])
    viz_human2.display(q2[i])

#     images.append(viewer.get_image())




F1_world = np.array(F1_world)
M1_world = np.array(M1_world)

import matplotlib.pyplot as plt
t = np.arange(len(F1_world))
t = np.arange(4000)
fig, axs = plt.subplots(2, 3, figsize=(15, 7))

labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

data_FM1 = np.hstack([F1_world, M1_world])
data_FM2 = np.hstack([F2_world, M2_world])


for j, ax in enumerate(axs.flatten()):

    
    ax.plot(t, data_FM1[:4000, j], label="Force plate")
    ax.plot(t, data_FM2[:4000, j], label="Force plate")

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
