
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
from utils.utils import find_col


urdf_path = "DATA/urdf_scaled/Vinc/Christine_scaled.urdf"# Human base
path_joint = "DATA/Vinc/Christine/Trial110/joints_filtered_FF.csv"
q_ref_df = pd.read_csv(path_joint)
q_ref = q_ref_df.to_numpy(dtype=float)
urdf_name = "human.urdf"
urdf_meshes_path = "motif/model/human_urdf"
model_h, coll_h, vis_h, _ = build_human_model(urdf_path, urdf_meshes_path)
print(model_h.nq)
print(model_h.gravity)

data_h = model_h.createData()

df = pd.read_csv("/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Vinc/Christine/Trial110.csv")

mks_dict, start_sample_dict = read_mks_data(df, start_sample=0, converter=1000.0)
mks_names = start_sample_dict.keys()
import meshcat.geometry as g

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



for name in mks_names:

    add_sphere(viewer, f"world/{name}", radius=0.01, color=0xff0000)


for i in range(len(q_ref)):

    # for i, frame in enumerate(mks_dict):
    
    #     for name in mks_names:
    #         pos = frame[name].reshape(3,)
    #         place(viewer, name, pos)

            viz_human.display(q_ref[i])
