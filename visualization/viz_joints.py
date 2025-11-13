import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils.model_utils import *
script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(os.path.dirname(script_directory))
import meshcat
import meshcat_shapes
from pinocchio.visualize import MeshcatVisualizer
import pandas as pd
from utils.utils import read_mks_data

meshes= ['middle_pelvis_0','left_upperleg_0','right_upperleg_0','left_lowerleg_0','right_lowerleg_0','left_lowerleg_1','right_lowerleg_1',
         'right_foot_0','left_foot_0']

path_joint = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Vincent/Trial112_joints.csv"
q_ref_df = pd.read_csv(path_joint).iloc[:,1:]
q_ref = q_ref_df.to_numpy(dtype=float)
urdf_name = "human.urdf"
urdf_path = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/urdf_scaled/Vincent_scaled.urdf"# Human base
urdf_meshes_path = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/motif/model/human_urdf/meshes"
model_h, coll_h, vis_h, _ = build_human_model(urdf_path, urdf_meshes_path)
data_h = model_h.createData()

# Shared Meshcat
viewer = meshcat.Visualizer()

# Visualizers
viz_human = MeshcatVisualizer(model_h, coll_h, vis_h)
viz_human.initViewer(viewer, open=True)
viz_human.viewer.delete()  # clear if relaunch
viz_human.loadViewerModel("ref",color=[0.0, 1.0, 0.0, 0.8])
# Cacher tous les visuals d'abord
for geom in vis_h.geometryObjects:
    node_name = viz_human.getViewerNodeName(geom, pin.GeometryType.VISUAL)
    viz_human.viewer[node_name].set_property("visible", False)
#n'afficher que ceux qui contiennent """
for geom in vis_h.geometryObjects:
    for mesh in meshes:

        if mesh in geom.name:   # ton critère
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
for i in range(len(q_ref)):
    viz_human.display(q_ref[i])
