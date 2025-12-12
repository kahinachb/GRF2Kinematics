import sys
import time
import numpy as np
import pinocchio as pin
import hppfcl as fcl
from pinocchio.visualize import GepettoVisualizer
from utils.viz_utils import place_gep
import pandas as pd

path_joint = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique/joints_test.csv"
q_ref_df = pd.read_csv(path_joint)
q_ref = q_ref_df.to_numpy(dtype=float)

path_joint_pred = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique/best_q_test.csv"
q_pred_df = pd.read_csv(path_joint_pred)
q_pred = q_pred_df.to_numpy(dtype=float)

# Build 2-DoF model
model = pin.Model()
geom_model = pin.GeometryModel()

L1, L2 = 1.0, 1.0  
m1, m2 = 1.0, 1.0   
body_radius = 0.1

base_shape = fcl.Sphere(body_radius)
base_geom  = pin.GeometryObject("base", 0, pin.SE3.Identity(), base_shape)
base_geom.meshColor = np.array([1.0, 0.0, 0.0, 1.0])
geom_model.addGeometryObject(base_geom)

parent_id       = 0  # universe

####joint 1
joint1_placement = pin.SE3.Identity()
joint1_id = model.addJoint(parent_id, pin.JointModelRY(), joint1_placement,"joint1") #The joints rotate around Y-axis :So each joint only transmits moments around Y

inertia1 = pin.Inertia.FromSphere(m1, body_radius) 
body1_placement = pin.SE3.Identity()
body1_placement.translation = np.array([0.0, 0.0, L1 / 2.0])
model.appendBodyToJoint(joint1_id, inertia1, body1_placement)

shape1 = fcl.Cylinder(body_radius, L1)
shape1_placement = pin.SE3.Identity()
shape1_placement.translation = np.array([0.0, 0.0, L1 / 2.0])
geom1 = pin.GeometryObject("link1", joint1_id, shape1_placement, shape1)
geom1.meshColor = np.array([0.2, 0.2, 0.8, 1.0])
geom_model.addGeometryObject(geom1)


####joint 2
joint2_placement = pin.SE3.Identity()
joint2_placement.translation = np.array([0.0, 0.0, L1])
joint2_id = model.addJoint(joint1_id,pin.JointModelRY(),joint2_placement,"joint2")


inertia2 = pin.Inertia.FromSphere(m2, body_radius)
body2_placement = pin.SE3.Identity()
body2_placement.translation = np.array([0.0, 0.0, L2 / 2.0])
model.appendBodyToJoint(joint2_id, inertia2, body2_placement)

shape2 = fcl.Cylinder(body_radius, L2)
shape2_placement = pin.SE3.Identity()
shape2_placement.translation = np.array([0.0, 0.0, L2 / 2.0])
geom2 = pin.GeometryObject("link2", joint2_id, shape2_placement, shape2)
geom2.meshColor = np.array([0.8, 0.2, 0.2, 1.0])
geom_model.addGeometryObject(geom2)


visual_model = geom_model

viz = GepettoVisualizer(model, geom_model, visual_model)

try:
    viz.initViewer()
except ImportError as err:
    print("Error while initializing the viewer. Install gepetto-viewer.")
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("two_link_arm")
except AttributeError as err:
    print("Error while loading the viewer model. Start gepetto-viewer-server.")
    print(err)
    sys.exit(0)
viz.viewer.gui.addXYZaxis('world/base_frame', [255, 0., 0, 1.], 0.1, 0.15)
place_gep(viz, 'world/base_frame', pin.SE3(np.eye(3), np.matrix([0, 0, 0]).T))


model_pred = pin.Model(model)
geom_pred  = pin.GeometryModel()

# dupliquer les géométries en changeant les couleurs pour q_pred
for geom_obj in geom_model.geometryObjects:
    new_geom = pin.GeometryObject(geom_obj)
    new_geom.name = geom_obj.name + "_pred"
    
    # couleur différente (bleu)
    new_geom.meshColor = np.array([0.0, 0.8, 1.0, 1.0])

    # décalage du modèle prédictif sur l'axe X pour éviter le chevauchement
    new_geom.placement.translation[1] += 0.5  

    geom_pred.addGeometryObject(new_geom)

# un visualizer séparé mais dans le même viewer
viz_pred = GepettoVisualizer(model_pred, geom_pred, geom_pred)

viz_pred.viewer = viz.viewer   # utiliser la même fenêtre
viz_pred.initViewer()
viz_pred.loadViewerModel("pred")


data = model.createData()
data_pred = model_pred.createData()


v = np.zeros(model.nv)
a = np.zeros(model.nv)

def compute_My(q1, q2, model, data, joint1_id, q, v, a):
    """Calcule My (wrench monde à la base) pour une config (q1,q2)."""
    q[:] = [q1, q2]

    # inverse dynamics
    tau = pin.rnea(model, data, q, v, a)
    # kinematics pour avoir oMi à jour
    pin.forwardKinematics(model, data, q, v, a)

    wrench_base = data.f[joint1_id]         # wrench dans le repère du joint1
    oM1 = data.oMi[joint1_id]               # SE3 joint1 dans le monde
    f_world = oM1.act(wrench_base)          # wrench dans le monde

    M = f_world.angular                     # [Mx, My, Mz]
    return M[1] 

q = pin.neutral(model)  
viz.display(q)
input()
q = pin.neutral(model_pred)  
viz.display(q)
for i in range(len(q_ref)):
    viz.display(q_ref[i])          
    viz_pred.display(q_pred[i])  

    q1_ref= q_ref[i][0]
    q2_ref=  q_ref[i][1]
    My_ref= compute_My(q1_ref, q2_ref, model, data, joint1_id, q, v, a)

    q1_pred= q_pred[i][0]
    q2_pred=  q_pred[i][1]
    My_pred= compute_My(q1_pred, q2_pred, model, data, joint1_id, q, v, a)

    print("My_ref",My_ref)
    print("My_pred",My_pred)
    time.sleep(0.03)
    input()