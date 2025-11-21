import sys
import time
import numpy as np
import pinocchio as pin
import hppfcl as fcl
from pinocchio.visualize import GepettoVisualizer
from utils.viz_utils import place_gep

# Build 2-DoF model
model      = pin.Model()
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


data = model.createData()
q = pin.neutral(model)  

q1 = np.deg2rad(0.0)
q2 = np.deg2rad(0.0)
q[:] = [q1, q2]
viz.display(q)

# input()
N = 200      #random poses
dt = 0.05    

v = np.zeros(model.nv)
a = np.zeros(model.nv)

# joints  = np.zeros((N, 2))   # q1, q2  rad
# wrenchY = np.zeros((N, 1))   # My  Nm

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
    return M[1]                             # My

def generate_split(N, joints_file, wrench_file,
                   model, data, joint1_id, q, v, a):
    joints  = np.zeros((N, 2), dtype=np.float32)
    wrenchY = np.zeros((N, 1), dtype=np.float32)

    for k in range(N):
        q1 = np.random.uniform(-np.pi/4, +np.pi/4)   # [-45°, +45°]
        q2 = np.random.uniform(-np.pi/2, +np.pi/2)   # [-90°, +90°]

        My = compute_My(q1, q2, model, data, joint1_id, q, v, a)

        joints[k, 0] = q1
        joints[k, 1] = q2
        wrenchY[k, 0] = My

        # Optionnel debug
        # print(f"{k:5d} q1={np.degrees(q1):6.1f}deg, q2={np.degrees(q2):6.1f}deg, My={My: .3f}")

    # NPY (tu peux aussi sauver en CSV si tu veux)
    np.save(joints_file, joints)
    np.save(wrench_file, wrenchY)
    print(f"Saved {N} samples to {joints_file} / {wrench_file}")

def generate_test_trajectory(N_test, joints_file, wrench_file,
                             model, data, joint1_id, q, v, a):
    joints  = np.zeros((N_test, 2), dtype=np.float32)
    wrenchY = np.zeros((N_test, 1), dtype=np.float32)

    # temps normalisé [0,1]
    t = np.linspace(0, 1, N_test)

    # Exemple de trajectoire "humaine" un peu smooth
    # q1: légère flexion/extension hanche
    q1_traj = 0.3 * np.sin(2 * np.pi * t)          # amplitude ~0.3 rad ~ 17°
    # q2: mouvement plus grand
    q2_traj = 0.8 * np.sin(4 * np.pi * t + 0.5)    # amplitude ~0.8 rad ~ 45°

    # Tu peux clipper pour rester dans tes bornes :
    q1_traj = np.clip(q1_traj, -np.pi/4, +np.pi/4)
    q2_traj = np.clip(q2_traj, -np.pi/2, +np.pi/2)

    for k in range(N_test):
        q1 = q1_traj[k]
        q2 = q2_traj[k]

        My = compute_My(q1, q2, model, data, joint1_id, q, v, a)

        joints[k, 0] = q1
        joints[k, 1] = q2
        wrenchY[k, 0] = My

        # Optionnel : visualiser la trajectoire au lieu des random poses
        viz.display(q)
        time.sleep(0.01)

    np.save(joints_file, joints)
    np.save(wrench_file, wrenchY)
    print(f"Saved test traj of length {N_test} to {joints_file} / {wrench_file}")


N_train = 20000
N_val   = 4000

generate_split(
    N_train,
    "joints_train.npy",
    "wrench_train.npy",
    model, data, joint1_id, q, v, a
)

generate_split(
    N_val,
    "joints_val.npy",
    "wrench_val.npy",
    model, data, joint1_id, q, v, a
)

N_test = 1000
generate_test_trajectory(
    N_test,
    "joints_test.npy",
    "wrench_test.npy",
    model, data, joint1_id, q, v, a
)


# for k in range(N):
#     q1 = np.random.uniform(-np.pi/4, np.pi/4)     #45~135deg    
#     q2 = np.random.uniform(-np.pi/2, np.pi/2)  #-90~+90
#     q[:] = [q1, q2]

#     #inverse dynamics
#     tau = pin.rnea(model, data, q, v, a)   #torque at joint1 and joint2
#     pin.forwardKinematics(model, data, q, v, a)


#     print(f"q1={np.degrees(q1):6.1f}deg, q2={np.degrees(q2):6.1f}deg "
#           f"tau = [{tau[0]: .3f}, {tau[1]: .3f}]")
    
#     wrench1 = data.f[joint1_id] #f N and M Nm in local frames, M1 == tau1 
#     print("wrench1",wrench1)

#     # wrench2 = data.f[joint2_id]
#     # print("wrench2",wrench2)

#     oM1 = data.oMi[joint1_id]               # SE3 for joint1 in world
#     f_world = oM1.act(wrench1)              # wrench expressed in world,  Fz N = (m₁ + m₂) × g = 2 kg × 9.81 m/s²

#     F = f_world.linear                      # 3D force vector
#     M = f_world.angular                     # 3D moment (unused here, but available)

#     print("f_world",f_world)
    
#     joints[k, 0]  = q1
#     joints[k, 1]  = q2
#     wrenchY[k, 0] = M[1]

#     viz.display(q)
#     time.sleep(dt)
#     # input()
