import sys
import time
import numpy as np
import pinocchio as pin
import hppfcl as fcl
from pinocchio.visualize import GepettoVisualizer
from utils.viz_utils import place_gep
from utils.model_utils import compute_My,build_2dof_model

model, data, geom_model,joint1_id = build_2dof_model()
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

q = pin.neutral(model)  

q1 = np.deg2rad(0.0)
q2 = np.deg2rad(0.0)
q[:] = [q1, q2]
viz.display(q)
v = np.zeros(model.nv)
a = np.zeros(model.nv)

# input()
N = 200      #random poses
dt = 0.05    



# joints  = np.zeros((N, 2))   # q1, q2  rad
# wrenchY = np.zeros((N, 1))   # My  Nm


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
        # viz.display(q)
        # time.sleep(0.01)
        # input()

    # NPY (tu peux aussi sauver en CSV si tu veux)
    np.save(joints_file, joints)
    np.save(wrench_file, wrenchY)
    print(f"Saved {N} samples to {joints_file} / {wrench_file}")

def generate_trajectory(N_test, joints_file, wrench_file,
                             model, data, joint1_id, q, v, a):
    joints  = np.zeros((N_test, 2), dtype=np.float32)
    wrenchY = np.zeros((N_test, 1), dtype=np.float32)

    # temps normalisé [0,1]
    t = np.linspace(0, 1, N_test)
 
    q1_traj = 0.3 * np.sin(2 * np.pi * t)         
    q2_traj = 0.8 * np.sin(4 * np.pi * t + 0.5)  
    # clipper pour rester dans  bornes :
    q1_traj = np.clip(q1_traj, -np.pi/4, +np.pi/4)
    q2_traj = np.clip(q2_traj, -np.pi/2, +np.pi/2)

    for k in range(N_test):
        q1 = q1_traj[k]
        q2 = q2_traj[k]

        My = compute_My(q1, q2, model, data, joint1_id, q, v, a)

        joints[k, 0] = q1
        joints[k, 1] = q2
        wrenchY[k, 0] = My

        # viz.display(q)
        # time.sleep(0.1)

    np.save(joints_file, joints)
    np.save(wrench_file, wrenchY)
    print(f"Saved test traj of length {N_test} to {joints_file} / {wrench_file}")


N_train = 20000
N_val   = 4000

generate_split(
    N_train,
    "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique/joints_train.npy",
    "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique/wrench_train.npy",
    model, data, joint1_id, q, v, a
)

generate_split(
    N_val,
    "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique/joints_val.npy",
    "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique/wrench_val.npy",
    model, data, joint1_id, q, v, a
)

N_test = 1000
generate_split(
    N_test,
    "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique/joints_test.npy",
    "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique/wrench_test.npy",
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
