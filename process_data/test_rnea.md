tau = pin.rnea(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])
pin.forwardKinematics(model_h, data_h, q_ref[i, :], v_ref[i, :], a_ref[i, :])
wrench_base = data_h.f[1]         
oM1 = data_h.oMi[1]              
f_world = oM1.act(wrench_base)          
M = f_world.angular


quat = pin.Quaternion(q_ref[i, 3:7])
rot_base = quat.matrix()
pos_bassin = q_ref[i, 0:3]

# Projection Force/Moment RNEA dans World
rnea_forces_world[i, :] = rot_base @ tau[0:3]
rnea_moments_world[i, :] = (rot_base @ tau[3:6] ) + np.cross(pos_bassin, rnea_forces_world[i, :])
print(tau)
print(wrench_base)
print(rnea_forces_world[i, :])
print(rnea_moments_world[i, :] )
print(f_world)

for i, name in enumerate(model_h.names):
    print(i, name, model_h.joints[i]) 
    