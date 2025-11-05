import numpy as np
from scipy.spatial.transform import Rotation as R

# matrice de conversion (Unity -> Z-up right-handed)
R_conv_mat = np.array([[0,0,1],
                       [-1,0,0],
                       [0,1,0]])
Rconv = R.from_matrix(R_conv_mat)
q_conv = Rconv.as_quat()   # [x,y,z,w]

# quaternion Unity (exemple) : format Unity [x,y,z,w]
q_unity = np.array([0.1, 0.2, 0.3, 0.9])
Runity = R.from_quat(q_unity)

# conversion par similarité (même effet que q_new = q_conv * q_unity * q_conv^{-1})
Rnew = Rconv * Runity * Rconv.inv()
q_new = Rnew.as_quat()   # quaternion dans le repère Z-up (format [x,y,z,w])

print("q_conv:", q_conv)
print("q_unity:", q_unity)
print("q_new:", q_new)

