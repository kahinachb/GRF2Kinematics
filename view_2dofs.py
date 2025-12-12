import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Paramètres du modèle ===
L1 = 1.0
L2 = 1.0

q1 = np.deg2rad(30)   # DOF 1 : rot Z
q2 = np.deg2rad(45)   # DOF 2 : rot Y local

# === Repère global ===
O = np.array([0, 0, 0])

# Rotation Z
Rz = np.array([
    [np.cos(q1), -np.sin(q1), 0],
    [np.sin(q1),  np.cos(q1), 0],
    [0, 0, 1]
])

# Articulation 1
A1 = O
A2 = A1 + Rz @ np.array([0, 0, L1])

# Rotation locale Y
Ry = np.array([
    [ np.cos(q2), 0, np.sin(q2)],
    [ 0,          1, 0         ],
    [-np.sin(q2), 0, np.cos(q2)]
])

R2 = Rz @ Ry
A3 = A2 + R2 @ np.array([0, 0, L2])

# === Visualisation ===
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# Segments
ax.plot([A1[0], A2[0]], [A1[1], A2[1]], [A1[2], A2[2]], linewidth=6, color="blue", label ="segment 1")
ax.plot([A2[0], A3[0]], [A2[1], A3[1]], [A2[2], A3[2]], linewidth=6, color="red", label ="segment 2")

# Articulations
ax.scatter(*A1, color="black", s=100, label ="joint 1")
ax.scatter(*A2, color="green", s=100, label ="joint 2")

# Repère global
ax.quiver(0,0,0, 0.5,0,0, color='r', linewidth=5)
ax.text(0.55,0,0,'X')
ax.quiver(0,0,0, 0,0.5,0, color='g', linewidth=5)
ax.text(0,0.55,0,'Y')
ax.quiver(0,0,0, 0,0,0.5, color='b', linewidth=5)
ax.text(0,0,0.55,'Z')

# ==========================
#  FLECHE COURBE autour de Y au joint 1
# ==========================
r = 0.3                             # rayon de l’arc
theta = np.linspace(0, np.pi, 60)  # portion de cercle
x_arc = A1[0] + r * np.cos(theta)
y_arc = np.full_like(theta, A1[1])  # Y constant
z_arc = A1[2] + r * np.sin(theta)

# Arc
ax.plot(x_arc, y_arc, z_arc, color='orange', linewidth=2)
ax.legend()

# Pointe de flèche : direction tangente à l’arc
x_end, y_end, z_end = x_arc[-1], y_arc[-1], z_arc[-1]
dx = x_arc[-1] - x_arc[-2]
dz = z_arc[-1] - z_arc[-2]
norm = np.sqrt(dx**2 + dz**2)
dx /= norm
dz /= norm

ax.quiver(x_end, y_end, z_end,
          dx*0.12, 0, dz*0.12,
          color='orange', linewidth=2)

ax.text(x_end+dx*0.15, y_end, z_end+dz*0.15,
        r"$M_y$", color='orange')
# ==========================

# Pas de grille
ax.grid(False)

# Limites et aspect
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(0, 2.5)
ax.set_box_aspect([1,1,1])

plt.show()
