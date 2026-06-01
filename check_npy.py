import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================
# Remplace ce chemin par ton fichier .npy
file_path_k = Path("/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/HUMANOIDS_NPY/Kahina/squat/joints.npy")
# Exemple :
file_path_j = Path("/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/HUMANOIDS_NPY/Kahina/squat/kinetics.npy")

# =========================================================
# LOAD FILE
# =========================================================
data = np.load(file_path_k)

print("File loaded:", file_path_k)
print("Shape:", data.shape)
print("Type:", data.dtype)
print("\nFirst 5 rows:")
print(data[:5])
data = np.load(file_path_j)

print("File loaded:", file_path_j)
print("Shape:", data.shape)
print("Type:", data.dtype)
print("\nFirst 5 rows:")
print(data[:5])
# =========================================================
# DISPLAY FIRST FEW ROWS
# =========================================================



# =========================================================
# OPTIONAL LABELS
# =========================================================
if "lower_body_joints" in file_path_j.name:
    labels = [
        "R Hip Flex/Ext", "R Hip Abd/Add", "R Hip Int/Ext Rot",
        "R Knee Flex/Ext", "R Ankle Flex/Ext", "R Ankle Abd/Add",
        "L Hip Flex/Ext", "L Hip Abd/Add", "L Hip Int/Ext Rot",
        "L Knee Flex/Ext", "L Ankle Flex/Ext", "L Ankle Abd/Add"
    ]

elif "kinetics" in file_path_k.name:
    labels = [
        "R Fx", "R Fy", "R Fz", "R Mx", "R My", "R Mz", "R COPx", "R COPy", "R COPz",
        "L Fx", "L Fy", "L Fz", "L Mx", "L My", "L Mz", "L COPx", "L COPy", "L COPz"
    ]

elif "all_joints" in file_path_j.name:
    labels = [f"DOF {i}" for i in range(data.shape[1])]

else:
    labels = [f"Col {i}" for i in range(data.shape[1])]

# =========================================================
# DISPLAY FIRST FEW ROWS
# =========================================================
print("\nFirst 5 rows:")
# print(data[:5])

# =========================================================
# PLOT ALL CHANNELS
# =========================================================
plt.figure(figsize=(14, 8))

for i in range(data.shape[1]):
    plt.plot(data[:, 6], label=labels[i])

plt.xlabel("Frame")
plt.ylabel("Value")
plt.title(file_path_j.name)
plt.legend(fontsize=8, ncol=3)
plt.tight_layout()
plt.show()