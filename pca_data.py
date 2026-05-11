import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

j = ["delta_x","delta_y","delta_z","delta_rx","delta_ry","delta_rz",
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",

    "Lumbar_flex_ext", "Lumbar_lateral_flex",
    "Lcalvicule_x",
    "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
    "Lelbow_flex_ext", "Lelbow_pron_supi",
    "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
    "rcalvicule_x",
    "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot",
    "Relbow_flex_ext", "Relbow_pron_supi",
]
root = Path("./processed_data_feet")

all_data = []

for joints_file in root.rglob("all_joints.npy"):

    data = np.load(joints_file)

    # data shape = (T, 35)

    all_data.append(data)

# concaténation
X = np.concatenate(all_data, axis=0)

print("Shape globale :", X.shape)

# -------------------------------------------------
# Standardisation
# -------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# PCA
# -------------------------------------------------

pca = PCA()

pca.fit(X_scaled)

# variance cumulée
cumvar = np.cumsum(pca.explained_variance_ratio_)

# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(8,5))

plt.plot(cumvar)

plt.xlabel("Nombre de composantes")
plt.ylabel("Variance cumulée")

# Find the number of components for 95% variance
n_95 = np.argmax(cumvar >= 0.95) + 1
print(f"Nombre de dimensions indépendantes (pour 95% de variance): {n_95}")

# Add a horizontal line to your plot for clarity
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.axvline(x=n_95, color='g', linestyle='--', label=f'{n_95} Composantes')
plt.legend()

plt.grid()


plt.show()

print(pca.components_[0])