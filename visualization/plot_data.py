import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier
df = pd.read_csv("DATA/generated_data/subject_01_squat_variant_860_dz-0.135_dx-0.025_dy+0.035_q.csv")
df2= pd.read_csv("DATA/generated_data/subject_02_squat_variant_000_dz+0.025_dx+0.070_dy+0.020_q.csv")

df_grfm = pd.read_csv("DATA/generated_data/subject_01_squat_variant_860_dz-0.135_dx-0.025_dy+0.035_grfm.csv")
df2_grfm= pd.read_csv("DATA/generated_data/subject_02_squat_variant_000_dz+0.025_dx+0.070_dy+0.020_grfm.csv")

# -------------------------
# GROUPES
# -------------------------
group1 = ['FF_X','FF_Y','FF_Z','FF_quatx','FF_quaty','FF_quatz','FF_quatw']
group2 = [
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",]
group3 = [
    "Lumbar_flex_ext", "Lumbar_lateral_flex",
    "Lcalvicule_x",
    "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
    "Lelbow_flex_ext", "Lelbow_pron_supi",
    "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
    "rcalvicule_x",
    "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot",
    "Relbow_flex_ext", "Relbow_pron_supi",]

# -------------------------
# FONCTION PLOT
def plot_group(dataframe, columns, title):
    data = dataframe[columns]

    n = len(columns)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        axes[i].plot(data[col])
        axes[i].set_title(col)
        axes[i].grid(True)

    # cacher les axes inutiles
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    

plot_group(df, group1, "File 1 - Freeflyer")
plot_group(df2, group1, "File 2 - Freeflyer")

# -------------------------
# Lower body
# -------------------------
plot_group(df, group2, "File 1 - Lower Body")
plot_group(df2, group2, "File 2 - Lower Body")

# -------------------------
# Upper body
# -------------------------
mid = len(group3) // 2

plot_group(df, group3[:mid], "File 1 - Upper Body (1)")
plot_group(df2, group3[:mid], "File 2 - Upper Body (1)")

plot_group(df, group3[mid:], "File 1 - Upper Body (2)")
plot_group(df2, group3[mid:], "File 2 - Upper Body (2)")
plt.show()
def plot_grfm(df, title):
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))

    sides = [1, 2]
    categories = ["Force", "Moment", "COP"]

    for j, side in enumerate(sides):

        # --- Force ---
        force_cols = [f"F{x}{side}_glob" for x in ["x", "y", "z"]]
        axes[0, j].plot(df[force_cols])
        axes[0, j].set_title(f"Side {side} - Force")
        axes[0, j].legend(force_cols)
        axes[0, j].grid()

        # --- Moment ---
        moment_cols = [f"M{x}{side}_glob" for x in ["x", "y", "z"]]
        axes[1, j].plot(df[moment_cols])
        axes[1, j].set_title(f"Side {side} - Moment")
        axes[1, j].legend(moment_cols)
        axes[1, j].grid()

        # --- COP ---
        cop_cols = [f"COP{x}{side}_glob" for x in ["x", "y", "z"]]
        axes[2, j].plot(df[cop_cols])
        axes[2, j].set_title(f"Side {side} - COP")
        axes[2, j].legend(cop_cols)
        axes[2, j].grid()

    fig.suptitle(title)
    plt.tight_layout()


plot_grfm(df_grfm, "Generated data - GRFM")
plot_grfm(df2_grfm, "Vinc/Jeremy data - GRFM")    
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

def load_data(is_synth=True):
    # Charge tous tes fichiers ici pour constituer une grande matrice (T, 35)
    # Retourne une matrice X de forme (N_total, 35)
    pass

def compare_metrics():
    X_synth = load_data(is_synth=True)
    X_real = load_data(is_synth=False)
    
    # 1. PCA Visualization
    pca = PCA(n_components=2)
    X_all = np.vstack([X_synth, X_real])
    pca.fit(X_all)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(pca.transform(X_synth)[:, 0], pca.transform(X_synth)[:, 1], alpha=0.1, label='Synthétique', s=1)
    plt.scatter(pca.transform(X_real)[:, 0], pca.transform(X_real)[:, 1], alpha=0.3, label='Réel', s=2)
    plt.legend(); plt.title("PCA du mouvement : Synthétique vs Réel")
    plt.show()
    
    # 2. Corrélation Fz vs Accélération Bassin (FF[2])
    # Exemple pour une seule séquence
    # corr_synth = pearsonr(forces_z, acc_pelvis)[0]
    print("Vérifie bien si la corrélation Force/Mouvement est cohérente !")