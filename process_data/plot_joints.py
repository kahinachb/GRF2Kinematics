import pandas as pd
import matplotlib.pyplot as plt

# Charger les fichiers
df_joints = pd.read_csv("DATA/Vinc/Jovana/Trial111/joints_filtered_.csv")
df_kinetics = pd.read_csv("DATA/Vinc/Jovana/Trial111/kinetics_glob_filtered_.csv")
df_kinetics= pd.read_csv("DATA/generated_103/subject_103_squat_variant_000_dz+0.125_dx-0.100_dy+0.065_grfm.csv")
df_joints= pd.read_csv("DATA/generated_103/subject_103_squat_variant_000_dz+0.125_dx-0.100_dy+0.065_q.csv")
# -------------------------
# GROUPES
# -------------------------
group1 = ['FF_X','FF_Y','FF_Z','FF_quatx','FF_quaty','FF_quatz','FF_quatw']
# group1 = ["delta_x","delta_y","delta_z","delta_rx","delta_ry","delta_rz"]
# group1 = ["root_joint","root_joint.1","root_joint.2",
                #  "root_joint.3","root_joint.4","root_joint.5","root_joint.6"]
group2 = [
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",
]

# group2 = ["right_hip_Z", "right_hip_X", "right_hip_Y",
#     "right_knee_Z", "right_ankle_Z", "right_ankle_X",
#     "left_hip_Z",  "left_hip_X",  "left_hip_Y",
#     "left_knee_Z", "left_ankle_Z", "left_ankle_X",]

group3 = [
    "Lumbar_flex_ext", "Lumbar_lateral_flex",
    "Lcalvicule_x",
    "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
    "Lelbow_flex_ext", "Lelbow_pron_supi",
    "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
    "rcalvicule_x",
    "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot",
    "Relbow_flex_ext", "Relbow_pron_supi",
]

# -------------------------
# FONCTION PLOT
# -------------------------

def plot_group(columns, title):
    data = df_joints[columns]

    n = len(columns)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        axes[i].plot(data[col])
        axes[i].set_title(col)
        axes[i].grid()

    # cacher axes inutiles
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_forces_moments():
    """Trace les efforts de chaque côté (Vinc : plateforme 1 = droite, 2 = gauche)."""
    sides = {
        "Right": "1",
        "Left": "2",
    }
    quantities = (
        ("F", "x", "Force (N)"),
        ("F", "y", "Force (N)"),
        ("F", "z", "Force (N)"),
        ("M", "x", "Moment (N.m)"),
        ("M", "y", "Moment (N.m)"),
        ("M", "z", "Moment (N.m)"),
    )

    fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True)

    for col_idx, (side, plate) in enumerate(sides.items()):
        for row_idx, (quantity, component, unit) in enumerate(quantities):
            label = f"{quantity}{component}"
            axes[row_idx, col_idx].plot(
                df_kinetics[f"{label}{plate}_glob"],
                color="tab:blue" if quantity == "F" else "tab:orange",
            )
            axes[row_idx, col_idx].set_title(f"{label} - {side}")
            axes[row_idx, col_idx].set_ylabel(unit)
            axes[row_idx, col_idx].grid(True, alpha=0.3)

        axes[-1, col_idx].set_xlabel("Frame")

    fig.suptitle("Ground reaction forces and moments")
    fig.tight_layout()
    plt.show()


# -------------------------
# PLOTS
# -------------------------

# 1) Freeflyer
plot_group(group1, "Freeflyer (translation + rotation)")

# 2) Membres inférieurs
plot_group(group2, "Lower Body Joints")

# 3) Haut du corps → split en 2 figures
mid = len(group3) // 2

plot_group(group3[:mid], "Upper Body Joints ")
plot_group(group3[mid:], "Upper Body Joints")

# 4) Forces et moments : une colonne par côté
plot_forces_moments()
