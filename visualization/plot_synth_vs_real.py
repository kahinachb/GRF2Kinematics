import pandas as pd
import matplotlib.pyplot as plt

# chemins
file1 = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_PE_sin_aug/Jeremy_Trial111_kinetics.csv"
file2 = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_PE_sin_aug/s1_squat_variant_980_dz-0.080_dx+0.023_dy-0.017_kinetics.csv"
# file1 = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Vinc/Jeremy/Trial111/kinetics_glob_filtered.csv"
# file2 = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/generated_human_like_motions_csv/generated_human_like_motions_csv/kinetics_glob_filtered_squat_variant_980_dz-0.080_dx+0.023_dy-0.017.csv"
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# df1.columns = df1.columns.str.replace("_glob", "")

# composantes à comparer
components = [
    "Fx1", "Fy1", "Fz1", "Mx1", "My1", "Mz1", "COPx1", "COPy1", "COPz1"
]

n = len(components)

plt.figure(figsize=(15, 2.5 * n))

for i, comp in enumerate(components):
    ax = plt.subplot(n, 1, i + 1)

    ax.plot(df1[comp].values, label="real_data", linewidth=1)
    ax.plot(df2[comp].values, label="synth_data", linewidth=1)

    ax.set_title(comp)
    ax.grid(True)
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.show()

components = [
    "Fx2", "Fy2", "Fz2", "Mx2", "My2", "Mz2", "COPx2", "COPy2", "COPz2"
]

n = len(components)

plt.figure(figsize=(15, 2.5 * n))

for i, comp in enumerate(components):
    ax = plt.subplot(n, 1, i + 1)

    ax.plot(df1[comp].values, label="real_data", linewidth=1)
    ax.plot(df2[comp].values, label="synth_data", linewidth=1)

    ax.set_title(comp)
    ax.grid(True)
    if i == 0:
        ax.legend()

plt.tight_layout()
plt.show()