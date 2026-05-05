import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

path_joint_pred = "inference_results_PE_sin_cross_synth_guided1/s1_squat_variant_980_dz-0.080_dx+0.023_dy-0.017_prediction.csv"
path_joint = "DATA/generated_human_like_motions_csv_new/generated_human_like_motions_csv/joint_filtered_squat_variant_980_dz-0.080_dx+0.023_dy-0.017.csv"

dofs = [
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",
]

# Load data
q_ref_df = pd.read_csv(path_joint, usecols=dofs).iloc[1:, :]
q_pred_df = pd.read_csv(path_joint_pred)[dofs]

# Match length
min_len = min(len(q_ref_df), len(q_pred_df))
q_ref_df = q_ref_df.iloc[:min_len]
q_pred_df = q_pred_df.iloc[:min_len]

# Subplots
n_dofs = len(dofs)
n_cols = 3
n_rows = math.ceil(n_dofs / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten()

# Plot
for i, dof in enumerate(dofs):
    ax = axes[i]
    
    ref = q_ref_df[dof].values
    pred = q_pred_df[dof].values
    
    # Metrics
    mse = np.mean((ref - pred) ** 2)
    rmse = np.sqrt(mse)
    print(mse)
    
    # Plot
    ax.plot(ref, label="Ref", color = 'black')
    ax.plot(pred, label="Pred", color= "red")
    
    ax.set_title(f"{dof}\nMSE={mse:.4f} | RMSE={rmse:.4f}", fontsize=8)
    ax.grid()
    

    if i == 0:
        ax.legend()

# Remove empty axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.subplots_adjust(hspace=0.6)
plt.show()