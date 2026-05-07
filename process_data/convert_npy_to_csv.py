import numpy as np
import csv

input_npy = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_PE_sin_cross_selfs_best_guided/Jeremy_Trial111_prediction.npy"
output_csv = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_PE_sin_cross_selfs_best_guided/Jeremy_Trial111_prediction.csv"

data = np.load(input_npy)   # shape (N, 2)

with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([#"FF_X","FF_Y","FF_Z","FF_quatx","FF_quaty","FF_quatz","FF_quatw", 
        "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",
    # "Lumbar_flex_ext", "Lumbar_lateral_flex",
    # "Lcalvicule_x",
    # "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
    # "Lelbow_flex_ext", "Lelbow_pron_supi",
    # "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
    # "rcalvicule_x",
    # "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot",
    # "Relbow_flex_ext", "Relbow_pron_supi"
    ])  # header
    writer.writerows(data)

print(f"Fichier CSV créé : {output_csv}")

# input_npy = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/synth_data/s1/squat_variant_980_dz-0.080_dx+0.023_dy-0.017/kinetics.npy"
# output_csv = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/inference_results_PE_sin_aug/s1_squat_variant_980_dz-0.080_dx+0.023_dy-0.017_kinetics.csv"


# data = np.load(input_npy)   # shape (N, 2)

# with open(output_csv, mode="w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(['Fx1', 'Fy1', 'Fz1', 'Mx1', 'My1', 'Mz1', 'COPx1', 'COPy1', 'COPz1',
#     'Fx2', 'Fy2', 'Fz2', 'Mx2', 'My2', 'Mz2', 'COPx2', 'COPy2', 'COPz2'
#     ])  # header
#     writer.writerows(data)

# print(f"Fichier CSV créé : {output_csv}")

