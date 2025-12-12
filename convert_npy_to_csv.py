import numpy as np
import csv

# ==== paramètres ====
input_npy = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique/joints_test.npy"
output_csv = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/minimal_model_statique/joints_test.csv"

# ==== chargement ====
data = np.load(input_npy)   # shape (N, 2)

# ==== écriture dans CSV ====
with open(output_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["q1", "q2"])  # header
    writer.writerows(data)

print(f"Fichier CSV créé : {output_csv}")
