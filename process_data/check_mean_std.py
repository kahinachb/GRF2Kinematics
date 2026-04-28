# import pandas as pd

# # Chemin vers ton fichier CSV
# file_path = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Vinc/Jeremy/Trial111/kinetics_glob_filtered_.csv"

# # Lire le fichier
# df = pd.read_csv(file_path)

# # Vérifier que les colonnes existent
# required_cols = ["My1_glob", "My2_glob"]
# for col in required_cols:
#     if col not in df.columns:
#         raise ValueError(f"Colonne manquante : {col}")

# # Extraire les données
# my1 = df["My1_glob"]
# my2 = df["My2_glob"]

# # Calculs
# mean_my1 = my1.mean()
# std_my1 = my1.std()

# mean_my2 = my2.mean()
# std_my2 = my2.std()

# # Affichage
# print("My1_glob:")
# print(f"  Moyenne = {mean_my1}")
# print(f"  Std     = {std_my1}")

# print("\nMy2_glob:")
# print(f"  Moyenne = {mean_my2}")
# print(f"  Std     = {std_my2}")

# mean_my1 = my1.mean()
# std_my1 = my1.std()

# print(f"My1_glob = {mean_my1:.3f} ± {std_my1:.3f}")
# print(f"My2_glob = {mean_my2:.3f} ± {std_my2:.3f}")

import pandas as pd
from pathlib import Path

root = Path("/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Vinc")
files = list(root.rglob("kinetics_glob_filtered_.csv"))

trial_results = []

# =========================
# 1. PAR TRIAL
# =========================
for file in files:
    try:
        df = pd.read_csv(file)

        subject = file.parts[-3]
        task = file.parts[-2]

        my1 = df["Mz1_glob"].dropna()
        my2 = df["Mz2_glob"].dropna()

        trial_results.append({
            "subject": subject,
            "task": task,
            "My1_mean": my1.mean(),
            "My1_std": my1.std(),
            "My2_mean": my2.mean(),
            "My2_std": my2.std()
        })

        print(f"{subject} | {task}")
        print(f"  My1 = {my1.mean():.3f} ± {my1.std():.3f}")
        print(f"  My2 = {my2.mean():.3f} ± {my2.std():.3f}")

    except Exception as e:
        print(f"Erreur avec {file}: {e}")

trial_df = pd.DataFrame(trial_results)

# =========================
# 2. PAR SUBJECT (tous les trials concaténés)
# =========================
print("\n===== PAR SUBJECT =====")

subject_values_my1 = {}
subject_values_my2 = {}

for file in files:
    df = pd.read_csv(file)
    subject = file.parts[-3]

    if subject not in subject_values_my1:
        subject_values_my1[subject] = []
        subject_values_my2[subject] = []

    subject_values_my1[subject].append(df["Mz1_glob"].dropna())
    subject_values_my2[subject].append(df["Mz2_glob"].dropna())

for subject in subject_values_my1:
    my1_all = pd.concat(subject_values_my1[subject])
    my2_all = pd.concat(subject_values_my2[subject])

    print(f"{subject}")
    print(f"  My1 = {my1_all.mean():.3f} ± {my1_all.std():.3f}")
    print(f"  My2 = {my2_all.mean():.3f} ± {my2_all.std():.3f}")

# =========================
# 3. GLOBAL (tous subjects + trials)
# =========================
print("\n===== GLOBAL =====")

all_my1 = []
all_my2 = []

for file in files:
    df = pd.read_csv(file)
    all_my1.append(df["Mz1_glob"].dropna())
    all_my2.append(df["Mz2_glob"].dropna())

all_my1 = pd.concat(all_my1)
all_my2 = pd.concat(all_my2)

print(f"My1 GLOBAL = {all_my1.mean():.3f} ± {all_my1.std():.3f}")
print(f"My2 GLOBAL = {all_my2.mean():.3f} ± {all_my2.std():.3f}")