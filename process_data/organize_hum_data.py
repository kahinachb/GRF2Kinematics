import os
import shutil

base_path = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/HUMANOIDS"

subjects = ['Kahina', 'Laure', 'Marie', 'Maxime', 'Mohamed', 'Thanh', 'Thomas', 'Zoe', 'Zoe02', 'Kahina02',
            'Laure02', 'Marie02', 'Maxime02', 'Mohamed02', 'Thanh02']

for subject in subjects:
    subject_path = os.path.join(base_path, subject)
    
    # Vérifie que le dossier existe
    if not os.path.isdir(subject_path):
        print(f"Dossier manquant: {subject_path}")
        continue

    # Nouveau dossier destination
    dest_path = os.path.join(base_path, f"{subject}_8kg")
    os.makedirs(dest_path, exist_ok=True)

    # Parcours des fichiers
    for file_name in os.listdir(subject_path):
        if "8kg" in file_name:
            src_file = os.path.join(subject_path, file_name)
            dest_file = os.path.join(dest_path, file_name)

            # Déplacement
            shutil.move(src_file, dest_file)
            print(f"Déplacé: {src_file} -> {dest_file}")