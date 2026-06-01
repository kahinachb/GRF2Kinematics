import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

# --- Paramètres du filtre Butterworth ---
fs = 100.0       # Fréquence d'échantillonnage en Hz
cutoff = 10.0    # Fréquence de coupure en Hz
order = 4        # Ordre du filtre (4 est un standard courant)

# Fonction pour créer et appliquer le filtre
def apply_butterworth_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # filtfilt applique le filtre sur chaque colonne (axis=0) sans créer de déphasage
    # padlen est ajusté automatiquement, mais on vérifie que la donnée est assez longue
    if data.shape[0] > 27: # filtfilt a besoin de suffisamment de points
        y = filtfilt(b, a, data, axis=0)
        return y
    else:
        print("Attention : données trop courtes pour être filtrées.")
        return data

# Dossier source et dossier de destination
dossier_source = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/HUMANOIDS"
dossier_destination = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/HUMANOIDS_NPY"

# Les 3 tâches définies
taches = ["squat", "squat_attelle", "squat_attelle_poids"]

# Ordre précis des colonnes souhaité
joints_columns = [
    "root_joint", "root_joint.1", "root_joint.2", "root_joint.3",
    "root_joint.4", "root_joint.5", "root_joint.6", 
    "right_hip_Z", "right_hip_X", "right_hip_Y",
    "right_knee_Z", "right_ankle_Z", "right_ankle_X",
    "left_hip_Z", "left_hip_X", "left_hip_Y",
    "left_knee_Z", "left_ankle_Z", "left_ankle_X", 
    "middle_lumbar_Z", "middle_lumbar_X",
    "left_clavicle_joint_X",
    "left_shoulder_Z", "left_shoulder_X", "left_shoulder_Y",
    "left_elbow_Z", "left_elbow_Y",
    "middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y",
    "right_clavicle_joint_X",
    "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",
    "right_elbow_Z", "right_elbow_Y"
]

kinetics_columns = [
    "Fx1", "Fy1", "Fz1", "Mx1", "My1", "Mz1", "Cx1", "Cy1", "Cz1",
    "Fx2", "Fy2", "Fz2", "Mx2", "My2", "Mz2", "Cx2", "Cy2", "Cz2"
]

# Création du dossier de destination s'il n'existe pas
if not os.path.exists(dossier_destination):
    os.makedirs(dossier_destination)

# Parcours des dossiers des sujets dans le dossier source
for sujet in os.listdir(dossier_source):
    chemin_sujet_source = os.path.join(dossier_source, sujet)
    
    if os.path.isdir(chemin_sujet_source):
        
        for tache in taches:
            csv_joints = os.path.join(chemin_sujet_source, f"{tache}_joints.csv")
            csv_kinetics = os.path.join(chemin_sujet_source, f"{tache}_kinetics.csv")
            
            # Si les deux fichiers existent pour cette tâche
            if os.path.exists(csv_joints) and os.path.exists(csv_kinetics):
                
                chemin_tache_dest = os.path.join(dossier_destination, sujet, tache)
                os.makedirs(chemin_tache_dest, exist_ok=True)
                
                # --- Traitement des joints (Extraction + Filtrage) ---
                try:
                    df_joints = pd.read_csv(csv_joints)
                    data_joints = df_joints[joints_columns].to_numpy()
                    
                    # Application du filtre
                    data_joints_filtered = apply_butterworth_filter(data_joints, cutoff, fs, order)
                    
                    # Sauvegarde
                    np.save(os.path.join(chemin_tache_dest, "joints.npy"), data_joints_filtered)
                except Exception as e:
                    print(f"Erreur lors du traitement de {csv_joints} : {e}")
                    
                # --- Traitement de la kinetics (Extraction + Filtrage) ---
                try:
                    df_kinetics = pd.read_csv(csv_kinetics)
                    data_kinetics = df_kinetics[kinetics_columns].to_numpy()
                    
                    # Application du filtre
                    data_kinetics_filtered = apply_butterworth_filter(data_kinetics, cutoff, fs, order)
                    
                    # Sauvegarde
                    np.save(os.path.join(chemin_tache_dest, "kinetics.npy"), data_kinetics_filtered)
                except Exception as e:
                    print(f"Erreur lors du traitement de {csv_kinetics} : {e}")
            else:
                # Optionnel : avertissement si la tâche est manquante
                print(f"Information: Fichiers manquants pour le sujet '{sujet}' (Tâche: {tache}). Ignorée.")

print("Conversion et filtrage terminés !")