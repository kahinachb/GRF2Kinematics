import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=7, fs=300, order=4):
    """Applique un filtre Butterworth passe-bas sur l'axe du temps."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def process_grf_to_pelvis(path_grf, path_pos_pelvis, cutoff=7, fs=300):
    # 1. Chargement des données
    grf_df = pd.read_csv(path_grf)
    pos_df = pd.read_csv(path_pos_pelvis)
    
    # On récupère la position du pelvis (X, Y, Z dans le labo)
    # Ajuste les colonnes [1:4] si ton fichier joints_corr.csv est différent
    pos_pelvis = pos_df.iloc[:, 1:4].to_numpy() 

    # 2. Extraction des forces et moments globaux (Origin 0,0,0)
    f1 = grf_df[['Fx1', 'Fy1', 'Fz1']].to_numpy()
    m1_glob = grf_df[['Mx1_glob', 'My1_glob', 'Mz1_glob']].to_numpy() / 1000.0 # Nmm -> Nm
    
    f2 = grf_df[['Fx2', 'Fy2', 'Fz2']].to_numpy()
    m2_glob = grf_df[['Mx2_glob', 'My2_glob', 'Mz2_glob']].to_numpy() / 1000.0

    # 3. Filtrage (Force, Moment et Position)
    f1_f = butter_lowpass_filter(f1, cutoff, fs)
    m1_f = butter_lowpass_filter(m1_glob, cutoff, fs)
    
    f2_f = butter_lowpass_filter(f2, cutoff, fs)
    m2_f = butter_lowpass_filter(m2_glob, cutoff, fs)
    
    p_f = butter_lowpass_filter(pos_pelvis, cutoff, fs)

    # 4. Calcul du transport de moment séparé
    # Formule : M_pelvis = M_glob - (pos_pelvis x Force)
    n_samples = len(f1_f)
    m1_pelvis = np.zeros((n_samples, 3))
    m2_pelvis = np.zeros((n_samples, 3))

    for i in range(n_samples):
        # Plateforme 1
        m1_pelvis[i, :] = m1_f[i] - np.cross(p_f[i], f1_f[i])
        # Plateforme 2
        m2_pelvis[i, :] = m2_f[i] - np.cross(p_f[i], f2_f[i])

    # 5. Création du DataFrame final
    res = pd.DataFrame({
        'time': np.arange(n_samples) / fs,
        # Plateforme 1 au pelvis
        'Fx1': f1_f[:, 0], 'Fy1': f1_f[:, 1], 'Fz1': f1_f[:, 2],
        'Mx1_pelvis': m1_pelvis[:, 0], 'My1_pelvis': m1_pelvis[:, 1], 'Mz1_pelvis': m1_pelvis[:, 2],
        # Plateforme 2 au pelvis
        'Fx2': f2_f[:, 0], 'Fy2': f2_f[:, 1], 'Fz2': f2_f[:, 2],
        'Mx2_pelvis': m2_pelvis[:, 0], 'My2_pelvis': m2_pelvis[:, 1], 'Mz2_pelvis': m2_pelvis[:, 2]
    })
    
    return res

# --- Utilisation ---
path_grf = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais/subject01/cmjs/kinetics.csv"
path_joints= f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais/subject01/cmjs/joints_corr.csv"
path_output = f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais/subject01/cmjs/kinetics_filtered_pelvis.csv"
df_final = process_grf_to_pelvis(path_grf, path_joints)
df_final.to_csv(path_output, index=False)