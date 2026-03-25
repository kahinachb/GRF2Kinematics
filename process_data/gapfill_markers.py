import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def process_and_save_markers(path_to_csv, method='spline', order=3, plot=True):
    df = pd.read_csv(path_to_csv)
    df_filled = df.copy()
    
    marker_cols = [c for c in df.columns if c.lower().endswith(('_x', '_y', '_z'))]
    
    df_with_nans = df_filled[marker_cols].replace(0, np.nan)
    
    gaps_detected = df_with_nans.isna().sum().sum()
    if gaps_detected == 0:
        print("Info : Aucun trou détecté dans le fichier.")
    else:
        print(f"Info : {gaps_detected} points manquants détectés. Début du gapfilling...")

    # Interpolation
    df_filled[marker_cols] = df_with_nans.interpolate(method=method, order=order, limit_direction='both')
    
    df_filled[marker_cols] = df_filled[marker_cols].ffill().bfill()

    if plot and gaps_detected > 0:

        mask_gaps = df_with_nans.isna().any()
        markers_affected = sorted(list(set([c[:-2] for c in mask_gaps[mask_gaps].index])))
        
        for marker in markers_affected[:3]: # Limité aux 3 premiers pour l'exemple
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            fig.suptitle(f"Correction du marker : {marker}")
            for i, ax_name in enumerate(['_x', '_y', '_z']):
                col = next((c for c in df.columns if c.lower() == f"{marker.lower()}{ax_name}"), None)
                if col:
                    axes[i].plot(df.index, df_with_nans[col], 'ro', markersize=2, alpha=0.3, label='Original')
                    axes[i].plot(df_filled.index, df_filled[col], 'b-', linewidth=1, label='Interpolé')
                    axes[i].set_ylabel(ax_name[1:].upper())
            axes[0].legend()
            plt.tight_layout()
            plt.show()


    base, ext = os.path.splitext(path_to_csv)
    output_path = f"{base}_filled{ext}"
    
    df_filled.to_csv(output_path, index=False)
    print(f"Succès : Fichier sauvegardé sous '{output_path}'")
    
    return df_filled, output_path

base_path = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais"
subjects = ['subject01','subject02','subject03', 'subject04', 'subject05', 'subject06', 'subject07', 'subject08', 
            'subject09', 'subject10', 'subject11', 'subject12', 'subject13', 'subject14', 
            'subject15', 'subject16']
task_keywords = ['bend', 'dyna', 'lufe', 'luyo', 'static2', 'walk', 'cmjs']

for subject in subjects:
    subject_dir = os.path.join(base_path, subject)
    
    if not os.path.exists(subject_dir):
        print(f"❌ {subject_dir}")
        continue
        
    print(f"\n--- {subject} ---")
    
    try:
        actual_task_folders = [d for d in os.listdir(subject_dir) if os.path.isdir(os.path.join(subject_dir, d))]
    except OSError:
        continue

    for task_folder in actual_task_folders:
        if any(key in task_folder.lower() for key in task_keywords):
            path_to_csv = os.path.join(subject_dir, task_folder, "markers.csv")
            
            if os.path.exists(path_to_csv):
                process_and_save_markers(path_to_csv, method='spline', order=3, plot=True)
            else:
                pass

print("\n🚀 Traitement terminé sur l'ensemble des sujets.")