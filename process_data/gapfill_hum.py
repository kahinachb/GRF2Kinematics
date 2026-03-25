import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def process_and_save_markers(path_to_csv, method='spline', order=3, plot=True):
    """Lit, gapfill et sauvegarde le fichier avec le suffixe _filled."""
    if not os.path.exists(path_to_csv):
        return
    
    df = pd.read_csv(path_to_csv)
    df_filled = df.copy()
    
    marker_cols = [c for c in df.columns if c.lower().endswith(('_x', '_y', '_z'))]
    
    # 0 or vide -> NaN
    df_with_nans = df_filled[marker_cols].replace(0, np.nan)
    gaps_detected = df_with_nans.isna().sum().sum()
    
    # Interpolation
    df_filled[marker_cols] = df_with_nans.interpolate(method=method, order=order, limit_direction='both')
    df_filled[marker_cols] = df_filled[marker_cols].ffill().bfill()

    # Plot 
    if plot and gaps_detected > 0:
        mask_gaps = df_with_nans.isna().any()
        markers_affected = sorted(list(set([c[:-2] for c in mask_gaps[mask_gaps].index])))
        
        for marker in markers_affected[:1]: # 1 plot par fichier pour ne pas bloquer
            fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
            fig.suptitle(f"Gapfill : {os.path.basename(path_to_csv)} | Marker : {marker}")
            for i, ax_name in enumerate(['_x', '_y', '_z']):
                col = next((c for c in df.columns if c.lower() == f"{marker.lower()}{ax_name}"), None)
                if col:
                    axes[i].plot(df.index, df_with_nans[col], 'ro', markersize=2, alpha=0.3, label='Original')
                    axes[i].plot(df_filled.index, df_filled[col], 'b-', linewidth=1, label='Interpolé')
            axes[0].legend()
            plt.tight_layout()
            plt.show()

    base, ext = os.path.splitext(path_to_csv)
    output_path = f"{base}_filled{ext}"
    df_filled.to_csv(output_path, index=False)
    print(f"✅ Fichier traité : {os.path.basename(output_path)} ({gaps_detected} points corrigés)")

base_path_humanoids = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Vinc"
# subjects = ['Kahina', 'Laure', 'Marie', 'Maxime', 'Mohamed', 'Thanh', 'Thomas', 'Zoe', 'Zoe02', 'Kahina02','Laure02',
#             'Marie02','Maxime02','Mohamed02','Thanh02']
# task_keywords = ['squat']

subjects = ['Christine', 'Jeremy', 'Jovana', 'Maria', 'Serge', 'Subject1', 
                'Vincent']
task_keywords = ['trial']



for subject in subjects:
    subject_dir = os.path.join(base_path_humanoids, subject)
    
    if not os.path.exists(subject_dir):
        print(f"❌: {subject_dir}")
        continue
        
    print(f"\n--- subject : {subject} ---")
    
    all_files = os.listdir(subject_dir)
    
    for filename in all_files:
        if filename.lower().endswith(".csv") and "_filled" not in filename and "_forces" not in filename and "_joints" not in filename:
            if any(key in filename.lower() for key in task_keywords):
                full_path = os.path.join(subject_dir, filename)
                process_and_save_markers(full_path, method='spline', plot=True)

