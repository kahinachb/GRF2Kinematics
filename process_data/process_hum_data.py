import pandas as pd
import numpy as np
import os
def transform_pf_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    Rm = np.array([
        [ -1, 0,  0],
        [0,  -1,  0],
        [ 0,  0, -1]
    ])
    # print(np.linalg.det(Rm))
    
    center_fp1 = np.array([300, 250, 0])
    center_fp2 = np.array([300, -340, 0])
    
    results = []

    def calculate_global(row, prefix, center):
        f_loc = np.array([row[f'Fx{prefix}'], row[f'Fy{prefix}'], row[f'Fz{prefix}']])
        m_loc = np.array([row[f'Mx{prefix}'], row[f'My{prefix}'], row[f'Mz{prefix}']])

 
        #  F_global = Rm @ F_local
        f_glob = Rm @f_loc
        
        # M_global = Rm @ M_local + (Center x F_global)
        m_glob = (Rm @m_loc) + np.cross(center, f_glob)
        
        # cop in global frame
        if abs(f_glob[2]) > 0.1: 
            cop_x = -m_glob[1] / f_glob[2]
            cop_y = m_glob[0] / f_glob[2]
        else:
            cop_x, cop_y = np.nan, np.nan
            
        return {
            f'Fx{prefix}_glob': f_glob[0], f'Fy{prefix}_glob': f_glob[1], f'Fz{prefix}_glob': f_glob[2],
            f'Mx{prefix}_glob': m_glob[0], f'My{prefix}_glob': m_glob[1], f'Mz{prefix}_glob': m_glob[2],
            f'COPx{prefix}_glob': cop_x, f'COPy{prefix}_glob': cop_y
        }

    new_data = []
    for _, row in df.iterrows():
        res1 = calculate_global(row, '1', center_fp1)
        res2 = calculate_global(row, '2', center_fp2)
        # Fusion des dictionnaires
        new_data.append({**res1, **res2})
    
    df_transformed = pd.DataFrame(new_data)
    
    df_transformed.to_csv(output_csv, index=False)
    print(f"Traitement terminé. Données sauvegardées dans : {output_csv}")

subjects = ['Kahina', 'Laure', 'Marie', 'Maxime', 'Mohamed', 'Thanh', 'Thomas', 'Zoe', 'Zoe02', 'Kahina02',
            'Laure02', 'Marie02', 'Maxime02', 'Mohamed02', 'Thanh02']

base_path = "/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/HUMANOIDS"

all_subjects = os.listdir(base_path)

for subject in all_subjects:
    
    if not any(subject.startswith(base) for base in subjects):
        continue
    
    subject_path = os.path.join(base_path, subject)

    if not os.path.isdir(subject_path):
        continue

    for file in os.listdir(subject_path):
        if file.endswith("_kinetics.csv"):

            task = file.replace("_kinetics.csv", "")

            input_csv = os.path.join(subject_path, file)
            output_csv = os.path.join(subject_path, f"{task}_kinetics_global.csv")

            if os.path.exists(output_csv):
                print(f"Skip {output_csv}")
                continue

            transform_pf_data(input_csv, output_csv)