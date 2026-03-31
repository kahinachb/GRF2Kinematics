import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from scipy import signal  # Nécessaire pour le rééchantillonnage

class BiomechanicsDataProcessor:
    """Process biomechanical force and joint angle data with downsampling to 100Hz."""
    
    def __init__(self, data_dir: str, output_dir: str, target_fs: int = 100):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_fs = target_fs
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Colonnes de forces (Kinetics)
        self.force_columns_left = ['Fx1', 'Fy1', 'Fz1', 'Mx1_glob', 'My1_glob', 'Mz1_glob']
        self.force_columns_right = ['Fx2', 'Fy2', 'Fz2', 'Mx2_glob', 'My2_glob', 'Mz2_glob']
        
        # Colonnes d'angles (Joints)
        self.angle_columns_left = [
            'left_hip_Z', 'left_hip_X', 'left_hip_Y',
            'left_knee_Z', 'left_ankle_Z', 'left_ankle_X'
        ]
        self.angle_columns_right = [
            'right_hip_Z', 'right_hip_X', 'right_hip_Y',
            'right_knee_Z', 'right_ankle_Z', 'right_ankle_X'
        ]
        
    def _resample(self, data: np.ndarray, current_fs: int) -> np.ndarray:
        """Rééchantillonne les données vers target_fs (100Hz)."""
        if current_fs == self.target_fs:
            return data
        
        # Calcul du nouveau nombre de points
        duration_sec = len(data) / current_fs
        new_n_samples = int(duration_sec * self.target_fs)
        
        # Resample le long de l'axe du temps (axis=0)
        return signal.resample(data, new_n_samples, axis=0)

    def process_csv_file(self, force_file: Path, angle_file: Path, current_fs: int) -> Tuple[np.ndarray, np.ndarray]:
        # Lecture des forces
        force_df = pd.read_csv(force_file)
        forces_combined = np.concatenate([
            force_df[self.force_columns_left].values, 
            force_df[self.force_columns_right].values
        ], axis=1)
        
        # Lecture des angles
        angle_df = pd.read_csv(angle_file)
        joints_combined = np.concatenate([
            angle_df[self.angle_columns_left].values, 
            angle_df[self.angle_columns_right].values
        ], axis=1)
        
        # Synchronisation initiale des longueurs
        min_len = min(len(forces_combined), len(joints_combined))
        forces_combined = forces_combined[:min_len]
        joints_combined = joints_combined[:min_len]
        
        # --- RÉÉCHANTILLONNAGE À 100Hz ---
        if current_fs != self.target_fs:
            forces_combined = self._resample(forces_combined, current_fs)
            joints_combined = self._resample(joints_combined, current_fs)
        
        return forces_combined, joints_combined

    def organize_by_subject_trials(self) -> Dict[str, Dict[str, Dict[str, Path]]]:
        organized_data = {}
        all_csv = list(self.data_dir.rglob("*.csv"))
        
        for file_path in all_csv:
            filename = file_path.name.lower()
            parts = file_path.parts 
            
            if len(parts) < 3: continue
                
            subject_name = parts[-3]
            task_name = parts[-2]
            trial_id = f"{subject_name}_{task_name}"
            
            if subject_name not in organized_data:
                organized_data[subject_name] = {}
            if trial_id not in organized_data[subject_name]:
                organized_data[subject_name][trial_id] = {}

            if 'kinetics_filtered' in filename:
                organized_data[subject_name][trial_id]['forces'] = file_path
            elif 'joints' in filename:
                organized_data[subject_name][trial_id]['joints'] = file_path

        final_data = {}
        for subj, trials in organized_data.items():
            valid_trials = {t: p for t, p in trials.items() if 'forces' in p and 'joints' in p}
            if valid_trials:
                final_data[subj] = valid_trials
        return final_data

    def process_all_data(self):
        print(f"Analyse du dossier : {self.data_dir}")
        organized_data = self.organize_by_subject_trials()
        
        for subject_name, subject_trials in organized_data.items():
            # DÉTECTION DE LA FRÉQUENCE : 300Hz si "subject" est dans le nom, sinon 100Hz
            current_fs = 300 if "subject" in subject_name.lower() else 100
            print(f"\nSujet : {subject_name} ({current_fs}Hz -> {self.target_fs}Hz)")
            
            subject_dir = self.output_dir / subject_name
            subject_dir.mkdir(exist_ok=True)
            
            for trial_id, file_paths in subject_trials.items():
                try:
                    forces, joints = self.process_csv_file(file_paths['forces'], file_paths['joints'], current_fs)
                    
                    trial_dir = subject_dir / trial_id
                    trial_dir.mkdir(exist_ok=True)
                    
                    np.save(trial_dir / 'forces_100.npy', forces.astype(np.float32))
                    np.save(trial_dir / 'joints_100.npy', joints.astype(np.float32))
                    print(f"  [OK] {trial_id} : {forces.shape[0]} timesteps (100Hz)")
                except Exception as e:
                    print(f"  [ERREUR] {trial_id} : {e}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess data with Resampling to 100Hz')
    parser.add_argument('--data_dir', type=str, required=True, help='Chemin vers DATA/')
    parser.add_argument('--output_dir', type=str, default='./processed_data_100Hz', help='Dossier de sortie')
    
    args = parser.parse_args()
    processor = BiomechanicsDataProcessor(args.data_dir, args.output_dir)
    processor.process_all_data()

if __name__ == "__main__":
    main()