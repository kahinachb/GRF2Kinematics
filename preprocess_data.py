"""
Data Preprocessing Script
Converts CSV files containing force and joint angle data to NPY format
for LSTM training on biomechanical data.

Data organization expected:
DATA/subject/trial107_forces.csv and trial107_joints.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import re

class BiomechanicsDataProcessor:
    """Process biomechanical force and joint angle data from CSV to NPY format."""
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing CSV files organized by subject with trial files
            output_dir: Directory to save processed NPY files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define force columns (6D GRFM for each foot)
        self.force_columns_right = ['FX1', 'FY1', 'FZ1', 'MX1', 'MY1', 'MZ1']
        self.force_columns_left = ['FX2', 'FY2', 'FZ2', 'MX2', 'MY2', 'MZ2']
        
        # Define joint angle columns
        self.angle_columns_left = [
            'left_hip_Z', 'left_hip_X', 'left_hip_Y',
            'left_knee_Z', 'left_ankle_Z', 'left_ankle_X'
        ]
        self.angle_columns_right = [
            'right_hip_Z', 'right_hip_X', 'right_hip_Y',
            'right_knee_Z', 'right_ankle_Z', 'right_ankle_X'
        ]
        
    def process_csv_file(self, force_file: Path, angle_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single pair of force and angle CSV files.
        
        Args:
            force_file: Path to force data CSV
            angle_file: Path to joint angle data CSV
            
        Returns:
            Tuple of (forces_array, joints_array)
        """
        # Read force data
        force_df = pd.read_csv(force_file)
        print("shape force", force_df.shape )
        
        # Check if columns exist, print available columns if not
        missing_force_cols = []
        for col in self.force_columns_right + self.force_columns_left:
            if col not in force_df.columns:
                missing_force_cols.append(col)
        
        if missing_force_cols:
            print(f"Warning: Missing force columns {missing_force_cols}")
            print(f"Available columns in {force_file.name}: {list(force_df.columns)[:20]}...")
            # Try to find similar columns
            self._suggest_similar_columns(force_df.columns, missing_force_cols)
        
        forces_right = force_df[self.force_columns_right].values
        forces_left = force_df[self.force_columns_left].values
        forces_combined = np.concatenate([forces_left, forces_right], axis=1)  # Shape: (timesteps, 12)
        print("len(forces_combined)",len(forces_combined))
        
        # Read angle data
        angle_df = pd.read_csv(angle_file)
        print("angle_df shape", angle_df.shape )
        
        # Check if columns exist
        missing_angle_cols = []
        for col in self.angle_columns_left + self.angle_columns_right:
            if col not in angle_df.columns:
                missing_angle_cols.append(col)
        
        if missing_angle_cols:
            print(f"Warning: Missing angle columns {missing_angle_cols}")
            print(f"Available columns in {angle_file.name}: {list(angle_df.columns)[:20]}...")
            self._suggest_similar_columns(angle_df.columns, missing_angle_cols)
        
        joints_left = angle_df[self.angle_columns_left].values
        joints_right = angle_df[self.angle_columns_right].values
        joints_combined = np.concatenate([joints_left, joints_right], axis=1)  # Shape: (timesteps, 12)
        print("len(forces_combined)",len(forces_combined))
        
        # Ensure same length (trim to minimum if needed)
        min_len = min(len(forces_combined), len(forces_combined))
        print("min_len", min_len)
        forces_combined = forces_combined[:]
        joints_combined = joints_combined[:]
        
        return forces_combined, joints_combined
    
    def _suggest_similar_columns(self, available_cols, missing_cols):
        """Suggest similar column names that might be the correct ones."""
        for missing in missing_cols:
            similar = [col for col in available_cols if missing.lower() in col.lower() or col.lower() in missing.lower()]
            if similar:
                print(f"  Possible match for '{missing}': {similar}")
    
    def organize_by_subject_trials(self) -> Dict[str, Dict[str, Dict[str, Path]]]:
        """
        Organize files by subject and trial.
        
        Expected directory structure:
        DATA/
            subject1/
                trial107_forces.csv
                trial107_joints.csv
                trial108_forces.csv
                trial108_joints.csv
                ...
            subject2/
                ...
        
        Returns:
            Nested dictionary: {subject: {trial: {type: file_path}}}
        """
        organized_data = {}
        
        # Iterate through subjects
        for subject_dir in self.data_dir.iterdir():
            if not subject_dir.is_dir():
                continue
                
            subject_name = subject_dir.name
            organized_data[subject_name] = {}
            
            # Find all force and joint files
            force_files = {}
            joint_files = {}
            
            for file_path in subject_dir.iterdir():
                if not file_path.is_file() or not file_path.suffix == '.csv':
                    continue
                
                filename = file_path.name.lower()
                
                # Extract trial number using regex
                # Handles formats like: trial107_forces.csv, trial_107_forces.csv, etc.
                trial_match = re.search(r'trial[_]?(\d+)', filename)
                
                if trial_match:
                    trial_num = trial_match.group(1)
                    trial_name = f"trial{trial_num}"
                    
                    if 'force' in filename:
                        force_files[trial_name] = file_path
                    elif 'joint' in filename:
                        joint_files[trial_name] = file_path
            
            # Match force and joint files by trial
            for trial_name in force_files.keys():
                if trial_name in joint_files:
                    organized_data[subject_name][trial_name] = {
                        'forces': force_files[trial_name],
                        'joints': joint_files[trial_name]
                    }
                else:
                    print(f"Warning: No matching joint file for {subject_name}/{trial_name}")
            
            # Check for unmatched joint files
            for trial_name in joint_files.keys():
                if trial_name not in force_files:
                    print(f"Warning: No matching force file for {subject_name}/{trial_name}")
        
        return organized_data
    
    def process_all_data(self):
        """Process all data files and save as NPY format."""
        print("Organizing data by subject and trial...")
        organized_data = self.organize_by_subject_trials()
        
        if not organized_data:
            print("No data found! Please check your data directory structure.")
            print(f"Looking in: {self.data_dir}")
            return
        
        # Process each subject
        for subject_name, subject_trials in organized_data.items():
            print(f"\nProcessing subject: {subject_name}")
            subject_dir = self.output_dir / subject_name
            subject_dir.mkdir(exist_ok=True)
            
            # Process each trial
            for trial_name, file_paths in subject_trials.items():
                print(f"  Processing {trial_name}")
                
                try:
                    # Process CSV files
                    forces, joints = self.process_csv_file(
                        file_paths['forces'],
                        file_paths['joints']
                    )
                    
                    # Save as NPY files - use trial name as folder
                    trial_dir = subject_dir / trial_name
                    trial_dir.mkdir(exist_ok=True)
                    
                    np.save(trial_dir / 'forces.npy', forces.astype(np.float32))
                    np.save(trial_dir / 'joints.npy', joints.astype(np.float32))
                    
                    print(f"    Saved: forces shape {forces.shape}, joints shape {joints.shape}")
                    
                except Exception as e:
                    print(f"    Error processing {subject_name}/{trial_name}: {str(e)}")
        
        print("\nData processing complete!")
        self.print_summary()
    
    def print_summary(self):
        """Print summary of processed data."""
        print("\n" + "="*50)
        print("PROCESSED DATA SUMMARY")
        print("="*50)
        
        total_subjects = 0
        total_trials = 0
        
        for subject_dir in self.output_dir.iterdir():
            if not subject_dir.is_dir():
                continue
            
            total_subjects += 1
            subject_trials = 0
            
            for trial_dir in subject_dir.iterdir():
                if not trial_dir.is_dir():
                    continue
                
                force_file = trial_dir / 'forces.npy'
                angle_file = trial_dir / 'joints.npy'
                
                if force_file.exists() and angle_file.exists():
                    subject_trials += 1
                    total_trials += 1
                    
                    # Load to check shapes
                    forces = np.load(force_file)
                    joints = np.load(angle_file)
                    print(f"{subject_dir.name}/{trial_dir.name}: {forces.shape[0]} timesteps")
            
            print(f"Subject {subject_dir.name}: {subject_trials} trials")
        
        print(f"\nTotal: {total_subjects} subjects, {total_trials} trials")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Preprocess biomechanical data from CSV to NPY')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing CSV files (e.g., DATA/ with subject folders containing trial files)')
    parser.add_argument('--output_dir', type=str, default='./processed_data',
                        help='Directory to save processed NPY files')
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = BiomechanicsDataProcessor(args.data_dir, args.output_dir)
    processor.process_all_data()


if __name__ == "__main__":
    main()