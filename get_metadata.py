import pandas as pd
import numpy as np
import yaml
import os

def extract_height_from_markers(markers_file):
    """
    Extract height from marker data using the first frame (neutral position).
    Height is estimated as the vertical distance from foot to head markers.
    Automatically detects marker set type.
    
    Parameters:
    -----------
    markers_file : str
        Path to the markers CSV file
        
    Returns:
    --------
    float : Estimated height in the same units as marker data (likely mm or m)
    """
    # Read the markers file
    df = pd.read_csv(markers_file)
    
    # Get column names to detect marker set type
    columns = df.columns.tolist()
    
    # Use first frame (index 0) for neutral standing position
    first_frame = df.iloc[0]
    
    # Detect which marker set is being used
    if 'LFHD_Z' in columns:
        # Original marker set
        print("Detected marker set: Original (LFHD, RFHD, LTOE, RTOE)")
        
        # Get the position of head markers from first frame
        head_z = (first_frame['LFHD_Z'] + first_frame['RFHD_Z']) / 2
        
        # Get the position of toe markers from first frame
        toe_z = (first_frame['LTOE_Z'] + first_frame['RTOE_Z']) / 2
        
        # Alternative with heel
        heel_z = (first_frame['LHEE_Z'] + first_frame['RHEE_Z']) / 2
        height_from_heel = head_z - heel_z
        
        # Calculate height
        height = head_z - toe_z
        
        print(f"Height (head to toe): {height:.2f}")
        print(f"Height (head to heel): {height_from_heel:.2f}")
        
    elif 'SV_z' in columns:
        # New marker set with SV (vertex)
        print("Detected marker set: New (SV, RFM1, LFM1)")
        
        # SV = vertex (top of head)
        head_z = first_frame['SV_z']
        
        # Use first metatarsal markers (RFM1, LFM1) or calcaneus (RFCC, LFCC)
        # First metatarsal (toe)
        if 'RFM1_z' in columns and 'LFM1_z' in columns:
            toe_z = (first_frame['RFM1_z'] + first_frame['LFM1_z']) / 2
            print(f"Using metatarsal markers (RFM1, LFM1) for foot reference")
        else:
            toe_z = None
        
        # Calcaneus (heel)
        if 'RFCC_z' in columns and 'LFCC_z' in columns:
            heel_z = (first_frame['RFCC_z'] + first_frame['LFCC_z']) / 2
            height_from_heel = head_z - heel_z
            print(f"Height (head to heel/calcaneus): {height_from_heel:.2f}")
        else:
            height_from_heel = None
        
        # Calculate height
        if toe_z is not None:
            height = head_z - toe_z
            print(f"Height (head to toe/metatarsal): {height:.2f}")
        elif height_from_heel is not None:
            height = height_from_heel
            print(f"Using heel-based height: {height:.2f}")
        else:
            raise ValueError("Could not find appropriate foot markers for height calculation")
    
    else:
        raise ValueError("Unknown marker set. Could not find expected head or foot markers.")
    
    return height


def extract_weight_from_forces(forces_file):
    """
    Extract weight from force plate data using the first frame (neutral position).
    Weight is calculated from the vertical ground reaction force (FZ) during standing.
    
    Parameters:
    -----------
    forces_file : str
        Path to the forces CSV file
        
    Returns:
    --------
    tuple : (weight_N, weight_kg) - Weight in Newtons and kilograms
    """
    # Read the forces file
    df = pd.read_csv(forces_file)
    columns = df.columns.tolist()
    
    # Use first frame for neutral standing position
    first_frame = df.iloc[0]
    
    # Calculate total vertical force (sum of both force plates)
    # FZ1 and FZ2 are vertical forces (typically negative in standard configuration)
    if 'FZ1' in columns:
        total_fz = first_frame['FZ1'] + first_frame['FZ2']
    elif 'Fz1' in columns:
        total_fz = first_frame['Fz1'] + first_frame['Fz2']
    else:
        raise ValueError("Could not find force columns (FZ1/FZ2 or Fz1/Fz2)")
    
    # Weight is the absolute value of the vertical force during standing
    # We use absolute value because force plates often record downward forces as negative
    weight_N = abs(total_fz)
    
    # Calculate weight in kg (assuming g = 9.81 m/s²)
    weight_kg = weight_N / 9.81
    
    print(f"Weight: {weight_N:.2f} N ({weight_kg:.2f} kg)")
    
    return weight_N, weight_kg


def main(markers_file, forces_file, subject, output_dir=None):
    """
    Main function to extract height and weight from biomechanical data.
    Saves results as subject.yaml in the subject folder.
    
    Parameters:
    -----------
    markers_file : str
        Path to the markers CSV file
    forces_file : str
        Path to the forces CSV file
    output_dir : str, optional
        Directory to save the subject.yaml file. If None, uses the directory of markers_file
    """
    print("=" * 60)
    print("Extracting Height and Weight from Biomechanical Data")
    print("=" * 60)
    
    print("\n--- Extracting Height ---")
    height = extract_height_from_markers(markers_file)
    
    print("\n--- Extracting Weight ---")
    weight_N, weight_kg = extract_weight_from_forces(forces_file)
    
    # Convert height to meters if it appears to be in millimeters
    height_m = height / 1000 if height > 100 else height
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Height: {height:.2f} mm ({height_m:.2f} m)")
    print(f"Weight: {weight_N:.2f} N ({weight_kg:.2f} kg)")
    print("=" * 60)
    
    # Prepare results dictionary for YAML
    results = {
        'height_mm': float(height),
        'height_m': float(height_m),
        'weight_N': float(weight_N),
        'weight_kg': float(weight_kg)
    }
    
    # Determine output directory
    if output_dir is None:
        output_dir = f"/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/Anais/{subject}"
    
    # Create output path for subject.yaml
    yaml_path = os.path.join(output_dir, f'{subject}.yaml')
    
    # Save results to YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nResults saved to: {yaml_path}")
    
    return results


if __name__ == "__main__":
    subjects = ["subject01", "subject02", "subject03","subject04", "subject05", "subject06","subject07", "subject08", "subject09",
                "subject10", "subject11", "subject12", "subject13", "subject14", "subject15", "subject16"]
    task ="static2" 
    for subject in subjects:
        markers_file = f"DATA/Anais/{subject}/{task}/markers.csv"
        forces_file = f"DATA/Anais/{subject}/{task}/kinetics.csv"
        
    # Run the extraction
    # Output will be saved as subject.yaml in the same directory as markers_file
        try:
            results = main(markers_file, forces_file, subject)
        except:
            print(f'{task} not found')
    print("\nDone! Check subject.yaml in the subject folder.")