import torch
joint_limits = {
    # --- LEFT LEG ---
    "left_hip_Z": [-1.047196667, 3.14159],
    "left_hip_X": [-1.047196667, 1.570795],
    "left_hip_Y": [-1.570795, 1.047196667],
    "left_knee_Z": [0.0, 3.14159],
    "left_ankle_Z": [-1.570795, 0.7853975],
    "left_ankle_X": [-0.523598333, 0.785398],

    # --- RIGHT LEG ---
    "right_hip_Z": [-1.047196667, 3.14159],
    "right_hip_X": [-1.047196667, 1.570795],
    "right_hip_Y": [-1.570795, 1.047196667],
    "right_knee_Z": [0.0, 3.14159],
    "right_ankle_Z": [-1.570795, 0.7853975],
    "right_ankle_X": [-0.523598333, 0.785398],

    # --- SPINE ---
    "middle_lumbar_Z": [-0.785398, 3.14159],
    "middle_lumbar_X": [-0.785398, 0.785398],

    # --- LEFT ARM ---
    "left_clavicle_joint_X": [-0.349066, 1.0472],
    "left_shoulder_Z": [-3.14159, 3.14159],
    "left_shoulder_X": [-1.0472, 3.14159],
    "left_shoulder_Y": [-1.5708, 3.14159],
    "left_elbow_Z": [0.0, 2.617991667],
    "left_elbow_Y": [-0.349066, 3.14159],

    # --- RIGHT ARM ---
    "right_clavicle_joint_X": [-0.349066, 1.0472],
    "right_shoulder_Z": [-3.14159, 3.14159],
    "right_shoulder_X": [-1.0472, 3.14159],
    "right_shoulder_Y": [-1.5708, 3.14159],
    "right_elbow_Z": [0.0, 2.617991667],
    "right_elbow_Y": [-0.349066, 3.14159],

    # --- HEAD ---
    "middle_cervical_Z": [-1.5708, 1.5708],
    "middle_cervical_X": [-1.5708, 1.5708],
    "middle_cervical_Y": [-1.5708, 1.5708],
}

def get_limit_tensors(joint_limits, device):
    # Ordre exact de tes joints (35 colonnes)
    joint_names = [
        "delta_x","delta_y","delta_z","delta_rx","delta_ry","delta_rz", # 0-5
        "right_hip_Z", "right_hip_X", "right_hip_Y",                   # 6-8
        "right_knee_Z", "right_ankle_Z", "right_ankle_X",              # 9-11
        "left_hip_Z", "left_hip_X", "left_hip_Y",                      # 12-14
        "left_knee_Z", "left_ankle_Z", "left_ankle_X",                 # 15-17
        "middle_lumbar_Z", "middle_lumbar_X",                          # 18-19
        "left_clavicle_joint_X",                                       # 20
        "left_shoulder_Z", "left_shoulder_X", "left_shoulder_Y",       # 21-23
        "left_elbow_Z", "left_elbow_Y",                                # 24-25
        "middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y", # 26-28
        "right_clavicle_joint_X",                                      # 29
        "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",    # 30-32
        "right_elbow_Z", "right_elbow_Y"                               # 33-34
    ]
    
    q_min = torch.tensor([-1e6]*6 + [joint_limits[name][0] for name in joint_names[6:]], device=device)
    q_max = torch.tensor([1e6]*6 + [joint_limits[name][1] for name in joint_names[6:]], device=device)
    
    return q_min, q_max