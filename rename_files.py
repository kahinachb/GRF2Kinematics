import os

base_dir = "processed_data"
subjects = ["Christine","Jeremy","Jovana","Maria","Serge","Vincent"]
for subject in os.listdir(base_dir):
    if subject in subjects:
        subject_dir = os.path.join(base_dir, subject)
    else:
        continue
    if not os.path.isdir(subject_dir):
        continue

    for task in os.listdir(subject_dir):
        task_dir = os.path.join(subject_dir, task)
        if not os.path.isdir(task_dir):
            continue

        old_forces = os.path.join(task_dir, "forces_300.npy")
        new_forces = os.path.join(task_dir, "forces.npy")

        old_joints = os.path.join(task_dir, "joints_300.npy")
        new_joints = os.path.join(task_dir, "joints.npy")

        if os.path.exists(old_forces):
            os.rename(old_forces, new_forces)
            print(f"Renamed: {old_forces} -> {new_forces}")

        if os.path.exists(old_joints):
            os.rename(old_joints, new_joints)
            print(f"Renamed: {old_joints} -> {new_joints}")
