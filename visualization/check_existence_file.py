import os

DATA_ROOT = "DATA/Anais"

missing_files = []
found_files = []

for subject in os.listdir(DATA_ROOT):
    subject_dir = os.path.join(DATA_ROOT, subject)
    if not os.path.isdir(subject_dir):
        continue

    for task in os.listdir(subject_dir):
        task_dir = os.path.join(subject_dir, task)
        if not os.path.isdir(task_dir):
            continue

        joints_path = os.path.join(task_dir, "joints.csv")

        if os.path.isfile(joints_path):
            found_files.append(joints_path)
        else:
            missing_files.append(joints_path)

print(f"✔️ Fichiers trouvés : {len(found_files)}")
print(f"❌ Fichiers manquants : {len(missing_files)}")

if missing_files:
    print("\nFichiers joints.csv manquants :")
    for f in missing_files:
        print(" -", f)
