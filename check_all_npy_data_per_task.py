import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================================
# 1) MAPPING MANUEL POUR LA PREMIERE CATEGORIE
# ==========================================================

manual_mapping = {
    "vincent": {
        "trial109": "hulahoop",
        "trial110": "hulahoop",
        "trial112": "squat",
        "trial113": "baisser_haut"
    },

    "jovana": {
        "trial108": "hulahoop",
        "trial109": "hulahoop",
        "trial110": "hulahoop",
        "trial111": "squat",
        "trial112": "baisser_haut"
    },

    "christine": {
        "trial107": "hulahoop",
        "trial108": "gauche_droit",
        "trial109": "hulahoop",
        "trial110": "squat",
        "trial111": "baisser_haut"
    },

    "jeremy": {
        "trial108": "hulahoop",
        "trial109": "gauche_droit",
        "trial110": "gauche_droit",
        "trial111": "squat"
    },

    "maria": {
        "trial108": "hulahoop",
        "trial109": "hulahoop",
        "trial112": "baisser_haut",
        "trial114": "squat"
    },

    "serge": {
        "trial108": "hulahoop",
        "trial109": "hulahoop",
        "trial110": "hulahoop",
        "trial111": "squat"
    },

    "subject1": {
        "trial108": "gauche_droit",
        "trial109": "hulahoop",
        "trial110": "hulahoop",
        "trial111": "squat"
    }
}


# ==========================================================
# 2) NORMALISATION AUTOMATIQUE DES NOMS
# ==========================================================

def normalize_task_name(name: str):
    """
    Regroupe automatiquement les variantes.
    """

    name = name.lower()

    # ---------------- WALK ----------------
    if "walk" in name:
        return "walk"

    # ---------------- STATIC ----------------
    if "static" in name:
        return "static"

    # ---------------- SQUAT ----------------
    if "squat" in name:
        return "squat"

    # ---------------- BEND ----------------
    if "bend" in name:
        return "bend"

    # ---------------- DYNA ----------------
    if "dyna" in name:
        return "dyna"

    # ---------------- LUFE ----------------
    if "lufe" in name:
        return "lufe"

    # ---------------- LUYO ----------------
    if "luyo" in name:
        return "luyo"

    return "other"

# ==========================================================
# 3) DETECTION DE LA TACHE
# ==========================================================

def get_task(subject_name, trial_name):

    subject_lower = subject_name.lower()
    trial_lower = trial_name.lower()

    # --------------------------------------
    # CAS 1 : mapping manuel
    # --------------------------------------

    if subject_lower in manual_mapping:

        if trial_lower in manual_mapping[subject_lower]:
            return manual_mapping[subject_lower][trial_lower]
        else:
            return "unknown"

    # --------------------------------------
    # CAS 2 : détection automatique
    # --------------------------------------

    return normalize_task_name(trial_lower)


# ==========================================================
# 4) ANALYSE PRINCIPALE
# ==========================================================

def analyze_tasks(dataset_root):

    root = Path(dataset_root)

    if not root.exists():
        print("Dataset introuvable")
        return

    # statistiques par tâche
    task_stats = defaultdict(lambda: {
        "frames": 0,
        "seconds": 0,
        "files": 0
    })

    total_frames = 0
    total_seconds = 0

    # ======================================================
    # Parcours des fichiers
    # ======================================================
    for kinetics_file in root.rglob("kinetics_feet.npy"):

        try:

            trial_dir = kinetics_file.parent
            subject_dir = trial_dir.parent

            subject_name = subject_dir.name
            trial_name = trial_dir.name

            # --------------------------------------------------
            # Détection de la tâche
            # --------------------------------------------------

            task = get_task(subject_name, trial_name)

            # --------------------------------------------------
            # Détection fréquence
            # --------------------------------------------------

            filename = kinetics_file.name.lower()

            if "300" in filename:
                fs = 300
            else:
                fs = 100

            # --------------------------------------------------
            # Chargement
            # --------------------------------------------------

            data = np.load(kinetics_file, mmap_mode='r')

            n_frames = data.shape[0]
            duration = n_frames / fs

            # --------------------------------------------------
            # Stats
            # --------------------------------------------------

            task_stats[task]["frames"] += n_frames
            task_stats[task]["seconds"] += duration
            task_stats[task]["files"] += 1

            total_frames += n_frames
            total_seconds += duration

        except Exception as e:
            print(f"Erreur avec {kinetics_file}: {e}")

    # ======================================================
    # AFFICHAGE FINAL
    # ======================================================
    print("\n" + "="*80)
    print("REPARTITION DES TACHES")
    print("="*80)

    print(f"{'TASK':<15} | {'FILES':<8} | {'FRAMES':<12} | {'SECONDS':<12} | {'% FRAMES'}")
    print("-"*80)

    sorted_tasks = sorted(
        task_stats.items(),
        key=lambda x: x[1]["frames"],
        reverse=True
    )

    for task, stats in sorted_tasks:

        percentage = 100 * stats["frames"] / total_frames

        print(
            f"{task:<15} | "
            f"{stats['files']:<8} | "
            f"{stats['frames']:<12} | "
            f"{stats['seconds']:<12.2f} | "
            f"{percentage:.2f}%"
        )

    print("\n" + "="*80)
    print(f"TOTAL FRAMES  : {total_frames}")
    print(f"TOTAL SECONDS : {total_seconds:.2f}")
    print(f"TOTAL HOURS   : {total_seconds/3600:.2f}")


def run_task_pca(dataset_root):
    root = Path(dataset_root)
    # Dictionary to store joint data lists for each task
    task_data_accumulator = defaultdict(list)

    # 1. Collect Joint Data per Task
    print("Chargement des données par tâche...")
    for joints_file in root.rglob("all_joints.npy"):
        try:
            trial_dir = joints_file.parent
            subject_dir = trial_dir.parent
            
            task = get_task(subject_dir.name, trial_dir.name)
            if task == "unknown" or task == "other":
                continue

            data = np.load(joints_file) # shape (T, 35)
            task_data_accumulator[task].append(data)
        except Exception as e:
            print(f"Erreur avec {joints_file}: {e}")

    # 2. Process PCA for each task and Plot
    plt.figure(figsize=(10, 6))
    print("\nCalcul des PCA...")

    # We sort by task name to keep the plot legend consistent
    for task in sorted(task_data_accumulator.keys()):
        # Concatenate all trials for this specific task
        X_task = np.concatenate(task_data_accumulator[task], axis=0)
        
        # Scaling is mandatory for PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_task)

        # Run PCA
        pca = PCA()
        pca.fit(X_scaled)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        
        eigenvalues = pca.explained_variance_

        sorted_eigenvalues = np.sort(eigenvalues)[::-1]

        print("Valeurs propres :")
        for i, val in enumerate(sorted_eigenvalues):
            print(f"PC{i+1}: {val:.4f}")
            
        # Find the 95% threshold index
        n_95 = np.argmax(cumvar >= 0.95) + 1
        
        # Plotting
        plt.plot(cumvar, label=f"{task} (95% @ {n_95} dim)")
        print(f"Tâche: {task:<12} | Dimensions indépendantes (95%): {n_95}")

    # Formatting the plot
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.3)
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance cumulée")
    plt.title("Complexité du mouvement par tâche (Joint Angles)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    analyze_tasks("./processed_data_feet")
    run_task_pca("./processed_data_feet")