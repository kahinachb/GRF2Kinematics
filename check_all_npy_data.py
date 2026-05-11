import numpy as np
from pathlib import Path

def analyze_hybrid_structure(output_dir: str):
    """
    Analyse les fichiers NPY avec une détection de fréquence contextuelle.
    """
    root = Path(output_dir)
    if not root.exists():
        print(f"Erreur : Le dossier {output_dir} n'existe pas.")
        return

    results = {
        300: {"steps": 0, "files": 0, "durations": []},
        100: {"steps": 0, "files": 0, "durations": []}
    }

    print(f"{'SUJET':<12} | {'ESSAI':<20} | {'FICHIER':<16} | {'FREQ':<6} | {'DURÉE (s)'}")
    print("-" * 80)

    # Parcourt tous les fichiers .npy récursivement
    for file_path in sorted(root.rglob("kinetics*.npy")):
        trial_dir = file_path.parent
        subject_dir = trial_dir.parent
        
        filename = file_path.name
        subject_name = subject_dir.name
        
        # --- LOGIQUE DE DÉTERMINATION DE LA FRÉQUENCE ---
        if filename == "forces_300.npy":
            fs = 300
        elif filename == "kinetics.npy":
            if "subject" in subject_name.lower():
                fs = 100
            else:
                fs = 100
        else:
            continue # Ignore les autres fichiers (ex: joints.npy)

        # Chargement léger des métadonnées
        try:
            data = np.load(file_path, mmap_mode='r')
            n_steps = data.shape[0]
            duration = n_steps / fs
            
            results[fs]["steps"] += n_steps
            results[fs]["files"] += 1
            results[fs]["durations"].append(duration)

            # print(f"{subject_name[:12]:<12} | {trial_dir.name[:20]:<20} | {filename:<16} |{n_steps} |{fs:<6} | {duration:.2f}s")

            # print(f"   -> Exemple première ligne : {data[0]}")

            # print("\n=== STRUCTURE DES DONNÉES ===")
            # print(f"Forces shape : {data.shape}")
            # print(f"Forces dtype : {data.dtype}")

            joints_path = trial_dir / "all_joints.npy"
            # if joints_path.exists():
            #     joints_data = np.load(joints_path, mmap_mode='r')
            #     # print(f"   -> Exemple première ligne : {joints_data[0]}")
            #     print(f"Joints shape : {joints_data.shape}")
            #     print(f"Joints dtype : {joints_data.dtype}")
            # input()


        except Exception as e:
            print(f"Erreur lecture {file_path.name} dans {subject_name}: {e}")

    # --- SYNTHÈSE FINALE ---
    print("\n" + "="*60)
    print("RÉCAPITULATIF PAR FRÉQUENCE")
    print("="*60)
    
    for fs in [300, 100]:
        res = results[fs]
        if res["files"] > 0:
            total_min = sum(res["durations"]) / 60
            avg_dur = np.mean(res["durations"])
            print(f"--- Groupe {fs} Hz ---")
            print(f"  Nombre de fichiers : {res['files']}")
            print(f"  Total timesteps    : {res['steps']}")
            print(f"  Durée totale       : {total_min:.2f} minutes")
            print(f"  Durée moyenne      : {avg_dur:.2f} secondes / essai")
            print("-" * 40)

    print(f"TOTAL GLOBAL : {results[300]['files'] + results[100]['files']} fichiers analysés.")

if __name__ == "__main__":
    analyze_hybrid_structure("./processed_data_feet")