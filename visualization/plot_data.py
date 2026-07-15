import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from scipy.stats import pearsonr

# --- Configuration des chemins et mapping ---
manual_mapping = {
    "Vincent": {"Trial112": "squat"},
    "Jovana": {"Trial111": "squat"},
    "Christine": {"Trial110": "squat"},
    "Jeremy": {"Trial111": "squat"},
    "Maria": {"Trial114": "squat"},
    "Serge": {"Trial111": "squat"},
    "Subject1": {"Trial111": "squat"}
}

def is_squat_task(subject_name, task_name):
    task_lower = task_name.lower()
    if "squat" in task_lower: return True
    if subject_name in manual_mapping:

        return manual_mapping[subject_name].get(task_name) == "squat"
    return False

def load_data(is_synth=True):
    all_X, all_F = [], []
    
    if is_synth:
        base_path = Path("DATA/generated_data/")
        for f_grfm in base_path.glob("*_grfm.csv"):
            f_q = f_grfm.with_name(f_grfm.name.replace("_grfm.csv", "_q.csv"))
            if f_q.exists():
                # On lit les valeurs. Ajuste ici si tu as des headers à ignorer
                all_X.append(pd.read_csv(f_q).values)
                all_F.append(pd.read_csv(f_grfm).values)
    else:
        base_path = Path("DATA/Vinc/")
        for subject_dir in base_path.iterdir():
            for trial_dir in subject_dir.iterdir():
                if is_squat_task(subject_dir.name, trial_dir.name):
                    f_q = trial_dir / "joints_filtered_FF.csv"
                    f_grfm = trial_dir / "kinetics_glob_filtered.csv"
     
                    if f_q.exists() and f_grfm.exists():
                        print("test")
                        # Attention: Assure-toi que les fichiers réels n'ont pas de colonnes 'Time'
                        # sinon fais : pd.read_csv(f_q).drop(columns=['Time']).values
                        all_X.append(pd.read_csv(f_q).values)
                        all_F.append(pd.read_csv(f_grfm).values)
                        
    # Empilement pour créer un grand dataset de comparaison
    return np.vstack(all_X), np.vstack(all_F)

def compare_datasets():
    print("Chargement des données...")
    X_synth, F_synth = load_data(is_synth=True)
    X_real, F_real = load_data(is_synth=False)
    
    # --- 1. PCA : Comparaison de la distribution cinématique ---
    pca = PCA(n_components=2)
    X_all = np.vstack([F_synth, F_real])
    pca.fit(X_all)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(pca.transform(F_synth)[:, 0], pca.transform(F_synth)[:, 1], alpha=0.05, label='Synthétique', s=1)
    plt.scatter(pca.transform(F_real)[:, 0], pca.transform(F_real)[:, 1], alpha=0.3, label='Réel', s=2)
    plt.legend()
    plt.title("PCA des mouvements articulaires : Synthétique vs Réel")
    plt.show()
    
    # --- 2. Analyse physique : Corrélation Fz vs Accélération Bassin ---
    # Remplace index 2 par la colonne correspondant à Fz (verticale) et le bassin
    # Supposons ici que Fz est à l'index 2 et Bassin à l'index 2 aussi
    fz_idx = 2 
    bassin_idx = 2
    
    def get_acc(X):
        # Accélération = dérivée seconde de la position
        vel = np.gradient(X[:, bassin_idx])
        return np.gradient(vel)

    acc_synth = get_acc(X_synth)
    acc_real = get_acc(X_real)
    
    corr_synth = pearsonr(F_synth[:, fz_idx], acc_synth)[0]
    corr_real = pearsonr(F_real[:, fz_idx], acc_real)[0]
    
    print(f"\n--- Analyse Physique ---")
    print(f"Corrélation Fz/Accél. (Synthétique) : {corr_synth:.3f}")
    print(f"Corrélation Fz/Accél. (Réel) : {corr_real:.3f}")
    
    if abs(corr_synth - corr_real) > 0.2:
        print("ATTENTION : Le lien physique force/mouvement est significativement différent.")
    else:
        print("BONNE NOUVELLE : La cohérence physique semble proche.")

if __name__ == "__main__":
    compare_datasets()