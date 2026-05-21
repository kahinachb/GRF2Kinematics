import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from utils.utils import is_squat_task # Ta fonction d'origine

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# CONFIGURATION : CHOISIS LE SUJET ET LA TÂCHE DE CHAQUE DATASET
# ==============================================================================
# Exemple Dataset A
SUJET_A = "jeremy"      # Nom du dossier sujet (ex: vincent, maria, subject1...)
TACHE_A = "trial111"      # Nom exact du sous-dossier de la tâche pour ce sujet

# Exemple Dataset B
SUJET_B = "kahina"  # Nom du dossier sujet du dataset B
TACHE_B = "squat"      # Nom exact du sous-dossier de la tâche pour ce sujet

# ==============================================================================

data_root = Path("processed_data_feet")
LABELS = [
    "Fx1","Fy1","Fz1","Mx1","My1","Mz1","COPx1","COPy1","COPz1",
    "Fx2","Fy2","Fz2","Mx2","My2","Mz2","COPx2","COPy2","COPz2"
]

def charger_fichier_specifique(nom_sujet, nom_tache):
    """Recherche et charge le fichier kinetics_feet.npy de manière insensible à la casse."""
    if not data_root.exists():
        print(f"Erreur : Le dossier racine '{data_root}' n'existe pas.")
        return None

    # Recherche du dossier sujet
    dossier_sujet = None
    for d in data_root.iterdir():
        if d.is_dir() and d.name.lower() == nom_sujet.lower():
            dossier_sujet = d
            break
            
    if dossier_sujet is None:
        print(f"Erreur : Le sujet '{nom_sujet}' est introuvable dans {data_root}")
        return None

    # Recherche du dossier tâche
    dossier_tache = None
    for d in dossier_sujet.iterdir():
        if d.is_dir() and d.name.lower() == nom_tache.lower():
            dossier_tache = d
            break

    if dossier_tache is None:
        print(f"Erreur : La tâche '{nom_tache}' est introuvable pour le sujet '{nom_sujet}'")
        print(f"Tâches disponibles pour {nom_sujet} : {[d.name for d in dossier_sujet.iterdir() if d.is_dir()]}")
        return None

    # Vérification et chargement du fichier npy
    kinetics_file = dossier_tache / "kinetics_feet.npy"
    if not kinetics_file.exists():
        print(f"Erreur : Le fichier 'kinetics_feet.npy' n'existe pas dans {dossier_tache}")
        return None

    data = np.load(kinetics_file)
    if data.ndim != 2 or data.shape[1] != 18:
        print(f"Erreur : Le fichier chargé a une forme incorrecte {data.shape}, attendu (N, 18)")
        return None

    return data

# --- CHARGEMENT ---
print("--- Chargement des fichiers spécifiques ---")
data_A = charger_fichier_specifique(SUJET_A, TACHE_A)
data_B = charger_fichier_specifique(SUJET_B, TACHE_B)

# --- TRACÉ DES GRAPHIQUES ---
if data_A is not None and data_B is not None:
    name_A = f"{SUJET_A} ({TACHE_A})"
    name_B = f"{SUJET_B} ({TACHE_B})"

    # ==========================================
    # 1. GRAPHIQUE DES FORCES (Fx, Fy, Fz)
    # ==========================================
    fig_f, axes_f = plt.subplots(2, 2, figsize=(14, 8), sharey='row')
    for idx in [0, 1, 2]: axes_f[0, 0].plot(data_A[:, idx], label=LABELS[idx])
    axes_f[0, 0].set_title(f"Forces right - {name_A}"); axes_f[0, 0].legend(); axes_f[0, 0].grid(True, alpha=0.3)
    for idx in [0, 1, 2]: axes_f[0, 1].plot(data_B[:, idx], label=LABELS[idx])
    axes_f[0, 1].set_title(f"Forces right - {name_B}"); axes_f[0, 1].legend(); axes_f[0, 1].grid(True, alpha=0.3)
    for idx in [9, 10, 11]: axes_f[1, 0].plot(data_A[:, idx], label=LABELS[idx])
    axes_f[1, 0].set_title(f"Forces left - {name_A}"); axes_f[1, 0].legend(); axes_f[1, 0].grid(True, alpha=0.3)
    for idx in [9, 10, 11]: axes_f[1, 1].plot(data_B[:, idx], label=LABELS[idx])
    axes_f[1, 1].set_title(f"Forces left - {name_B}"); axes_f[1, 1].legend(); axes_f[1, 1].grid(True, alpha=0.3)
    fig_f.suptitle("Comparaison des Forces (F)", fontsize=14, weight='bold')
    fig_f.tight_layout()

    # ==========================================
    # 2. GRAPHIQUE DES MOMENTS (Mx, My, Mz)
    # ==========================================
    fig_m, axes_m = plt.subplots(2, 2, figsize=(14, 8), sharey='row')
    for idx in [3, 4, 5]: axes_m[0, 0].plot(data_A[:, idx], label=LABELS[idx])
    axes_m[0, 0].set_title(f"Moments right - {name_A}"); axes_m[0, 0].legend(); axes_m[0, 0].grid(True, alpha=0.3)
    for idx in [3, 4, 5]: axes_m[0, 1].plot(data_B[:, idx], label=LABELS[idx])
    axes_m[0, 1].set_title(f"Moments right - {name_B}"); axes_m[0, 1].legend(); axes_m[0, 1].grid(True, alpha=0.3)
    for idx in [12, 13, 14]: axes_m[1, 0].plot(data_A[:, idx], label=LABELS[idx])
    axes_m[1, 0].set_title(f"Moments left - {name_A}"); axes_m[1, 0].legend(); axes_m[1, 0].grid(True, alpha=0.3)
    for idx in [12, 13, 14]: axes_m[1, 1].plot(data_B[:, idx], label=LABELS[idx])
    axes_m[1, 1].set_title(f"Moments left - {name_B}"); axes_m[1, 1].legend(); axes_m[1, 1].grid(True, alpha=0.3)
    fig_m.suptitle("Comparaison des Moments (M)", fontsize=14, weight='bold')
    fig_m.tight_layout()

    # ==========================================
    # 3. GRAPHIQUE DES COPs (COPx, COPy, COPz)
    # ==========================================
    fig_cop, axes_cop = plt.subplots(2, 2, figsize=(14, 8), sharey='row')
    for idx in [6, 7, 8]: axes_cop[0, 0].plot(data_A[:, idx], label=LABELS[idx])
    axes_cop[0, 0].set_title(f"COP right - {name_A}"); axes_cop[0, 0].legend(); axes_cop[0, 0].grid(True, alpha=0.3)
    for idx in [6, 7, 8]: axes_cop[0, 1].plot(data_B[:, idx], label=LABELS[idx])
    axes_cop[0, 1].set_title(f"COP right - {name_B}"); axes_cop[0, 1].legend(); axes_cop[0, 1].grid(True, alpha=0.3)
    for idx in [15, 16, 17]: axes_cop[1, 0].plot(data_A[:, idx], label=LABELS[idx])
    axes_cop[1, 0].set_title(f"COP left - {name_A}"); axes_cop[1, 0].legend(); axes_cop[1, 0].grid(True, alpha=0.3)
    for idx in [15, 16, 17]: axes_cop[1, 1].plot(data_B[:, idx], label=LABELS[idx])
    axes_cop[1, 1].set_title(f"COP left - {name_B}"); axes_cop[1, 1].legend(); axes_cop[1, 1].grid(True, alpha=0.3)
    fig_cop.suptitle("Comparaison des Centres de Pression (COP)", fontsize=14, weight='bold')
    fig_cop.tight_layout()
    
    plt.show()
else:
    print("\nÉchec du tracé. Veuillez corriger les noms de dossiers affichés dans les erreurs ci-dessus.")