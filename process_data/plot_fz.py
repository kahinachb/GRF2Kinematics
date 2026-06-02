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
from scipy.signal import find_peaks
# ==============================================================================
# CONFIGURATION : CHOISIS LE SUJET ET LA TÂCHE DE CHAQUE DATASET
# ==============================================================================
# Exemple Dataset A
SUJET_A = "Jeremy"      # Nom du dossier sujet (ex: vincent, maria, subject1...)
TACHE_A = "trial111"      # Nom exact du sous-dossier de la tâche pour ce sujet

# Exemple Dataset B
SUJET_B = "laure"  # Nom du dossier sujet du dataset B
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
    kinetics_file = dossier_tache / "kinetics_glob.npy"
    if not kinetics_file.exists():
        print(f"Erreur : Le fichier 'kinetics_feet.npy' n'existe pas dans {dossier_tache}")
        return None

    data = np.load(kinetics_file)
    if data.ndim != 2 or data.shape[1] != 18:
        print(f"Erreur : Le fichier chargé a une forme incorrecte {data.shape}, attendu (N, 18)")
        return None

    return data

# --- CHARGEMENT ---
# print("--- Chargement des fichiers spécifiques ---")
# data_A = charger_fichier_specifique(SUJET_A, TACHE_A)
# data_B = charger_fichier_specifique(SUJET_B, TACHE_B)

# # --- TRACÉ DES GRAPHIQUES ---
# if data_A is not None and data_B is not None:
#     name_A = f"{SUJET_A} ({TACHE_A})"
#     name_B = f"{SUJET_B} ({TACHE_B})"

#     # ==========================================
#     # 1. GRAPHIQUE DES FORCES (Fx, Fy, Fz)
#     # ==========================================
#     fig_f, axes_f = plt.subplots(2, 2, figsize=(14, 8), sharey='row')
#     for idx in [0, 1, 2]: axes_f[0, 0].plot(data_A[:, idx], label=LABELS[idx])
#     axes_f[0, 0].set_title(f"Forces right - {name_A}"); axes_f[0, 0].legend(); axes_f[0, 0].grid(True, alpha=0.3)
#     for idx in [0, 1, 2]: axes_f[0, 1].plot(data_B[:, idx], label=LABELS[idx])
#     axes_f[0, 1].set_title(f"Forces right - {name_B}"); axes_f[0, 1].legend(); axes_f[0, 1].grid(True, alpha=0.3)
#     for idx in [9, 10, 11]: axes_f[1, 0].plot(data_A[:, idx], label=LABELS[idx])
#     axes_f[1, 0].set_title(f"Forces left - {name_A}"); axes_f[1, 0].legend(); axes_f[1, 0].grid(True, alpha=0.3)
#     for idx in [9, 10, 11]: axes_f[1, 1].plot(data_B[:, idx], label=LABELS[idx])
#     axes_f[1, 1].set_title(f"Forces left - {name_B}"); axes_f[1, 1].legend(); axes_f[1, 1].grid(True, alpha=0.3)
#     fig_f.suptitle("Comparaison des Forces (F)", fontsize=14, weight='bold')
#     fig_f.tight_layout()

#     # ==========================================
#     # 2. GRAPHIQUE DES MOMENTS (Mx, My, Mz)
#     # ==========================================
#     fig_m, axes_m = plt.subplots(2, 2, figsize=(14, 8), sharey='row')
#     for idx in [3, 4, 5]: axes_m[0, 0].plot(data_A[:, idx], label=LABELS[idx])
#     axes_m[0, 0].set_title(f"Moments right - {name_A}"); axes_m[0, 0].legend(); axes_m[0, 0].grid(True, alpha=0.3)
#     for idx in [3, 4, 5]: axes_m[0, 1].plot(data_B[:, idx], label=LABELS[idx])
#     axes_m[0, 1].set_title(f"Moments right - {name_B}"); axes_m[0, 1].legend(); axes_m[0, 1].grid(True, alpha=0.3)
#     for idx in [12, 13, 14]: axes_m[1, 0].plot(data_A[:, idx], label=LABELS[idx])
#     axes_m[1, 0].set_title(f"Moments left - {name_A}"); axes_m[1, 0].legend(); axes_m[1, 0].grid(True, alpha=0.3)
#     for idx in [12, 13, 14]: axes_m[1, 1].plot(data_B[:, idx], label=LABELS[idx])
#     axes_m[1, 1].set_title(f"Moments left - {name_B}"); axes_m[1, 1].legend(); axes_m[1, 1].grid(True, alpha=0.3)
#     fig_m.suptitle("Comparaison des Moments (M)", fontsize=14, weight='bold')
#     fig_m.tight_layout()

#     # ==========================================
#     # 3. GRAPHIQUE DES COPs (COPx, COPy, COPz)
#     # ==========================================
#     fig_cop, axes_cop = plt.subplots(2, 2, figsize=(14, 8), sharey='row')
#     for idx in [6, 7, 8]: axes_cop[0, 0].plot(data_A[:, idx], label=LABELS[idx])
#     axes_cop[0, 0].set_title(f"COP right - {name_A}"); axes_cop[0, 0].legend(); axes_cop[0, 0].grid(True, alpha=0.3)
#     for idx in [6, 7, 8]: axes_cop[0, 1].plot(data_B[:, idx], label=LABELS[idx])
#     axes_cop[0, 1].set_title(f"COP right - {name_B}"); axes_cop[0, 1].legend(); axes_cop[0, 1].grid(True, alpha=0.3)
#     for idx in [15, 16, 17]: axes_cop[1, 0].plot(data_A[:, idx], label=LABELS[idx])
#     axes_cop[1, 0].set_title(f"COP left - {name_A}"); axes_cop[1, 0].legend(); axes_cop[1, 0].grid(True, alpha=0.3)
#     for idx in [15, 16, 17]: axes_cop[1, 1].plot(data_B[:, idx], label=LABELS[idx])
#     axes_cop[1, 1].set_title(f"COP left - {name_B}"); axes_cop[1, 1].legend(); axes_cop[1, 1].grid(True, alpha=0.3)
#     fig_cop.suptitle("Comparaison des Centres de Pression (COP)", fontsize=14, weight='bold')
#     fig_cop.tight_layout()
    
#     plt.show()
# else:
#     print("\nÉchec du tracé. Veuillez corriger les noms de dossiers affichés dans les erreurs ci-dessus.")

DICTIONNAIRE_DATASET_A = {
    "vincent": "trial112",
    "jeremy": "trial111",
    "christine": "trial110",
    "jovana": "trial111",
    "serge": "trial111",
    "maria": "trial114",
    "subject1": "trial111"
}

def lisser_signal(signal, window=9):
    """Lissage pour nettoyer le bruit de la dérivée numérique."""
    return np.convolve(signal, np.ones(window)/window, mode='same')

def detecter_pics(deriv_signal, prominence_factor=0.4):
    """Trouve les pics positifs et négatifs importants sur la dérivée."""
    std_val = np.std(deriv_signal)
    pics_pos, _ = find_peaks(deriv_signal, prominence=std_val * prominence_factor, distance=25)
    pics_neg, _ = find_peaks(-deriv_signal, prominence=std_val * prominence_factor, distance=25)
    return pics_pos, pics_neg

# --- 2. COLLECTE DE TOUS LES TRIALS ---
tous_les_trials = []

if not data_root.exists():
    raise FileNotFoundError(f"Le dossier '{data_root}' est introuvable.")

for subject_dir in sorted(data_root.iterdir()):
    if not subject_dir.is_dir():
        continue
    
    subject_name = subject_dir.name.lower()
    
    for task_dir in subject_dir.iterdir():
        if not task_dir.is_dir():
            continue
            
        task_name = task_dir.name.lower()
        kinetics_file = task_dir / "kinetics_glob.npy"
        
        if not kinetics_file.exists():
            continue
            
        # Vérification des règles Dataset A et Dataset B
        appartient_a_A = subject_name in DICTIONNAIRE_DATASET_A and task_name == DICTIONNAIRE_DATASET_A[subject_name]
        appartient_a_B = subject_name not in DICTIONNAIRE_DATASET_A and "squat" in task_name
        
        if appartient_a_A:
            tous_les_trials.append((subject_dir.name, task_dir.name, kinetics_file, "Dataset A"))
        elif appartient_a_B:
            tous_les_trials.append((subject_dir.name, task_dir.name, kinetics_file, "Dataset B"))

# --- 3. AFFICHAGE INDIVIDUEL (UNE FIGURE PAR TRIAL) ---
print(f"Nombre total de figures à générer : {len(tous_les_trials)}")

for sub, task, file_path, origin in tous_les_trials:
    
    # Chargement et calculs pour ce fichier spécifique
    data = np.load(file_path)
    Fz_total = data[:, 2] + data[:, 11] # Fz1 + Fz2
    dFz = lisser_signal(np.gradient(Fz_total))
    pics_pos, pics_neg = detecter_pics(dFz)
    
    # Création d'une figure dédiée à ce trial
    fig, (ax_fz, ax_dfz) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # --- Graphique du Haut : Force Totale (Fz) ---
    ax_fz.plot(Fz_total, color="black", linewidth=2, label="Fz Totale")
    ax_fz.set_ylabel("Force (N)", fontsize=11)
    ax_fz.grid(True, alpha=0.3)
    ax_fz.legend(loc="upper right")
    
    # --- Graphique du Bas : Vraie Dérivée (dFz/dt) ---
    ax_dfz.plot(dFz, color="purple", linewidth=1.5, label="dFz/dt (Verticale)")
    ax_dfz.axhline(0, color="gray", linestyle="--", alpha=0.7) # Ligne du zéro pour la dérivée
    
    # Ajout des étoiles (rouges pour les max, bleues pour les min)
    ax_dfz.scatter(pics_pos, dFz[pics_pos], color="red", marker="*", s=150, zorder=5, label="Pics Positifs")
    ax_dfz.scatter(pics_neg, dFz[pics_neg], color="blue", marker="*", s=150, zorder=5, label="Pics Négatifs")
    
    ax_dfz.set_ylabel("dFz/dt (N/frame)", fontsize=11)
    ax_dfz.set_xlabel("Frames (Temps)", fontsize=11)
    ax_dfz.grid(True, alpha=0.3)
    ax_dfz.legend(loc="upper right")
    
    # Projection des lignes verticales pour lier la force et les pics détectés
    for p in np.concatenate([pics_pos, pics_neg]):
        ax_fz.axvline(x=p, color='red', linestyle=':', alpha=0.5)
        ax_dfz.axvline(x=p, color='red', linestyle=':', alpha=0.5)
        
    # Titre personnalisé combinant Nom Sujet + Nom de la tâche + Origine
    plt.suptitle(f"Sujet : {sub}  |  Tâche : {task}  ({origin})", fontsize=14, weight='bold')
    plt.tight_layout()

# Lancement de l'affichage global à la fin du script
    plt.show()