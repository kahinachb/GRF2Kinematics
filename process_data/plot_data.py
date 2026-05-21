import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from utils.utils import is_squat_task # Ta fonction d'origine

# --- 1. CONFIGURATION ET CLASSIFICATION DES DATASETS ---
data_root = Path("processed_data_feet")

# Liste officielle de tes sujets pour le Dataset A (en minuscules)
SUJETS_DATASET_A = {"vincent", "christine", "jovana", "jeremy", "maria", "subject1", "serge"}

all_kinetics = []
dataset_labels = []
subjects_found = set()

# Récupération et tri des dossiers sujets
subjects = sorted([d for d in data_root.iterdir() if d.is_dir()])

# --- 2. CHARGEMENT FILTRÉ ET LABELLISATION ---
for subject_dir in subjects:
    subject_name = subject_dir.name.lower()
    task_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]

    for task_dir in task_dirs:
        task_name = task_dir.name.lower()

        # Application de ton filtre de tâche
        if not is_squat_task(subject_name, task_name):
            continue

        kinetics_file = task_dir / "kinetics_feet.npy"
        joints_file = task_dir / "all_joints.npy"

        if kinetics_file.exists() and joints_file.exists():
            # Chargement de la cinétique (shape: n_frames, 18)
            data = np.load(kinetics_file)
            
            if data.ndim == 2 and data.shape[1] == 18:
                all_kinetics.append(data)
                subjects_found.add(subject_name)
                
                # Attribution du Dataset selon ta règle
                label = "Dataset_A" if subject_name in SUJETS_DATASET_A else "Dataset_B"
                dataset_labels.extend([label] * data.shape[0])

# --- 3. VÉRIFICATION DES DONNÉES ---
X_all = np.vstack(all_kinetics)
labels_all = np.array(dataset_labels)

print(f"--- Rapport de chargement ---")
print(f"Sujets identifiés avec squats : {sorted(list(subjects_found))}")
print(f"Total de frames cinétiques accumulées : {X_all.shape[0]}")
print(f"Répartition - A: {np.sum(labels_all == 'Dataset_A')} frames | B: {np.sum(labels_all == 'Dataset_B')} frames\n")

# --- 4. GRAPH_1 : ANALYSE GLOBALE VIA PCA (SOUS-ÉCHANTILLONNÉE) ---
# Normalisation obligatoire pour la PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Sous-échantillonnage (1 frame sur 5) pour éviter de figer le plot et la RAM
step = 5
X_pca_plot = X_pca[::step]
labels_plot = labels_all[::step]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_pca_plot[:, 0], y=X_pca_plot[:, 1], 
    hue=labels_plot, 
    alpha=0.2, # Transparence pour voir la densité des superpositions
    palette={"Dataset_A": "#1f77b4", "Dataset_B": "#ff7f0e"}, # Bleu et Orange standard
    s=8
)
plt.title(f"Visualisation PCA (1 frame / {step}) — 18 composantes")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance expliquée)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance expliquée)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# --- 5. GRAPH_2 : BOXPLOTS DES 18 COMPOSANTES ---
# Création du DataFrame global
df_kinetics = pd.DataFrame(X_all, columns=[f"Comp_{i+1}" for i in range(18)])
df_kinetics["Dataset"] = labels_all

# Format long pour Seaborn
df_long = pd.melt(
    df_kinetics, 
    id_vars=["Dataset"], 
    value_vars=[f"Comp_{i+1}" for i in range(18)],
    var_name="Composante", 
    value_name="Valeur"
)

plt.figure(figsize=(16, 8))
sns.boxplot(
    data=df_long, 
    x="Composante", 
    y="Valeur", 
    hue="Dataset", 
    palette={"Dataset_A": "#1f77b4", "Dataset_B": "#ff7f0e"},
    fliersize=0.5 # Outliers très petits pour ne pas polluer la vue
)
plt.xticks(rotation=45)
plt.title("Comparaison des distributions boîte par boîte (18 composantes)")
plt.xlabel("Composantes")
plt.ylabel("Valeurs")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


LABELS = [
    "Fx1","Fy1","Fz1","Mx1","My1","Mz1","COPx1","COPy1","COPz1",
    "Fx2","Fy2","Fz2","Mx2","My2","Mz2","COPx2","COPy2","COPz2"
]

sample_A, sample_B = None, None
subjects = sorted([d for d in data_root.iterdir() if d.is_dir()])

# --- RECHERCHE DES EXEMPLES ---
# for subject_dir in subjects:
#     subject_name = subject_dir.name.lower()
#     for task_dir in [d for d in subject_dir.iterdir() if d.is_dir()]:
#         task_name = task_dir.name.lower()
#         if not is_squat_task(subject_name, task_name):
#             continue
        
#         kinetics_file = task_dir / "kinetics_feet.npy"
#         if kinetics_file.exists():
#             data = np.load(kinetics_file)
#             if data.ndim == 2 and data.shape[1] == 18:
#                 if subject_name in SUJETS_DATASET_A and sample_A is None:
#                     sample_A = (f"A: {subject_name}", data)
#                 elif subject_name not in SUJETS_DATASET_A and sample_B is None:
#                     sample_B = (f"B: {subject_name}", data)
#     if sample_A and sample_B:
#         break

