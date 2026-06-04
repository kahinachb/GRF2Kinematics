import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# ==========================================
# CONFIGURATION
# ==========================================
DATA_ROOT = Path("/home/kchalabi/Documents/THESE/datasets_kinetics/GRF2Kinematics/DATA/npy")
CONTACT_THRESHOLD = 20.0  # Seuil en Newtons pour considérer que le pied touche le sol
VERTICAL_AXIS_IDX = 1     # 2 si Z est vertical, 1 si Y est vertical. (Fz ou Fy)

# Indices basés sur ta structure : 
# Fx,Fy,Fz,Mx,My,Mz,Copx,Copy,Copz (right) puis (left)
IDX = {
    'R_F': slice(0, 3),   'R_M': slice(3, 6),   'R_CoP': slice(6, 9),
    'L_F': slice(9, 12),  'L_M': slice(12, 15), 'L_CoP': slice(15, 18)
}

def analyze_dataset():
    if not DATA_ROOT.exists():
        print(f"Erreur : Le dossier {DATA_ROOT} n'existe pas.")
        return

    all_R_CoP_contact = []
    all_L_CoP_contact = []
    
    issues_found = {
        'noisy_swing': [],
        'negative_fz': [],
        'extreme_cop': []
    }

    total_frames = 0
    
    print("Démarrage du scan du dataset...\n")
    
    for subject_dir in sorted(DATA_ROOT.iterdir()):
        if not subject_dir.is_dir(): continue
            
        for task_dir in sorted(subject_dir.iterdir()):
            if not task_dir.is_dir(): continue
                
            kinetics_path = task_dir / "kinetics_feet_corr.npy"
            if not kinetics_path.exists(): continue
                
            data = np.load(kinetics_path)
            T = data.shape[0]
            total_frames += T
            
            # --- 1. Extraction des données ---
            R_F = data[:, IDX['R_F']]
            R_CoP = data[:, IDX['R_CoP']]
            L_F = data[:, IDX['L_F']]
            L_CoP = data[:, IDX['L_CoP']]
            
            R_Fz = R_F[:, VERTICAL_AXIS_IDX]
            L_Fz = L_F[:, VERTICAL_AXIS_IDX]
            
            # --- 2. Détection des contacts ---
            # On utilise la valeur absolue au cas où l'axe serait inversé
            R_contact = np.abs(R_Fz) > CONTACT_THRESHOLD
            L_contact = np.abs(L_Fz) > CONTACT_THRESHOLD
            
            # --- 3. Vérification du Bruit en Phase de Vol ---
            # Si le pied n'est pas en contact, le CoP devrait idéalement être à 0
            R_swing_CoP_mag = np.linalg.norm(R_CoP[~R_contact], axis=1) if np.any(~R_contact) else np.array([0])
            L_swing_CoP_mag = np.linalg.norm(L_CoP[~L_contact], axis=1) if np.any(~L_contact) else np.array([0])
            
            if np.max(R_swing_CoP_mag) > 0.05 or np.max(L_swing_CoP_mag) > 0.05:
                issues_found['noisy_swing'].append(f"{subject_dir.name}/{task_dir.name}")
                
            # --- 4. Vérification de la polarité de Fz ---
            # La force de réaction devrait être majoritairement positive (ou négative selon ta convention)
            # On signale s'il y a de fortes valeurs du côté opposé
            if np.any(R_Fz < -CONTACT_THRESHOLD) or np.any(L_Fz < -CONTACT_THRESHOLD):
                issues_found['negative_fz'].append(f"{subject_dir.name}/{task_dir.name}")

            # --- 5. Accumulation des CoP (uniquement pendant le contact) ---
            # Pour vérifier l'empreinte anatomique
            if np.any(R_contact): all_R_CoP_contact.append(R_CoP[R_contact])
            if np.any(L_contact): all_L_CoP_contact.append(L_CoP[L_contact])

    # ==========================================
    # AFFICHAGE DU RAPPORT
    # ==========================================
    print("="*40)
    print("📊 RAPPORT DE DIAGNOSTIC BIOMÉCANIQUE")
    print("="*40)
    print(f"Total des frames analysées : {total_frames:,}")
    
    print("\n⚠️  1. BRUIT EN PHASE DE VOL (Swing Phase Noise)")
    if issues_found['noisy_swing']:
        print(f"  -> {len(issues_found['noisy_swing'])} essais ont un CoP non-nul alors que le pied est en l'air.")
        print("  ! ACTION REQUISE : Dans ton script de dataloader, force le CoP (et idéalement M et F) à 0 quand Fz < 20N.")
    else:
        print("  -> OK : Les phases de vol sont bien nettoyées.")

    print("\n⚠️  2. POLARITÉ DE LA FORCE VERTICALE")
    if issues_found['negative_fz']:
        print(f"  -> {len(issues_found['negative_fz'])} essais ont des forces de réaction inversées (négatives).")
        print("  ! ACTION REQUISE : Vérifie l'orientation de ton axe vertical. La GRF devrait pointer dans un seul sens.")
    else:
        print("  -> OK : La polarité est constante.")

    # ==========================================
    # GÉNÉRATION DES GRAPHIQUES (Empreintes CoP)
    # ==========================================
    if all_R_CoP_contact and all_L_CoP_contact:
        R_CoP_cat = np.vstack(all_R_CoP_contact)
        L_CoP_cat = np.vstack(all_L_CoP_contact)
        
        # On suppose que X est l'axe antéro-postérieur et Y le médio-latéral (ou Z selon ton repère)
        # Ajuste les indices [:, 0] et [:, 1] selon comment tes axes locaux sont définis
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Empreinte locale Pied GAUCHE (en contact)")
        # On affiche 10000 points max pour ne pas surcharger le plot
        idx_L = np.random.choice(len(L_CoP_cat), min(10000, len(L_CoP_cat)), replace=False)
        plt.scatter(L_CoP_cat[idx_L, 0], L_CoP_cat[idx_L, 2], alpha=0.1, s=1, c='blue')
        plt.axis('equal'); plt.grid(True)
        plt.xlabel("Axe local 1 (m)"); plt.ylabel("Axe local 2 (m)")
        
        plt.subplot(1, 2, 2)
        plt.title("Empreinte locale Pied DROIT (en contact)")
        idx_R = np.random.choice(len(R_CoP_cat), min(10000, len(R_CoP_cat)), replace=False)
        plt.scatter(R_CoP_cat[idx_R, 0], R_CoP_cat[idx_R, 2], alpha=0.1, s=1, c='red')
        plt.axis('equal'); plt.grid(True)
        plt.xlabel("Axe local 1 (m)"); plt.ylabel("Axe local 2 (m)")
        
        plt.tight_layout()
        plt.savefig("cop_footprint_check.png")
        print("\n📸 Un graphique 'cop_footprint_check.png' a été généré.")
        print("  ! VÉRIFICATION : Regarde l'image. Les nuages de points doivent ressembler à la plante d'un pied gauche et droit.")
        print("                   Si le nuage fait des mètres de long ou forme une sphère géante, tes matrices de rotation locale sont fausses.")

if __name__ == "__main__":
    analyze_dataset()