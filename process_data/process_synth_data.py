import sys
import os
import glob
import pandas as pd
import pinocchio as pin
import numpy as np
import meshcat
import meshcat_shapes
from pinocchio.visualize import MeshcatVisualizer
import time
import matplotlib.pyplot as plt

# Configuration des chemins
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils.model_utils import build_human_model, get_foot_pose
from utils.utils import find_col
from utils.viz_utils import add_sphere, place, set_tf, safe_place

# --- CONFIGURATION ---

fps = 100
dt = 1.0 / fps
input_dir = "DATA/generated_q_csv"

urdf_meshes_path = "motif/model/human_urdf"

needed_markers = [
    'r_mankle_study', 'r_ankle_study', 'r_toe_study', 'r_5meta_study', 'r_calc_study',
    'L_mankle_study', 'L_ankle_study', 'L_toe_study', 'L_5meta_study', 'L_calc_study'
]

# --- CHARGEMENT DU MODÈLE ---


# --- INITIALISATION MESHCAT ---
import meshcat.geometry as g

def draw_force_arrow(viewer, name, cop, force, color=0xff0000, scale=0.001):
    """
    Affiche une ligne partant du COP pour représenter la force.
    color: format hexadécimal (ex: 0xff0000 pour rouge)
    """
    # Calcul du point d'arrivée
    end_point = cop + (force * scale)
    
    # Création des points de la ligne (doit être un array 3x2)
    points = np.array([cop, end_point]).astype(np.float32).T
    
    # Envoi au viewer Meshcat
    viewer[name].set_object(g.Line(g.PointsGeometry(points), 
                                   g.LineBasicMaterial(color=color, linewidth=3)))
# viewer = meshcat.Visualizer()
# viz_human = MeshcatVisualizer(model_h, coll_h, vis_h)
# viz_human.initViewer(viewer, open=True)
# viz_human.loadViewerModel("ref", color=[0.0, 1.0, 0.0, 0.8])

# Repères de pieds et pelvis
# for side in ["R", "L"]:
#     meshcat_shapes.frame(viewer[f"Foot_{side}"], axis_length=0.2, axis_thickness=0.01)
# meshcat_shapes.frame(viewer["pelvis"], axis_length=0.2, axis_thickness=0.01)

# # Sphères de COP
# add_sphere(viewer, "world/COP_right", radius=0.015, color=0xFF8800) # Orange
# add_sphere(viewer, "world/COP_left",  radius=0.015, color=0x0000FF) # Bleu
# add_sphere(viewer, "world/COP_platform_global", radius=0.018, color=0x00FF00) # Vert
# add_sphere(viewer, "world/COP_RNEA",  radius=0.018, color=0xFF0000) # Rouge
import re

# --- BOUCLE SUR LES TRIALS ---
joint_files = glob.glob(os.path.join(input_dir, "*_q.csv"))
pattern = re.compile(
    r"^(subject_\d+)_(.+)_q\.csv$"
)
for path_joint in joint_files:
    filename = os.path.basename(path_joint)

    match = pattern.match(filename)
    if not match:
        continue

    subject = match.group(1)
    trial_id = match.group(2)

    print(subject)
    print(trial_id)

    path_kinetics = os.path.join(input_dir, f"{subject}_{trial_id}_grfm.csv")
    output_path = os.path.join(input_dir, f"feet_frame_{trial_id}")
    print(path_kinetics)
    print(output_path)

    if os.path.exists(output_path):
        print(f"Skipping: {trial_id} (déjà traité)")
        continue

    if not os.path.exists(path_kinetics): continue
    
    print(f"Processing: {trial_id}")
    q_ref = pd.read_csv(path_joint).to_numpy(dtype=float)
    df_cop = pd.read_csv(path_kinetics)
    
    # Calcul v et a pour RNEA
    n_samples = len(q_ref)
    urdf_dir = "DATA/urdf_scaled/URDFS"

    pattern = re.compile(rf"^({subject})_(\d+(?:\.\d+)?)kg\.urdf$")
    urdf_path = None

    for f in os.listdir(urdf_dir):
        if pattern.match(f):
            urdf_path = os.path.join(urdf_dir, f)
            break

    if urdf_path is None:
        raise FileNotFoundError(f"Aucun URDF pour {subject}")

    print(urdf_path)
    input()

    model_h, coll_h, vis_h, _ = build_human_model(urdf_path, urdf_meshes_path)
    data_h = model_h.createData()
    nv = model_h.nv
    v_ref = np.zeros((n_samples, nv))
    a_ref = np.zeros((n_samples, nv))
    for i in range(n_samples - 1):
        v_ref[i, :] = pin.difference(model_h, q_ref[i, :], q_ref[i+1, :]) / dt
    for i in range(n_samples - 1):
        a_ref[i, :] = (v_ref[i+1, :] - v_ref[i, :]) / dt

    results_feet = []
    data_plot = {
        'R': {'F': [], 'M': [], 'COP': []},
        'L': {'F': [], 'M': [], 'COP': []}
    }

    for i in range(n_samples):
        q_curr = q_ref[i, :]
        
        # 1. Mise à jour Cinématique et Dynamique (RNEA)
        tau = pin.rnea(model_h, data_h, q_curr, v_ref[i, :], a_ref[i, :])
        pin.forwardKinematics(model_h, data_h, q_curr)
        pin.updateFramePlacements(model_h, data_h)

        # 2. Calcul du COP RNEA (Global)
        pos_bassin = q_curr[0:3]
        quat_p = pin.Quaternion(q_curr[6], q_curr[3], q_curr[4], q_curr[5])
        R_p = quat_p.toRotationMatrix()
        
        # Wrench au bassin via RNEA
        f_world_rnea = data_h.oMi[1].act(data_h.f[1]) 
        F_rnea = f_world_rnea.linear
        # Moment au centre monde (0,0,0) induit par RNEA
        M_rnea = (R_p @ tau[3:6]) + np.cross(pos_bassin, F_rnea)
        
        cop_rnea = np.array([-M_rnea[1]/F_rnea[2], M_rnea[0]/F_rnea[2], 0.0])

        # 3. Extraction et calcul COP Global Plateformes
        F_w1 = np.array([df_cop["Fx1_glob"][i], df_cop["Fy1_glob"][i], df_cop["Fz1_glob"][i]])
        F_w2 = np.array([df_cop["Fx2_glob"][i], df_cop["Fy2_glob"][i], df_cop["Fz2_glob"][i]])
        M_w1 = np.array([df_cop["Mx1_glob"][i], df_cop["My1_glob"][i], df_cop["Mz1_glob"][i]])
        M_w2 = np.array([df_cop["Mx2_glob"][i], df_cop["My2_glob"][i], df_cop["Mz2_glob"][i]])
        cop_w1 = np.array([df_cop["COPx1_glob"][i], df_cop["COPy1_glob"][i], df_cop["COPz1_glob"][i]])
        cop_w2 = np.array([df_cop["COPx2_glob"][i], df_cop["COPy2_glob"][i], df_cop["COPz2_glob"][i]])
        
        Fz_total = F_w1[2] + F_w2[2]
        if Fz_total > 10: # Évite division par zéro si sujet en l'air
            cop_global = (F_w1[2]*cop_w1 + F_w2[2]*cop_w2) / Fz_total
            cop_global[2] = 0.0
        else:
            cop_global = np.array([0,0,0])

        # 4. Transformation Repère Pieds
        mks_pos = {n: data_h.oMf[model_h.getFrameId(n)].translation for n in needed_markers}
        T_w_fR = get_foot_pose(mks_pos, side='right')
        T_w_fL = get_foot_pose(mks_pos, side='left')

        # Pied Droit
        R_fR, P_fR = T_w_fR[:3, :3], T_w_fR[:3, 3]
        F_locR = R_fR.T @ F_w1
        M_locR = R_fR.T @ (M_w1 - np.cross(P_fR, F_w1))
        cop_locR = R_fR.T @ (cop_w1 - P_fR)

        # Pied Gauche
        R_fL, P_fL = T_w_fL[:3, :3], T_w_fL[:3, 3]
        F_locL = R_fL.T @ F_w2
        M_locL = R_fL.T @ (M_w2 - np.cross(P_fL, F_w2))
        cop_locL = R_fL.T @ (cop_w2 - P_fL)

        # --- BACK TO WORLD ---
        
        # 1. Re-transformation de la Force vers World
        F_world_check = R_fR @ F_locR
        
        # 2. Re-transformation du Moment vers World
        # Attention : On doit inverser le transport de moment
        # M_w = (R @ M_loc) + (P_foot ^ F_world)
        M_world_check = (R_fR @ M_locR) + np.cross(P_fR, F_world_check)
        
        # 3. Re-transformation du COP vers World
        # cop_w = (R @ cop_loc) + P_foot
        cop_world_check = (R_fR @ cop_locR) + P_fR

        # --- COMPARAISON (Vérification de l'erreur) ---
        error_F = np.linalg.norm(F_w1 - F_world_check)
        error_M = np.linalg.norm(M_w1 - M_world_check)
        error_COP = np.linalg.norm(cop_w1 - cop_world_check)

        if error_F > 1e-6 or error_M > 1e-6:
            print(f"!!! Erreur de calcul au frame {i} !!!")
            print(f"Erreur Force: {error_F}, Erreur Moment: {error_M}, Erreur COP: {error_COP}")
        else: 
            print("no problem")


        # 5. Affichage
        # set_tf(viewer, "pelvis", pin.SE3(R_p, pos_bassin).homogeneous)
        # set_tf(viewer, "Foot_R", T_w_fR)
        # set_tf(viewer, "Foot_L", T_w_fL)
        
        # safe_place(viewer, "COP_right", cop_w1)
        # safe_place(viewer, "COP_left", cop_w2)
        # safe_place(viewer, "COP_platform_global", cop_global)
        # safe_place(viewer, "COP_RNEA", cop_rnea)
        
        # draw_force_arrow(viewer, "force_R", cop_w1, F_w1, color=0xFF8800)
        # draw_force_arrow(viewer, "force_L", cop_w2, F_w2, color=0x0000FF)

        # viz_human.display(q_curr)

        # 6. Sauvegarde des données pieds
        results_feet.append({
            'Fx1': F_locR[0], 'Fy1': F_locR[1], 'Fz1': F_locR[2],
            'Mx1': M_locR[0], 'My1': M_locR[1], 'Mz1': M_locR[2],
            'COPx1': cop_locR[0], 'COPy1': cop_locR[1], 'COPz1': cop_locR[2],
            'Fx2': F_locL[0], 'Fy2': F_locL[1], 'Fz2': F_locL[2],
            'Mx2': M_locL[0], 'My2': M_locL[1], 'Mz2': M_locL[2],
            'COPx2': cop_locL[0], 'COPy2': cop_locL[1], 'COPz2': cop_locL[2]
        })

        data_plot['R']['F'].append(F_locR)
        data_plot['R']['M'].append(M_locR)
        data_plot['R']['COP'].append(cop_locR)
        data_plot['L']['F'].append(F_locL)
        data_plot['L']['M'].append(M_locL)
        data_plot['L']['COP'].append(cop_locL)
    
    def plot_side_data(side_key, full_name, t_id):
        F = np.array(data_plot[side_key]['F'])
        M = np.array(data_plot[side_key]['M'])
        COP = np.array(data_plot[side_key]['COP'])
        time_axis = np.arange(len(F)) * dt

        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(f'Trial: {t_id}\nEfforts Local Foot {full_name}', fontsize=12)

        # Force Plot
        axs[0].plot(time_axis, F[:, 0], 'r', label='Fx (AP)')
        axs[0].plot(time_axis, F[:, 1], 'g', label='Fy (Vert)')
        axs[0].plot(time_axis, F[:, 2], 'b', label='Fz (ML)')
        axs[0].set_ylabel('Force [N]')
        axs[0].legend(loc='upper right')
        axs[0].grid(True)

        # Moment Plot
        axs[1].plot(time_axis, M[:, 0], 'r', label='Mx')
        axs[1].plot(time_axis, M[:, 1], 'g', label='My')
        axs[1].plot(time_axis, M[:, 2], 'b', label='Mz')
        axs[1].set_ylabel('Moment [Nm]')
        axs[1].grid(True)

        # COP Plot
        axs[2].plot(time_axis, COP[:, 0], 'r', label='COPx')
        axs[2].plot(time_axis, COP[:, 1], 'g', label='COPy (Height)')
        axs[2].plot(time_axis, COP[:, 2], 'b', label='COPz')
        axs[2].set_ylabel('COP Local [m]')
        axs[2].set_xlabel('Time [s]')
        axs[2].grid(True)

        plt.tight_layout()
        plt.savefig(f"plot_{side_key}_{t_id}.png")
        # plt.show()

    # Affichage des plots (bloquant jusqu'à fermeture)
    # plot_side_data('R', 'RIGHT', trial_id)
    # plot_side_data('L', 'LEFT', trial_id)

    # Sauvegarde CSV du trial
    # out_csv = os.path.join(input_dir, f"feet_frame_{trial_id}")
    pd.DataFrame(results_feet).to_csv(output_path, index=False)
    print(f"   -> Terminé et sauvegardé dans : {output_path}")

print("\n--- TRAITEMENT TERMINÉ ---")