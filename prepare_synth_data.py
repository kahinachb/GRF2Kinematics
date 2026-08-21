#convert csv to npy for Vinc and synth data.
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

from utils.model_utils import build_human_model, get_foot_pose

KINETICS_COLS = [
    'Fx1_glob', 'Fy1_glob', 'Fz1_glob', 'Mx1_glob', 'My1_glob', 'Mz1_glob', 'COPx1_glob', 'COPy1_glob', 'COPz1_glob',
    'Fx2_glob', 'Fy2_glob', 'Fz2_glob', 'Mx2_glob', 'My2_glob', 'Mz2_glob', 'COPx2_glob', 'COPy2_glob', 'COPz2_glob'
]

# Réorganisation demandée des Joints (Freeflyer -> Right Leg -> Left Leg -> Upper Body)
JOINTS_REORDER = [
    "FF_X", "FF_Y", "FF_Z", "FF_quatx", "FF_quaty", "FF_quatz", "FF_quatw", 
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot", "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot", "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add", 
    "Lumbar_flex_ext", "Lumbar_lateral_flex", "Lcalvicule_x",
    "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot", "Lelbow_flex_ext", "Lelbow_pron_supi",
    "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot", "rcalvicule_x",
    "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot", "Relbow_flex_ext", "Relbow_pron_supi"
]

# Same marker-based foot frames and right/left output ordering as
# process_data/glob_to_feet.py for the Vinc dataset.
FOOT_MARKERS = [
    'r_mankle_study', 'r_ankle_study', 'r_toe_study', 'r_5meta_study', 'r_calc_study',
    'L_mankle_study', 'L_ankle_study', 'L_toe_study', 'L_5meta_study', 'L_calc_study',
]
URDF_MESHES_PATH = "motif/model/human_urdf"
CONTACT_THRESHOLD_N = 20.0


def _wrench_in_foot_frame(force_world, moment_world, cop_world, foot_pose):
    """Express a world wrench and COP in one local foot frame.

    The output convention intentionally matches glob_to_feet.py: the first
    9-feature block is the right foot and the second is the left foot.
    """
    rotation, position = foot_pose[:3, :3], foot_pose[:3, 3]
    force_world = force_world.copy()

    # Match the real-data converter's plate vertical-force convention.
    if force_world[2] < -CONTACT_THRESHOLD_N:
        force_world[2] *= -1.0
    if abs(force_world[2]) <= CONTACT_THRESHOLD_N:
        return np.zeros(3), np.zeros(3), np.zeros(3)

    force_local = rotation.T @ force_world
    moment_local = rotation.T @ (moment_world - np.cross(position, force_world))
    cop_local = rotation.T @ (cop_world - position)
    return force_local, moment_local, cop_local


def global_grfm_to_feet(raw_joints, global_grfm, model_h):
    """Convert generated global GRFM to right/left local-foot GRFM.

    ``raw_joints`` must use ``JOINTS_REORDER`` (global free-flyer followed by
    the 29 articulated angles).  In generated_data, plate 1 is the left foot
    and plate 2 is the right foot.  The returned array is deliberately
    reordered to right then left so it matches the real kinetics_feet.npy
    convention produced by glob_to_feet.py.
    """
    if len(raw_joints) != len(global_grfm):
        raise ValueError(f"q/GRFM length mismatch: {len(raw_joints)} vs {len(global_grfm)}")

    data_h = model_h.createData()
    local_grfm = np.zeros_like(global_grfm, dtype=np.float32)
    marker_ids = {name: model_h.getFrameId(name) for name in FOOT_MARKERS}
    if any(frame_id >= model_h.nframes for frame_id in marker_ids.values()):
        missing = [name for name, frame_id in marker_ids.items() if frame_id >= model_h.nframes]
        raise ValueError(f"URDF is missing required foot-marker frames: {missing}")

    for i, (q_curr, wrench) in enumerate(zip(raw_joints, global_grfm)):
        pin.forwardKinematics(model_h, data_h, q_curr)
        pin.updateFramePlacements(model_h, data_h)
        markers = {name: data_h.oMf[frame_id].translation for name, frame_id in marker_ids.items()}
        right_foot = get_foot_pose(markers, side='right')
        left_foot = get_foot_pose(markers, side='left')

        f_left, m_left, cop_left = _wrench_in_foot_frame(
            wrench[0:3], wrench[3:6], wrench[6:9], left_foot
        )
        f_right, m_right, cop_right = _wrench_in_foot_frame(
            wrench[9:12], wrench[12:15], wrench[15:18], right_foot
        )
        local_grfm[i] = np.concatenate([f_right, m_right, cop_right, f_left, m_left, cop_left])

    return local_grfm

#ff to delta
def process_folder_to_local_delta(input_folder, output_base, make_feet_kinetics=False,
                                  urdf_dir="DATA/10_urdf"):
    input_path = Path(input_folder)
    output_path = Path(output_base)
    joint_files = glob.glob(str(input_path / "*q_doc.csv"))
    models = {}
    
    for j_file in joint_files:
        trial_id = os.path.basename(j_file).replace("_q_doc", "").replace(".csv", "")

        parts = trial_id.split("_")
        # print(parts)

        # subject_name = "_".join(parts[:2])   # subject_01
        subject_name = parts[0]
        # print(subject_name)
        # input()
        variant_idx = parts.index("variant")
        task_name = "_".join(parts[variant_idx:])
        
        k_file = input_path / f"{trial_id}_grfm_doc.csv"
        print(k_file)
        if not k_file.exists(): continue
        
        trial_dir = output_path / subject_name / task_name
        # Optionnel : skip si déjà fait
        # if (trial_dir / "all_joints.npy").exists(): continue

        try:
            # --- CHARGEMENT ---
            df_k = pd.read_csv(k_file)
            df_j = pd.read_csv(j_file)

            # --- PRÉPARATION DES COLONNES JOINTS ---
            old_ff_cols = df_j.columns[:7]
            ff_rename = {old: new for old, new in zip(old_ff_cols, JOINTS_REORDER[:7])}
            df_j = df_j.rename(columns=ff_rename)
            raw_joints = df_j[JOINTS_REORDER].values.astype(np.float32)

            # --- CALCUL DU FREEFLYER LOCAL ---
            # Positions (X, Y, Z) et Quaternions (x, y, z, w)
            pos_global = raw_joints[:, 0:3]
            rot_global = R.from_quat(raw_joints[:, 3:7])

            # Matrices de rotation à l'instant t
            r_t = rot_global[:-1] 
            # Positions à t et t+1
            p_t = pos_global[:-1]
            p_next = pos_global[1:]
            # Rotations à t+1
            r_next = rot_global[1:]

            # 1. Translation locale : R_t.inv() * (p_next - p_t)
            # .apply() sur un objet Rotation Scipy fait exactement la multiplication matricielle R.T @ delta_p
            local_delta_pos = r_t.inv().apply(p_next - p_t)

            # 2. Rotation locale : R_rel = R_t.inv() * R_next
            local_delta_rot_obj = r_t.inv() * r_next
            local_delta_rotvec = local_delta_rot_obj.as_rotvec()

            pose_joints = raw_joints[1:, 7:]

            # --- CONCATÉNATION (3 pos + 3 rotvec + 29 angles = 35 cols) ---
            arr_j_final = np.hstack([
                local_delta_pos, 
                local_delta_rotvec, 
                pose_joints
            ]).astype(np.float32)

            # --- TRAITEMENT KINETICS ---
            rename_map = {col: col.replace('footR', 'fR').replace('footL', 'fL') for col in df_k.columns}
            df_k = df_k.rename(columns=rename_map)
            arr_k = df_k[KINETICS_COLS].values.astype(np.float32)
            
            # Alignement : on prend à partir de la frame 1 pour matcher les deltas
            arr_k_sync = arr_k[1:]

            # --- SAUVEGARDE ---
            trial_dir.mkdir(parents=True, exist_ok=True)
            np.save(trial_dir / "kinetics_deltaf.npy", arr_k_sync)
            np.save(trial_dir / "all_joints_deltaf.npy", arr_j_final)
            print(f" {trial_dir }/ all_joints_deltaf.npy")

            if make_feet_kinetics:
                if subject_name not in models:
                    urdf_path = Path(urdf_dir) / f"{subject_name}_scaled.urdf"
                    if not urdf_path.exists():
                        raise FileNotFoundError(f"URDF not found: {urdf_path}")
                    models[subject_name] = build_human_model(str(urdf_path), URDF_MESHES_PATH)[0]
                feet_grfm = global_grfm_to_feet(raw_joints, arr_k, models[subject_name])
                # Same frame-1 alignment as the global kinetics and q target.
                np.save(trial_dir / "kinetics_feet_deltaf.npy", feet_grfm[1:])
            
            print(f"  [OK] {trial_id} : {arr_j_final.shape}")

        except Exception as e:
            print(f"  [ERROR] {trial_id} : {e}")


if __name__ == "__main__":
    # Dossier où se trouvent tes fichiers en vrac
    IN_FOLDER = "DATA/Christine_npy"
    # Dossier où tu veux créer tes dossiers de trials
    OUT_FOLDER = "DATA/synth_christine"
    
    process_folder_to_local_delta(IN_FOLDER, OUT_FOLDER, make_feet_kinetics=True)
    print("\n--- Opération terminée ---")
