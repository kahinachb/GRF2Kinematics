"""Convertit les CSV de ``DATA/christine_ref`` au format du training.

Sorties par defaut::

    DATA/christine_ref_npy/Christine/variant_000/kinetics_deltaf.npy
    DATA/christine_ref_npy/Christine/variant_000/all_joints_deltaf.npy

La cinetique conserve la convention des CSV generes : GRFM 1 = pied gauche,
GRFM 2 = pied droit. La premiere frame est retiree pour l'aligner avec les
deltas locaux du free-flyer.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


KINETICS_COLUMNS = [
    "Fx1_glob", "Fy1_glob", "Fz1_glob",
    "Mx1_glob", "My1_glob", "Mz1_glob",
    "COPx1_glob", "COPy1_glob", "COPz1_glob",
    "Fx2_glob", "Fy2_glob", "Fz2_glob",
    "Mx2_glob", "My2_glob", "Mz2_glob",
    "COPx2_glob", "COPy2_glob", "COPz2_glob",
]

FREE_FLYER_COLUMNS = [
    "FF_X", "FF_Y", "FF_Z",
    "FF_quatx", "FF_quaty", "FF_quatz", "FF_quatw",
]

JOINT_COLUMNS = [
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",
    "Lumbar_flex_ext", "Lumbar_lateral_flex", "Lcalvicule_x",
    "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
    "Lelbow_flex_ext", "Lelbow_pron_supi",
    "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
    "rcalvicule_x",
    "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot",
    "Relbow_flex_ext", "Relbow_pron_supi",
]


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {source}: {missing}")


def convert(q_csv: Path, grfm_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    joints_frame = pd.read_csv(q_csv)
    grfm_frame = pd.read_csv(grfm_csv)
    require_columns(joints_frame, FREE_FLYER_COLUMNS + JOINT_COLUMNS, q_csv)
    require_columns(grfm_frame, KINETICS_COLUMNS, grfm_csv)

    if len(joints_frame) != len(grfm_frame):
        raise ValueError(
            f"Nombre de frames different: q={len(joints_frame)}, "
            f"GRFM={len(grfm_frame)}"
        )
    if len(joints_frame) < 2:
        raise ValueError("Au moins deux frames sont necessaires pour calculer les deltas.")

    free_flyer = joints_frame[FREE_FLYER_COLUMNS].to_numpy(dtype=np.float64)
    positions = free_flyer[:, :3]
    quaternions = free_flyer[:, 3:7]
    quaternion_norms = np.linalg.norm(quaternions, axis=1)
    if not np.isfinite(free_flyer).all() or not np.isfinite(quaternion_norms).all():
        raise ValueError(f"Valeurs non finies dans {q_csv}")
    if np.any(quaternion_norms < 1e-8):
        raise ValueError(f"Quaternion nul dans {q_csv}")

    # Rotation normalise les quaternions et utilise la convention (x, y, z, w).
    rotations = Rotation.from_quat(quaternions)
    rotations_t = rotations[:-1]
    delta_positions_local = rotations_t.inv().apply(positions[1:] - positions[:-1])
    delta_rotations_local = (rotations_t.inv() * rotations[1:]).as_rotvec()

    # La cible du modele est [delta FF(6), jambe D(6), jambe G(6), haut(17)].
    articulated = joints_frame[JOINT_COLUMNS].to_numpy(dtype=np.float32)[1:]
    all_joints_delta = np.concatenate(
        [delta_positions_local, delta_rotations_local, articulated], axis=1
    ).astype(np.float32)

    # Le CSV christine_ref est deja [gauche, droite], comme le training synth.
    kinetics_delta = grfm_frame[KINETICS_COLUMNS].to_numpy(dtype=np.float32)[1:]

    if not np.isfinite(all_joints_delta).all() or not np.isfinite(kinetics_delta).all():
        raise ValueError("Les tableaux convertis contiennent des NaN ou des valeurs infinies.")
    if all_joints_delta.shape[1] != 35 or kinetics_delta.shape[1] != 18:
        raise AssertionError(
            f"Dimensions inattendues: joints={all_joints_delta.shape}, "
            f"GRFM={kinetics_delta.shape}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    kinetics_path = output_dir / "kinetics_deltaf.npy"
    joints_path = output_dir / "all_joints_deltaf.npy"
    np.save(kinetics_path, kinetics_delta)
    np.save(joints_path, all_joints_delta)

    print(f"GRFM   : {kinetics_path} {kinetics_delta.shape}")
    print(f"Joints : {joints_path} {all_joints_delta.shape}")
    print("Convention GRFM : [gauche (0:9), droite (9:18)]")
    return kinetics_path, joints_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("DATA/christine_ref"),
        help="Dossier contenant les deux CSV christine_ref.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("DATA/christine_ref_npy/Christine/variant_000"),
        help="Dossier dans lequel enregistrer les deux fichiers NPY.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        args.input_dir / "Christine_Trial110_variant_000_q_doc.csv",
        args.input_dir / "Christine_Trial110_variant_000_grfm_doc.csv",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
