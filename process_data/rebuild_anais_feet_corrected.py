"""Rebuild Anais local-foot kinetics with a per-trial force-plate mapping.

Why this script exists
----------------------
The Anais recordings do not use one fixed association between force plates and
anatomical feet.  The old ``glob_to_feet.py`` hard-codes plate 1 as left and
plate 2 as right, which is incorrect for many trials.  This script infers the
mapping independently for every trial from the robust distance between each
active plate CoP and the two reconstructed foot frames.

For every processed trial it writes:

``<output-root>/<subject>/<trial>/kinetics_feet.csv``
    18 channels in the fixed anatomical order right then left:
    ``[F_R(3), M_R(3), CoP_R(3), F_L(3), M_L(3), CoP_L(3)]``.

``<output-root>/<subject>/<trial>/kinetics_feet.npy``
    The same data as a ``float32`` array, shape ``(T, 18)``.

``<output-root>/<subject>/<trial>/all_joints.npy``
    A validated copy of the existing target array, so the output root can be
    passed directly to the Anais training scripts.

``<output-root>/mapping_report.csv``
    Provenance and plate-mapping diagnostics for every trial.

The existing ``processed_data_feet`` is never overwritten unless an explicit
output path is supplied and ``--overwrite`` is passed.

Example
-------
python process_data/rebuild_anais_feet_corrected.py
python process_data/rebuild_anais_feet_corrected.py --tasks luyo
python process_data/rebuild_anais_feet_corrected.py --dry-run --tasks luyo
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin

# Keep the project root importable when this file is run directly with
# ``python process_data/rebuild_anais_feet_corrected.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_utils import build_human_model, get_foot_pose


DEFAULT_SOURCE_ROOT = Path("DATA/Anais")
DEFAULT_URDF_DIR = Path("DATA/10_urdf")
DEFAULT_TARGETS_ROOT = Path("processed_data_feet")
DEFAULT_OUTPUT_ROOT = Path("processed_data_feet_anais_platefixed")

TASK_PREFIXES = ("bend", "dyna", "lufe", "luyo", "static2", "walk")
MARKER_FRAMES = (
    "r_mankle_study", "r_ankle_study", "r_toe_study", "r_5meta_study", "r_calc_study",
    "L_mankle_study", "L_ankle_study", "L_toe_study", "L_5meta_study", "L_calc_study",
)
OUTPUT_COLUMNS = (
    "Fx1", "Fy1", "Fz1", "Mx1", "My1", "Mz1", "COPx1", "COPy1", "COPz1",
    "Fx2", "Fy2", "Fz2", "Mx2", "My2", "Mz2", "COPx2", "COPy2", "COPz2",
)


@dataclass(frozen=True)
class PlateMapping:
    """An inferred mapping, with plate numbers represented as 1 or 2."""

    left_plate: int
    right_plate: int
    left_right_score_m: float
    right_left_score_m: float
    confidence_ratio: float


def canonical_task_name(name: str) -> str | None:
    """Recognise supported task folders, including ``bend2`` and ``luyo2``."""
    normalized = name.strip().lower()
    if "copie" in normalized:
        return None
    for task in TASK_PREFIXES:
        if normalized == task or normalized.startswith(f"{task}_") or normalized.startswith(f"{task}2"):
            return task
    return None


def parse_csv_list(value: str | None) -> set[str] | None:
    if value is None:
        return None
    entries = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not entries:
        raise ValueError("The supplied comma-separated list is empty.")
    return entries


def plate_columns(plate: int) -> tuple[list[str], list[str], list[str]]:
    if plate not in (1, 2):
        raise ValueError(f"Invalid plate number: {plate}")
    suffix = f"{plate}_glob"
    return (
        [f"Fx{suffix}", f"Fy{suffix}", f"Fz{suffix}"],
        [f"Mx{suffix}", f"My{suffix}", f"Mz{suffix}"],
        [f"COPx{suffix}", f"COPy{suffix}", f"COPz{suffix}"],
    )


def foot_positions(model, data, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return world rotations and ankle-centre origins for right and left feet."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    markers = {
        marker: data.oMf[model.getFrameId(marker)].translation
        for marker in MARKER_FRAMES
    }
    right_pose = get_foot_pose(markers, side="right")
    left_pose = get_foot_pose(markers, side="left")
    return right_pose[:3, :3], right_pose[:3, 3], left_pose[:3, :3], left_pose[:3, 3]


def load_trial_inputs(trial_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    joints_path = trial_dir / "joints_filtered_FF.csv"
    kinetics_path = trial_dir / "kinetics_glob_filtered.csv"
    if not joints_path.exists() or not kinetics_path.exists():
        raise FileNotFoundError(
            f"Expected joints_filtered_FF.csv and kinetics_glob_filtered.csv in {trial_dir}"
        )
    # The kinetics files have one fewer row than the FF file; this is the same
    # alignment used by the existing glob_to_feet script.
    q = pd.read_csv(joints_path).iloc[1:].to_numpy(dtype=np.float64)
    kinetics = pd.read_csv(kinetics_path)
    if len(q) != len(kinetics):
        raise ValueError(
            f"Frame mismatch in {trial_dir}: q={len(q)}, global kinetics={len(kinetics)}"
        )
    required = set()
    for plate in (1, 2):
        for group in plate_columns(plate):
            required.update(group)
    missing = sorted(required.difference(kinetics.columns))
    if missing:
        raise ValueError(f"Missing global kinetics columns in {kinetics_path}: {missing}")
    if not np.isfinite(q).all() or not np.isfinite(kinetics.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite value found in {trial_dir}")
    return q, kinetics


def has_trial_inputs(trial_dir: Path) -> bool:
    """Return whether this source trial contains the two aligned inputs."""
    return (
        (trial_dir / "joints_filtered_FF.csv").is_file()
        and (trial_dir / "kinetics_glob_filtered.csv").is_file()
    )


def infer_plate_mapping(
    model,
    data,
    q: np.ndarray,
    kinetics: pd.DataFrame,
    contact_threshold_n: float,
    sample_stride: int,
    min_confidence_ratio: float,
) -> PlateMapping:
    """Infer plate-to-foot mapping from active CoP-to-foot distances.

    The selected association is robust because it uses medians over the whole
    trial rather than a single posture.  CoPs are used only while the plate's
    global vertical force is above the contact threshold.
    """
    _, _, cop1_cols = plate_columns(1)
    _, _, cop2_cols = plate_columns(2)
    fz1 = kinetics["Fz1_glob"].to_numpy(dtype=float)
    fz2 = kinetics["Fz2_glob"].to_numpy(dtype=float)
    cop1 = kinetics[cop1_cols].to_numpy(dtype=float)
    cop2 = kinetics[cop2_cols].to_numpy(dtype=float)

    p1_to_left: list[float] = []
    p1_to_right: list[float] = []
    p2_to_left: list[float] = []
    p2_to_right: list[float] = []
    for frame in range(0, len(q), sample_stride):
        _, right_origin, _, left_origin = foot_positions(model, data, q[frame])
        if abs(fz1[frame]) > contact_threshold_n:
            p1_to_left.append(float(np.linalg.norm(cop1[frame] - left_origin)))
            p1_to_right.append(float(np.linalg.norm(cop1[frame] - right_origin)))
        if abs(fz2[frame]) > contact_threshold_n:
            p2_to_left.append(float(np.linalg.norm(cop2[frame] - left_origin)))
            p2_to_right.append(float(np.linalg.norm(cop2[frame] - right_origin)))

    if not all((p1_to_left, p1_to_right, p2_to_left, p2_to_right)):
        raise ValueError("Insufficient active CoP samples to infer the plate mapping.")

    # Candidate A is the historical hard-coded convention. Candidate B swaps
    # the physical plates.  Smaller distance means CoP lies below that foot.
    score_plate1_left = float(np.median(p1_to_left) + np.median(p2_to_right))
    score_plate1_right = float(np.median(p1_to_right) + np.median(p2_to_left))
    best = min(score_plate1_left, score_plate1_right)
    worst = max(score_plate1_left, score_plate1_right)
    confidence = worst / max(best, 1e-8)
    if confidence < min_confidence_ratio:
        raise ValueError(
            "Ambiguous plate mapping: candidate distance scores are too close "
            f"({score_plate1_left:.3f} m vs {score_plate1_right:.3f} m, "
            f"ratio={confidence:.2f}x)."
        )
    if score_plate1_left < score_plate1_right:
        return PlateMapping(1, 2, score_plate1_left, score_plate1_right, confidence)
    return PlateMapping(2, 1, score_plate1_left, score_plate1_right, confidence)


def vertical_force_sign(kinetics: pd.DataFrame, plate: int, threshold_n: float) -> float:
    """Return a consistent wrench sign for a plate, normally +1 for Anais.

    A complete wrench (force *and* moment) is flipped only if the active
    vertical force convention is globally negative.  This is safer than
    changing Fz alone before translating the moment to the foot origin.
    """
    fz = kinetics[f"Fz{plate}_glob"].to_numpy(dtype=float)
    active = fz[np.abs(fz) > threshold_n]
    if not len(active):
        raise ValueError(f"Plate {plate} has no active samples.")
    return -1.0 if np.median(active) < 0.0 else 1.0


def local_wrench_and_cop(
    rotation_world_foot: np.ndarray,
    foot_origin_world: np.ndarray,
    force_world: np.ndarray,
    moment_world: np.ndarray,
    cop_world: np.ndarray,
    in_contact: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Express one plate wrench and CoP in its anatomical foot frame."""
    if not in_contact:
        return np.zeros(3), np.zeros(3), np.zeros(3)
    rotation_foot_world = rotation_world_foot.T
    force_local = rotation_foot_world @ force_world
    moment_local = rotation_foot_world @ (
        moment_world - np.cross(foot_origin_world, force_world)
    )
    cop_local = rotation_foot_world @ (cop_world - foot_origin_world)
    return force_local, moment_local, cop_local


def build_local_kinetics(
    model,
    data,
    q: np.ndarray,
    kinetics: pd.DataFrame,
    mapping: PlateMapping,
    contact_threshold_n: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build an array in the fixed order [right block, left block]."""
    force_cols_left, moment_cols_left, cop_cols_left = plate_columns(mapping.left_plate)
    force_cols_right, moment_cols_right, cop_cols_right = plate_columns(mapping.right_plate)
    force_left = kinetics[force_cols_left].to_numpy(dtype=float)
    moment_left = kinetics[moment_cols_left].to_numpy(dtype=float)
    cop_left = kinetics[cop_cols_left].to_numpy(dtype=float)
    force_right = kinetics[force_cols_right].to_numpy(dtype=float)
    moment_right = kinetics[moment_cols_right].to_numpy(dtype=float)
    cop_right = kinetics[cop_cols_right].to_numpy(dtype=float)
    fz_left = kinetics[f"Fz{mapping.left_plate}_glob"].to_numpy(dtype=float)
    fz_right = kinetics[f"Fz{mapping.right_plate}_glob"].to_numpy(dtype=float)
    sign_left = vertical_force_sign(kinetics, mapping.left_plate, contact_threshold_n)
    sign_right = vertical_force_sign(kinetics, mapping.right_plate, contact_threshold_n)

    output = np.zeros((len(q), 18), dtype=np.float32)
    right_contacts = 0
    left_contacts = 0
    for frame in range(len(q)):
        right_rotation, right_origin, left_rotation, left_origin = foot_positions(model, data, q[frame])
        right_contact = abs(fz_right[frame]) > contact_threshold_n
        left_contact = abs(fz_left[frame]) > contact_threshold_n
        force_r, moment_r, cop_r = local_wrench_and_cop(
            right_rotation, right_origin,
            sign_right * force_right[frame], sign_right * moment_right[frame], cop_right[frame],
            right_contact,
        )
        force_l, moment_l, cop_l = local_wrench_and_cop(
            left_rotation, left_origin,
            sign_left * force_left[frame], sign_left * moment_left[frame], cop_left[frame],
            left_contact,
        )
        # Block 1 is always anatomical RIGHT; block 2 is always anatomical LEFT.
        output[frame] = np.concatenate((force_r, moment_r, cop_r, force_l, moment_l, cop_l))
        right_contacts += int(right_contact)
        left_contacts += int(left_contact)

    if not np.isfinite(output).all():
        raise ValueError("Local kinetics contain non-finite values after conversion.")
    return output, {
        "right_contact_frames": right_contacts,
        "left_contact_frames": left_contacts,
        "right_force_sign": int(sign_right),
        "left_force_sign": int(sign_left),
    }


def validate_local_output(local: np.ndarray, contact_threshold_n: float) -> None:
    """Check output ordering and the zero-kinetics convention off contact."""
    if local.ndim != 2 or local.shape[1] != 18:
        raise ValueError(f"Expected local kinetics shape (T, 18), got {local.shape}")
    for force_slice, rest_slice, side in (
        (slice(0, 3), slice(3, 9), "right"),
        (slice(9, 12), slice(12, 18), "left"),
    ):
        unloaded = np.linalg.norm(local[:, force_slice], axis=1) < 1e-6
        if unloaded.any() and not np.allclose(local[unloaded, rest_slice], 0.0, atol=1e-6):
            raise ValueError(f"Non-zero moment or CoP found while the {side} foot is unloaded.")


def load_validated_targets(
    targets_root: Path,
    subject: str,
    trial: str,
    n_frames: int,
) -> np.ndarray:
    """Load a target array and ensure it is aligned with the rebuilt input."""
    source_target = targets_root / subject / trial / "all_joints.npy"
    if not source_target.exists():
        raise FileNotFoundError(f"Target file not found: {source_target}")
    joints = np.load(source_target)
    if joints.ndim != 2 or joints.shape[0] != n_frames or joints.shape[1] < 35:
        raise ValueError(
            f"Target shape {joints.shape} does not match local kinetics "
            f"({n_frames}, 18) for {subject}/{trial}."
        )
    if not np.isfinite(joints).all():
        raise ValueError(f"Non-finite target value found in {source_target}")
    return joints


def write_trial(
    output_root: Path,
    targets_root: Path,
    subject: str,
    trial: str,
    local: np.ndarray,
    overwrite: bool,
    copy_targets: bool,
) -> None:
    target_dir = output_root / subject / trial
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "kinetics_feet.csv"
    npy_path = target_dir / "kinetics_feet.npy"
    if not overwrite and (csv_path.exists() or npy_path.exists()):
        raise FileExistsError(
            f"Refusing to overwrite {target_dir}. Use --overwrite or another --output-root."
        )
    pd.DataFrame(local, columns=OUTPUT_COLUMNS).to_csv(csv_path, index=False)
    np.save(npy_path, local.astype(np.float32, copy=False))

    if copy_targets:
        joints = load_validated_targets(targets_root, subject, trial, len(local))
        np.save(target_dir / "all_joints.npy", joints.astype(np.float32, copy=False))


def iter_trials(source_root: Path, subjects: set[str] | None, tasks: set[str] | None):
    for subject_dir in sorted(path for path in source_root.glob("subject[0-9][0-9]") if path.is_dir()):
        subject = subject_dir.name.lower()
        if subjects is not None and subject not in subjects:
            continue
        for trial_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            task = canonical_task_name(trial_dir.name)
            if task is None or (tasks is not None and task not in tasks):
                continue
            # A few directories are unprocessed originals (for example,
            # subject13/bend) or incomplete recordings.  They cannot be
            # reconstructed safely and are not part of the training targets.
            if not has_trial_inputs(trial_dir):
                print(f"[SKIP] {subject_dir.name}/{trial_dir.name}: missing aligned joint or global-kinetics CSV.")
                continue
            yield subject_dir.name, trial_dir.name, task, trial_dir


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--urdf-dir", type=Path, default=DEFAULT_URDF_DIR)
    parser.add_argument("--targets-root", type=Path, default=DEFAULT_TARGETS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subjects", help="Comma-separated list, e.g. subject01,subject02.")
    parser.add_argument("--tasks", help="Comma-separated canonical tasks, e.g. luyo,walk.")
    parser.add_argument("--contact-threshold", type=float, default=20.0, help="Global |Fz| contact threshold in N.")
    parser.add_argument("--mapping-stride", type=int, default=10, help="Use every Nth frame while inferring mapping.")
    parser.add_argument("--min-mapping-confidence", type=float, default=1.5)
    parser.add_argument("--no-copy-targets", action="store_true", help="Do not copy all_joints.npy to the output tree.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Infer and validate mappings without writing files.")
    return parser.parse_args()


def main(args) -> None:
    if args.contact_threshold <= 0:
        raise ValueError("--contact-threshold must be positive.")
    if args.mapping_stride < 1:
        raise ValueError("--mapping-stride must be at least one.")
    if args.min_mapping_confidence <= 1.0:
        raise ValueError("--min-mapping-confidence must be greater than one.")
    for name, path in (("source root", args.source_root), ("URDF directory", args.urdf_dir)):
        if not path.exists():
            raise FileNotFoundError(f"{name.capitalize()} does not exist: {path.resolve()}")
    if not args.no_copy_targets and not args.targets_root.exists():
        raise FileNotFoundError(f"Targets root does not exist: {args.targets_root.resolve()}")

    subjects = parse_csv_list(args.subjects)
    tasks = parse_csv_list(args.tasks)
    if tasks is not None:
        unknown = tasks.difference(TASK_PREFIXES)
        if unknown:
            raise ValueError(f"Unsupported task names: {sorted(unknown)}. Choices: {TASK_PREFIXES}")
    trials = list(iter_trials(args.source_root, subjects, tasks))
    if not trials:
        raise ValueError("No Anais trials matched the requested subject/task filters.")
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    models: dict[str, tuple[pin.Model, pin.Data]] = {}
    report_rows: list[dict[str, object]] = []
    for index, (subject, trial, task, trial_dir) in enumerate(trials, start=1):
        urdf_path = args.urdf_dir / f"{subject}_scaled.urdf"
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found for {subject}: {urdf_path}")
        if subject not in models:
            model = build_human_model(str(urdf_path), str(args.urdf_dir))[0]
            models[subject] = (model, model.createData())
        model, data = models[subject]
        q, kinetics = load_trial_inputs(trial_dir)
        mapping = infer_plate_mapping(
            model, data, q, kinetics,
            contact_threshold_n=args.contact_threshold,
            sample_stride=args.mapping_stride,
            min_confidence_ratio=args.min_mapping_confidence,
        )
        local, metadata = build_local_kinetics(
            model, data, q, kinetics, mapping, args.contact_threshold
        )
        validate_local_output(local, args.contact_threshold)
        if not args.no_copy_targets:
            # Do this even in dry-run mode: a reconstruction is useful for
            # training only if its input and 35-DoF target are frame-aligned.
            load_validated_targets(args.targets_root, subject, trial, len(local))
        if not args.dry_run:
            write_trial(
                args.output_root, args.targets_root, subject, trial, local,
                overwrite=args.overwrite,
                copy_targets=not args.no_copy_targets,
            )
        report = {
            "subject": subject,
            "trial": trial,
            "task": task,
            "n_frames": len(local),
            "left_plate": mapping.left_plate,
            "right_plate": mapping.right_plate,
            "score_plate1_left_plate2_right_m": f"{mapping.left_right_score_m:.6f}",
            "score_plate1_right_plate2_left_m": f"{mapping.right_left_score_m:.6f}",
            "mapping_confidence_ratio": f"{mapping.confidence_ratio:.3f}",
            **metadata,
        }
        report_rows.append(report)
        print(
            f"[{index:03d}/{len(trials):03d}] {subject}/{trial}: "
            f"left=plate{mapping.left_plate}, right=plate{mapping.right_plate}, "
            f"confidence={mapping.confidence_ratio:.2f}x, frames={len(local)}"
        )

    if not args.dry_run:
        report_path = args.output_root / "mapping_report.csv"
        with open(report_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(report_rows[0]))
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"\n[FINISH] Rebuilt {len(report_rows)} trials in {args.output_root}")
        print(f"[REPORT] {report_path}")
    else:
        print(f"\n[DRY RUN] Validated {len(report_rows)} trials; no files were written.")


if __name__ == "__main__":
    main(parse_arguments())
