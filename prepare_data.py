"""
convert_to_npy.py
=================
Converts joints_filtered.csv and kinetics_filtered.csv to .npy files
for three dataset folders: Anais/, HUMANOIDS/, Vinc/

Expected folder structure:
    <ROOT>/Anais/<subject>/<task>/joints_filtered.csv
    <ROOT>/Anais/<subject>/<task>/kinetics_filtered.csv
    (same for HUMANOIDS and Vinc)

Output per trial  →  <ROOT>/npy/<dataset>/<subject>/<task>/
    lower_body_joints.npy   : 12 DOFs — right then left (hip, knee, ankle)
    all_joints.npy          : all_joints.npy : freeflyer + all articulated DOFs
    kinetics.npy            : 12 channels — right plate then left plate

Usage:
    python convert_to_npy.py --root /path/to/data
    python convert_to_npy.py --root /path/to/data --dry-run
    python convert_to_npy.py --root /path/to/data --plot
    python convert_to_npy.py --root /path/to/data --plot --plot-out report.png
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Plate convention: which plate number maps to which foot
DATASETS = {
    "Anais":     {"left_plate": 1, "right_plate": 2},  # plate 1 = Left, plate 2 = Right
    "HUMANOIDS": {"left_plate": 2, "right_plate": 1},  # plate 1 = Right, plate 2 = Left
    "Vinc":      {"left_plate": 2, "right_plate": 1},  # plate 1 = Right, plate 2 = Left
}

# ── Lower body DOFs — desired order: Right then Left ─────────────────────────
LOWER_ANAIS_VINC = [
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",
]

LOWER_HUMANOIDS = [
    "right_hip_Z", "right_hip_X", "right_hip_Y",
    "right_knee_Z", "right_ankle_Z", "right_ankle_X",
    "left_hip_Z",  "left_hip_X",  "left_hip_Y",
    "left_knee_Z", "left_ankle_Z", "left_ankle_X",
]

# Human-readable canonical names (shared by both naming conventions)
LOWER_CANONICAL = [
    "R Hip Flex/Ext",   "R Hip Abd/Add",   "R Hip Int/Ext Rot",
    "R Knee Flex/Ext",  "R Ankle Flex/Ext", "R Ankle Abd/Add",
    "L Hip Flex/Ext",   "L Hip Abd/Add",   "L Hip Int/Ext Rot",
    "L Knee Flex/Ext",  "L Ankle Flex/Ext", "L Ankle Abd/Add",
]

# ── Upper body DOFs (for display / ordering reference) ───────────────────────
UPPER_ANAIS_VINC = [
    "Lumbar_flex_ext", "Lumbar_lateral_flex",
    "Lcalvicule_x",
    "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
    "Lelbow_flex_ext", "Lelbow_pron_supi",
    "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
    "rcalvicule_x",
    "Rshoulder_flex_ext", "Rshoulder_abd_add", "Rshoulder_int_ext_rot",
    "Relbow_flex_ext", "Relbow_pron_supi",
]

UPPER_HUMANOIDS = [
    "middle_lumbar_Z", "middle_lumbar_X",
    "left_clavicle_joint_X",
    "left_shoulder_Z", "left_shoulder_X", "left_shoulder_Y",
    "left_elbow_Z", "left_elbow_Y",
    "middle_cervical_Z", "middle_cervical_X", "middle_cervical_Y",
    "right_clavicle_joint_X",
    "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",
    "right_elbow_Z", "right_elbow_Y",
]

UPPER_CANONICAL = [
    "Lumbar Flex/Ext",     "Lumbar Lateral Flex",
    "L Clavicle X",
    "L Shoulder Flex/Ext", "L Shoulder Abd/Add",  "L Shoulder Int/Ext Rot",
    "L Elbow Flex/Ext",    "L Elbow Pron/Sup",
    "Cervical Flex/Ext",   "Cervical Lat Bend",   "Cervical Int/Ext Rot",
    "R Clavicle X",
    "R Shoulder Flex/Ext", "R Shoulder Abd/Add",  "R Shoulder Int/Ext Rot",
    "R Elbow Flex/Ext",    "R Elbow Pron/Sup",
]

# all_joints.npy column order: lower (right→left) then upper body
ALL_JOINTS_ANAIS_VINC = LOWER_ANAIS_VINC + UPPER_ANAIS_VINC   # 12 + 17 = 29 DOFs
ALL_JOINTS_HUMANOIDS  = LOWER_HUMANOIDS  + UPPER_HUMANOIDS     # 12 + 17 = 29 DOFs
ALL_JOINTS_CANONICAL  = LOWER_CANONICAL  + UPPER_CANONICAL     # 29 canonical names


# ── Freeflyer columns to exclude ─────────────────────────────────────────────

# FF_ANAIS_VINC = ["FF_X","FF_Y","FF_Z","FF_quatx","FF_quaty","FF_quatz","FF_quatw"]
# FF_HUMANOIDS  = ["root_joint","root_joint.1","root_joint.2",
#                  "root_joint.3","root_joint.4","root_joint.5","root_joint.6"]

FF_ANAIS_VINC = ["delta_x","delta_y","delta_z","delta_rx","delta_ry","delta_rz"]
FF_HUMANOIDS  =["delta_x","delta_y","delta_z","delta_rx","delta_ry","delta_rz"]
# ── Expected full headers (for validation) ───────────────────────────────────
EXPECTED_JOINTS_ANAIS_VINC = FF_ANAIS_VINC + [
    "Lhip_flex_ext","Lhip_abd_add","Lhip_int_ext_rot","Lknee_flex_ext",
    "Lankle_flex_ext","Lankle_abd_add","Lumbar_flex_ext","Lumbar_lateral_flex",
    "Lcalvicule_x","Lshoulder_flex_ext","Lshoulder_abd_add","Lshoulder_int_ext_rot",
    "Lelbow_flex_ext","Lelbow_pron_supi","Cervical_flex_ext","Cervical_lat_bend",
    "Cervical_int_ext_rot","rcalvicule_x","Rshoulder_flex_ext","Rshoulder_abd_add",
    "Rshoulder_int_ext_rot","Relbow_flex_ext","Relbow_pron_supi",
    "Rhip_flex_ext","Rhip_abd_add","Rhip_int_ext_rot",
    "Rknee_flex_ext","Rankle_flex_ext","Rankle_abd_add",
]

EXPECTED_JOINTS_HUMANOIDS = FF_HUMANOIDS + [
    "left_hip_Z","left_hip_X","left_hip_Y","left_knee_Z","left_ankle_Z","left_ankle_X",
    "middle_lumbar_Z","middle_lumbar_X","left_clavicle_joint_X",
    "left_shoulder_Z","left_shoulder_X","left_shoulder_Y",
    "left_elbow_Z","left_elbow_Y","middle_cervical_Z","middle_cervical_X","middle_cervical_Y",
    "right_clavicle_joint_X",
    "right_shoulder_Z","right_shoulder_X","right_shoulder_Y",
    "right_elbow_Z","right_elbow_Y",
    "right_hip_Z","right_hip_X","right_hip_Y","right_knee_Z","right_ankle_Z","right_ankle_X",
]

EXPECTED_KINETICS = ["Fx1","Fy1","Fz1","Mx1","My1","Mz1","Fx2","Fy2","Fz2","Mx2","My2","Mz2", "COPx1","COPy1","COPz1","COPx2","COPy2","COPz2"]
# EXPECTED_KINETICS = ["Fx1_glob","Fy1_glob","Fz1_glob","Mx1_glob","My1_glob","Mz1_glob",
#                      "Fx2_glob","Fy2_glob","Fz2_glob","Mx2_glob","My2_glob","Mz2_glob",
#                      "COPx1_glob","COPy1_glob","COPz1_glob","COPx2_glob","COPy2_glob","COPz2_glob"]
# EXPECTED_KINETICS = ["Fx1","Fy1","Fz1","Mx1","My1","Mz1",
#                      "Fx2","Fy2","Fz2","Mx2","My2","Mz2",
#                      "COPx1","COPy1","COPz1","COPx2","COPy2","COPz2"]

# 
SAMPLING_RATE_HZ = 100  # data acquisition frequency

KINETICS_CANONICAL = [
    "R Fx (N)", "R Fy (N)", "R Fz (N)", "R Mx (Nm)", "R My (Nm)", "R Mz (Nm)","R Copx (m)", "R Copy (m)", "R Copz (m)",
    "L Fx (N)", "L Fy (N)", "L Fz (N)", "L Mx (Nm)", "L My (Nm)", "L Mz (Nm)","L Copx (m)", "L Copy (m)", "L Copz (m)"
]

ALL_JOINTS_WITH_FF_ANAIS_VINC = FF_ANAIS_VINC + ALL_JOINTS_ANAIS_VINC
ALL_JOINTS_WITH_FF_HUMANOIDS = FF_HUMANOIDS + ALL_JOINTS_HUMANOIDS

# ── Physiological validity ranges ────────────────────────────────────────────
FORCE_MIN,  FORCE_MAX  = -5000.0,  5000.0   # N
MOMENT_MIN, MOMENT_MAX =  -1000.0,   1000.0   # Nm
ANGLE_MIN,  ANGLE_MAX  =    -100.0,     100.0   # rad (~± 229°)

SAMPLING_HZ = 100   # acquisition frequency — used to convert frames to duration

# ── Acquisition parameters ────────────────────────────────────────────────────
SAMPLING_RATE_HZ = 100   # Hz — all datasets recorded at 100 Hz

# ── Per-dataset display colors ────────────────────────────────────────────────
DS_COLORS = {"Anais": "#4e79a7", "HUMANOIDS": "#f28e2b", "Vinc": "#59a14f"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _frames_to_min(n_frames):
    """Convert a frame count to minutes at SAMPLING_RATE_HZ."""
    return n_frames / SAMPLING_RATE_HZ / 60.0


def _fmt_duration(n_frames):
    """Return a human-readable duration string, e.g. '12.3 min (1h 14.5s)'."""
    total_sec = n_frames / SAMPLING_RATE_HZ
    minutes   = total_sec / 60.0
    if total_sec < 60:
        return f"{total_sec:.1f} s"
    if total_sec < 3600:
        return f"{minutes:.2f} min ({total_sec:.0f} s)"
    h  = int(total_sec // 3600)
    m  = (total_sec % 3600) / 60.0
    return f"{h}h {m:.1f} min  ({minutes:.1f} min total)"


def _build_kinetics_order(dataset_name):
    """Return kinetics column names in the order: right plate → left plate."""
    cfg  = DATASETS[dataset_name]
    r, l = cfg["right_plate"], cfg["left_plate"]
    return [f"Fx{r}",f"Fy{r}",f"Fz{r}",f"Mx{r}",f"My{r}",f"Mz{r}",f"COPx{r}",f"COPy{r}",f"COPz{r}",
            f"Fx{l}",f"Fy{l}",f"Fz{l}",f"Mx{l}",f"My{l}",f"Mz{l}",f"COPx{l}",f"COPy{l}",f"COPz{l}",]

    # suffix = "_glob"   # 🔥 

    # return [
    #     f"Fx{r}{suffix}", f"Fy{r}{suffix}", f"Fz{r}{suffix}",
    #     f"Mx{r}{suffix}", f"My{r}{suffix}", f"Mz{r}{suffix}",
    #     f"COPx{r}{suffix}", f"COPy{r}{suffix}", f"COPz{r}{suffix}",
    #     f"Fx{l}{suffix}", f"Fy{l}{suffix}", f"Fz{l}{suffix}",
    #     f"Mx{l}{suffix}", f"My{l}{suffix}", f"Mz{l}{suffix}",
    #     f"COPx{l}{suffix}", f"COPy{l}{suffix}", f"COPz{l}{suffix}",
    # ]


def _check_range(arr, vmin, vmax, label, path):
    """Return a warning list if any value falls outside [vmin, vmax]."""
    if arr.size == 0:
        return []
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if mn < vmin or mx > vmax:
        print(label)
        print(f"  [RANGE]  {label}: [{mn:.2f}, {mx:.2f}] outside [{vmin}, {vmax}]  →  {path}")
        return [f"  [RANGE]  {label}: [{mn:.2f}, {mx:.2f}] outside [{vmin}, {vmax}]  →  {path}"]
    return []


def _check_nan(arr, label, path):
    """Return a warning list if NaN values are detected."""
    n = int(np.isnan(arr).sum())
    if n:
        return [f"  [NAN]    {label}: {n} NaN values detected  →  {path}"]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# TRIAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_trial(joints_path, kinetics_path, dataset_name, out_dir, dry_run=False):
    """
    Read one trial and produce three .npy files:
        lower_body_joints.npy  — 12 DOFs: right hip/knee/ankle, then left
        all_joints.npy         — 29 DOFs: lower + upper body (no freeflyer)
        kinetics.npy           — 18 channels: right plate then left plate

    Returns a metadata dict used for the summary report.
    """
    meta = {
        "dataset":          dataset_name,
        "trial":            str(out_dir),
        "subject":          out_dir.parent.name,
        "task":             out_dir.name,
        "warnings":         [],
        "samples_joints":   0,
        "samples_kinetics": 0,
        "n_lower":          0,
        "n_all":            0,
        "ok":               True,
    }

    # ── Read joints ──────────────────────────────────────────────────────────
    try:
        df_j = pd.read_csv(joints_path)
    except Exception as e:
        meta["warnings"].append(f"  [ERROR]  Could not read joints file: {e}")
        meta["ok"] = False
        return meta

    actual_cols  = list(df_j.columns)
    is_humanoids = (dataset_name == "HUMANOIDS")
    expected_j   = EXPECTED_JOINTS_HUMANOIDS  if is_humanoids else EXPECTED_JOINTS_ANAIS_VINC
    ff_cols      = FF_HUMANOIDS               if is_humanoids else FF_ANAIS_VINC
    lower_cols   = LOWER_HUMANOIDS            if is_humanoids else LOWER_ANAIS_VINC
    all_cols     = ALL_JOINTS_WITH_FF_HUMANOIDS       if is_humanoids else ALL_JOINTS_WITH_FF_ANAIS_VINC

    # Validate header
    missing_hdr = set(expected_j) - set(actual_cols)
    extra_hdr   = set(actual_cols) - set(expected_j)
    if missing_hdr:
        meta["warnings"].append(f"  [HEADER] Missing joints columns: {sorted(missing_hdr)}")
    if extra_hdr:
        meta["warnings"].append(f"  [HEADER] Unexpected joints columns: {sorted(extra_hdr)}")

    # Check that all required DOF columns actually exist before extracting
    missing_lower = [c for c in lower_cols if c not in actual_cols]
    missing_all   = [c for c in all_cols   if c not in actual_cols]
    if missing_lower:
        meta["warnings"].append(f"  [ERROR]  Lower body columns not found: {missing_lower}")
        meta["ok"] = False
        return meta
    if missing_all:
        meta["warnings"].append(f"  [ERROR]  All-joints columns not found: {missing_all}")
        meta["ok"] = False
        return meta

    # Extract arrays
    arr_lower = df_j[lower_cols].values.astype(np.float32)  # (T, 12)
    arr_all   = df_j[all_cols].values.astype(np.float32)    # (T, 29)

    meta["samples_joints"] = len(df_j)
    meta["n_lower"]        = arr_lower.shape[1]   # should always be 12
    meta["n_all"]          = arr_all.shape[1]     # should always be 29

    # Sanity checks on angle values
    meta["warnings"] += _check_nan(arr_lower, "lower_body angles", joints_path)
    meta["warnings"] += _check_nan(arr_all,   "all joints angles", joints_path)
    meta["warnings"] += _check_range(arr_lower, ANGLE_MIN, ANGLE_MAX, "lower_body angles (rad)", joints_path)
    meta["warnings"] += _check_range(arr_all,   ANGLE_MIN, ANGLE_MAX, "all joints angles (rad)", joints_path)
    # ── Read kinetics ────────────────────────────────────────────────────────
    try:
        df_k = pd.read_csv(kinetics_path)
    except Exception as e:
        meta["warnings"].append(f"  [ERROR]  Could not read kinetics file: {e}")
        meta["ok"] = False
        return meta

    k_cols    = list(df_k.columns)
    missing_k = set(EXPECTED_KINETICS) - set(k_cols)
    extra_k   = set(k_cols) - set(EXPECTED_KINETICS)
    if missing_k:
        meta["warnings"].append(f"  [HEADER] Missing kinetics columns: {sorted(missing_k)}")
    if extra_k:
        meta["warnings"].append(f"  [HEADER] Unexpected kinetics columns: {sorted(extra_k)}")

    kinetics_order = _build_kinetics_order(dataset_name)
    missing_ko = [c for c in kinetics_order if c not in k_cols]
    if missing_ko:
        meta["warnings"].append(f"  [ERROR]  Kinetics reorder columns not found: {missing_ko}")
        meta["ok"] = False
        return meta

    arr_k = df_k[kinetics_order].values.astype(np.float32)  # (T, 12)
    meta["samples_kinetics"] = len(df_k)

    # Sanity checks on force/moment values
    meta["warnings"] += _check_nan(arr_k, "kinetics", kinetics_path)
 
    meta["warnings"] += _check_range(arr_k[:, [0,1,2,9,10,11]],   FORCE_MIN,  FORCE_MAX,  "forces (N)",   kinetics_path)
    meta["warnings"] += _check_range(arr_k[:, [3,4,5,12,13,14]], MOMENT_MIN, MOMENT_MAX, "moments (Nm)", kinetics_path)

    # Check temporal consistency between joints and kinetics
    if meta["samples_joints"] != meta["samples_kinetics"]:
        meta["warnings"].append(
            f"  [SYNC]   Frame count mismatch: joints={meta['samples_joints']} "
            f"vs kinetics={meta['samples_kinetics']}"
        )

    # ── Save .npy files ──────────────────────────────────────────────────────
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        # np.save(out_dir / "lower_body_joints_.npy", arr_lower)  # (T, 12)
        np.save(out_dir / "all_joints.npy",        arr_all)    # (T, 29)
        np.save(out_dir / "kinetics_feet_corr.npy",           arr_k)     # (T, 12)

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY TRAVERSAL
# ─────────────────────────────────────────────────────────────────────────────

def find_trials(root: Path, dataset_name: str):
    """
    Scan <root>/<dataset>/<subject>/<task>/ for pairs of
    joints_filtered.csv + kinetics_pelvis_filtered.csv.
    Returns a list of (joints_path, kinetics_path, output_dir).
    """
    base = root / dataset_name
    if not base.exists():
        return []
    trials = []
    for subject_dir in sorted(base.iterdir()):
        if not subject_dir.is_dir():
            continue
        for task_dir in sorted(subject_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            j = task_dir / "joints_filtered.csv"
            k = task_dir / "kinetics_feet_corr.csv"
            if j.exists() and k.exists():
                out_dir = root / "npy" / dataset_name / subject_dir.name / task_dir.name
                trials.append((j, k, out_dir))
    return trials


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_meta):
    sep = "─" * 72
    print(f"\n{'═'*72}")
    print("  CONVERSION SUMMARY")
    print(f"{'═'*72}")

    total_sj = total_sk = total_ok = total_warn = total_err = 0
    by_dataset = defaultdict(list)
    for m in all_meta:
        by_dataset[m["dataset"]].append(m)

    for ds, mlist in sorted(by_dataset.items()):
        n_ok   = sum(1 for m in mlist if m["ok"])
        n_sj   = sum(m["samples_joints"]   for m in mlist)
        n_sk   = sum(m["samples_kinetics"] for m in mlist)
        n_warn = sum(len(m["warnings"])    for m in mlist)
        lower_set = set(m["n_lower"] for m in mlist if m["ok"])
        all_set   = set(m["n_all"]   for m in mlist if m["ok"])
        print(f"\n  [{ds}]")
        dur_min = n_sj / SAMPLING_HZ / 60
        print(f"      Trials processed    : {n_ok}/{len(mlist)}")
        print(f"      Frames (joints)     : {n_sj:,}  ({dur_min:.2f} min @ {SAMPLING_HZ} Hz)")
        print(f"      Frames (kinetics)   : {n_sk:,}")
        print(f"      lower_body_joints   : {lower_set} DOFs  (expected: {{12}})")
        print(f"      all_joints          : {all_set} DOFs  (expected: {{35}})")
        print(f"      Warnings            : {n_warn}")
        total_sj   += n_sj;  total_sk  += n_sk
        total_ok   += n_ok;  total_warn += n_warn
        total_err  += len(mlist) - n_ok

    print(f"\n{sep}")
    print(f"  TOTALS")
    print(f"      Trials OK           : {total_ok}")
    print(f"      Trials with errors  : {total_err}")
    total_dur = total_sj / SAMPLING_HZ / 60
    print(f"      Frames (joints)     : {total_sj:,}  →  {total_dur:.2f} min  ({total_dur*60:.0f} s @ {SAMPLING_HZ} Hz)")
    print(f"      Frames (kinetics)   : {total_sk:,}")
    print(f"      Warnings            : {total_warn}")

    # Detailed warnings
    warnings_all = [(m["trial"], w) for m in all_meta for w in m["warnings"]]
    if warnings_all:
        print(f"\n{sep}\n  WARNING / ERROR DETAILS\n{sep}")
        prev = None
        for trial, w in warnings_all:
            if trial != prev:
                print(f"\n  [{trial}]"); prev = trial
            print(w)
    else:
        print(f"\n  All clear — no warnings detected.")

    print(f"\n{'═'*72}\n")

    # Cross-dataset consistency
    n_lower_vals = set(m["n_lower"] for m in all_meta if m["ok"])
    n_all_vals   = set(m["n_all"]   for m in all_meta if m["ok"])
    if n_lower_vals != {12}:
        print(f"  [WARN] Inconsistent lower body DOF counts across datasets: {n_lower_vals}")
    else:
        print("  [OK]  lower_body_joints: 12 DOFs across all datasets.")
    if n_all_vals != {35}:
        print(f"  [WARN] Inconsistent all_joints DOF counts across datasets: {n_all_vals}")
    else:
        print("  [OK]  all_joints: 35 DOFs across all datasets.")
    unit_ok = all(not any("RANGE" in w for w in m["warnings"]) for m in all_meta)
    if unit_ok:
        print("  [OK]  Value ranges consistent with expected units (N, Nm, rad).")
    else:
        print("  [WARN] Some files have out-of-range values — check units.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHICAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(all_meta, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import ListedColormap

    by_dataset = defaultdict(list)
    for m in all_meta:
        by_dataset[m["dataset"]].append(m)

    # Color palette
    C_OK   = "#2ecc71"
    C_WARN = "#f39c12"
    C_ERR  = "#e74c3c"
    C_BG   = "#fafafa"
    C_TEXT = "#2c3e50"
    C_HEAD = "#34495e"
    C_RIGHT = "#d6eaf8"   # light blue  → right side
    C_LEFT  = "#d5f5e3"   # light green → left side
    REGION_COLORS = {
        "Lumbar":   "#fdebd0",
        "Cervical": "#e8daef",
        "L ":       C_LEFT,
        "R ":       C_RIGHT,
    }

    def region_color(canon):
        for k, v in REGION_COLORS.items():
            if canon.startswith(k):
                return v
        return "white"

    fig = plt.figure(figsize=(22, 30), facecolor=C_BG)
    fig.suptitle("CSV → NPY Conversion Report", fontsize=18, fontweight="bold",
                 color=C_TEXT, y=0.995)

    outer = gridspec.GridSpec(5, 1, figure=fig, hspace=0.48,
                              top=0.97, bottom=0.03, left=0.04, right=0.97)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 1 — Global statistics
    # ══════════════════════════════════════════════════════════════════════════
    ax_stats = fig.add_subplot(outer[0])
    ax_stats.axis("off")
    ax_stats.set_title("① Overview", fontsize=13, fontweight="bold",
                        color=C_HEAD, loc="left", pad=6)

    inner_stats = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.35)
    ds_names = sorted(by_dataset.keys())

    # Bar chart: frame count + duration per dataset
    ax_frames = fig.add_subplot(inner_stats[0])
    ax_frames.set_facecolor(C_BG)
    frames_j  = [sum(m["samples_joints"]   for m in by_dataset[d]) for d in ds_names]
    frames_k  = [sum(m["samples_kinetics"] for m in by_dataset[d]) for d in ds_names]
    dur_min_j = [f / SAMPLING_HZ / 60 for f in frames_j]
    x = np.arange(len(ds_names)); w = 0.35
    bars1 = ax_frames.bar(x - w/2, frames_j, w, label="Joints",
                           color=[DS_COLORS.get(d,"#888") for d in ds_names], alpha=0.85)
    bars2 = ax_frames.bar(x + w/2, frames_k, w, label="Kinetics",
                           color=[DS_COLORS.get(d,"#888") for d in ds_names], alpha=0.45, hatch="//")
    ax_frames.set_xticks(x); ax_frames.set_xticklabels(ds_names, fontsize=10)
    ax_frames.set_ylabel("Frames", fontsize=9)
    ax_frames.set_title(f"Frames & duration per dataset  ({SAMPLING_HZ} Hz)", fontsize=10, fontweight="bold")
    ax_frames.legend(fontsize=8)
    ax_frames.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{int(v):,}"))
    for bar, dur in zip(bars1, dur_min_j):
        h = bar.get_height()
        if h > 0:
            ax_frames.text(bar.get_x()+bar.get_width()/2, h*1.01,
                           f"{int(h):,}\n({dur:.1f} min)",
                           ha="center", va="bottom", fontsize=7, linespacing=1.3)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax_frames.text(bar.get_x()+bar.get_width()/2, h*1.01, f"{int(h):,}",
                           ha="center", va="bottom", fontsize=7.5)
    ax_frames.spines[["top","right"]].set_visible(False)

    # Bar chart: trial status per dataset
    ax_trials = fig.add_subplot(inner_stats[1])
    ax_trials.set_facecolor(C_BG)
    n_ok_ds   = [sum(1 for m in by_dataset[d] if m["ok"] and not m["warnings"]) for d in ds_names]
    n_warn_ds = [sum(1 for m in by_dataset[d] if m["ok"] and m["warnings"])     for d in ds_names]
    n_err_ds  = [sum(1 for m in by_dataset[d] if not m["ok"])                   for d in ds_names]
    ax_trials.bar(x, n_ok_ds,   0.55, label="OK",    color=C_OK,   alpha=0.85)
    ax_trials.bar(x, n_warn_ds, 0.55, bottom=n_ok_ds,
                  label="Warning", color=C_WARN, alpha=0.85)
    bot2 = [a+b for a,b in zip(n_ok_ds, n_warn_ds)]
    ax_trials.bar(x, n_err_ds,  0.55, bottom=bot2, label="Error", color=C_ERR, alpha=0.85)
    ax_trials.set_xticks(x); ax_trials.set_xticklabels(ds_names, fontsize=10)
    ax_trials.set_ylabel("Trials", fontsize=9)
    ax_trials.set_title("Trial status per dataset", fontsize=10, fontweight="bold")
    ax_trials.legend(fontsize=8)
    ax_trials.spines[["top","right"]].set_visible(False)

    # Summary table
    ax_recap = fig.add_subplot(inner_stats[2])
    ax_recap.axis("off")
    total_ok_c  = sum(1 for m in all_meta if m["ok"] and not m["warnings"])
    total_warn_c= sum(1 for m in all_meta if m["ok"] and m["warnings"])
    total_err_c = sum(1 for m in all_meta if not m["ok"])
    total_sj    = sum(m["samples_joints"]   for m in all_meta)
    total_sk    = sum(m["samples_kinetics"] for m in all_meta)
    total_dur   = total_sj / SAMPLING_HZ / 60
    recap_rows = [
        ["Total trials",       str(len(all_meta))],
        ["OK",                 str(total_ok_c)],
        ["Warnings",           str(total_warn_c)],
        ["Errors",             str(total_err_c)],
        ["Frames (joints)",    f"{total_sj:,}"],
        ["Frames (kinetics)",  f"{total_sk:,}"],
        ["Total duration",     f"{total_dur:.2f} min"],
        ["Sampling rate",      f"{SAMPLING_HZ} Hz"],
        ["lower_body_joints",  "12 DOFs / trial"],
        ["all_joints",         "35 DOFs / trial"],
        ["kinetics",           "12 ch  / trial"],
    ]
    tbl = ax_recap.table(cellText=recap_rows, colLabels=["Metric", "Value"],
                          cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    STATUS_BG = {"OK": "#d5f5e3", "Warnings": "#fdebd0", "Errors": "#fadbd8"}
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor(C_HEAD); cell.set_text_props(color="white", fontweight="bold")
        else:
            key = recap_rows[r-1][0]
            cell.set_facecolor(STATUS_BG.get(key, "#eaf0fb" if r%2==0 else "white"))
    ax_recap.set_title("Global recap", fontsize=10, fontweight="bold", pad=4)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 2 — Per-trial status heatmap
    # ══════════════════════════════════════════════════════════════════════════
    ax_heat = fig.add_subplot(outer[1])
    ax_heat.axis("off")
    ax_heat.set_title("② Per-trial status  (green = OK · orange = warning · red = error)",
                       fontsize=13, fontweight="bold", color=C_HEAD, loc="left", pad=6)

    inner_heat = gridspec.GridSpecFromSubplotSpec(1, len(ds_names), subplot_spec=outer[1], wspace=0.06)

    for di, ds in enumerate(ds_names):
        ax_h = fig.add_subplot(inner_heat[di])
        ax_h.set_facecolor(C_BG)
        subj_map = defaultdict(list)
        for m in by_dataset[ds]:
            subj_map[m["subject"]].append(m)
        subjects  = sorted(subj_map.keys())
        max_tasks = max(len(v) for v in subj_map.values()) if subjects else 1
        grid      = np.full((len(subjects), max_tasks), -1.0)
        task_lbls = {}
        for si, subj in enumerate(subjects):
            for ti, m in enumerate(sorted(subj_map[subj], key=lambda x: x["task"])):
                grid[si, ti]      = 2.0 if not m["ok"] else (1.0 if m["warnings"] else 0.0)
                task_lbls[(si,ti)] = m["task"]

        cmap   = ListedColormap([C_OK, C_WARN, C_ERR])
        masked = np.ma.masked_where(grid < 0, grid)
        ax_h.imshow(masked, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")

        for si in range(len(subjects)):
            for ti in range(max_tasks):
                if grid[si, ti] >= 0:
                    lbl    = task_lbls.get((si,ti), "")
                    n_w    = len(sorted(subj_map[subjects[si]], key=lambda x: x["task"])[ti]["warnings"])
                    txt    = lbl if n_w == 0 else f"{lbl}\n({n_w}w)"
                    ax_h.text(ti, si, txt, ha="center", va="center",
                              fontsize=6.5, color="white", fontweight="bold")

        ax_h.set_xticks([]); ax_h.set_yticks(range(len(subjects)))
        ax_h.set_yticklabels(subjects, fontsize=8)
        ax_h.set_title(ds, fontsize=11, fontweight="bold", color=DS_COLORS.get(ds, C_TEXT), pad=4)
        ax_h.set_xlabel("← tasks →", fontsize=7.5, labelpad=3)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 3 — lower_body_joints.npy DOF table
    # ══════════════════════════════════════════════════════════════════════════
    ax_lower = fig.add_subplot(outer[2])
    ax_lower.axis("off")
    ax_lower.set_title(
        "③  lower_body_joints.npy  —  12 DOFs  (order: Right → Left)",
        fontsize=13, fontweight="bold", color=C_HEAD, loc="left", pad=6)

    rows_lower = [[str(i), c, av, hum, "R" if c.startswith("R ") else "L"]
                  for i,(c,av,hum) in enumerate(zip(LOWER_CANONICAL, LOWER_ANAIS_VINC, LOWER_HUMANOIDS))]
    tbl2 = ax_lower.table(
        cellText=[[r[0],r[1],r[2],r[3],r[4]] for r in rows_lower],
        colLabels=["#", "Canonical name", "Anais / Vinc  (CSV col)", "HUMANOIDS  (CSV col)", "Side"],
        cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl2.auto_set_font_size(False); tbl2.set_fontsize(9)
    for (r,c), cell in tbl2.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor(C_HEAD); cell.set_text_props(color="white", fontweight="bold")
        else:
            side = rows_lower[r-1][4]
            cell.set_facecolor(C_RIGHT if side == "R" else C_LEFT)
    # Visual separator between right (rows 1-6) and left (rows 7-12)
    for c in range(5):
        tbl2.get_celld()[(7, c)].set_edgecolor("#2980b9")
        tbl2.get_celld()[(7, c)].set_linewidth(2.5)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 4 — all_joints.npy DOF table (29 DOFs = lower + upper)
    # ══════════════════════════════════════════════════════════════════════════
    ax_all = fig.add_subplot(outer[3])
    ax_all.axis("off")
    ax_all.set_title(
        "④  all_joints.npy  —  35 DOFs  (lower body [0-11] + upper body [12-28], 7 freeflyer)",
        fontsize=13, fontweight="bold", color=C_HEAD, loc="left", pad=6)

    rows_all = [[str(i), c, av, hum]
                for i,(c,av,hum) in enumerate(zip(ALL_JOINTS_CANONICAL,
                                                   ALL_JOINTS_ANAIS_VINC,
                                                   ALL_JOINTS_HUMANOIDS))]
    tbl3 = ax_all.table(
        cellText=rows_all,
        colLabels=["#", "Canonical name", "Anais / Vinc  (CSV col)", "HUMANOIDS  (CSV col)"],
        cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl3.auto_set_font_size(False); tbl3.set_fontsize(8.5)
    for (r,c), cell in tbl3.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor(C_HEAD); cell.set_text_props(color="white", fontweight="bold")
        elif r <= 12:
            # Lower body (indices 0-11): same color coding as panel 3
            side = rows_all[r-1][1]
            cell.set_facecolor(C_RIGHT if side.startswith("R ") else C_LEFT)
        else:
            # Upper body (indices 12-28): region-based coloring
            cell.set_facecolor(region_color(rows_all[r-1][1]))
    # Visual separator between lower (rows 1-12) and upper body (rows 13-29)
    for c in range(4):
        tbl3.get_celld()[(13, c)].set_edgecolor("#e74c3c")
        tbl3.get_celld()[(13, c)].set_linewidth(2.5)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL 5 — kinetics.npy + conventions table
    # ══════════════════════════════════════════════════════════════════════════
    ax_kine = fig.add_subplot(outer[4])
    ax_kine.axis("off")
    ax_kine.set_title(
        "⑤  kinetics.npy  —  12 channels  (order: Right plate → Left plate)",
        fontsize=13, fontweight="bold", color=C_HEAD, loc="left", pad=6)

    inner_kine = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4], wspace=0.12)

    # Kinetics channel table
    ax_kt = fig.add_subplot(inner_kine[0])
    ax_kt.axis("off")
    rows_k = []
    for i, canon in enumerate(KINETICS_CANONICAL):
        side = "R" if i < 6 else "L"
        src_anais    = "2" if i < 6 else "1"   # Anais: right=plate2, left=plate1
        src_hum_vinc = "1" if i < 6 else "2"   # HUM/Vinc: right=plate1, left=plate2
        rows_k.append([str(i), canon, f"plate {src_anais}", f"plate {src_hum_vinc}", side])
    tbl4 = ax_kt.table(
        cellText=[[r[0],r[1],r[2],r[3],r[4]] for r in rows_k],
        colLabels=["#", "Channel (npy output)", "Anais source", "HUM/Vinc source", "Side"],
        cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl4.auto_set_font_size(False); tbl4.set_fontsize(9)
    for (r,c), cell in tbl4.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor(C_HEAD); cell.set_text_props(color="white", fontweight="bold")
        else:
            side = rows_k[r-1][4]
            cell.set_facecolor(C_RIGHT if side == "R" else C_LEFT)
    for c in range(5):
        tbl4.get_celld()[(7, c)].set_edgecolor("#2980b9")
        tbl4.get_celld()[(7, c)].set_linewidth(2.5)

    # Conventions & units table
    ax_conv = fig.add_subplot(inner_kine[1])
    ax_conv.axis("off")
    ax_conv.set_title("Plate conventions & validity ranges", fontsize=10, fontweight="bold", pad=4)

    conv_rows = [["Dataset", "Plate 1", "Plate 2"],
                 ["Anais",     "Left  ←", "Right →"],
                 ["HUMANOIDS", "Right →", "Left  ←"],
                 ["Vinc",      "Right →", "Left  ←"]]
    tbl5 = ax_conv.table(cellText=conv_rows[1:], colLabels=conv_rows[0],
                          cellLoc="center", loc="upper center", bbox=[0, 0.5, 1, 0.45])
    tbl5.auto_set_font_size(False); tbl5.set_fontsize(9.5)
    DS_BG = {"Anais":"#eaf3fb","HUMANOIDS":"#fef5e7","Vinc":"#eafaf1"}
    for (r,c), cell in tbl5.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor(C_HEAD); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(DS_BG.get(conv_rows[1:][r-1][0], "white"))

    unit_rows = [["Signal",      "Unit", "Valid range"],
                 ["Fx, Fy, Fz", "N",   f"[{FORCE_MIN:.0f}, {FORCE_MAX:.0f}]"],
                 ["Mx, My, Mz", "Nm",  f"[{MOMENT_MIN:.0f}, {MOMENT_MAX:.0f}]"],
                 ["Angles",     "rad", f"[{ANGLE_MIN:.1f}, {ANGLE_MAX:.1f}]"]]
    tbl6 = ax_conv.table(cellText=unit_rows[1:], colLabels=unit_rows[0],
                          cellLoc="center", loc="lower center", bbox=[0, 0.0, 1, 0.42])
    tbl6.auto_set_font_size(False); tbl6.set_fontsize(9.5)
    for (r,c), cell in tbl6.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor(C_HEAD); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8f9fa")

    # Legend
    legend_patches = [
        mpatches.Patch(color=C_OK,    label="OK — no warnings"),
        mpatches.Patch(color=C_WARN,  label="Warning (data issue)"),
        mpatches.Patch(color=C_ERR,   label="Error — invalid file"),
        mpatches.Patch(color=C_RIGHT, label="Right side"),
        mpatches.Patch(color=C_LEFT,  label="Left side"),
        mpatches.Patch(color="#fdebd0", label="Lumbar"),
        mpatches.Patch(color="#e8daef", label="Cervical"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=7,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.005))

    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"\n  [PLOT] Report saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert biomechanics CSV files to NPY")
    parser.add_argument("--root",     required=True, help="Root folder containing Anais/, HUMANOIDS/, Vinc/")
    parser.add_argument("--dry-run",  action="store_true", help="Analyse only, do not write .npy files")
    parser.add_argument("--plot",     action="store_true", help="Generate a graphical summary PNG report")
    parser.add_argument("--plot-out", default=None,        help="Path for the PNG report (default: <root>/synthesis_report.png)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[ERROR] Root folder not found: {root}"); sys.exit(1)

    print(f"\n{'═'*72}")
    print("  CSV → NPY CONVERSION")
    print(f"  Root  : {root}")
    print(f"  Mode  : {'DRY-RUN (no files written)' if args.dry_run else 'WRITE ENABLED'}")
    print(f"{'═'*72}\n")

    all_meta = []
    for ds in DATASETS:
        trials = find_trials(root, ds)
        if not trials:
            print(f"  [WARN] No trials found for {ds} in {root / ds}"); continue
        print(f"  [{ds}]  {len(trials)} trial(s) found")
        for joints_path, kinetics_path, out_dir in trials:
            rel = joints_path.parent.relative_to(root)
            print(f"       → {rel} ", end="", flush=True)
            meta = process_trial(joints_path, kinetics_path, ds, out_dir, dry_run=args.dry_run)
            all_meta.append(meta)
            status = "OK" if meta["ok"] and not meta["warnings"] else ("WARN" if meta["ok"] else "ERROR")
            dur = _fmt_duration(meta["samples_joints"])
            print(f"[{status}]  ({meta['samples_joints']} frames — {dur})")

    if not all_meta:
        print("  No trials processed. Check your folder structure."); sys.exit(1)

    print_summary(all_meta)

    if not args.dry_run:
        print(f"  NPY files saved under : {root / 'npy'}/")
        print("  Structure : npy/<dataset>/<subject>/<task>/{lower_body_joints,all_joints,kinetics}.npy\n")

    if args.plot:
        out_png = Path(args.plot_out) if args.plot_out else root / "synthesis_report.png"
        plot_summary(all_meta, out_png)


if __name__ == "__main__":
    main()