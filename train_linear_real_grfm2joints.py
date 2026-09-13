"""Entraine une Ridge GRFM -> 29 angles sur les squats reels de Vinc.

Evaluation leave-one-subject-out : le sujet donne par ``--test-subject`` est
totalement exclu des scalers et de l'ajustement, puis utilise uniquement pour
le test. Les GRFM sont celles du repere world (``kinetics_glob.npy``).
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train_linear_christine_npz import JOINT_NAMES, correlation_columns


# Dans les fichiers reels a 18 canaux: Mz droit=5, Mz gauche=14.
MZ_INDICES = (5, 14)


SQUAT_TRIALS = {
    "Vincent": "Trial112",
    "Jovana": "Trial111",
    "Christine": "Trial110",
    "Jeremy": "Trial111",
    "Maria": "Trial114",
    "Serge": "Trial111",
    "Subject1": "Trial111",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("processed_data_feet_Vinc"))
    parser.add_argument("--test-subject", choices=list(SQUAT_TRIALS), default="Christine")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Par defaut: results_linear_real_<sujet>.")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--ignore-mz", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--force-file", choices=["kinetics_glob.npy", "kinetics_feet.npy"],
                        default="kinetics_glob.npy")
    return parser.parse_args()


def squat_paths(root):
    paths = {}
    for subject, trial in SQUAT_TRIALS.items():
        folder = root / subject / trial
        required = [folder / "kinetics_glob.npy", folder / "all_joints.npy"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Fichiers manquants pour {subject}: {missing}")
        paths[subject] = folder
    return paths


def load_pair(folder, force_file):
    x = np.load(folder / force_file).astype(np.float64)
    joints = np.load(folder / "all_joints.npy").astype(np.float64)
    y = joints[:, 6:35]
    if x.shape != (len(x), 18) or y.shape != (len(y), 29):
        raise ValueError(f"Dimensions invalides dans {folder}: X={x.shape}, Y={y.shape}")
    if len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Donnees non alignees ou non finies dans {folder}")
    return x, y


def training_stats(folders, force_file):
    x = np.concatenate([load_pair(folder, force_file)[0] for folder in folders])
    y = np.concatenate([load_pair(folder, force_file)[1] for folder in folders])
    xm, xs = x.mean(0), x.std(0)
    ym, ys = y.mean(0), y.std(0)
    xs = np.maximum(xs, 1e-6)
    ys = np.maximum(ys, 1e-6)
    return xm, xs, ym, ys, len(x)


def fit(folders, force_file, xm, xs, ym, ys, ridge, ignore_mz):
    xtx = np.zeros((18, 18), dtype=np.float64)
    xty = np.zeros((18, 29), dtype=np.float64)
    for folder in folders:
        x, y = load_pair(folder, force_file)
        x = (x - xm) / xs
        y = (y - ym) / ys
        if ignore_mz:
            x[:, MZ_INDICES] = 0.0
        xtx += x.T @ x
        xty += x.T @ y
    return np.linalg.solve(xtx + ridge * np.eye(18), xty)


def evaluate(folder, force_file, weights, xm, xs, ym, ys, ignore_mz):
    x, reference = load_pair(folder, force_file)
    x = (x - xm) / xs
    if ignore_mz:
        x[:, MZ_INDICES] = 0.0
    prediction = (x @ weights) * ys + ym
    error = prediction - reference
    rmse_deg = np.degrees(np.sqrt(np.mean(error ** 2, axis=0)))
    cc = correlation_columns(reference, prediction)
    return reference, prediction, rmse_deg, cc


def print_metrics(subject, rmse, cc):
    print(f"\nTEST SUR LE SUJET JAMAIS VU : {subject}")
    print(f"{'DOF':29s} {'RMSE (deg)':>12s} {'CC':>9s}")
    print("-" * 53)
    for name, error, corr in zip(JOINT_NAMES, rmse, cc):
        print(f"{name:29s} {error:12.3f} {corr:9.3f}")
    print("-" * 53)
    global_rmse = np.sqrt(np.mean(rmse ** 2))
    print(f"RMSE moyenne: {np.mean(rmse):.3f} deg | "
          f"RMSE global: {global_rmse:.3f} deg | "
          f"CC median: {np.nanmedian(cc):.3f} | CC moyen: {np.nanmean(cc):.3f}")


def save_metrics(path, rmse, cc):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dof_index", "dof", "rmse_deg", "cc"])
        for index, (name, error, corr) in enumerate(zip(JOINT_NAMES, rmse, cc)):
            writer.writerow([index, name, error, corr])


def plot_metrics(path, subject, rmse, cc):
    positions = np.arange(29)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    axes[0].bar(positions, rmse, color="#4e79a7")
    axes[0].set_ylabel("RMSE (degres)")
    axes[0].set_title(f"Modele lineaire reel - sujet test: {subject}")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(positions, cc, color="#59a14f")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Coefficient de correlation (CC)")
    axes[1].set_xticks(positions, JOINT_NAMES, rotation=65, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    paths = squat_paths(args.data_root)
    test_folder = paths[args.test_subject]
    train_subjects = [subject for subject in SQUAT_TRIALS if subject != args.test_subject]
    train_folders = [paths[subject] for subject in train_subjects]
    output_dir = args.output_dir or Path(f"results_linear_real_{args.test_subject.lower()}")

    print(f"Sujets train ({len(train_subjects)}): {', '.join(train_subjects)}")
    print(f"Sujet test: {args.test_subject} ({SQUAT_TRIALS[args.test_subject]})")
    print(f"GRFM: {args.force_file} | ignore_mz={args.ignore_mz}")

    xm, xs, ym, ys, n_frames = training_stats(train_folders, args.force_file)
    print(f"Entrainement sur {n_frames:,} frames de squat reel...")
    weights = fit(train_folders, args.force_file, xm, xs, ym, ys,
                  args.ridge, args.ignore_mz)
    reference, prediction, rmse, cc = evaluate(
        test_folder, args.force_file, weights, xm, xs, ym, ys, args.ignore_mz)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "linear_model.npz", weights=weights, x_mean=xm,
             x_std=xs, y_mean=ym, y_std=ys, ignore_mz=args.ignore_mz,
             test_subject=args.test_subject, force_file=args.force_file,
             train_subjects=np.asarray(train_subjects))
    np.save(output_dir / "prediction.npy", prediction.astype(np.float32))
    np.save(output_dir / "reference.npy", reference.astype(np.float32))
    save_metrics(output_dir / "metrics_per_dof.csv", rmse, cc)
    plot_metrics(output_dir / "metrics_per_dof.png", args.test_subject, rmse, cc)
    print_metrics(args.test_subject, rmse, cc)
    print(f"\nResultats enregistres dans: {output_dir}")


if __name__ == "__main__":
    main()
