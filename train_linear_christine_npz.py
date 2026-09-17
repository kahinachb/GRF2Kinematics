"""Baseline Ridge sur les NPZ corriges de ``DATA/Christine_synthetic``.

Entrees (12): [F_left_world, M_left_foot_world,
               F_right_world, M_right_foot_world].
La meme regression est evaluee sur une variante synthetique tenue a l'ecart et
sur les champs ``reference_*`` correspondants contenus dans le NPZ.
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


JOINT_NAMES = [
    "Rhip_flex_ext", "Rhip_abd_add", "Rhip_int_ext_rot",
    "Rknee_flex_ext", "Rankle_flex_ext", "Rankle_abd_add",
    "Lhip_flex_ext", "Lhip_abd_add", "Lhip_int_ext_rot",
    "Lknee_flex_ext", "Lankle_flex_ext", "Lankle_abd_add",
    "Lumbar_flex_ext", "Lumbar_lateral_flex", "Lcalvicule_x",
    "Lshoulder_flex_ext", "Lshoulder_abd_add", "Lshoulder_int_ext_rot",
    "Lelbow_flex_ext", "Lelbow_pron_supi",
    "Cervical_flex_ext", "Cervical_lat_bend", "Cervical_int_ext_rot",
    "Rcalvicule_x", "Rshoulder_flex_ext", "Rshoulder_abd_add",
    "Rshoulder_int_ext_rot", "Relbow_flex_ext", "Relbow_pron_supi",
]


def correlation_columns(reference, prediction):
    reference = reference - reference.mean(axis=0)
    prediction = prediction - prediction.mean(axis=0)
    numerator = np.sum(reference * prediction, axis=0)
    denominator = np.sqrt(np.sum(reference ** 2, axis=0) *
                          np.sum(prediction ** 2, axis=0))
    return np.divide(numerator, denominator, out=np.full(29, np.nan),
                     where=denominator > 1e-12)


INPUT_KEYS = (
    "F_left_world", "M_left_foot_world",
    "F_right_world", "M_right_foot_world",
)
REFERENCE_INPUT_KEYS = (
    "reference_F_left_world", "reference_M_left_foot_world",
    "reference_F_right_world", "reference_M_right_foot_world",
)
MZ_INDICES = (5, 11)
TARGET_UNITS = {"q": "deg", "dq": "deg/s", "ddq": "deg/s2"}
# Ordre articulaire des NPZ: [jambe G (6), haut du corps (17), jambe D (6)].
# Ordre canonique du modele et de JOINT_NAMES: [jambe D, jambe G, haut du corps].
NPZ_TO_CANONICAL = np.r_[23:29, 0:6, 6:23]


def discover_variant_files(root):
    """Supporte l'ancien format plat et le format reference.npz + variants/."""
    nested = sorted((root / "variants").glob("*.npz"))
    if nested:
        return nested
    return [path for path in sorted(root.glob("*.npz")) if path.name != "reference.npz"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("DATA/Christine_synthetic"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_linear_christine_npz"))
    parser.add_argument("--target", choices=["q", "dq", "ddq"], default="q")
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--ignore-mz", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def target_29(array, target):
    start = 7 if target == "q" else 6
    result = np.asarray(array[:, start:start + 29], dtype=np.float64)
    if result.shape[1] != 29:
        raise ValueError(f"Cible {target} incompatible: {array.shape}")
    return result[:, NPZ_TO_CANONICAL]


def load_pair(path, target, reference=False):
    path = Path(path)
    reference_path = path.parent.parent / "reference.npz"
    # Nouveau format: la reference est centralisee hors des variantes.
    if reference and reference_path.is_file():
        with np.load(reference_path) as data:
            left = np.asarray(data["measured_grfm_left_world"], dtype=np.float64)
            right = np.asarray(data["measured_grfm_right_world"], dtype=np.float64)
            x = np.concatenate([left[:, :3], left[:, 3:6],
                                right[:, :3], right[:, 3:6]], axis=1)
            y = target_29(np.asarray(data[f"reference_{target}"]), target)
    else:
        with np.load(path) as data:
            input_keys = REFERENCE_INPUT_KEYS if reference else INPUT_KEYS
            missing = [key for key in input_keys if key not in data]
            target_key = f"reference_{target}" if reference else target
            if target_key not in data:
                missing.append(target_key)
            if missing:
                raise KeyError(f"Champs manquants dans {path}: {missing}")
            x = np.concatenate([np.asarray(data[key], dtype=np.float64)
                                for key in input_keys], axis=1)
            y = target_29(np.asarray(data[target_key]), target)
    if x.shape != (len(x), 12) or len(x) != len(y):
        raise ValueError(f"Dimensions invalides dans {path}: X={x.shape}, Y={y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Valeurs non finies dans {path}")
    return x, y


def split_files(files, seed, train_ratio, val_ratio):
    files = list(files)
    random.Random(seed).shuffle(files)
    n_train = int(len(files) * train_ratio)
    n_val = int(len(files) * val_ratio)
    return files[:n_train], files[n_train:n_train + n_val], files[n_train + n_val:]


def stats(paths, target):
    n = 0
    sx = np.zeros(12); sx2 = np.zeros(12)
    sy = np.zeros(29); sy2 = np.zeros(29)
    for path in paths:
        x, y = load_pair(path, target)
        n += len(x)
        sx += x.sum(0); sx2 += np.square(x).sum(0)
        sy += y.sum(0); sy2 += np.square(y).sum(0)
    xm, ym = sx / n, sy / n
    xs = np.sqrt(np.maximum(sx2 / n - xm ** 2, 1e-12))
    ys = np.sqrt(np.maximum(sy2 / n - ym ** 2, 1e-12))
    return xm, xs, ym, ys, n


def fit(paths, target, xm, xs, ym, ys, ridge, ignore_mz):
    xtx = np.zeros((12, 12)); xty = np.zeros((12, 29))
    for path in paths:
        x, y = load_pair(path, target)
        x = (x - xm) / xs
        y = (y - ym) / ys
        if ignore_mz:
            x[:, MZ_INDICES] = 0.0
        xtx += x.T @ x
        xty += x.T @ y
    return np.linalg.solve(xtx + ridge * np.eye(12), xty)


def evaluate(path, target, weights, xm, xs, ym, ys, ignore_mz, reference=False):
    x, y = load_pair(path, target, reference=reference)
    x = (x - xm) / xs
    if ignore_mz:
        x[:, MZ_INDICES] = 0.0
    prediction = (x @ weights) * ys + ym
    rmse = np.degrees(np.sqrt(np.mean((prediction - y) ** 2, axis=0)))
    return y, prediction, rmse, correlation_columns(y, prediction)


def print_metrics(title, rmse, cc, unit):
    print(f"\n{title}")
    print(f"{'DOF':29s} {('RMSE (' + unit + ')'):>15s} {'CC':>9s}")
    print("-" * 56)
    for name, error, corr in zip(JOINT_NAMES, rmse, cc):
        print(f"{name:29s} {error:15.3f} {corr:9.3f}")
    print("-" * 56)
    print(f"RMSE moyenne: {np.mean(rmse):.3f} {unit} | "
          f"RMSE global: {np.sqrt(np.mean(rmse ** 2)):.3f} {unit} | "
          f"CC median: {np.nanmedian(cc):.3f} | CC moyen: {np.nanmean(cc):.3f}")


def save_metrics(path, synth, real):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dof_index", "dof", "synth_rmse", "synth_cc",
                         "reference_rmse", "reference_cc"])
        for i, name in enumerate(JOINT_NAMES):
            writer.writerow([i, name, synth[0][i], synth[1][i], real[0][i], real[1][i]])


def plot_curves(path, y, prediction, rmse, cc, title, unit, dt=0.01):
    time = np.arange(len(y)) * dt
    y_plot, pred_plot = np.degrees(y), np.degrees(prediction)
    fig, axes = plt.subplots(6, 5, figsize=(22, 19), sharex=True)
    axes = axes.ravel()
    for i, name in enumerate(JOINT_NAMES):
        axes[i].plot(time, y_plot[:, i], color="black", lw=1, label="Reference")
        axes[i].plot(time, pred_plot[:, i], color="#d62728", lw=.9,
                     alpha=.85, label="Prediction")
        axes[i].set_title(f"{name}\nRMSE={rmse[i]:.2f} {unit} | CC={cc[i]:.2f}", fontsize=9)
        axes[i].grid(alpha=.25)
    axes[-1].axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_comparison(path, synth, real, unit):
    pos = np.arange(29); width = .4
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    axes[0].bar(pos-width/2, synth[0], width, label="Test synthetique")
    axes[0].bar(pos+width/2, real[0], width, label="Reference reelle")
    axes[0].set_ylabel(f"RMSE ({unit})"); axes[0].legend(); axes[0].grid(axis="y", alpha=.3)
    axes[1].bar(pos-width/2, synth[1], width)
    axes[1].bar(pos+width/2, real[1], width)
    axes[1].axhline(0, color="black", lw=.8); axes[1].set_ylabel("CC")
    axes[1].set_xticks(pos, JOINT_NAMES, rotation=65, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    args = parse_args()
    files = discover_variant_files(args.data_root)
    if not files:
        raise FileNotFoundError(f"Aucun NPZ dans {args.data_root}")
    train, val, test = split_files(files, args.seed, args.train_ratio, args.val_ratio)
    if not train or not test:
        raise ValueError("Le split doit contenir des fichiers train et test.")
    test_file = args.test_file or test[0]
    test_file = Path(test_file)
    if test_file not in test:
        raise ValueError(f"{test_file} n'appartient pas au test pour seed={args.seed}")

    print(f"NPZ: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Fichier test: {test_file.name} | cible={args.target} | ignore_mz={args.ignore_mz}")
    xm, xs, ym, ys, n = stats(train, args.target)
    print(f"Ajustement Ridge sur {n:,} frames...")
    weights = fit(train, args.target, xm, xs, ym, ys, args.ridge, args.ignore_mz)
    sy, sp, srmse, scc = evaluate(test_file, args.target, weights, xm, xs, ym, ys,
                                  args.ignore_mz, reference=False)
    ry, rp, rrmse, rcc = evaluate(test_file, args.target, weights, xm, xs, ym, ys,
                                  args.ignore_mz, reference=True)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "linear_model.npz", weights=weights, x_mean=xm, x_std=xs,
             y_mean=ym, y_std=ys, target=args.target, ignore_mz=args.ignore_mz)
    np.save(out / "synthetic_prediction.npy", sp.astype(np.float32))
    np.save(out / "reference_prediction.npy", rp.astype(np.float32))
    save_metrics(out / "metrics_per_dof.csv", (srmse, scc), (rrmse, rcc))
    unit = TARGET_UNITS[args.target]
    plot_curves(out / "synthetic_prediction_vs_target.png", sy, sp, srmse, scc,
                f"Test synthetique - {test_file.name}", unit)
    plot_curves(out / "reference_prediction_vs_target.png", ry, rp, rrmse, rcc,
                "Test sur les champs de reference reels", unit)
    plot_metric_comparison(out / "metrics_comparison.png", (srmse, scc), (rrmse, rcc), unit)
    print_metrics("TEST SYNTHETIQUE", srmse, scc, unit)
    print_metrics("REFERENCE REELLE", rrmse, rcc, unit)
    print(f"\nResultats: {out}")


if __name__ == "__main__":
    main()
