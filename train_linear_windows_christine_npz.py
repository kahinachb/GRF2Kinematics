"""Compare des regressions Ridge avec plusieurs fenetres GRFM causales.

Convention: GRFM[t-window+1:t+1] -> q[t]. Toutes les tailles sont entrainees
sur les memes fichiers et les memes instants cibles. L'evaluation commune
commence a ``max(windows)-1`` afin d'utiliser exactement les memes frames.
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from train_linear_christine_npz import (
    INPUT_KEYS, REFERENCE_INPUT_KEYS, JOINT_NAMES, load_pair, split_files,
    stats, correlation_columns,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("DATA/Christine_synthetic"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results_linear_windows_christine_npz"))
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 25, 50, 100, 200])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=.70)
    parser.add_argument("--val-ratio", type=float, default=.15)
    parser.add_argument("--max-train-windows", type=int, default=20000,
                        help="Nombre d'instants train echantillonnes (communs a toutes les tailles).")
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--ignore-mz", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def choose_coordinates(paths, first_frame, maximum, seed):
    """Echantillonne uniformement des couples (fichier, frame cible)."""
    counts = []
    for path in paths:
        x, _ = load_pair(path, "q")
        counts.append(max(0, len(x) - first_frame))
    total = sum(counts)
    if total == 0:
        raise ValueError("Aucune sequence assez longue pour la fenetre maximale.")
    number = min(maximum, total)
    selected = np.sort(np.random.default_rng(seed).choice(total, number, replace=False))
    cumulative = np.cumsum([0] + counts)
    coordinates = []
    for flat_index in selected:
        file_index = int(np.searchsorted(cumulative, flat_index, side="right") - 1)
        frame = first_frame + int(flat_index - cumulative[file_index])
        coordinates.append((file_index, frame))
    return coordinates, total


def normalized_pair(path, xm, xs, ym, ys, reference=False, ignore_mz=False):
    x, y = load_pair(path, "q", reference=reference)
    x = ((x - xm) / xs).astype(np.float32)
    y = ((y - ym) / ys).astype(np.float32)
    if ignore_mz:
        x[:, (5, 11)] = 0.0
    return x, y


def build_sampled_train(paths, coordinates, window, xm, xs, ym, ys, ignore_mz):
    features = np.empty((len(coordinates), window * 12), dtype=np.float32)
    targets = np.empty((len(coordinates), 29), dtype=np.float32)
    by_file = {}
    for row, (file_index, frame) in enumerate(coordinates):
        by_file.setdefault(file_index, []).append((row, frame))
    for file_index, rows in by_file.items():
        x, y = normalized_pair(paths[file_index], xm, xs, ym, ys,
                               ignore_mz=ignore_mz)
        for row, frame in rows:
            # Ordre explicite: plus ancien -> frame courante, puis 12 canaux.
            features[row] = x[frame-window+1:frame+1].reshape(-1)
            targets[row] = y[frame]
    return features, targets


def build_eval(path, window, first_frame, xm, xs, ym, ys, ignore_mz, reference):
    x, y = normalized_pair(path, xm, xs, ym, ys, reference, ignore_mz)
    frames = np.arange(first_frame, len(x))
    features = np.empty((len(frames), window * 12), dtype=np.float32)
    for row, frame in enumerate(frames):
        features[row] = x[frame-window+1:frame+1].reshape(-1)
    return features, y[frames]


def metrics(y_normalized, pred_normalized, ym, ys):
    y = y_normalized * ys + ym
    prediction = pred_normalized * ys + ym
    rmse = np.degrees(np.sqrt(np.mean((prediction - y) ** 2, axis=0)))
    cc = correlation_columns(y, prediction)
    return y, prediction, rmse, cc


def aggregate(rmse, cc):
    return {
        "rmse_mean_deg": float(np.mean(rmse)),
        "rmse_global_deg": float(np.sqrt(np.mean(rmse ** 2))),
        "cc_median": float(np.nanmedian(cc)),
        "cc_mean": float(np.nanmean(cc)),
    }


def plot_summary(path, rows):
    windows = [row["window_frames"] for row in rows]
    seconds = np.asarray(windows) * .01
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for domain, label, color in [("synth", "Test synthetique", "#4e79a7"),
                                  ("real", "Reference reelle", "#e15759")]:
        axes[0].plot(seconds, [row[f"{domain}_rmse_mean_deg"] for row in rows],
                     marker="o", label=label, color=color)
        axes[1].plot(seconds, [row[f"{domain}_cc_median"] for row in rows],
                     marker="o", label=label, color=color)
    axes[0].set_ylabel("RMSE moyenne (deg)")
    axes[1].set_ylabel("CC median")
    for axis in axes:
        axis.set_xlabel("Historique causal (s)")
        axis.set_xticks(seconds)
        axis.grid(alpha=.3)
        axis.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_per_dof(path, all_results, metric, ylabel):
    positions = np.arange(29)
    fig, axes = plt.subplots(2, 1, figsize=(17, 11), sharex=True)
    for window, result in all_results.items():
        axes[0].plot(positions, result[f"synth_{metric}"], marker=".", label=f"{window} frames")
        axes[1].plot(positions, result[f"real_{metric}"], marker=".", label=f"{window} frames")
    axes[0].set_title("Test synthetique"); axes[1].set_title("Reference reelle")
    for axis in axes:
        axis.set_ylabel(ylabel); axis.grid(alpha=.25); axis.legend(ncol=5, fontsize=8)
    axes[1].set_xticks(positions, JOINT_NAMES, rotation=65, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    args = parse_args()
    windows = sorted(set(args.windows))
    if not windows or windows[0] < 1:
        raise ValueError("Toutes les fenetres doivent etre >= 1.")
    files = sorted(args.data_root.glob("*.npz"))
    train, val, test = split_files(files, args.seed, args.train_ratio, args.val_ratio)
    if not train or not test:
        raise ValueError("Le split doit contenir des fichiers train et test.")
    test_file = test[0]
    first_frame = max(windows) - 1
    xm, xs, ym, ys, _ = stats(train, "q")
    coordinates, available = choose_coordinates(
        train, first_frame, args.max_train_windows, args.seed)
    print(f"NPZ: train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"Test: {test_file.name}")
    print(f"Instants train: {len(coordinates):,}/{available:,}; evaluation commune t>={first_frame}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, all_results = [], {}
    for window in windows:
        print(f"\nFenetre {window} frames ({window*.01:.2f} s), {window*12*29+29:,} parametres")
        train_x, train_y = build_sampled_train(
            train, coordinates, window, xm, xs, ym, ys, args.ignore_mz)
        model = Ridge(alpha=args.ridge, fit_intercept=True, solver="lsqr", tol=1e-4)
        model.fit(train_x, train_y)
        del train_x, train_y

        sx, sy = build_eval(test_file, window, first_frame, xm, xs, ym, ys,
                            args.ignore_mz, reference=False)
        rx, ry = build_eval(test_file, window, first_frame, xm, xs, ym, ys,
                            args.ignore_mz, reference=True)
        _, _, srmse, scc = metrics(sy, model.predict(sx), ym, ys)
        _, _, rrmse, rcc = metrics(ry, model.predict(rx), ym, ys)
        sa, ra = aggregate(srmse, scc), aggregate(rrmse, rcc)
        row = {"window_frames": window, "window_seconds": window*.01}
        row.update({f"synth_{key}": value for key, value in sa.items()})
        row.update({f"real_{key}": value for key, value in ra.items()})
        summary_rows.append(row)
        all_results[window] = {"synth_rmse": srmse, "synth_cc": scc,
                               "real_rmse": rrmse, "real_cc": rcc}
        np.savez(args.output_dir / f"model_window_{window}.npz",
                 coef=model.coef_, intercept=model.intercept_, x_mean=xm, x_std=xs,
                 y_mean=ym, y_std=ys, window=window, first_eval_frame=first_frame)
        print(f"  Synth: RMSE moy={sa['rmse_mean_deg']:.3f} deg, CC med={sa['cc_median']:.3f}")
        print(f"  Reel : RMSE moy={ra['rmse_mean_deg']:.3f} deg, CC med={ra['cc_median']:.3f}")

    with (args.output_dir / "window_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    with (args.output_dir / "metrics_per_dof.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window_frames", "dof_index", "dof", "synth_rmse_deg",
                         "synth_cc", "real_rmse_deg", "real_cc"])
        for window, result in all_results.items():
            for i, name in enumerate(JOINT_NAMES):
                writer.writerow([window, i, name, result["synth_rmse"][i],
                                 result["synth_cc"][i], result["real_rmse"][i],
                                 result["real_cc"][i]])
    plot_summary(args.output_dir / "window_comparison.png", summary_rows)
    plot_per_dof(args.output_dir / "rmse_per_dof.png", all_results, "rmse", "RMSE (deg)")
    plot_per_dof(args.output_dir / "cc_per_dof.png", all_results, "cc", "CC")
    print(f"\nResultats: {args.output_dir}")


if __name__ == "__main__":
    main()
