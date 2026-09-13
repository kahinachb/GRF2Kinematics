"""Compare visuellement les anciens/nouveaux synthétiques à christine_ref.

Les trois jeux sont supposés suivre la convention GRFM [gauche, droite].
Les fichiers synthétiques sont sous <root>/Christine/variant_*/ et la
référence sous <ref-root>/Christine/variant_000/.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


GRFM_NAMES = [
    "L Fx", "L Fy", "L Fz", "L Mx", "L My", "L Mz",
    "L CoPx", "L CoPy", "L CoPz",
    "R Fx", "R Fy", "R Fz", "R Mx", "R My", "R Mz",
    "R CoPx", "R CoPy", "R CoPz",
]

JOINT_NAMES = [
    "R hip flex/ext", "R hip abd/add", "R hip int/ext rot",
    "R knee flex/ext", "R ankle flex/ext", "R ankle abd/add",
    "L hip flex/ext", "L hip abd/add", "L hip int/ext rot",
    "L knee flex/ext", "L ankle flex/ext", "L ankle abd/add",
    "Lumbar flex/ext", "Lumbar lateral flex", "L clavicle X",
    "L shoulder flex/ext", "L shoulder abd/add", "L shoulder int/ext rot",
    "L elbow flex/ext", "L elbow pron/sup",
    "Cervical flex/ext", "Cervical lateral bend", "Cervical int/ext rot",
    "R clavicle X", "R shoulder flex/ext", "R shoulder abd/add",
    "R shoulder int/ext rot", "R elbow flex/ext", "R elbow pron/sup",
]

COLORS = {"Référence": "black", "Ancien synth.": "#d95f02", "Nouveau synth.": "#1b9e77"}


def load_synthetic(root: Path, filename: str, stride: int) -> np.ndarray:
    trials = sorted((root / "Christine").glob("variant_*"))
    if not trials:
        raise FileNotFoundError(f"Aucune variante trouvée dans {root / 'Christine'}")
    arrays = []
    for trial in trials:
        path = trial / filename
        if not path.exists():
            raise FileNotFoundError(path)
        arrays.append(np.load(path)[::stride])
    return np.concatenate(arrays).astype(np.float64, copy=False)


def robust_limits(arrays: list[np.ndarray], index: int) -> tuple[float, float]:
    values = np.concatenate([array[:, index] for array in arrays])
    low, high = np.quantile(values, [0.002, 0.998])
    if not np.isfinite(low + high) or np.isclose(low, high):
        center = float(np.nanmean(values)) if len(values) else 0.0
        return center - 1.0, center + 1.0
    margin = 0.04 * (high - low)
    return low - margin, high + margin


def distribution_grid(
    datasets: dict[str, np.ndarray], names: list[str], output: Path,
    ncols: int = 3, bins: int = 70,
) -> None:
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.15 * nrows))
    axes = np.asarray(axes).reshape(-1)
    arrays = list(datasets.values())
    for index, name in enumerate(names):
        ax = axes[index]
        low, high = robust_limits(arrays, index)
        edges = np.linspace(low, high, bins + 1)
        for label, values in datasets.items():
            ax.hist(
                values[:, index], bins=edges, density=True,
                histtype="step", linewidth=1.7, color=COLORS[label], label=label,
            )
        stats_lines = []
        for label, values in datasets.items():
            mean = values[:, index].mean()
            std = values[:, index].std()
            stats_lines.append(f"{label}: μ={mean:.4g}  σ={std:.4g}")
        ax.text(
            0.98, 0.97, "\n".join(stats_lines),
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.78,
                  "edgecolor": "0.75"},
        )
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.2)
        if index == 0:
            ax.legend(fontsize=8)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Distributions — référence vs anciens/nouveaux synthétiques", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, np.ndarray]:
    ref_mean = reference.mean(axis=0)
    ref_std = reference.std(axis=0)
    cand_mean = candidate.mean(axis=0)
    cand_std = candidate.std(axis=0)
    scale = ref_std + 1e-12
    return {
        "wasserstein_std": np.array([
            wasserstein_distance(reference[:, i], candidate[:, i]) / scale[i]
            for i in range(reference.shape[1])
        ]),
        "mean_shift_std": (cand_mean - ref_mean) / scale,
        "std_ratio": cand_std / scale,
    }


def metric_heatmap(
    old_metrics: dict[str, np.ndarray], new_metrics: dict[str, np.ndarray],
    names: list[str], output: Path, title: str,
) -> None:
    # Chaque ligne est une mesure; chaque colonne un canal.
    matrix = np.vstack([
        old_metrics["wasserstein_std"], new_metrics["wasserstein_std"],
        np.abs(old_metrics["mean_shift_std"]), np.abs(new_metrics["mean_shift_std"]),
        np.abs(np.log2(np.maximum(old_metrics["std_ratio"], 1e-8))),
        np.abs(np.log2(np.maximum(new_metrics["std_ratio"], 1e-8))),
    ])
    row_names = [
        "W/std ancien", "W/std nouveau",
        "|écart moyenne| ancien", "|écart moyenne| nouveau",
        "|log2 ratio std| ancien", "|log2 ratio std| nouveau",
    ]
    vmax = max(1.0, float(np.quantile(matrix[np.isfinite(matrix)], 0.95)))
    fig, ax = plt.subplots(figsize=(max(14, 0.48 * len(names)), 5.2))
    image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(names)), labels=names, rotation=65, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_names)), labels=row_names, fontsize=9)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label="Écart à la référence (plus faible = meilleur)")
    fig.tight_layout()
    fig.savefig(output, dpi=190)
    plt.close(fig)


def save_summary(
    path: Path, group: str, names: list[str], reference: np.ndarray,
    old: np.ndarray, new: np.ndarray,
) -> None:
    old_m = metrics(reference, old)
    new_m = metrics(reference, new)
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "group": group,
            "channel": name,
            "ref_mean": reference[:, i].mean(), "ref_std": reference[:, i].std(),
            "old_mean": old[:, i].mean(), "old_std": old[:, i].std(),
            "new_mean": new[:, i].mean(), "new_std": new[:, i].std(),
            "old_wasserstein_over_ref_std": old_m["wasserstein_std"][i],
            "new_wasserstein_over_ref_std": new_m["wasserstein_std"][i],
            "old_mean_shift_ref_std": old_m["mean_shift_std"][i],
            "new_mean_shift_ref_std": new_m["mean_shift_std"][i],
            "old_std_ratio": old_m["std_ratio"][i],
            "new_std_ratio": new_m["std_ratio"][i],
        })
    frame = pd.DataFrame(rows)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=Path("DATA/synth_christine"))
    parser.add_argument("--new-root", type=Path, default=Path("DATA/synth2_christine"))
    parser.add_argument("--ref-root", type=Path, default=Path("DATA/christine_ref_npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_christine_data_comparison"))
    parser.add_argument("--stride", type=int, default=20, help="Sous-échantillonnage temporel des synthétiques.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride doit être >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_grfm = load_synthetic(args.old_root, "kinetics_deltaf.npy", args.stride)
    new_grfm = load_synthetic(args.new_root, "kinetics_deltaf.npy", args.stride)
    old_joints = load_synthetic(args.old_root, "all_joints_deltaf.npy", args.stride)
    new_joints = load_synthetic(args.new_root, "all_joints_deltaf.npy", args.stride)
    ref_dir = args.ref_root / "Christine" / "variant_000"
    ref_grfm = np.load(ref_dir / "kinetics_deltaf.npy").astype(np.float64)
    ref_joints = np.load(ref_dir / "all_joints_deltaf.npy").astype(np.float64)

    grfm_sets = {"Référence": ref_grfm, "Ancien synth.": old_grfm, "Nouveau synth.": new_grfm}
    articulated_sets = {
        "Référence": ref_joints[:, 6:],
        "Ancien synth.": old_joints[:, 6:],
        "Nouveau synth.": new_joints[:, 6:],
    }
    ff_names = ["FF Δx", "FF Δy", "FF Δz", "FF Δrx", "FF Δry", "FF Δrz"]
    ff_sets = {"Référence": ref_joints[:, :6], "Ancien synth.": old_joints[:, :6], "Nouveau synth.": new_joints[:, :6]}

    distribution_grid(grfm_sets, GRFM_NAMES, args.output_dir / "grfm_distributions.png")
    distribution_grid(articulated_sets, JOINT_NAMES, args.output_dir / "joint_distributions.png")
    distribution_grid(ff_sets, ff_names, args.output_dir / "freeflyer_distributions.png")

    old_grfm_m, new_grfm_m = metrics(ref_grfm, old_grfm), metrics(ref_grfm, new_grfm)
    old_joint_m = metrics(ref_joints[:, 6:], old_joints[:, 6:])
    new_joint_m = metrics(ref_joints[:, 6:], new_joints[:, 6:])
    metric_heatmap(old_grfm_m, new_grfm_m, GRFM_NAMES, args.output_dir / "grfm_distance_heatmap.png", "Écarts des GRFM à christine_ref")
    metric_heatmap(old_joint_m, new_joint_m, JOINT_NAMES, args.output_dir / "joint_distance_heatmap.png", "Écarts des angles à christine_ref")

    summary_path = args.output_dir / "distribution_summary.csv"
    if summary_path.exists():
        summary_path.unlink()
    save_summary(summary_path, "GRFM", GRFM_NAMES, ref_grfm, old_grfm, new_grfm)
    save_summary(summary_path, "JOINT", JOINT_NAMES, ref_joints[:, 6:], old_joints[:, 6:], new_joints[:, 6:])
    save_summary(summary_path, "FREEFLYER", ff_names, ref_joints[:, :6], old_joints[:, :6], new_joints[:, :6])

    print(f"Figures et CSV sauvegardés dans {args.output_dir}")


if __name__ == "__main__":
    main()
