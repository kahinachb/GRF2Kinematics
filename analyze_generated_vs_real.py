"""Compare generated squat trajectories against Trial111 recordings.

Outputs a channel-level CSV summary, best-trajectory comparisons, and two
diagnostic figures in analysis_generated_102/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
GEN = ROOT / "DATA" / "generated_102"
REAL = ROOT / "DATA" / "Vinc" / "Jeremy" / "Trial111"
OUT = ROOT / "analysis_generated_102"
OUT.mkdir(exist_ok=True)


def load_pair(kind, real_name, real_column_map=None):
    real = pd.read_csv(REAL / real_name)
    files = sorted(GEN.glob(f"*_{kind}.csv"))
    bases = {p.name.removesuffix(f"_{kind}.csv") for p in files}
    other = "grfm" if kind == "q" else "q"
    paired = sorted(bases & {p.name.removesuffix(f"_{other}.csv")
                             for p in GEN.glob(f"*_{other}.csv")})
    files = [GEN / f"{b}_{kind}.csv" for b in paired]
    cols = list(real.columns)
    real_cols = [real_column_map.get(c, c) if real_column_map else c for c in cols]
    n = min(len(real), len(pd.read_csv(files[0])))
    arr = np.stack([pd.read_csv(p).loc[:n - 1, cols].to_numpy(float) for p in files])
    return real.loc[:n - 1, real_cols].to_numpy(float), arr, cols, real_cols, paired


def summarize(real, generated, cols, real_cols, bases, label):
    # RMSE normalized by real signal range avoids force/moment scale dominating.
    span = np.ptp(real, axis=0)
    scale = np.where(span > 1e-8, span, np.nanstd(real, axis=0) + 1e-8)
    rmse_by_traj = np.sqrt(np.mean((generated - real) ** 2, axis=1))
    normalized = rmse_by_traj / scale
    score = np.nanmean(normalized, axis=1)
    best = int(np.nanargmin(score))
    mean = generated.mean(axis=0)
    lo, hi = np.quantile(generated, [0.05, 0.95], axis=0)
    result = pd.DataFrame({
        "generated_channel": cols, "matched_real_channel": real_cols,
        "unit": "rad" if label == "q" else ["N" if c.startswith("F") else "N m" if c.startswith("M") else "m" for c in cols],
        "real_mean": real.mean(axis=0), "generated_mean": mean.mean(axis=0),
        "mean_bias": (mean - real).mean(axis=0),
        "real_range": span, "generated_mean_range": np.ptp(mean, axis=0),
        "ensemble_time_RMSE": np.sqrt(np.mean((mean-real)**2, axis=0)),
        "median_trajectory_RMSE": np.median(rmse_by_traj, axis=0),
        "real_inside_5_95_pct": ((real >= lo) & (real <= hi)).mean(axis=0) * 100,
        "best_trajectory_RMSE": rmse_by_traj[best],
    })
    result.to_csv(OUT / f"{label}_channel_summary.csv", index=False)
    pd.DataFrame({"trajectory": bases, "normalized_range_RMSE": score}).sort_values(
        "normalized_range_RMSE").to_csv(OUT / f"{label}_trajectory_ranking.csv", index=False)
    return result, best, score[best]


def plot_signals(real, generated, cols, best, names, title, path, unit):
    t = np.linspace(0, 100, len(real))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, name in zip(axes.flat, names):
        i = cols.index(name)
        avg = generated[:, :, i].mean(axis=0)
        low, high = np.quantile(generated[:, :, i], [.05, .95], axis=0)
        ax.fill_between(t, low, high, color="#f4a261", alpha=.35, label="generated 5–95%")
        ax.plot(t, avg, color="#e76f51", lw=1.3, label="generated mean")
        ax.plot(t, generated[best, :, i], color="#457b9d", lw=1, alpha=.85, label="best generated")
        ax.plot(t, real[:, i], color="#1d3557", lw=1.2, label="real")
        ax.set_title(name); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.supxlabel("Normalized recording duration (%)")
    fig.supylabel(unit)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    rq, gq, qcols, rqcols, bases = load_pair("q", "joints_filtered_FF.csv")
    # Generated plate 1 is spatially aligned with real plate 2, and conversely.
    swap = {c: c.replace("1_glob", "X_glob").replace("2_glob", "1_glob").replace("X_glob", "2_glob")
            for c in pd.read_csv(REAL / "kinetics_glob_filtered.csv", nrows=1).columns}
    rk, gk, kcols, rkcols, kbases = load_pair("grfm", "kinetics_glob_filtered.csv", swap)
    assert bases == kbases
    qs, qb, qscore = summarize(rq, gq, qcols, rqcols, bases, "q")
    ks, kb, kscore = summarize(rk, gk, kcols, rkcols, bases, "grfm")
    plot_signals(rq, gq, qcols, qb,
                 ["Lhip_flex_ext", "Lknee_flex_ext", "Lankle_flex_ext", "Rhip_flex_ext", "Rknee_flex_ext", "Rankle_flex_ext"],
                 "Lower-limb joint angles: real vs generated", OUT / "joint_angles_comparison.png", "radians")
    plot_signals(rk, gk, kcols, kb,
                 ["Fx1_glob", "Fy1_glob", "Fz1_glob", "Fx2_glob", "Fy2_glob", "Fz2_glob"],
                 "Ground-reaction forces: real vs generated", OUT / "ground_reaction_forces_comparison.png", "N")
    with open(OUT / "README.md", "w") as f:
        f.write("# Generated vs real Trial111 analysis\n\n")
        f.write(f"Aligned samples: {len(rq)}; paired generated trajectories: {len(bases)}.\n\n")
        f.write(f"Best joint-angle trajectory: `{bases[qb]}` (mean range-normalized RMSE {qscore:.3f}).\n\n")
        f.write(f"Best GRF/moment trajectory: `{bases[kb]}` (mean range-normalized RMSE {kscore:.3f}).\n\n")
        f.write("Kinetics use a plate swap: generated plate 1 is compared with real plate 2, and generated plate 2 with real plate 1.\n")


if __name__ == "__main__":
    main()
