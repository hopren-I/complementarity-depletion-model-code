"""Render current Figure 2 from the archived synthetic parameter-recovery repetitions.

Usage: python code/plot_parameter_recovery.py [--language en|zh|both]
       [--output-dir PATH]

Only this figure and its numerical summary are written. The original coverage
figures and all simulation inputs remain unchanged. No simulations are rerun.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

P = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(P / ".work/mpl"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ARMS = ("correct", "heteroscedastic", "correlated")
TARGETS = ("Vbi", "k")
COLORS = ("#165a8a", "#b45e00", "#623d8c")
TRUTHS = {"Vbi": 1.1, "k": 4.3e12}
EXPECTED = {
    ("Vbi", "correct"): (5.47, 94.5),
    ("Vbi", "heteroscedastic"): (5.21, 95.6),
    ("Vbi", "correlated"): (9.19, 69.1),
    ("k", "correct"): (2.09, 94.5),
    ("k", "heteroscedastic"): (2.09, 94.8),
    ("k", "correlated"): (3.47, 69.7),
}


def load_data():
    root = P
    paths = {
        "correct": P / "data/validation/bayes_repetitions.csv",
        "stress": P / "data/validation/stress_repetitions.csv",
    }
    sources = {}
    hashes = {}
    for key, path in paths.items():
        data = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        sources[key] = list(csv.DictReader(data.decode().splitlines()))
    groups = []
    for target in TARGETS:
        for arm in ARMS:
            source = sources["correct" if arm == "correct" else "stress"]
            rows = sorted(
                [r for r in source if r["target"] == target and (arm == "correct" or r["arm"] == arm)],
                key=lambda r: int(r["rep"]),
            )
            assert [int(r["rep"]) for r in rows] == list(range(1000))
            estimate = np.array([float(r["estimate"]) for r in rows])
            truth = np.array([float(r["truth"]) for r in rows])
            lower = np.array([float(r["lower"]) for r in rows])
            upper = np.array([float(r["upper"]) for r in rows])
            assert np.all(truth == TRUTHS[target])
            assert np.all(np.isfinite([estimate, truth, lower, upper]))
            assert np.all((lower <= estimate) & (estimate <= upper))
            covered = (lower <= truth) & (truth <= upper)
            np.testing.assert_array_equal(covered.astype(int), [int(r["covered"]) for r in rows])
            error = 100 * (estimate - truth) / truth
            mae = float(np.mean(np.abs(error)))
            coverage = float(100 * np.mean(covered))
            expected_mae, expected_coverage = EXPECTED[target, arm]
            assert round(mae, 2) == expected_mae
            assert round(coverage, 1) == expected_coverage
            groups.append({
                "target": target, "arm": arm, "truth": TRUTHS[target],
                "repetitions": len(rows), "error": error,
                "mean_absolute_relative_error_pct": mae,
                "mean_signed_relative_error_pct": float(np.mean(error)),
                "root_mean_squared_relative_error_pct": float(np.sqrt(np.mean(error ** 2))),
                "credible_interval_coverage_pct": coverage,
                "mean_credible_interval_relative_width_pct": float(np.mean(100 * (upper - lower) / truth)),
                "error_percentiles": dict(zip(
                    ("min", "q25", "median", "q75", "max"),
                    map(float, np.quantile(error, [0, .25, .5, .75, 1])),
                )),
            })
    return groups, hashes


def draw(groups, language, output):
    zh = language == "zh"
    plt.rcParams.update({
        "font.family": ["FandolHei", "DejaVu Sans"] if zh else ["DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10,
        "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "text.color": "black", "axes.labelcolor": "black",
        "mathtext.fontset": "dejavusans", "svg.fonttype": "path",
        "svg.hashsalt": "figure6-parameter-recovery",
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.75))
    fig.subplots_adjust(left=.115, right=.975, bottom=.19, top=.88)
    positions = [1, 2, 3, 5, 6, 7]
    # The jitter changes only horizontal display positions, never estimates.
    rng = np.random.default_rng(5206)
    for i, (group, x) in enumerate(zip(groups, positions)):
        color = COLORS[i % 3]
        error = group["error"]
        assert len(error) == 1000 and np.min(error) > -40 and np.max(error) < 44
        jitter = rng.uniform(-.26, .26, len(error))
        ax.scatter(x + jitter, error, s=3.2, alpha=.13, color=color,
                   linewidths=0, zorder=2)
        box = ax.boxplot([error], positions=[x], widths=.34, whis=1.5,
                         showfliers=False, patch_artist=True, manage_ticks=False,
                         boxprops={"edgecolor": color, "linewidth": 1.1},
                         medianprops={"color": "#18222b", "linewidth": 1.5},
                         whiskerprops={"color": color, "linewidth": 1.05},
                         capprops={"color": color, "linewidth": 1.05}, zorder=3)
        box["boxes"][0].set_facecolor(matplotlib.colors.to_rgba(color, .22))
        ax.text(x, 49, f'{group["credible_interval_coverage_pct"]:.1f}%',
                ha="center", va="center", fontsize=10, color=color, fontweight="bold")

    ax.axhline(0, color="#667078", lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(4, 1.4, "真值（0%）" if zh else "True value (0%)",
            ha="center", va="bottom", fontsize=8.2, color="#47525b",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0}, zorder=4)
    ax.text(4, 57, "95%可信区间的实际覆盖率" if zh else "Empirical coverage of 95% credible intervals",
            ha="center", va="center", fontsize=9.1, color="#39434b")
    ax.set(xlim=(.25, 7.75), ylim=(-40, 62),
           ylabel="反演参数相对误差（%）" if zh else "Relative error of recovered parameter (%)")
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.set_xticks([2, 6])
    ax.set_xticklabels([
        "$V_{\\mathrm{bi}}$\n" + ("真值：1.1 V" if zh else "True value: 1.1 V"),
        "$k$\n" + ("真值：" if zh else "True value: ") + "$4.3\\times10^{12}$ cm",
    ], linespacing=1.55)
    ax.tick_params(axis="x", length=0, pad=9)
    ax.tick_params(axis="y", length=3, width=.8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e7ebee", lw=.65)
    names = (["独立、同方差", "异方差", "相关误差"] if zh else
             ["Independent, equal variance", "Heteroscedastic", "Correlated errors"])
    fig.legend([Patch(facecolor=matplotlib.colors.to_rgba(c, .3), edgecolor=c) for c in COLORS],
               names, loc="upper center", bbox_to_anchor=(.54, .995), ncol=3,
               frameon=False, fontsize=8.7, columnspacing=1.8, handlelength=1.4)
    texts = [item.get_text() for item in fig.findobj(matplotlib.text.Text) if item.get_text()]
    if not zh:
        assert not any("\u4e00" <= ch <= "\u9fff" for text in texts for ch in text)
    for ext in ("png", "svg"):
        fig.savefig(output / f"fig_parameter_recovery_{language}.{ext}", dpi=600,
                    facecolor="white", metadata={"Date": None} if ext == "svg" else None)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "zh", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=P / ".work/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(str(P / "assets/fonts/FandolHei-Regular.otf"))
    groups, hashes = load_data()
    for language in (("en", "zh") if args.language == "both" else (args.language,)):
        draw(groups, language, args.output_dir)
    summary = {
        "generator": "code/plot_parameter_recovery.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs_sha256": hashes,
        "estimate": "posterior median in each independent repetition",
        "error_definition": "100 * (estimate - truth) / truth",
        "points_plotted": 6000,
        "box_definition": "25th-75th percentiles; median line; 1.5 IQR whiskers; all repetitions plotted",
        "coverage_definition": "fraction of per-repetition 95% credible intervals containing the fixed truth",
        "groups": [{k: v for k, v in group.items() if k != "error"} for group in groups],
        "figures_sha256": {
            "paper/figs/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.output_dir.glob("fig_parameter_recovery_*"))
            if path.suffix in (".png", ".svg")
        },
    }
    summary_path = (P / "qa/merged_parameter_recovery.json"
                    if args.output_dir.resolve() == (P / "paper/figs").resolve()
                    else args.output_dir / "fig_parameter_recovery_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "points": 6000, "groups": 6}))


if __name__ == "__main__":
    main()
