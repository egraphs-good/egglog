"""Plot the three-trial native ParaBit scaling sweep."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot(source, output, staged=None):
    samples = defaultdict(list)
    for path in [source] + ([staged] if staged else []):
        with path.open() as f:
            for row in csv.DictReader(f):
                family, _, name = row["case"].split("/")
                size = int(Path(name).stem.rsplit("-", 1)[1])
                samples[family, row["mode"], size].append(row)
    modes = [("lean-cert", "Certified Lean", "#005f73", "o"),
             ("lean-rel", "Relational Lean", "#2c79ad", "s"),
             ("egg-track", "egg tracking", "#774795", "D"),
             ("egg-explain", "egg materializing", "#b44e1c", "^")]
    if staged:
        modes.insert(1, ("lean-staged", "Staged certified Lean", "#508128", "X"))
    families = [("mirror-add", "Mirrored adders"), ("guarded-assoc", "Guarded associativity"),
                ("mux-add", "Adder through multiplexers")]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), sharey=True)
    for ax, (family, title) in zip(axes, families):
        sizes = sorted({n for f, _, n in samples if f == family})
        for mode, label, color, mark in modes:
            values = []
            for size in sizes:
                rows = samples[family, mode, size]
                complete = (sorted(r["trial"] for r in rows) == ["1", "2", "3"]
                            and all(r["status"] == "solved" for r in rows))
                values.append(median(float(r["solver_ms"]) for r in rows) if complete else float("nan"))
            ax.plot(sizes, values, marker=mark, color=color, label=label, linewidth=1.7,
                    linestyle="--" if mode == "lean-staged" else "-")
        ax.set(xscale="log", yscale="log", title=title, xlabel="Leaves")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.minorticks_off()
        ax.grid(axis="y", color="#dddddd")
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Median native solver time (ms)")
    handles = [Line2D([], [], marker=m, color=c, label=l, linestyle="--" if k == "lean-staged" else "-")
               for k, l, c, m in modes]
    fig.legend(handles=handles, loc="lower center", ncol=len(modes), frameon=False)
    fig.suptitle("Single-threaded ParaBit searches completed in all three trials")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(output, dpi=160)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--staged", type=Path, help="include the measured alternate certified schedule")
    args = p.parse_args()
    plot(args.input, args.output, args.staged)
