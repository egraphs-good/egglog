"""Time the bounds-check analysis schedule after seeding both single-threaded engines."""
import argparse
import csv
from pathlib import Path
from statistics import median
import tempfile

from compare import ROOT, check_load, sample

SCHEMA = """(datatype E (Reg i64) (Maximum E E) (Next E) (Copy E) (Yes) (InBounds E E))
(relation atMost (E E))
(function lower (E) E :merge (Maximum old new))
(ruleset ranges)
(ruleset writes)
(ruleset copies)
(ruleset checks)
(rule ((= k (Maximum x y))) ((atMost x k)) :ruleset ranges)
(rule ((= k (Maximum x y))) ((atMost y k)) :ruleset ranges)
(rule ((= n (Next (Maximum x y)))) ((set (lower n) x) (set (lower n) y)) :ruleset writes)
(rewrite (Copy x) x :ruleset copies)
(rule ((= c (InBounds i n)) (atMost i k) (= k (lower n))) ((union c (Yes))) :ruleset checks)
(unstable-combined-ruleset range-writes ranges writes)
(Yes)"""
SCHEDULE = "(run-schedule (seq range-writes copies writes checks))"


def rust_inputs(directory, roots):
    setup = [SCHEMA]
    checks = []
    for i in range(roots):
        left, right = f"(Reg {2*i})", f"(Reg {2*i+1})"
        bound = f"(Maximum {left} {right})"
        length = f"(Next (Copy {bound}))"
        test = f"(InBounds {left} {length})"
        setup.append(test)
        checks.append(f"(check (= {test} (Yes)) (= (lower {length}) {bound}))")
    paths = [directory / name for name in ("setup.egg", "schedule.egg", "checks.egg")]
    for path, text in zip(paths, ("\n".join(setup), SCHEDULE, "\n".join(checks))):
        path.write_text(text)
    return paths


def measure(directory, roots, trials, writer):
    commands = {
        "Lean": (ROOT / "lean/.lake/build/bin/compiler-bench", "bounds-schedule", [str(roots)]),
        "Rust": (ROOT / "target/release/examples/lean_comparison", "plain", rust_inputs(directory, roots)),
    }
    timings = {engine: [] for engine in commands}
    for trial in range(trials + 1):
        idle = check_load()
        engines = ["Rust", "Lean"] if trial % 2 else ["Lean", "Rust"]
        for position, engine in enumerate(engines, 1):
            _, ns, wall = sample(*commands[engine], 0, 120)
            writer.writerow([roots, trial, position, engine, ns, wall, idle])
            if trial:
                timings[engine].append(ns / 1e6)
    lean, rust = (median(timings[engine]) for engine in ("Lean", "Rust"))
    print(f"{roots} roots: certified Lean {lean:.3f} ms, Rust {rust:.3f} ms, Lean/Rust {lean/rust:.2f}×")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--trials", type=int, default=8, help="measured pairs, following one warm-up pair")
    parser.add_argument("--csv", type=Path, required=True, help="new file to receive every schedule sample")
    args = parser.parse_args()
    if min(args.roots) < 1 or args.trials < 1:
        parser.error("root counts and trial count must be positive")
    with tempfile.TemporaryDirectory() as tmp, args.csv.open("x", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["roots", "trial", "position", "engine", "schedule_ns", "wall_ns", "idle_pct"])
        for count in args.roots:
            measure(Path(tmp), count, args.trials, writer)
            output.flush()


if __name__ == "__main__":
    main()
