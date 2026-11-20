#!/usr/bin/env python3
"""Compare the same single-threaded schedule and check every reported table size."""

import argparse
import ctypes
import os
from pathlib import Path
from random import Random
from statistics import median, quantiles
import subprocess
import tempfile
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SIGNATURE = "(datatype E (Zero) (One) (Succ E) (Add E E) (Mul E E))"
WORKLOADS = {
    "identities": [
        "(rewrite (Add x (Zero)) x)",
        "(rewrite (Mul x (One)) x)",
        "(rewrite (Mul x (Zero)) (Zero))",
    ],
    "commute": [
        "(rewrite (Add x y) (Add y x))",
        "(rewrite (Mul x y) (Mul y x))",
    ],
    "distribute": [
        "(rewrite (Mul x (Add y z)) (Add (Mul x y) (Mul x z)))",
        "(rewrite (Add x (Add y z)) (Add (Add x y) z))",
    ],
    "nonlinear": ["(rewrite (Add x x) (Mul (Succ (One)) x))"],
}


def generate(case, size):
    setup = [SIGNATURE, *WORKLOADS[case], "(let $n0 (One))"]
    checks = []
    # Distinct shared atoms with O(n) input; all workloads run on the same signature.
    for i in range(1, size):
        setup.append(f"(let $n{i} (Succ $n{i - 1}))")
    for i in range(size):
        x, y, z = (f"$n{(i + offset) % size}" for offset in (0, 3, 7))
        if case == "identities":
            term = f"(Add (Mul (Add {x} (Zero)) (One)) (Mul {y} (Zero)))"
            expected = x
        elif case == "commute":
            term = f"(Mul (Add {x} {y}) {z})"
            expected = f"(Mul {z} (Add {y} {x}))"
        elif case == "distribute":
            term = f"(Mul {x} (Add {y} (Add {z} {x})))"
            expected = f"(Add (Add (Mul {x} {y}) (Mul {x} {z})) (Mul {x} {x}))"
        else:
            term = f"(Add {x} {x})"
            expected = f"(Mul (Succ (One)) {x})"
            setup.append(f"(Add {x} {y})")  # Repeated-variable guard should reject these.
        setup.append(f"(let $t{i} {term})")
        if i in {0, size // 2, size - 1}:
            checks.append(f"(check (= $t{i} {expected}))")
    checks.extend(f"(print-size {op})" for op in ("Zero", "One", "Succ", "Add", "Mul"))
    return "\n".join(setup), "(run 8)", "\n".join(checks)


def cpu_ticks():
    """Return total and idle ticks from all CPUs."""
    if sys.platform == "darwin":
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        ticks = (ctypes.c_uint32 * 4)()
        count = ctypes.c_uint32(4)
        if lib.host_statistics(lib.mach_host_self(), 3, ticks, ctypes.byref(count)) != 0:
            raise RuntimeError("cannot sample host CPU load")
        return sum(ticks), ticks[2]
    fields = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    return sum(fields), fields[3] + fields[4]


def check_load():
    total, idle = cpu_ticks()
    time.sleep(1)
    new_total, new_idle = cpu_ticks()
    idle_pct = 100 * (new_idle - idle) / (new_total - total)
    print(f"Host {idle_pct:.1f}% CPU idle, load {os.getloadavg()[0]:.2f}", flush=True)
    if idle_pct < 60:
        raise SystemExit("Busy host; retry the benchmark when CPU load subsides")
    return round(idle_pct, 1)


def sample(binary, mode, paths, count, timeout):
    start = time.perf_counter_ns()
    result = subprocess.run(
        [str(binary), mode, *paths], check=True, stdout=subprocess.PIPE, text=True,
        env={**os.environ, "LEAN_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}, timeout=timeout,
    )
    wall_ns = time.perf_counter_ns() - start
    *rows, timing = result.stdout.splitlines()
    if not timing.startswith("schedule_ns=") or len(rows) != count:
        raise ValueError(f"unexpected output: {result.stdout!r}")
    return tuple(map(int, rows)), int(timing.split("=")[1]), wall_ns


def math_workload(rounds):
    source = (ROOT / "tests/math-microbenchmark.egg").read_text()
    setup, _ = source.split("(run 11)", 1)
    names = "Diff Integral Add Sub Mul Div Pow Ln Sqrt Sin Cos Const Var".split()
    checks = [
        '(check (= (Add (Var "x") (Var "y")) (Add (Var "y") (Var "x"))))',
        '(check (= (Mul (Const 2) (Var "x")) (Mul (Var "x") (Const 2))))',
    ]
    checks.extend(f"(print-size {name})" for name in names)
    checks = "\n".join(checks)
    return setup, f"(run {rounds})", checks


def paths_workload(edges):
    # The recursive rules come from tests/web-demo/path.egg. Extend the input
    # chain and verify its complete transitive closure after a bounded run.
    source = (ROOT / "tests/web-demo/path.egg").read_text()
    rules, _ = source.split("(edge 1 2)", 1)
    seed = [f"(edge {i} {i + 1})" for i in range(edges)]
    checks = f"(check (path 0 {edges}))\n(print-size edge)\n(print-size path)"
    return rules + "\n".join(seed), f"(run {edges + 1})", checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--case", choices=WORKLOADS, action="append")
    parser.add_argument("--proofs", action="store_true", help="also time Rust proof generation")
    parser.add_argument("--enforce", action="store_true", help="fail if any schedule ratio exceeds 1.30")
    parser.add_argument("--math-rounds", type=int, choices=range(1, 12),
                        help="run the repository math benchmark alone, with the given round bound")
    parser.add_argument("--timeout", type=float, default=120, help="seconds allowed for each process")
    parser.add_argument("--paths", type=int, metavar="EDGES",
                        help="run the repository path rules on a chain with this many edges")
    args = parser.parse_args()
    if args.paths is not None and (args.paths < 1 or args.math_rounds is not None):
        parser.error("choose either positive --paths EDGES or --math-rounds")
    if args.size < 9 or args.runs < 1 or args.warmup < 0:
        parser.error("expected size >= 9, runs >= 1, warmup >= 0")
    rust = ROOT / "target/release/examples/lean_comparison"
    lean = ROOT / "lean/.lake/build/bin/lean-egglog"
    modes = {"Rust": (rust, "plain"), "Lean": (lean, "--time-run")}
    if args.math_rounds or args.paths:
        modes["Lean"] = lean, "--time-raw"
    if args.proofs:
        modes["Rust proofs"] = rust, "proofs"
    rng = Random(42)
    passes = True
    print("case          engine       schedule ms   IQR ms   wall ms   ratio    table sizes")
    with tempfile.TemporaryDirectory() as tmp:
        paths = [Path(tmp) / f"{phase}.egg" for phase in ("setup", "run", "checks")]
        cases = ["math"] if args.math_rounds else (["paths"] if args.paths else (args.case or WORKLOADS))
        for case in cases:
            check_load()
            if case == "math":
                parts = math_workload(args.math_rounds)
            elif case == "paths":
                parts = paths_workload(args.paths)
            else:
                parts = generate(case, args.size)
            for path, content in zip(paths, parts):
                path.write_text(content + "\n")
            samples = {mode: [] for mode in modes}
            count = parts[2].count("(print-size ")
            expected = (args.paths, args.paths * (args.paths + 1) // 2) if case == "paths" else None
            for i in range(args.warmup + args.runs):
                order = list(modes)
                rng.shuffle(order)
                for mode in order:
                    rows, ns, wall = sample(*modes[mode], paths, count, args.timeout)
                    if expected is not None and rows != expected:
                        raise ValueError(f"{case}: {mode} rows {rows} differ from {expected}")
                    expected = rows
                    if i >= args.warmup:
                        samples[mode].append((ns, wall))
            baseline = median(ns for ns, _ in samples["Rust"])
            for mode, values in samples.items():
                ns = median(ns for ns, _ in values)
                ratio = ns / baseline
                wall = median(w for _, w in values)
                q = quantiles([t / 1e6 for t, _ in values], method="inclusive") if len(values) > 1 else [0, 0, 0]
                print(f"{case:13} {mode:11} {ns/1e6:11.3f} {q[2] - q[0]:8.3f} {wall/1e6:9.2f} {ratio:7.3f}    {expected}", flush=True)
                if mode == "Lean" and ratio > 1.30:
                    passes = False
    if args.enforce and not passes:
        raise SystemExit("Lean exceeded 1.30x the Rust schedule time")


if __name__ == "__main__":
    main()
