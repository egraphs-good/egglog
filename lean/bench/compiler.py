"""Compare the single-threaded Rust and verified Lean passes from native IR to native IR."""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from random import Random
from statistics import median, quantiles

from compare import ROOT, check_load

# Rust's primitive i64 costs one. Reg costs zero so open variables have
# the same additive cost as in the certified compiler.
BOOLEAN_SETUP = """
(datatype E (Reg i64 :cost 0) (Yes) (No) (Neg E) (Conj E E) (Disj E E) (Mux E E E))
(ruleset algebra)
(ruleset control)
(rewrite (Conj x (Yes)) x :ruleset algebra)
(rewrite (Disj (No) x) x :ruleset algebra)
(rewrite (Neg (Neg x)) x :ruleset algebra)
(rewrite (Mux c x x) x :ruleset control)
"""

STRENGTH_SETUP = """
(datatype E (Reg i64 :cost 0) (One) (Plus E E :cost 2) (Times E E :cost 4) (Pow2 E) (Shl E E))
(ruleset simplify)
(ruleset shift)
(rewrite (Plus x x) (Times x (Pow2 (One))) :ruleset simplify)
(rewrite (Plus (One) (One)) (Pow2 (One)) :ruleset simplify)
(rule ((= p (Times x y)) (= y (Pow2 n))) ((union p (Shl x n))) :ruleset shift)
(rule ((= p (Times x y)) (= x (One)) (= y (One))) ((union p (One))) :ruleset shift)
"""

WORKLOADS = {
    "boolean": (BOOLEAN_SETUP, "(run-schedule control (repeat 4 (seq algebra control)))"),
    "strength": (STRENGTH_SETUP, "(run-schedule simplify shift)"),
}


def program(workload, start, size):
    if size == 1:
        reg = f"(Reg {start})"
        second = f"(Reg {start + 1})"
        if workload == "boolean":
            guard = f"(Neg (Neg {second}))"
            return f"(Mux {guard} (Conj {reg} (Yes)) (Disj (No) {reg}))", reg
        if start % 3 == 0:
            return f"(Plus {reg} {reg})", f"(Shl {reg} (One))"
        if start % 3 == 1:
            return f"(Times {reg} (Plus (One) (One)))", f"(Shl {reg} (One))"
        return f"(Times {reg} (Pow2 {second}))", f"(Shl {reg} {second})"
    left, right = size // 2, size - size // 2
    a, x = program(workload, start, left)
    b, y = program(workload, start + left, right)
    op = "Conj" if workload == "boolean" else "Plus"
    return f"({op} {a} {b})", f"({op} {x} {y})"


def run_compiler(command, env):
    """Collect the child's own CPU time and peak RSS, including parsing and checks."""
    start = time.perf_counter_ns()
    deadline = time.monotonic() + 120
    with tempfile.TemporaryFile(mode="w+") as output, subprocess.Popen(
        command, stdout=output, stderr=subprocess.STDOUT, env=env
    ) as child:
        while True:
            pid, status, usage = os.wait4(child.pid, os.WNOHANG)
            if pid:
                child.returncode = os.waitstatus_to_exitcode(status)
                break
            if time.monotonic() >= deadline:
                child.kill()
                raise subprocess.TimeoutExpired(command, 120)
            time.sleep(0.005)
        output.seek(0)
        message = output.read().strip()
        if child.returncode:
            raise RuntimeError(f"{command[0]} exited {child.returncode}: {message}")
        if not message.startswith("compiler_ns="):
            raise RuntimeError(f"unexpected compiler output: {message!r}")
        ns = int(message.removeprefix("compiler_ns="))
        rss_bytes = usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024
        return ns, usage.ru_utime + usage.ru_stime, rss_bytes, time.perf_counter_ns() - start


def measure(workload, size, args, ledger):
    check_load()
    source, expected = program(workload, 0, size)
    input_nodes = source.count("(")
    output_nodes = expected.count("(")
    theory, phases = WORKLOADS[workload]
    with tempfile.TemporaryDirectory() as tmp:
        setup, schedule, expr, want = [Path(tmp, s) for s in ("setup", "run", "input", "expected")]
        setup.write_text(theory)
        expr.write_text(source)
        want.write_text(expected)
        schedule.write_text(f"(let $src (bench-seed))\n{phases}\n(extract $src)\n")
        modes = {
            "Rust": [ROOT / "target/release/examples/lean_comparison", "compiler", setup, schedule, expr, want],
            "Lean certified": [ROOT / "lean/.lake/build/bin/compiler-bench", workload, expr, want],
        }
        samples = {k: [] for k in modes}
        rng = Random(42)
        for n in range(args.warmup + args.runs):
            warmup = n < args.warmup
            trial = n + 1 if warmup else n - args.warmup + 1
            for attempt in range(1, 4):
                order = list(modes)
                rng.shuffle(order)
                pair = [run_compiler(modes[name], {
                    **os.environ, "LEAN_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"})
                    for name in order]
                # The timed pass is CPU-bound and nested inside the measured
                # process. A long wall-clock pause can include machine sleep.
                # Save both samples and repeat the whole pair in this case.
                accepted = all(ns <= 2 * cpu * 1e9 + 250_000_000 for ns, cpu, _, _ in pair)
                if ledger:
                    writer, stream = ledger
                    for position, (name, (ns, cpu, rss, wall)) in enumerate(zip(order, pair), 1):
                        writer.writerow([workload, size, input_nodes, output_nodes, warmup, trial,
                                         attempt, accepted, position, name, ns, wall, f"{cpu:.6f}", rss])
                    stream.flush()
                if accepted:
                    if not warmup:
                        for name, (ns, _, _, _) in zip(order, pair):
                            samples[name].append(ns / 1e6)
                    break
                print(f"Interrupted paired trial {trial} on {workload}/{size}, attempt {attempt}", flush=True)
                check_load()
            else:
                raise SystemExit("Three interrupted pairs; unable to measure an uninterrupted compiler pass")
        print(f"{workload} compiler, {size} leaves ({input_nodes} input / {output_nodes} output nodes)")
        baseline = median(samples["Rust"])
        for mode, values in samples.items():
            elapsed = median(values)
            q = quantiles(values, method="inclusive") if len(values) > 1 else [0, 0, 0]
            print(f"{mode:15} {elapsed:9.3f} ms  IQR {q[2] - q[0]:6.3f} ms  {elapsed / baseline:5.2f}x")
        return median(samples["Lean certified"]) / baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=["all", *WORKLOADS], default="all")
    parser.add_argument("--size", type=int, nargs="+", default=[128], help="one or more open IR leaf counts")
    parser.add_argument("--csv", type=Path, help="save every validated sample to a fresh CSV file")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    if min(*args.size, args.runs) < 1 or args.warmup < 0:
        parser.error("expected a positive size and run count and a nonnegative warmup")
    selected = WORKLOADS if args.workload == "all" else [args.workload]
    with ExitStack() as stack:
        ledger = None
        if args.csv:
            stream = stack.enter_context(args.csv.open("x", newline=""))
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["workload", "leaves", "input_nodes", "output_nodes", "warmup",
                             "trial", "attempt", "accepted_pair", "position", "engine", "compiler_ns",
                             "process_wall_ns", "process_cpu_s", "peak_rss_bytes"])
            ledger = writer, stream
        ratios = [measure(name, size, args, ledger) for size in args.size for name in selected]
    if args.enforce and any(ratio > 1.30 for ratio in ratios):
        raise SystemExit("Certified compiler exceeded 1.30x Rust time")


if __name__ == "__main__":
    main()
