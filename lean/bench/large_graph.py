#!/usr/bin/env python3
"""Scale recursive path closure, a cyclic join, and the repository's calculus rules."""
import argparse
import csv
import os
import resource
import subprocess
import threading
import time
from pathlib import Path
from random import Random
from statistics import median
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.compare import ROOT, check_load, math_workload, paths_workload
from parabit.compare import resident_bytes

SCALES = {"paths": (512, 1024, 2048), "triangle": (16, 32, 64),
          "windmill": (256, 1024, 4096), "math": (12,), "cyk": (801, 7821)}
ENGINES = {"rust": [ROOT / "target/release/examples/lean_comparison", "signal"],
           "lean": [ROOT / "lean/.lake/build/bin/lean-egglog", "--signal-raw"]}


def triangle_input(size, dense):
    # One directed relation for all three atoms of a cyclic self-join.
    if dense:
        # Directed A -> B -> C -> A. Any choice of one vertex from each
        # partition gives three ordered triangles, one per starting vertex.
        edges = ((side * size + a, (side + 1) % 3 * size + b)
                 for side in range(3) for a in range(size) for b in range(size))
        expected = (3 * size**2, 3 * size**3)
        witnesses = [(0, size, 2 * size), (3 * size - 1, size - 1, 2 * size - 1)]
    else:
        # Every blade is one undirected triangle with a shared center.
        # A pairwise join at the hub has quadratic intermediate cardinality.
        edges = ((x, y) for i in range(size)
                 for a, b in [(0, 2*i+1), (0, 2*i+2), (2*i+1, 2*i+2)]
                 for x, y in [(a, b), (b, a)])
        expected = (6 * size, 6 * size)
        witnesses = [(0, 1, 2), (2 * size, 2 * size - 1, 0)]
    setup = ["(relation edge (i64 i64))", "(relation triangle (i64 i64 i64))",
             "(rule ((edge x y) (edge y z) (edge z x)) ((triangle x y z)))"]
    setup.extend(f"(edge {x} {y})" for x, y in edges)
    checks = [f"(check (triangle {x} {y} {z}))" for x, y, z in witnesses]
    checks.extend(["(print-size edge)", "(print-size triangle)"])
    return ("\n".join(setup), "(run 2)", "\n".join(checks)), expected


def workload(name, size):
    if name == "cyk":
        from bench.cyk import cyk_workload
        return cyk_workload(size)
    if name == "math":
        # Preserve the initial terms, rewrite theory and all 13 count checks.
        return math_workload(size), None
    if name == "paths":
        return paths_workload(size), (size, size * (size + 1) // 2)
    return triangle_input(size, name == "triangle")


def run_observed(command, seconds, max_rss):
    # A single parent monotonic clock observes both executables. Wait for the
    # worker to finish setup, then time from releasing it to its end-of-run
    # signal. Checking, output formatting and teardown follow the end signal.
    start = time.perf_counter_ns()
    deadline = time.monotonic() + seconds
    env = {**os.environ, "LEAN_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}
    spans = {}
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, env=env) as child:
        def exchange():
            spans["ready"] = child.stdout.readline()
            if spans["ready"] == b"ready\n":
                released = time.perf_counter_ns()
                try:
                    child.stdin.write(b"go\n")
                    child.stdin.flush()
                except BrokenPipeError:
                    return
                spans["done"] = child.stdout.readline()
                spans["schedule"] = (time.perf_counter_ns() - released) / 1e6
            spans["out"] = child.stdout.read()
            spans["eof"] = time.perf_counter_ns()
        reader = threading.Thread(target=exchange)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        reader.start()
        while True:
            try:
                status = child.wait(timeout=min(0.1, max(0, deadline - time.monotonic())))
                break
            except subprocess.TimeoutExpired:
                high_rss = resident_bytes(child.pid) > max_rss
                if not high_rss and time.monotonic() < deadline:
                    continue
                child.kill()
                child.wait()
                status = "memory-limit" if high_rss else "timeout"
                break
        reader.join()
        elapsed = (spans.get("eof", time.perf_counter_ns()) - start) / 1e6
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime) * 1000
        errors.seek(0)
        detail = errors.read().decode(errors="replace")[-500:]
        return status, elapsed, cpu, spans, detail


def sample(engine, paths, expected, count, args):
    status, wall, cpu, spans, detail = run_observed(
        [*ENGINES[engine], *paths], args.seconds, int(args.rss_gib * 1024**3))
    if status != 0:
        return (status if isinstance(status, str) else "error"), wall, cpu, "", "", (), detail
    if spans.get("ready") != b"ready\n" or spans.get("done") != b"done\n":
        raise RuntimeError(f"{engine} failed timing handshake: {spans}")
    *counts, timing = spans["out"].decode().splitlines()
    if not timing.startswith("schedule_ns=") or len(counts) != count:
        raise RuntimeError(f"{engine} malformed benchmark output: {spans}")
    rows = tuple(map(int, counts))
    ms = spans["schedule"]
    child_ms = int(timing.removeprefix("schedule_ns=")) / 1e6
    if expected is not None and rows != expected:
        return "mismatch", wall, cpu, ms, child_ms, rows, f"expected {expected}"
    return "ok", wall, cpu, ms, child_ms, rows, ""


def run(args, selections=None, make_workload=workload):
    check_load()
    failed = False
    if selections is None:
        selections = [(name, size) for name, sizes in SCALES.items() for size in sizes]
        if args.case:
            selections = [(args.case, s) for s in args.size or SCALES[args.case]]
    rng = Random(42)
    with tempfile.TemporaryDirectory() as tmp, args.output.open("x", newline="") as file:
        fields = "case size engine trial status wall_ms cpu_ms schedule_ms child_schedule_ms counts detail host_idle_pct".split()
        writer = csv.DictWriter(file, fields, lineterminator="\n")
        writer.writeheader()
        paths = [Path(tmp, name + ".egg") for name in ("setup", "run", "checks")]
        for name, size in selections:
            parts, expected = make_workload(name, size)
            for path, content in zip(paths, parts):
                path.write_text(content + "\n")
            count = parts[2].count("(print-size ")
            active = list(ENGINES)
            times = {mode: [] for mode in active}
            for trial in range(1 - args.warmup, args.runs + 1):
                if not active:
                    break
                idle_pct = check_load()
                engines = active.copy()
                rng.shuffle(engines)
                for engine in engines:
                    status, wall, cpu, ms, child_ms, rows, detail = sample(engine, paths, expected, count, args)
                    writer.writerow(dict(case=name, size=size, engine=engine, trial=trial,
                                         status=status, wall_ms=wall, cpu_ms=cpu, schedule_ms=ms, child_schedule_ms=child_ms,
                                         counts=" ".join(map(str, rows)), detail=detail, host_idle_pct=idle_pct))
                    file.flush()
                    if status != "ok":
                        failed = True
                        print(f"{name} {size}, {engine}, trial {trial}: {status} {detail}", flush=True)
                        active.remove(engine)
                    else:
                        if expected is None:
                            expected = rows  # Calculus cross-checks every table against the first run.
                        if trial > 0:
                            times[engine].append(ms)
            medians = {mode: round(median(v), 3) for mode, v in times.items() if len(v) == args.runs}
            ratio = ""
            if "rust" in medians and "lean" in medians:
                ratio = f", Lean/Rust {median(times['lean']) / median(times['rust']):.3f}"
            print(f"{name} {size}, schedule ms {medians}{ratio}, table sizes {expected}", flush=True)
    return int(failed)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--case", choices=SCALES)
    p.add_argument("--size", type=int, action="append")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1, help="warmup trials numbered zero and below")
    p.add_argument("--seconds", type=float, default=30, help="wall limit per process")
    p.add_argument("--rss-gib", type=float, default=6, help="resident-memory ceiling sampled every 100 ms")
    args = p.parse_args()
    if min(args.runs, args.seconds, args.rss_gib) <= 0 or args.warmup < 0:
        p.error("expected positive budgets and a nonnegative warmup count")
    if args.size and (not args.case or min(args.size) < 1):
        p.error("--size requires --case and positive sizes")
    sys.exit(run(args))
