#!/usr/bin/env python3
"""Measure complete native solver processes on the pinned ParaBit CAV corpus."""
import argparse
import csv
import math
import os
import re
import random
import ctypes
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from bench.compare import check_load

MODES = ("egg-track", "egg-explain", "lean-rel", "lean-cert")
ALL_MODES = (*MODES, "egg-coverage", "lean-staged")
FIELDS = ["case", "mode", "trial", "status", "wall_ms", "solver_ms", "runner_ms", "detail"]


def command(mode, case, args):
    binaries = HERE.parent / ".lake/build/bin"
    if mode.startswith("egg-"):
        binary = args.artifact / "parabit/target/release/parabit-timed"
        return [binary, mode.removeprefix("egg-"), case]
    if mode in {"lean-cert", "lean-staged"}:
        options = ["--staged"] if mode == "lean-staged" else []
        return [binaries / "parabit-certified", *options, case, str(args.cert_rounds)]
    return [binaries / "parabit-relational", HERE / "theory.egg", case, str(args.rounds)]


def resident_bytes(pid):
    if sys.platform == "darwin":
        # struct proc_taskinfo starts with two uint64 fields: virtual and resident size.
        buf = ctypes.create_string_buffer(256)
        used = LIBPROC.proc_pidinfo(pid, 4, 0, buf, len(buf))
        return int.from_bytes(buf.raw[8:16], sys.byteorder) if used >= 16 else 0
    try:
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except FileNotFoundError:
        return 0


LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib") if sys.platform == "darwin" else None


def run_limited(cmd, seconds, *, env=None, cwd=None, stdin=None, max_rss=8 * 1024**3):
    """Capture a single worker and preserve its output when an external limit fires."""
    if env is None:
        env = {**os.environ, "LEAN_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1"}
    start = time.perf_counter_ns()
    with subprocess.Popen(cmd, env=env, cwd=cwd, stdin=stdin, stderr=subprocess.PIPE, stdout=subprocess.PIPE) as child:
        deadline = start / 1e9 + seconds
        while True:
            left = deadline - time.perf_counter()
            try:
                out, err = child.communicate(timeout=max(0, min(0.1, left)))
                return child.returncode, round((time.perf_counter_ns() - start) / 1e6, 3), out, err
            except subprocess.TimeoutExpired:
                limit = resident_bytes(child.pid) > max_rss
                if not limit and time.perf_counter() < deadline:
                    continue
                child.kill()
                out, err = child.communicate()
                return "memory-limit" if limit else "timeout", round((time.perf_counter_ns() - start) / 1e6, 3), out, err


def measure(mode, case, args):
    status, elapsed, out, err = run_limited(command(mode, case, args), args.seconds)
    if isinstance(status, str):
        marker = re.search(rb"^search=(true|false) ([0-9]+)$", err, re.M)
        return status, elapsed, None, None, marker[0].decode() if marker else ""
    if status:
        msg = err.decode(errors="replace").strip()
        return "error", elapsed, None, None, msg[-500:] or f"exit {status}"
    # The upstream width-condition code also writes diagnostics to stdout.
    count = 3 if mode.startswith("egg-") else 2
    status, *times = out.decode().strip().splitlines()[-count:]
    if status not in {"solved", "round-limit", "unproved"}:
        raise ValueError(f"unexpected output {out!r}")
    times = dict(t.split("=") for t in times)
    solver_ms = int(times["solver_ns"]) / 1e6
    runner_ms = int(times["runner_ns"]) / 1e6 if "runner_ns" in times else None
    return status, round(elapsed, 3), solver_ms, runner_ms, ""


def run(args):
    base = args.artifact / "benchmarks"
    cases = sorted(base.glob("*/bwlang/*.bwlang"))
    if not cases:
        raise SystemExit("no artifact BWLang cases found")
    modes = args.mode or MODES
    if args.case_list or args.unfinished:
        if args.case_list:
            selected = set(args.case_list.read_text().splitlines())
        else:
            with args.unfinished.open() as previous:
                selected = {r["case"] for r in csv.DictReader(previous)
                            if r["mode"] == "egg-explain" and r["status"] in {"timeout", "memory-limit"}}
        cases = [p for p in cases if p.relative_to(base).as_posix() in selected]
        if len(cases) != len(selected):
            raise SystemExit("case list does not match the artifact")
    check_load()
    # Warm the executable mappings before timing fresh processes. Each new
    # Rust process still initializes its own quanta clock outside solver_ns.
    if cases and not (args.case_list or args.unfinished):
        for mode in modes:
            measure(mode, cases[0], args)
    rng = random.Random(42)
    with args.output.open("x", newline="") as out:
        writer = csv.DictWriter(out, FIELDS, lineterminator="\n")
        writer.writeheader()
        for trial in range(1, args.trials + 1):
            rng.shuffle(cases)
            for i, case in enumerate(cases):
                if i % 25 == 0:
                    check_load()
                order = list(modes)
                rng.shuffle(order)
                for mode in order:
                    status, ms, solver_ms, runner_ms, detail = measure(mode, case, args)
                    writer.writerow(dict(case=case.relative_to(base).as_posix(), mode=mode,
                                         trial=trial, status=status, wall_ms=ms, solver_ms=solver_ms, runner_ms=runner_ms, detail=detail))
                    out.flush()
            print(f"Completed trial {trial} over {len(cases)} cases", flush=True)


def summarize(paths):
    runs = defaultdict(list)
    trials = defaultdict(set)
    for path in paths:
        with path.open() as f:
            reader = csv.DictReader(f)
            if not {"mode", "case", "trial", "status", "solver_ms"}.issubset(reader.fieldnames or []):
                raise SystemExit(f"{path}: expected a ParaBit timing CSV")
            for r in reader:
                mode, case = r["mode"], r["case"]
                if mode not in ALL_MODES or not case:
                    raise SystemExit(f"{path}:{reader.line_num}: invalid mode or case")
                try:
                    trial = int(r["trial"])
                except (TypeError, ValueError):
                    raise SystemExit(f"{path}:{reader.line_num}: invalid trial {r['trial']!r}") from None
                if trial < 1 or trial in trials[mode, case]:
                    raise SystemExit(f"{path}:{reader.line_num}: invalid or duplicate {mode} trial {trial} for {case}")
                trials[mode, case].add(trial)
                runs[mode, case].append(r)
            if reader.line_num <= 1:
                raise SystemExit(f"{path}: empty timing sweep")

    # All selected engines in an ordinary comparison run the same cases and
    # trials. The longer follow-up reruns its own subset under a separate budget.
    for family in (MODES, ("egg-coverage",), ("lean-staged",)):
        cases = {case for mode, case in runs if mode in family}
        modes = {mode for mode, _ in runs if mode in family}
        if not cases:
            continue
        last_trial = max(max(ids) for (mode, _), ids in trials.items() if mode in family)
        for mode in sorted(modes):
            for case in sorted(cases):
                ids = trials.get((mode, case), set())
                if len(ids) != last_trial:
                    raise SystemExit(f"Incomplete sweep: {mode} {case} has trials {sorted(ids)}, expected 1..{last_trial}")
    common = {}
    field = "solver_ms"
    for mode in ALL_MODES:
        subset = {case: samples for (m, case), samples in runs.items() if mode == m}
        if not subset:
            continue
        solved = {c: statistics.median(float(r[field]) for r in rs)
                  for c, rs in subset.items() if all(r["status"] == "solved" for r in rs)}
        counts = Counter(tuple(sorted({r["status"] for r in rs})) for rs in subset.values())
        print(f"{mode:12} {len(solved)}/{len(subset)}", dict(counts))
        common[mode] = solved
    for lean in ("lean-rel", "lean-cert", "lean-staged"):
        for egg in ("egg-track", "egg-explain"):
            if lean not in common or egg not in common:
                continue
            joint = sorted(common[lean].keys() & common[egg].keys())
            if not joint:
                continue
            a, b = ([common[m][c] for c in joint] for m in (lean, egg))
            ratio = math.exp(statistics.fmean(math.log(x / y) for x, y in zip(a, b)))
            within = sum(x <= 1.30 * y for x, y in zip(a, b))
            print(f"{lean} / {egg}: {len(joint)} joint solves, solver medians "
                  f"{statistics.median(a):.3f} / {statistics.median(b):.3f} ms, "
                  f"geomean paired ratio {ratio:.3f}, {within}/{len(joint)} within 1.30x")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="action", required=True)
    bench = subs.add_parser("run")
    bench.add_argument("artifact", type=Path)
    bench.add_argument("output", type=Path)
    bench.add_argument("--mode", choices=ALL_MODES, action="append")
    bench.add_argument("--seconds", type=float, default=2.0, help="wall limit per process")
    bench.add_argument("--rounds", type=int, default=12)
    bench.add_argument("--cert-rounds", type=int, default=6)
    bench.add_argument("--trials", type=int, default=1)
    selection = bench.add_mutually_exclusive_group()
    selection.add_argument("--case-list", type=Path)
    selection.add_argument("--unfinished", type=Path, metavar="SWEEP_CSV",
                           help="retry egg explanations that reached the wall or memory limit in a previous sweep")
    summary = subs.add_parser("summary")
    summary.add_argument("csv", type=Path, nargs="+")
    args = p.parse_args()
    if args.action == "summary":
        summarize(args.csv)
    elif args.seconds <= 0 or min(args.trials, args.rounds, args.cert_rounds) < 1:
        p.error("budgets and number of trials must be positive")
    else:
        run(args)


if __name__ == "__main__":
    main()
