#!/usr/bin/env python3
"""Benchmark math-microbenchmark across egglog thread counts.

The harness intentionally always passes ``--no-decomp``.  It builds the
release egglog binary by default, performs per-configuration warmups, then
interleaves measured runs to reduce ordering bias.  Aggregate results are
written as CSV and raw plus aggregate results are written as JSON.

Examples:

    python3 scripts/math_scalability.py
    python3 scripts/math_scalability.py --threads 1,2,4,8,12 --repetitions 7
    python3 scripts/math_scalability.py --no-build --output-dir /private/tmp
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    import resource
except ImportError:  # pragma: no cover - resource is available on Unix/macOS.
    resource = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "egglog"
DEFAULT_BENCHMARK = REPO_ROOT / "tests" / "math-microbenchmark.egg"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "target" / "math-scalability"
DEFAULT_THREADS = (1, 2, 4, 8, 12)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def thread_counts(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of integers"
        ) from error
    if not parsed or any(thread <= 0 for thread in parsed):
        raise argparse.ArgumentTypeError("thread counts must be greater than zero")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("thread counts must be unique")
    return parsed


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure math-microbenchmark scalability. Every benchmark command "
            "uses --no-decomp."
        )
    )
    parser.add_argument(
        "--threads",
        type=thread_counts,
        default=DEFAULT_THREADS,
        metavar="N,N,...",
        help="thread counts to measure (default: 1,2,4,8,12)",
    )
    parser.add_argument(
        "--warmups",
        type=nonnegative_int,
        default=1,
        help="warmup runs per thread count (default: 1)",
    )
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=5,
        help="measured runs per thread count (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=300.0,
        metavar="SECONDS",
        help="timeout for each egglog invocation (default: 300)",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help="egglog binary to run (default: target/release/egglog)",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=DEFAULT_BENCHMARK,
        help="benchmark input (default: tests/math-microbenchmark.egg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for timestamped CSV and JSON results",
    )
    parser.add_argument(
        "--label",
        default="",
        help="optional label stored in the output metadata",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="use the existing binary without running cargo build --release",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the build and benchmark commands without running them",
    )
    return parser.parse_args(argv)


def clean_runtime_environment() -> tuple[dict[str, str], list[str]]:
    """Remove ambient egglog logging/tuning settings for reproducible runs."""
    environment = os.environ.copy()
    removed = sorted(
        key
        for key in environment
        if key == "RUST_LOG" or key.startswith("EGGLOG_")
    )
    for key in removed:
        del environment[key]
    return environment, removed


def git_output(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *arguments],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def benchmark_command(
    binary: Path, benchmark: Path, threads: int
) -> list[str]:
    # Keep --no-decomp unconditional: results from this harness should remain
    # comparable while decomposition is disabled for this workload.
    return [
        str(binary),
        "--no-decomp",
        "-j",
        str(threads),
        str(benchmark),
    ]


def display_command(command: Sequence[str]) -> str:
    import shlex

    return shlex.join(command)


def child_cpu_times() -> tuple[float, float] | None:
    """Return cumulative (user, system) CPU seconds for completed children."""
    if resource is None:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    except (AttributeError, OSError):
        return None
    return usage.ru_utime, usage.ru_stime


def run_once(
    command: Sequence[str],
    *,
    environment: dict[str, str],
    timeout: float,
) -> dict[str, float | None]:
    cpu_before = child_cpu_times()
    started = time.perf_counter_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"benchmark exceeded the {timeout:g}s timeout: "
            f"{display_command(command)}"
        ) from error
    elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = f"\n{stderr}" if stderr else ""
        raise RuntimeError(
            f"benchmark exited with status {completed.returncode}: "
            f"{display_command(command)}{detail}"
        )
    cpu_after = child_cpu_times()
    user_cpu_seconds: float | None = None
    system_cpu_seconds: float | None = None
    if cpu_before is not None and cpu_after is not None:
        user_cpu_seconds = max(0.0, cpu_after[0] - cpu_before[0])
        system_cpu_seconds = max(0.0, cpu_after[1] - cpu_before[1])
    total_cpu_seconds = (
        user_cpu_seconds + system_cpu_seconds
        if user_cpu_seconds is not None and system_cpu_seconds is not None
        else None
    )
    return {
        "elapsed_seconds": elapsed_seconds,
        "user_cpu_seconds": user_cpu_seconds,
        "system_cpu_seconds": system_cpu_seconds,
        "cpu_seconds": total_cpu_seconds,
        "effective_utilized_cores": (
            total_cpu_seconds / elapsed_seconds
            if total_cpu_seconds is not None
            else None
        ),
    }


def summarize(
    samples: dict[int, list[dict[str, float | None]]],
    ordered_threads: Sequence[int],
) -> list[dict[str, Any]]:
    baseline_threads = min(ordered_threads)
    baseline_mean = statistics.fmean(
        sample["elapsed_seconds"] for sample in samples[baseline_threads]
    )
    results: list[dict[str, Any]] = []
    for threads in ordered_threads:
        thread_samples = samples[threads]
        values = [sample["elapsed_seconds"] for sample in thread_samples]
        assert all(value is not None for value in values)
        elapsed_values = [float(value) for value in values]
        mean = statistics.fmean(elapsed_values)
        speedup = baseline_mean / mean
        efficiency = speedup * baseline_threads / threads
        cpu_available = all(
            sample["cpu_seconds"] is not None for sample in thread_samples
        )
        mean_user_cpu = (
            statistics.fmean(
                float(sample["user_cpu_seconds"]) for sample in thread_samples
            )
            if cpu_available
            else None
        )
        mean_system_cpu = (
            statistics.fmean(
                float(sample["system_cpu_seconds"]) for sample in thread_samples
            )
            if cpu_available
            else None
        )
        mean_cpu = (
            statistics.fmean(
                float(sample["cpu_seconds"]) for sample in thread_samples
            )
            if cpu_available
            else None
        )
        results.append(
            {
                "threads": threads,
                "repetitions": len(elapsed_values),
                "mean_seconds": mean,
                "median_seconds": statistics.median(elapsed_values),
                "min_seconds": min(elapsed_values),
                "max_seconds": max(elapsed_values),
                "stddev_seconds": statistics.stdev(elapsed_values)
                if len(elapsed_values) > 1
                else 0.0,
                "mean_user_cpu_seconds": mean_user_cpu,
                "mean_system_cpu_seconds": mean_system_cpu,
                "mean_cpu_seconds": mean_cpu,
                "effective_utilized_cores": mean_cpu / mean
                if mean_cpu is not None
                else None,
                "speedup_vs_baseline": speedup,
                "parallel_efficiency_pct": efficiency * 100.0,
            }
        )
    return results


CSV_COLUMNS = (
    "threads",
    "repetitions",
    "mean_seconds",
    "median_seconds",
    "min_seconds",
    "max_seconds",
    "stddev_seconds",
    "mean_user_cpu_seconds",
    "mean_system_cpu_seconds",
    "mean_cpu_seconds",
    "effective_utilized_cores",
    "speedup_vs_baseline",
    "parallel_efficiency_pct",
)


def write_outputs(
    output_dir: Path, document: dict[str, Any]
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"math-scalability-{timestamp}"
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"

    with csv_path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(document["results"])
    with json_path.open("x", encoding="utf-8") as output:
        json.dump(document, output, indent=2, sort_keys=True)
        output.write("\n")
    return csv_path, json_path


def print_summary(results: Sequence[dict[str, Any]], baseline_threads: int) -> None:
    print()
    print("math-microbenchmark scalability (--no-decomp)")
    print(
        f"{'threads':>7}  {'mean (s)':>10}  {'stddev':>10}  "
        f"{'speedup':>8}  {'efficiency':>10}  {'CPU cores':>9}"
    )
    for result in results:
        cpu_cores = result["effective_utilized_cores"]
        cpu_cores_display = f"{cpu_cores:.2f}" if cpu_cores is not None else "n/a"
        print(
            f"{result['threads']:>7d}  "
            f"{result['mean_seconds']:>10.4f}  "
            f"{result['stddev_seconds']:>10.4f}  "
            f"{result['speedup_vs_baseline']:>7.2f}x  "
            f"{result['parallel_efficiency_pct']:>9.1f}%  "
            f"{cpu_cores_display:>9}"
        )
    print(
        f"Speedup and efficiency use the {baseline_threads}-thread mean as "
        "the baseline. CPU cores is (child user + system CPU time) / wall time."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    binary = args.binary.expanduser().resolve()
    benchmark = args.benchmark.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    environment, removed_environment = clean_runtime_environment()
    build_command = [
        "cargo",
        "build",
        "--release",
        "--manifest-path",
        str(REPO_ROOT / "Cargo.toml"),
        "--bin",
        "egglog",
    ]

    if args.dry_run:
        if not args.no_build:
            print(display_command(build_command))
        for threads in args.threads:
            print(
                f"# {args.warmups} warmup(s), {args.repetitions} measured run(s)"
            )
            print(display_command(benchmark_command(binary, benchmark, threads)))
        print(f"# output directory: {output_dir}")
        return 0

    if not benchmark.is_file():
        raise RuntimeError(f"benchmark input does not exist: {benchmark}")
    if not args.no_build:
        print("Building release egglog...", flush=True)
        subprocess.run(build_command, cwd=REPO_ROOT, check=True)
    if not binary.is_file():
        raise RuntimeError(
            f"egglog binary does not exist: {binary} "
            "(remove --no-build or pass --binary)"
        )

    print(
        "Benchmarking "
        + ", ".join(f"{threads}t" for threads in args.threads)
        + f" with {args.warmups} warmup(s) and "
        + f"{args.repetitions} measured run(s) each.",
        flush=True,
    )
    if removed_environment:
        print(
            "Ignoring ambient runtime settings: "
            + ", ".join(removed_environment),
            flush=True,
        )

    for threads in args.threads:
        command = benchmark_command(binary, benchmark, threads)
        for warmup in range(args.warmups):
            print(
                f"  warmup {warmup + 1}/{args.warmups}: {threads} threads",
                flush=True,
            )
            run_once(command, environment=environment, timeout=args.timeout)

    samples = {threads: [] for threads in args.threads}
    raw_samples: list[dict[str, Any]] = []
    # Alternate traversal direction each round to reduce monotonic thermal and
    # background-load bias without adding random, irreproducible ordering.
    for repetition in range(args.repetitions):
        order = args.threads if repetition % 2 == 0 else tuple(reversed(args.threads))
        for threads in order:
            sample = run_once(
                benchmark_command(binary, benchmark, threads),
                environment=environment,
                timeout=args.timeout,
            )
            samples[threads].append(sample)
            raw_samples.append(
                {
                    "threads": threads,
                    "repetition": repetition + 1,
                    **sample,
                }
            )
            elapsed = sample["elapsed_seconds"]
            cpu_cores = sample["effective_utilized_cores"]
            cpu_display = (
                f", {cpu_cores:.2f} effective CPU cores"
                if cpu_cores is not None
                else ""
            )
            print(
                f"  run {repetition + 1}/{args.repetitions}: "
                f"{threads:>2} threads  {elapsed:.4f}s{cpu_display}",
                flush=True,
            )

    results = summarize(samples, args.threads)
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "benchmark": str(benchmark),
        "binary": str(binary),
        "no_decomp": True,
        "threads": list(args.threads),
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "timeout_seconds": args.timeout,
        "command_template": [
            str(binary),
            "--no-decomp",
            "-j",
            "{threads}",
            str(benchmark),
        ],
        "git": {
            "commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain")),
        },
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "cpu_time_available": child_cpu_times() is not None,
        "effective_utilized_cores_definition": (
            "(child user CPU seconds + child system CPU seconds) / wall seconds"
        ),
        "ignored_environment_variables": removed_environment,
    }
    document = {
        "metadata": metadata,
        "samples": raw_samples,
        "results": results,
    }
    csv_path, json_path = write_outputs(output_dir, document)

    print_summary(results, min(args.threads))
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
