
#!/usr/bin/env python3
"""
bench.py — egglog benchmark harness

Commands:
  run <commit1> <commit2>   — benchmark two commits and save the diff report
  run --compare-unstaged    — benchmark HEAD vs current working-tree changes
  diff                      — print the most recently saved diff report

Improvement policy:
  IMPROVEMENT iff at least one benchmark improved >=3% AND net average change is negative.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
BENCH_DIR = REPO_ROOT / "benchmarks"
TEST_DIR = REPO_ROOT / "tests"
EGGLOG = REPO_ROOT / "target" / "release" / "egglog"

BENCHMARKS = [
    "hardboiled_conv1d_32.egg",
    "hardboiled_conv1d_128.egg",
    "luminal-llama.egg",
    "python_array_optimize.egg",
    "cykjson.egg",
    "eggcc-extraction.egg",
]

HYPERFINE_RUNS = 15
WARMUP_RUNS = 3
IMPROVE_THRESHOLD = 3.0
UNCHANGED_BAND = 0.5


# ── git helpers ───────────────────────────────────────────────────────────────


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def _resolve(ref: str) -> tuple[str, str, str]:
    """Return (full_hash, short_hash, commit_subject) for any ref."""
    full = _git("rev-parse", ref)
    short = _git("rev-parse", "--short", ref)
    subject = _git("log", "-1", "--format=%s", full)
    return full, short, subject


def _current_ref() -> str:
    """Branch name if on a branch, else full hash (detached HEAD)."""
    try:
        return _git("symbolic-ref", "--short", "HEAD")
    except subprocess.CalledProcessError:
        return _git("rev-parse", "HEAD")


# ── benchmarking ─────────────────────────────────────────────────────────────


def benchmark_state(label: str) -> dict[str, tuple[float, float]]:
    """Build the current working tree and hyperfine all benchmarks.

    Returns {bench_name: (mean_s, stddev_s)}.
    """
    print(f"\n  [{label}] Building egglog (release)...")
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(REPO_ROOT / "Cargo.toml")],
        check=True,
    )

    results: dict[str, tuple[float, float]] = {}
    print(f"\n  [{label}] Benchmarking ({HYPERFINE_RUNS} runs each)\n")
    print(f"    {'Benchmark':<42} {'Mean (s)':>9}  {'± Stddev':>9}")
    print("    " + "─" * 62)

    for bench in BENCHMARKS:
        src = TEST_DIR / bench
        if not src.exists():
            print(f"    SKIP  {bench}")
            continue

        shell_cmd = shlex.join([str(EGGLOG), str(src)]) + " >/dev/null 2>&1"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_json = tf.name

        try:
            proc = subprocess.run(
                [
                    "hyperfine",
                    "--warmup",
                    str(WARMUP_RUNS),
                    "--runs",
                    str(HYPERFINE_RUNS),
                    "--export-json",
                    tmp_json,
                    shell_cmd,
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(f"    FAIL  {bench}")
                if proc.stderr:
                    print("          " + proc.stderr.splitlines()[0])
                continue

            with open(tmp_json) as f:
                hf = json.load(f)

            r = hf["results"][0]
            mean = r["mean"]
            stddev = r.get("stddev", 0.0)
            results[bench] = (mean, stddev)
            print(f"    {bench:<42} {mean:>9.3f}  ±{stddev:>8.3f}")
        finally:
            os.unlink(tmp_json)

    return results


# ── diff formatting ───────────────────────────────────────────────────────────


def format_diff(
    label1: str,
    desc1: str,
    times1: dict[str, tuple[float, float]],
    label2: str,
    desc2: str,
    times2: dict[str, tuple[float, float]],
    timestamp: str,
) -> str:
    lines: list[str] = []
    lines += [
        "# Benchmark diff",
        "",
        f"  Generated  : {timestamp}",
        f"  Baseline   : {label1}  —  {desc1}",
        f"  Comparison : {label2}  —  {desc2}",
        "",
        f"  {'Benchmark':<42} {'Before (s)':>10}  {'After (s)':>10}  {'Δ (s)':>8}  {'Δ %':>7}",
        "  " + "─" * 88,
    ]

    faster = slower = unchanged = missing = 0
    measured_pcts: list[float] = []
    has_big_improvement = False

    for bench in BENCHMARKS:
        t1 = times1.get(bench)
        t2 = times2.get(bench)

        if t1 is None or t2 is None:
            lines.append(f"  {bench:<42} {'—':>10}  {'—':>10}  {'—':>8}  {'—':>7}  (missing)")
            missing += 1
            continue

        p, c = t1[0], t2[0]
        delta = c - p
        pct = delta / p * 100 if p else 0.0
        sign = "+" if delta >= 0 else ""
        measured_pcts.append(pct)

        if abs(pct) < UNCHANGED_BAND:
            marker = "  ·"
            unchanged += 1
        elif delta < 0:
            marker = "  ▼ faster"
            faster += 1
            if -pct >= IMPROVE_THRESHOLD:
                has_big_improvement = True
        else:
            marker = "  ▲ slower"
            slower += 1

        lines.append(f"  {bench:<42} {p:>10.3f}  {c:>10.3f}" f"  {sign}{delta:>7.3f}  {sign}{pct:>6.1f}%{marker}")

    lines += [
        "",
        f"  Summary: {faster} faster  ·  {slower} slower  ·  {unchanged} unchanged" f"  ·  {missing} missing",
    ]

    if measured_pcts:
        avg_pct = sum(measured_pcts) / len(measured_pcts)
        avg_sign = "+" if avg_pct >= 0 else ""
        lines.append(f"  Overall average Δ: {avg_sign}{avg_pct:.2f}%")
        lines.append("")

        if has_big_improvement and avg_pct < 0:
            lines.append(
                f"  VERDICT: IMPROVEMENT\n"
                f"    At least one benchmark improved ≥{IMPROVE_THRESHOLD:.0f}% and the net\n"
                f"    average change is negative — overall performance is better."
            )
        elif has_big_improvement:
            lines.append(
                f"  VERDICT: MIXED\n"
                f"    A benchmark improved ≥{IMPROVE_THRESHOLD:.0f}% but the net average change is\n"
                f"    positive (regression elsewhere). Not classified as an improvement."
            )
        elif avg_pct < 0:
            lines.append(
                f"  VERDICT: MODEST GAIN\n"
                f"    Net average change is negative but no single benchmark improved\n"
                f"    ≥{IMPROVE_THRESHOLD:.0f}%. Not classified as a significant improvement."
            )
        else:
            lines.append(
                f"  VERDICT: REGRESSION\n" f"    Net average change is positive — overall performance worsened."
            )

    lines.append("")
    return "\n".join(lines)


def _save_diff(report: str, slug: str, timestamp: str) -> Path:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    safe_ts = timestamp.replace(":", "-")
    out_path = BENCH_DIR / f"diff_{slug}_{safe_ts}.md"
    out_path.write_text(report)
    return out_path


# ── run ──────────────────────────────────────────────────────────────────────


def cmd_run(argv: list[str]) -> None:
    if not shutil.which("hyperfine"):
        sys.exit("hyperfine not found — install with: cargo install hyperfine")

    if argv == ["--compare-unstaged"]:
        _run_compare_unstaged()
    elif len(argv) == 2:
        _run_compare_commits(argv[0], argv[1])
    else:
        sys.exit("Usage:\n" "  bench.py run <commit1> <commit2>\n" "  bench.py run --compare-unstaged")


def _run_compare_commits(ref1: str, ref2: str) -> None:
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        sys.exit("Working tree is dirty. Stash or commit your changes first.")

    full1, short1, subject1 = _resolve(ref1)
    full2, short2, subject2 = _resolve(ref2)
    orig = _current_ref()

    try:
        _git("checkout", full1)
        times1 = benchmark_state(short1)
        _git("checkout", full2)
        times2 = benchmark_state(short2)
    finally:
        _git("checkout", orig)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    report = format_diff(
        f"{short1} ({full1[:12]})",
        subject1,
        times1,
        f"{short2} ({full2[:12]})",
        subject2,
        times2,
        timestamp,
    )
    print(report)
    out = _save_diff(report, f"{short1}_vs_{short2}", timestamp)
    print(f"  Saved to {out}")


def _run_compare_unstaged() -> None:
    diff_stat = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "HEAD", "--stat"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not diff_stat:
        sys.exit("No working-tree changes found relative to HEAD.")

    full_head, short_head, subject_head = _resolve("HEAD")

    # Benchmark working-tree state (with changes) first, then HEAD as baseline.
    times_wt = benchmark_state("working-tree")

    _git("stash", "push", "--include-untracked", "-m", "bench.py temp stash")
    try:
        times_head = benchmark_state(short_head)
    finally:
        _git("stash", "pop")

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    report = format_diff(
        f"{short_head} ({full_head[:12]})",
        subject_head,
        times_head,
        "working-tree",
        "(unstaged changes)",
        times_wt,
        timestamp,
    )
    print(report)
    out = _save_diff(report, f"{short_head}_vs_unstaged", timestamp)
    print(f"  Saved to {out}")


# ── diff ──────────────────────────────────────────────────────────────────────


def cmd_diff() -> None:
    reports = sorted(BENCH_DIR.glob("diff_*.md"))
    if not reports:
        sys.exit("No diff reports found. Run 'bench.py run' first.")
    latest = reports[-1]
    print(f"  (from {latest.name})\n")
    print(latest.read_text())


# ── dispatch ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        sys.exit(1)

    cmd, *rest = argv
    if cmd == "run":
        cmd_run(rest)
    elif cmd == "diff":
        cmd_diff()
    else:
        print(__doc__)
        sys.exit(1)

