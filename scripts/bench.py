#!/usr/bin/env python3
"""
bench.py — egglog benchmark harness

Commands:
  run <commit1> <commit2>   — benchmark two commits and save the diff report
  run --compare-unstaged    — benchmark HEAD vs current working-tree changes
  diff                      — print the most recently saved diff report

Each state is built in its own temporary git worktree (the unstaged state
builds in the main checkout), so the checkout you are working in is never
touched. The two binaries' runs are interleaved per benchmark — alternating
back-to-back, swapping which binary goes first every round — so slow drift in
machine state (thermals, cache pressure) hits both states equally instead of
biasing whichever was measured last.

Temporary worktrees are removed on exit, including on SIGINT/SIGTERM; ones
orphaned by a hard kill (SIGKILL) are swept up by the next invocation.

Improvement policy:
  IMPROVEMENT iff at least one benchmark improved >=3% AND net average change is negative.
"""

import atexit
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
BENCH_DIR = REPO_ROOT / "benchmarks"
TEST_DIR = REPO_ROOT / "tests"

BENCHMARKS = [
    "hardboiled_conv1d_32.egg",
    "hardboiled_conv1d_128.egg",
    "luminal-llama.egg",
    "python_array_optimize.egg",
    "cykjson.egg",
    "eggcc-extraction.egg",
    # "gemma.egg",
    # "gemma4_moe.egg",
    "llama.egg",
    "paged_llama.egg",
    "qwen.egg",
    "qwen3_moe.egg",
    "whisper.egg",
]

TIMED_RUNS = 15
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


# ── temporary worktrees ──────────────────────────────────────────────────────

_WORKTREE_PREFIX = "egglog-bench-"
_temp_worktrees: list[Path] = []


def _remove_worktree(path: Path) -> None:
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force", str(path)],
        capture_output=True,
    )
    # `git worktree remove` refuses in some states (e.g. a build was killed
    # mid-write); make sure the directory is gone either way, then drop any
    # dangling registration.
    shutil.rmtree(path, ignore_errors=True)
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "worktree", "prune"],
        capture_output=True,
    )


def _cleanup_worktrees() -> None:
    while _temp_worktrees:
        _remove_worktree(_temp_worktrees.pop())


def _sweep_stale_worktrees() -> None:
    """Remove bench worktrees left behind by a previous killed run."""
    out = _git("worktree", "list", "--porcelain")
    for line in out.splitlines():
        if line.startswith("worktree "):
            path = Path(line[len("worktree ") :])
            if path.name.startswith(_WORKTREE_PREFIX):
                print(f"  Removing stale bench worktree {path}")
                _remove_worktree(path)


def _add_worktree(commit: str, label: str) -> Path:
    """Check `commit` out into a fresh temporary worktree and return its path."""
    path = Path(tempfile.mkdtemp(prefix=f"{_WORKTREE_PREFIX}{label}-"))
    path.rmdir()  # `git worktree add` wants to create the directory itself.
    _temp_worktrees.append(path)
    _git("worktree", "add", "--detach", str(path), commit)
    return path


def _exit_on_signal(signum: int, _frame) -> None:
    # Turn the signal into a normal exit so `finally` blocks and the atexit
    # worktree cleanup run.
    sys.exit(128 + signum)


atexit.register(_cleanup_worktrees)
for _sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
    if _sig is not None:
        signal.signal(_sig, _exit_on_signal)


# ── benchmarking ─────────────────────────────────────────────────────────────


def _build(manifest_dir: Path, label: str) -> Path:
    """Build egglog (release) in `manifest_dir` and return the binary path."""
    print(f"\n  [{label}] Building egglog (release)...")
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(manifest_dir / "Cargo.toml")],
        check=True,
    )
    return manifest_dir / "target" / "release" / "egglog"


def _time_run(binary: Path, src: Path) -> float | None:
    """One timed run; returns wall-clock seconds, or None if the run failed."""
    start = time.perf_counter()
    proc = subprocess.run(
        [str(binary), str(src)],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    elapsed = time.perf_counter() - start
    return elapsed if proc.returncode == 0 else None


def benchmark_interleaved(
    label1: str,
    bin1: Path,
    label2: str,
    bin2: Path,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    """Time all benchmarks, interleaving runs of the two binaries.

    Both binaries run against the main checkout's `tests/` directory (same
    inputs for both states) with the repo root as working directory.

    Returns ({bench: (mean_s, stddev_s)}, {bench: (mean_s, stddev_s)}).
    """
    results1: dict[str, tuple[float, float]] = {}
    results2: dict[str, tuple[float, float]] = {}

    print(f"\n  Benchmarking ({TIMED_RUNS} interleaved runs each, {WARMUP_RUNS} warmup)\n")
    print(f"    {'Benchmark':<42} {label1:>15}  {label2:>15}")
    print("    " + "─" * 76)

    for bench in BENCHMARKS:
        src = TEST_DIR / bench
        if not src.exists():
            print(f"    SKIP  {bench}")
            continue

        times1: list[float] = []
        times2: list[float] = []
        failed = False

        for _ in range(WARMUP_RUNS):
            if _time_run(bin1, src) is None or _time_run(bin2, src) is None:
                failed = True
                break

        for i in range(TIMED_RUNS):
            if failed:
                break
            # Swap the order every round so neither binary always runs first.
            pair = [(bin1, times1), (bin2, times2)]
            if i % 2:
                pair.reverse()
            for binary, acc in pair:
                t = _time_run(binary, src)
                if t is None:
                    failed = True
                    break
                acc.append(t)

        if failed:
            print(f"    FAIL  {bench}")
            continue

        mean1, stddev1 = statistics.mean(times1), statistics.stdev(times1)
        mean2, stddev2 = statistics.mean(times2), statistics.stdev(times2)
        results1[bench] = (mean1, stddev1)
        results2[bench] = (mean2, stddev2)
        print(f"    {bench:<42} {mean1:>7.3f} ±{stddev1:>6.3f}  {mean2:>7.3f} ±{stddev2:>6.3f}")

    return results1, results2


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
    _sweep_stale_worktrees()
    try:
        if argv == ["--compare-unstaged"]:
            _run_compare_unstaged()
        elif len(argv) == 2:
            _run_compare_commits(argv[0], argv[1])
        else:
            sys.exit("Usage:\n" "  bench.py run <commit1> <commit2>\n" "  bench.py run --compare-unstaged")
    finally:
        _cleanup_worktrees()


def _run_compare_commits(ref1: str, ref2: str) -> None:
    full1, short1, subject1 = _resolve(ref1)
    full2, short2, subject2 = _resolve(ref2)

    wt1 = _add_worktree(full1, short1)
    wt2 = _add_worktree(full2, short2)
    bin1 = _build(wt1, short1)
    bin2 = _build(wt2, short2)
    times1, times2 = benchmark_interleaved(short1, bin1, short2, bin2)

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

    wt_head = _add_worktree(full_head, short_head)
    bin_head = _build(wt_head, short_head)
    bin_wt = _build(REPO_ROOT, "working-tree")
    times_head, times_wt = benchmark_interleaved(short_head, bin_head, "working-tree", bin_wt)

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
