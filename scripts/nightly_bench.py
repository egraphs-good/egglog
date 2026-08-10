#!/usr/bin/env python3
"""
nightly_bench.py — egglog nightly benchmark harness

Builds the release `egglog` binary, runs every benchmark `.egg` file under
`tests/` through `hyperfine` in several configurations, and writes a
self-contained HTML report (plus a machine-readable `results.json`) into an
output directory.

Each program is benchmarked in these configurations:

  * standard    — `egglog -j 1 <file>`          (single thread, the baseline)
  * 2 threads   — `egglog -j 2 <file>`
  * 4 threads   — `egglog -j 4 <file>`
  * 8 threads   — `egglog -j 8 <file>`
  * proof       — `egglog --proof-testing <file>` (single thread), for programs
                  that support proofs (mirrors `file_supports_proofs` minus the
                  known-unsupported exclusion list used by tests/files.rs)

The report is a single table (one row per benchmark, one column per
configuration above), rendered with eval-live (https://github.com/oflatt/eval-live)
for in-browser filtering. eval-live is a Python dependency; see
`scripts/requirements.txt`.

Gating:
  * Every individual run has a 2-minute timeout. A run that exceeds it is
    killed and reported as "timeout".
  * A benchmark is skipped entirely if neither its standard run nor its proof
    run reaches 50ms — too fast to measure reliably. Programs under
    `tests/proofs/` require proofs, so only their proof run is considered.

Usage:
  nightly_bench.py [output_dir]      # default output_dir: <repo>/nightly/output

This is the entry point used by `make nightly` for the nightly dashboard at
nightly.cs.washington.edu.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
TEST_DIR = REPO_ROOT / "tests"
EGGLOG = REPO_ROOT / "target" / "release" / "egglog"

# Benchmarks faster than this (single-run wall clock) are skipped.
MIN_BENCH_SECONDS = 0.050

# Every individual run is capped at two minutes.
RUN_TIMEOUT = 120

# Programs that are expected to fail or are reproduction snippets are not
# benchmarks; skip them entirely.
EXCLUDE_SUBSTRINGS = ("fail-typecheck", "repro-", "/repro")

# Programs that statically support proofs but are excluded from proof-mode
# benchmarking (too slow, or known correctness bugs). Mirrors the
# `proof_unsupported_file_list` in tests/files.rs.
PROOF_UNSUPPORTED_FILES = (
    "math-microbenchmark.egg",
    "rectangle.egg",
    "eggcc-2mm.egg",
    "subsume.egg",
    "subsume-relation.egg",
)

# Prefix used to cap each run at RUN_TIMEOUT, e.g. ["timeout", "120"]. Set by
# calibrate_timeout(): only used when the `timeout` binary's own overhead is
# small enough not to pollute measurements (GNU timeout is ~1ms; some
# reimplementations add ~100ms, which would swamp sub-100ms benchmarks).
TIMEOUT_PREFIX: list[str] = []

# (key, label, threads, proof) — also the report's column order.
CONFIGS = [
    ("standard", "Standard", 1, False),
    ("threads2", "2 threads", 2, False),
    ("threads4", "4 threads", 4, False),
    ("threads8", "8 threads", 8, False),
    ("proof", "Proof", 1, True),
]
THREAD_KEYS = ("standard", "threads2", "threads4", "threads8")


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ── git helpers ───────────────────────────────────────────────────────────────


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def commit_info() -> dict[str, str]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "subject": _git("log", "-1", "--format=%s"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


# ── benchmark discovery ─────────────────────────────────────────────────────────


def discover_benchmarks() -> list[Path]:
    files = sorted(TEST_DIR.rglob("*.egg"))
    return [
        f for f in files if not any(sub in f.as_posix() for sub in EXCLUDE_SUBSTRINGS)
    ]


def requires_proofs(path: Path) -> bool:
    """Programs under tests/proofs/ only make sense with proofs enabled."""
    return path.parent.name == "proofs"


def proof_excluded(path: Path) -> bool:
    return path.name in PROOF_UNSUPPORTED_FILES


def egglog_cmd(path: Path, threads: int, proof: bool) -> list[str]:
    cmd = [str(EGGLOG), "-j", str(threads)]
    if proof:
        cmd.append("--proof-testing")
    cmd.append(str(path))
    return cmd


# ── probing & measuring ─────────────────────────────────────────────────────────


def probe(path: Path, threads: int, proof: bool) -> tuple[bool, float, bool]:
    """Run a config once. Returns (ok, elapsed_seconds, timed_out).

    ok is False if egglog exited non-zero (e.g. a program that does not support
    proofs in proof-testing mode) or if the run exceeded RUN_TIMEOUT.
    """
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            egglog_cmd(path, threads, proof),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=RUN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - start, True
    return proc.returncode == 0, time.perf_counter() - start, False


def runs_for(probe_seconds: float) -> tuple[int, int]:
    """Pick (warmup, runs) for hyperfine: more samples for fast benchmarks,
    fewer for slow ones so the whole sweep stays within a reasonable budget."""
    if probe_seconds < 0.5:
        return 2, 10
    if probe_seconds < 2.0:
        return 2, 7
    if probe_seconds < 5.0:
        return 1, 5
    return 1, 3


def calibrate_timeout() -> None:
    """Decide whether wrapping runs in `timeout` is cheap enough to use.

    GNU coreutils `timeout` adds ~1ms; some reimplementations add ~100ms, which
    would dominate fast benchmarks. We only use the wrapper when its overhead is
    negligible; either way the per-run cap is also enforced by probe() and by a
    subprocess-level backstop in hyperfine()."""
    global TIMEOUT_PREFIX
    if not shutil.which("timeout"):
        log("Note: `timeout` not found; per-run cap enforced via probe only.")
        return
    best = min(_time_true() for _ in range(3))
    if best < 0.025:
        TIMEOUT_PREFIX = ["timeout", str(RUN_TIMEOUT)]
    else:
        log(f"Note: `timeout` overhead is {best * 1000:.0f}ms; not wrapping runs "
            "with it (per-run cap still enforced via probe and backstop).")


def _time_true() -> float:
    start = time.perf_counter()
    subprocess.run(["timeout", str(RUN_TIMEOUT), "true"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - start


def hyperfine(path: Path, threads: int, proof: bool, warmup: int, runs: int) -> dict | None:
    # When available, wrap each run in `timeout` so no single execution exceeds
    # RUN_TIMEOUT; hyperfine then aborts the benchmark (reported as a timeout).
    shell_cmd = (
        (" ".join(TIMEOUT_PREFIX) + " " if TIMEOUT_PREFIX else "")
        + shlex.join(egglog_cmd(path, threads, proof))
        + " >/dev/null 2>&1"
    )
    # Aggregate backstop: even without the per-run wrapper, never let the whole
    # measurement run away (e.g. a non-deterministic hang in a repeat run).
    backstop = RUN_TIMEOUT * (warmup + runs) + 30
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_json = tf.name
    try:
        proc = subprocess.run(
            [
                "hyperfine",
                "--warmup", str(warmup),
                "--runs", str(runs),
                "--export-json", tmp_json,
                shell_cmd,
            ],
            capture_output=True,
            text=True,
            timeout=backstop,
        )
        if proc.returncode != 0:
            return None
        with open(tmp_json) as f:
            return json.load(f)["results"][0]
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(tmp_json)


def measure_cell(path: Path, threads: int, proof: bool,
                 probe_result: tuple[bool, float, bool] | None = None) -> dict:
    """Measure one (benchmark, configuration) cell. Returns a dict with either
    a measured mean or a status: 'timeout' / 'error'."""
    ok, elapsed, timed_out = probe_result or probe(path, threads, proof)
    if not ok:
        return {"status": "timeout" if timed_out else "error"}

    warmup, runs = runs_for(elapsed)
    hf = hyperfine(path, threads, proof, warmup, runs)
    if hf is None:
        # hyperfine aborts on a non-zero exit; with the timeout wrapper that
        # almost always means a run was killed at RUN_TIMEOUT.
        return {"status": "timeout"}
    return {
        "mean": hf["mean"],
        "stddev": hf.get("stddev", 0.0),
        "min": hf["min"],
        "max": hf["max"],
        "runs": len(hf.get("times", [])),
    }


# ── main sweep ──────────────────────────────────────────────────────────────────


def ensure_rustup_on_path() -> None:
    """Put the rustup shim dir (~/.cargo/bin) first on PATH.

    The nightly host has a system `cargo` ahead of the rustup shim; it is too
    old for this crate's 2024 edition and ignores rust-toolchain.toml. The shim
    reads rust-toolchain.toml and installs/uses the pinned toolchain on demand.
    hyperfine and other cargo-installed tools live here too."""
    cargo_bin = str(Path.home() / ".cargo" / "bin")
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p != cargo_bin]
    os.environ["PATH"] = os.pathsep.join([cargo_bin, *parts])


def build() -> None:
    log("Building egglog (release)...")
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "egglog",
         "--manifest-path", str(REPO_ROOT / "Cargo.toml")],
        check=True,
    )
    if not EGGLOG.exists():
        sys.exit(f"egglog binary not found at {EGGLOG}")


def run_sweep() -> tuple[list[dict], list[dict]]:
    """Returns (rows, skipped). Each row is {name, cells: {config_key: cell}}."""
    benchmarks = discover_benchmarks()
    log(f"Discovered {len(benchmarks)} candidate program(s) under {TEST_DIR}\n")

    rows: list[dict] = []
    skipped: list[dict] = []

    headers = "  ".join(f"{label:>10}" for _, label, _, _ in CONFIGS)
    log(f"    {'Benchmark':<40} {headers}")
    log("    " + "─" * (40 + len(CONFIGS) * 12))

    for path in benchmarks:
        name = path.relative_to(TEST_DIR).as_posix()
        req = requires_proofs(path)

        # Qualification probes: the single-thread standard run and the proof run.
        std_probe = None if req else probe(path, 1, False)
        proof_probe = None if proof_excluded(path) else probe(path, 1, True)
        proof_supported = proof_probe is not None and proof_probe[0]

        std_qualifies = std_probe is not None and std_probe[0] and std_probe[1] >= MIN_BENCH_SECONDS
        proof_qualifies = proof_supported and proof_probe[1] >= MIN_BENCH_SECONDS

        if not (std_qualifies or proof_qualifies):
            reason = "errored" if (req and not proof_supported) else "too-fast"
            skipped.append({"name": name, "reason": reason})
            continue

        cells: dict[str, dict] = {}
        for key, _label, threads, proof in CONFIGS:
            if proof:
                if not proof_supported:
                    cells[key] = {"status": "na"}  # proofs unsupported / excluded
                else:
                    cells[key] = measure_cell(path, threads, True, proof_probe)
            else:
                if req:
                    cells[key] = {"status": "na"}  # needs proofs; no plain run
                else:
                    pr = std_probe if threads == 1 else None
                    cells[key] = measure_cell(path, threads, False, pr)

        # Enforce the threshold on measured data: the cold qualification probe
        # can overshoot 50ms for a program whose warmed runs are all faster.
        measured = [c["mean"] for c in cells.values() if "mean" in c]
        if not measured or max(measured) < MIN_BENCH_SECONDS:
            skipped.append({"name": name, "reason": "too-fast"})
            continue

        rows.append({"name": name, "cells": cells})

        def fmt(key: str) -> str:
            c = cells[key]
            return f"{c['mean']:>10.3f}" if "mean" in c else f"{c.get('status', '—'):>10}"
        log(f"    {name:<40} " + "  ".join(fmt(k) for k, *_ in CONFIGS))

    rows.sort(key=lambda r: r["cells"].get("standard", {}).get("mean", 0.0)
              or r["cells"].get("proof", {}).get("mean", 0.0), reverse=True)

    n_fast = sum(s["reason"] == "too-fast" for s in skipped)
    log(f"\n  Benchmarked {len(rows)}; skipped {n_fast} under "
        f"{int(MIN_BENCH_SECONDS * 1000)}ms, "
        f"{len(skipped) - n_fast} errored/proof-only-unsupported.")
    return rows, skipped


# ── report rendering ──────────────────────────────────────────────────────────


def render_html(rows: list[dict], skipped: list[dict], meta: dict) -> str:
    import eval_live

    # Flatten the nested cells structure into eval-live row objects: one row per
    # benchmark, one column per configuration. A measured cell becomes its mean
    # (a number, so eval-live's column filter offers a numeric dropdown); an
    # "na" cell becomes "—" (not applicable, e.g. a non-proof run of a
    # proofs-only program); "timeout"/"error" cells keep their status string.
    bench_rows = []
    for r in rows:
        cells = r["cells"]
        entry: dict = {"name": r["name"]}
        for key, label, _, _ in CONFIGS:
            cell = cells.get(key, {"status": "na"})
            if "mean" in cell:
                entry[label] = round(cell["mean"], 4)
            elif cell.get("status") == "na":
                entry[label] = "—"
            else:
                entry[label] = cell.get("status", "error")
        bench_rows.append(entry)

    data: dict = {"Benchmarks": bench_rows}
    if skipped:
        data["Skipped"] = [{"name": s["name"], "reason": s["reason"]} for s in skipped]

    css = eval_live.css()
    js = eval_live.js()
    # Embedded inside a <script> block, so neutralize any "</script>" breakout.
    # ("<\/" is just "</" inside a JS string literal, so the JSON stays valid.)
    data_json = json.dumps(data).replace("</", "<\\/")

    commit = meta.get("commit_short") or "unknown"
    commit_full = meta.get("commit", "")
    commit_link = (
        f'<a href="https://github.com/egraphs-good/egglog/commit/{escape(commit_full)}">'
        f'{escape(commit)}</a>'
        if commit_full else escape(commit)
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>egglog nightly benchmarks</title>
  <style>
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      margin: 0; padding: 2rem 3rem;
      background: #f5f6f8; color: #1a1a1a;
    }}
    {css}
  </style>
</head>
<body>
  <h1>egglog nightly benchmarks</h1>
  <p>
    Commit {commit_link} &middot;
    branch <code>{escape(meta.get('branch', '?'))}</code> &middot;
    {escape(meta.get('subject', ''))} &middot;
    Generated {generated}
  </p>
  <p>
    All times in seconds (mean). Each run is capped at a {RUN_TIMEOUT // 60}-minute timeout.
    Programs whose standard and proof runs are both under
    {int(MIN_BENCH_SECONDS * 1000)}ms are omitted.
    Raw data: <a href="results.json">results.json</a>.
  </p>
  <div id="tables"></div>
  <script>
    {js}
    initEvalLive("tables", {data_json}, "egglog nightly");
  </script>
</body>
</html>"""


def main() -> int:
    ensure_rustup_on_path()
    if not shutil.which("hyperfine"):
        sys.exit("hyperfine not found — install with: cargo install hyperfine")
    # Fail fast: the report uses eval-live, but it is only needed at the very
    # end, after a sweep that can take a long time. Check it up front.
    try:
        import eval_live  # noqa: F401
    except ImportError:
        sys.exit("eval-live not found — install with: "
                 "pip install -r scripts/requirements.txt")

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "nightly" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    calibrate_timeout()
    build()
    meta = commit_info()
    rows, skipped = run_sweep()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **meta,
        "min_bench_seconds": MIN_BENCH_SECONDS,
        "run_timeout_seconds": RUN_TIMEOUT,
        "configs": [{"key": k, "label": l, "threads": t, "proof": p} for k, l, t, p in CONFIGS],
        "rows": rows,
        "skipped": skipped,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "index.html").write_text(render_html(rows, skipped, meta))
    log(f"\n  Wrote report to {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
