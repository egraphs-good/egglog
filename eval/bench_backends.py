#!/usr/bin/env python3
"""Benchmark egglog across the cross product of {backend} x {mode}.

Backends:  bridge (default in-memory) and duckdb (`--duckdb`).
Modes:     normal, term-encoding (`--term-encoding`), proofs (`--proofs`).

For every (benchmark file, backend, mode) cell we run the egglog CLI as a
subprocess `--runs` times (after `--warmup` discarded runs) and record the
wall-clock timings. Cells that error (non-zero exit, timeout) are recorded in
an `errors` table instead. Results are written to a JSON database that
`eval-live` renders interactively (`--serve`).

Note: duckdb is term-encoding-only, so `duckdb-normal` and
`duckdb-term-encoding` exercise the same engine and should report similar
numbers; both are kept so the grid is a true cross product.

Usage:
    python3 eval/bench_backends.py                 # default benchmarks, build + run + save
    python3 eval/bench_backends.py --serve          # also open the eval-live viewer
    python3 eval/bench_backends.py path/to/dir      # benchmark every .egg under a dir
    python3 eval/bench_backends.py --runs 5 --warmup 1 --timeout 600
    python3 eval/bench_backends.py --justserve      # skip benchmarking, view existing results
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]

# (name, extra CLI flags) for each axis of the cross product.
BACKENDS = [
    ("bridge", []),
    ("duckdb", ["--duckdb"]),
]
MODES = [
    ("normal", []),
    ("term-encoding", ["--term-encoding"]),
    ("proofs", ["--proofs"]),
]


def conditions():
    """The cross product backends x modes, as (name, flags) pairs."""
    for backend, bflags in BACKENDS:
        for mode, mflags in MODES:
            yield (f"{backend}-{mode}", backend, mode, bflags + mflags)


class BenchDB:
    """Minimal results database, serialized to the JSON shape eval-live reads:
    {"timings": [...], "errors": [...]}."""

    def __init__(self):
        self.timings = []
        self.errors = []

    def add_timing(self, benchmark, backend, mode, condition, timing_list):
        self.timings.append({
            "benchmark": benchmark,
            "backend": backend,
            "mode": mode,
            "condition": condition,
            "timing_list": timing_list,
        })

    def add_error(self, benchmark, backend, mode, condition, error):
        self.errors.append({
            "benchmark": benchmark,
            "backend": backend,
            "mode": mode,
            "condition": condition,
            "error": error,
        })

    def to_dict(self):
        return {"timings": self.timings, "errors": self.errors}

    def save_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def build_egglog(release: bool) -> Path:
    """Build the egglog CLI (needs the `bin` feature) and return its path."""
    profile = ["--release"] if release else []
    print(f"Building egglog ({'release' if release else 'debug'})...", flush=True)
    subprocess.run(
        ["cargo", "build", *profile, "--features", "bin", "--bin", "egglog"],
        cwd=WORKSPACE, check=True,
    )
    target = "release" if release else "debug"
    binary = WORKSPACE / "target" / target / "egglog"
    if not binary.exists():
        sys.exit(f"egglog binary not found at {binary}")
    return binary


def find_benchmarks(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.egg"))


def run_once(binary: Path, flags: list[str], bench: Path, timeout: float):
    """Run one invocation. Returns (elapsed_seconds, None) on success or
    (None, error_message) on failure/timeout."""
    cmd = [str(binary), *flags, str(bench)]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"timeout after {timeout}s"
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        msg = tail[-1] if tail else f"exit code {proc.returncode}"
        return None, f"exit {proc.returncode}: {msg}"
    return elapsed, None


def bench_one(binary, bench, rel, condition, backend, mode, flags, args, db):
    # Warm-up runs (discarded): pay one-time costs (page cache, etc.).
    for _ in range(args.warmup):
        run_once(binary, flags, bench, args.timeout)

    timings = []
    for _ in range(args.runs):
        elapsed, err = run_once(binary, flags, bench, args.timeout)
        if err is not None:
            db.add_error(rel, backend, mode, condition, err)
            print(f"    {condition:24} ERROR: {err}", flush=True)
            return
        timings.append(round(elapsed, 6))

    db.add_timing(rel, backend, mode, condition, timings)
    mean = sum(timings) / len(timings)
    print(f"    {condition:24} mean {mean:8.3f}s  (runs: {timings})", flush=True)


def serve_results(results_path: Path, port: int):
    """Embed results + graphs into a self-contained HTML page and serve it."""
    import http.server
    import webbrowser
    import eval_live

    css = eval_live.css()
    js = eval_live.js()
    results_json = results_path.read_text()
    eval_live_py = eval_live.pyodide_lib()
    graph_script_path = Path(__file__).resolve().parent / "graphs.py"
    graph_script = graph_script_path.read_text() if graph_script_path.exists() else ""

    pyodide_tag = ""
    init_graphs_args = ""
    if graph_script:
        pyodide_tag = '<script src="https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js"></script>'
        init_graphs_args = f", {json.dumps(graph_script)}, {json.dumps(eval_live_py)}"

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>egglog backend eval</title>
<style>body {{ font-family: system-ui, sans-serif; margin: 0; padding: 2rem 3rem;
background: #f5f6f8; color: #1a1a1a; }} {css}</style>
{pyodide_tag}</head><body>
<div id="tables"></div>
<script>{js}
initEvalLive("tables", {results_json}, "egglog backends"{init_graphs_args});</script>
</body></html>"""

    page_bytes = page.encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(page_bytes)))
            self.end_headers()
            self.wfile.write(page_bytes)

        def log_message(self, *_):
            pass

    server = http.server.HTTPServer(("", port), Handler)
    url = f"http://localhost:{port}"
    print(f"\nServing eval-live at {url}  (Ctrl-C to stop)")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=None,
                        help="benchmark file or directory (default: paper-benchmarks)")
    parser.add_argument("--runs", type=int, default=3, help="timed runs per cell (default 3)")
    parser.add_argument("--warmup", type=int, default=1, help="discarded warm-up runs (default 1)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="per-run timeout in seconds (default 300)")
    parser.add_argument("--output", default=str(WORKSPACE / "eval" / "results.json"),
                        help="results JSON path")
    parser.add_argument("--debug", action="store_true",
                        help="use the debug build instead of release")
    parser.add_argument("--serve", action="store_true", help="open the eval-live viewer after running")
    parser.add_argument("--justserve", action="store_true", help="skip benchmarking; just serve results")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.justserve:
        serve_results(Path(args.output), args.port)
        return

    binary = build_egglog(release=not args.debug)

    bench_path = Path(args.path) if args.path else (WORKSPACE / "paper-benchmarks")
    if not bench_path.is_absolute():
        bench_path = (WORKSPACE / bench_path)
    benchmarks = find_benchmarks(bench_path)
    if not benchmarks:
        sys.exit(f"no .egg benchmarks found under {bench_path}")

    conds = list(conditions())
    print(f"\n{len(benchmarks)} benchmark(s) x {len(conds)} condition(s), "
          f"{args.runs} run(s) each (warmup {args.warmup}, timeout {args.timeout}s)\n")

    db = BenchDB()
    for i, bench in enumerate(benchmarks, 1):
        rel = str(bench.relative_to(WORKSPACE)) if str(bench).startswith(str(WORKSPACE)) else str(bench)
        print(f"[{i}/{len(benchmarks)}] {rel}", flush=True)
        for condition, backend, mode, flags in conds:
            bench_one(binary, bench, rel, condition, backend, mode, flags, args, db)
        db.save_json(args.output)  # incremental: write after each benchmark

    print(f"\nResults written to {args.output}")
    if args.serve:
        serve_results(Path(args.output), args.port)


if __name__ == "__main__":
    main()
