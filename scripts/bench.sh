#!/usr/bin/env bash
# Usage:
#   ./scripts/bench.sh run        -- build and time all benchmarks, append results to benchmarks/result.csv
#   ./scripts/bench.sh archive    -- rename result.csv to a timestamped file (keeps the run)
#   ./scripts/bench.sh clear      -- delete result.csv (discards the run)
#   ./scripts/bench.sh report     -- print a table showing how each benchmark changed over time
#   ./scripts/bench.sh plot       -- render a chart (PNG via matplotlib, or ASCII fallback)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmarks"
mkdir -p "$BENCH_DIR"

BENCHMARKS=(
  hardboiled_conv1d_32.egg
  hardboiled_conv1d_128.egg
  luminal-llama.egg
  python_array_optimize.egg
  cykjson.egg
  eggcc-extraction.egg
)

EGGLOG="$REPO_ROOT/target/release/egglog"
TEST_DIR="$REPO_ROOT/tests"
RESULT_CSV="$BENCH_DIR/result.csv"

# ── run ──────────────────────────────────────────────────────────────────────

cmd_run() {
  if [[ -f "$RESULT_CSV" ]]; then
    echo "Error: $RESULT_CSV already exists."
    echo "  archive — keep results: ./scripts/bench.sh archive"
    echo "  clear   — discard results: ./scripts/bench.sh clear"
    exit 1
  fi

  echo "Building egglog (release)..."
  cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" 2>&1

  TIMESTAMP="$(date +%Y-%m-%dT%H:%M:%S)"

  echo "timestamp,benchmark,time_s" > "$RESULT_CSV"

  echo "Running benchmarks at $TIMESTAMP"
  printf "%-40s %s\n" "benchmark" "time (s)"
  printf "%-40s %s\n" "─────────────────────────────────────────" "────────"

  for bench in "${BENCHMARKS[@]}"; do
    src="$TEST_DIR/$bench"
    if [[ ! -f "$src" ]]; then
      echo "  SKIP $bench (not found at $src)"
      continue
    fi

    START=$(date +%s%3N)   # milliseconds
    "$EGGLOG" "$src" > /dev/null 2>&1
    STATUS=$?
    END=$(date +%s%3N)

    ELAPSED_MS=$(( END - START ))
    ELAPSED_S=$(awk "BEGIN { printf \"%.3f\", $ELAPSED_MS / 1000 }")

    if [[ $STATUS -ne 0 ]]; then
      printf "%-40s FAILED (exit %d)\n" "$bench" "$STATUS"
    else
      printf "%-40s %s\n" "$bench" "$ELAPSED_S"
      echo "$TIMESTAMP,$bench,$ELAPSED_S" >> "$RESULT_CSV"
    fi
  done

  echo
  echo "Results appended to $RESULT_CSV"
}

# ── archive ───────────────────────────────────────────────────────────────────

cmd_archive() {
  if [[ ! -f "$RESULT_CSV" ]]; then
    echo "Nothing to archive: $RESULT_CSV does not exist."
    exit 1
  fi
  DEST="$BENCH_DIR/$(date +%Y-%m-%dT%H:%M:%S).csv"
  mv "$RESULT_CSV" "$DEST"
  echo "Archived to $DEST"
}

# ── clear ─────────────────────────────────────────────────────────────────────

cmd_clear() {
  if [[ ! -f "$RESULT_CSV" ]]; then
    echo "Nothing to clear: $RESULT_CSV does not exist."
    exit 0
  fi
  rm "$RESULT_CSV"
  echo "Deleted $RESULT_CSV"
}

# ── report ───────────────────────────────────────────────────────────────────

cmd_report() {
  python3 - "$BENCH_DIR" "${BENCHMARKS[@]}" <<'EOF'
import sys, csv, os, glob
from collections import defaultdict

bench_dir = sys.argv[1]
benchmarks = sys.argv[2:]

# Load result.csv plus all archived timestamped CSVs
data = defaultdict(list)  # bench -> [(timestamp, time_s)]
csvs = sorted(glob.glob(os.path.join(bench_dir, "*.csv")))
if not csvs:
    print("No benchmark data found. Run './scripts/bench.sh run' first.")
    sys.exit(0)

for path in csvs:
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["benchmark"]].append((row["timestamp"], float(row["time_s"])))

# Sort each benchmark's runs chronologically
for b in data:
    data[b].sort()

all_timestamps = sorted({ts for runs in data.values() for ts, _ in runs})

SPARK = " ▁▂▃▄▅▆▇█"

def sparkline(values):
    if len(values) < 2:
        return "─" * len(values)
    lo, hi = min(values), max(values)
    span = hi - lo or 1
    return "".join(SPARK[min(8, int((v - lo) / span * 8))] for v in values)

def pct_change(first, last):
    if first == 0:
        return "N/A"
    c = (last - first) / first * 100
    sign = "+" if c >= 0 else ""
    return f"{sign}{c:.1f}%"

# Print summary table
print()
print(f"{'Benchmark':<42} {'Runs':>4}  {'First (s)':>9}  {'Latest (s)':>10}  {'Change':>8}  Trend")
print("─" * 100)
for bench in benchmarks:
    runs = data.get(bench, [])
    if not runs:
        print(f"  {bench:<40} {'no data':>4}")
        continue
    timestamps, times = zip(*runs)
    spark = sparkline(times)
    print(f"  {bench:<40} {len(runs):>4}  {times[0]:>9.3f}  {times[-1]:>10.3f}  {pct_change(times[0], times[-1]):>8}  {spark}")

print()
print(f"Range: {all_timestamps[0]} → {all_timestamps[-1]}  ({len(all_timestamps)} runs)")
print()

# Per-benchmark detail
for bench in benchmarks:
    runs = data.get(bench, [])
    if not runs:
        continue
    print(f"  {bench}")
    for ts, t in runs:
        bar = "█" * max(1, int(t / max(r for _, r in runs) * 30))
        print(f"    {ts}  {t:7.3f}s  {bar}")
    print()
EOF
}

# ── plot ─────────────────────────────────────────────────────────────────────

cmd_plot() {
  python3 - "$BENCH_DIR" "${BENCHMARKS[@]}" <<'EOF'
import sys, csv, os, glob
from collections import defaultdict

bench_dir = sys.argv[1]
benchmarks = sys.argv[2:]

data = defaultdict(list)
csvs = sorted(glob.glob(os.path.join(bench_dir, "*.csv")))
if not csvs:
    print("No benchmark data found. Run './scripts/bench.sh run' first.")
    sys.exit(0)

for path in csvs:
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["benchmark"]].append((row["timestamp"], float(row["time_s"])))

for b in data:
    data[b].sort()

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    fig, ax = plt.subplots(figsize=(12, 6))
    for bench in benchmarks:
        runs = data.get(bench, [])
        if not runs:
            continue
        dates = [datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S") for ts, _ in runs]
        times = [t for _, t in runs]
        label = bench.replace(".egg", "")
        ax.plot(dates, times, marker="o", label=label)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    ax.set_xlabel("Date")
    ax.set_ylabel("Time (s)")
    ax.set_title("egglog benchmark runtimes over time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(bench_dir, "bench_plot.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")

except ImportError:
    # ASCII fallback: one chart per benchmark using terminal width
    import shutil
    WIDTH = shutil.get_terminal_size((80, 24)).columns - 20

    for bench in benchmarks:
        runs = data.get(bench, [])
        if not runs:
            continue
        timestamps, times = zip(*runs)
        hi = max(times)
        print(f"\n  {bench}")
        for ts, t in zip(timestamps, times):
            bar_len = max(1, int(t / hi * WIDTH))
            bar = "█" * bar_len
            print(f"  {ts}  {t:7.3f}s  {bar}")
    print()
EOF
}

# ── diff ─────────────────────────────────────────────────────────────────────

cmd_diff() {
  python3 - "$BENCH_DIR" "$RESULT_CSV" "${BENCHMARKS[@]}" <<'EOF'
import sys, csv, os, glob
from collections import defaultdict

bench_dir  = sys.argv[1]
result_csv = sys.argv[2]
benchmarks = sys.argv[3:]

def load_latest(path):
    """Return {benchmark: latest_time} from a CSV (last row wins per benchmark)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["benchmark"]] = float(row["time_s"])
    return out

if not os.path.exists(result_csv):
    print("No result.csv found. Run './scripts/bench.sh run' first.")
    sys.exit(1)

# Find the most recent archived CSV (exclude result.csv itself)
archives = sorted(f for f in glob.glob(os.path.join(bench_dir, "*.csv"))
                  if os.path.basename(f) != "result.csv")
if not archives:
    print("No archived runs found. Run 'archive' after a previous run to create a baseline.")
    sys.exit(1)

prev_csv  = archives[-1]
prev      = load_latest(prev_csv)
curr      = load_latest(result_csv)

print(f"\n  Comparing:")
print(f"    baseline : {os.path.basename(prev_csv)}")
print(f"    current  : result.csv")
print()
print(f"  {'Benchmark':<42} {'Before (s)':>10}  {'After (s)':>10}  {'Δ (s)':>8}  {'Δ %':>7}  ")
print("  " + "─" * 88)

faster = slower = unchanged = missing = 0

for bench in benchmarks:
    p = prev.get(bench)
    c = curr.get(bench)
    if p is None or c is None:
        status = "N/A"
        print(f"  {bench:<42} {'—':>10}  {'—':>10}  {'—':>8}  {'—':>7}  (missing in one run)")
        missing += 1
        continue
    delta   = c - p
    pct     = delta / p * 100 if p else 0
    sign    = "+" if delta >= 0 else ""
    if abs(pct) < 0.5:
        marker = "  ·"
        unchanged += 1
    elif delta < 0:
        marker = "  ▼ faster"
        faster += 1
    else:
        marker = "  ▲ slower"
        slower += 1
    print(f"  {bench:<42} {p:>10.3f}  {c:>10.3f}  {sign}{delta:>7.3f}  {sign}{pct:>6.1f}%{marker}")

print()
print(f"  Summary: {faster} faster  ·  {slower} slower  ·  {unchanged} unchanged  ·  {missing} missing")
print()
EOF
}

# ── dispatch ─────────────────────────────────────────────────────────────────

case "${1:-}" in
  run)     cmd_run ;;
  archive) cmd_archive ;;
  clear)   cmd_clear ;;
  diff)    cmd_diff ;;
  report)  cmd_report ;;
  plot)    cmd_plot ;;
  *)
    echo "Usage: $0 {run|archive|clear|diff|report|plot}"
    echo "  run     — build and time all benchmarks, append to benchmarks/result.csv"
    echo "  archive — rename result.csv to a timestamped file (keep the results)"
    echo "  clear   — delete result.csv (discard the results)"
    echo "  diff    — compare result.csv against the last archived run"
    echo "  report  — print a summary table and per-benchmark history"
    echo "  plot    — render a chart (PNG if matplotlib available, else ASCII)"
    exit 1
    ;;
esac
