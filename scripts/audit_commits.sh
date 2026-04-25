#!/usr/bin/env bash
# Rebuild and benchmark every experiment commit, then report pairwise comparisons.
# Usage: ./scripts/audit_commits.sh [--from <commit>]
# Output: benchmarks/audit_<timestamp>.csv  and  a summary table to stdout

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmarks"
EGGLOG="$REPO_ROOT/target/release/egglog"
TEST_DIR="$REPO_ROOT/tests"

BENCHMARKS=(
  hardboiled_conv1d_32.egg
  hardboiled_conv1d_128.egg
  luminal-llama.egg
  python_array_optimize.egg
  cykjson.egg
  eggcc-extraction.egg
)

TIMESTAMP="$(date +%Y-%m-%dT%H:%M:%S)"
AUDIT_CSV="$BENCH_DIR/audit_${TIMESTAMP}.csv"
mkdir -p "$BENCH_DIR"

# ── collect commits to audit ──────────────────────────────────────────────────
# All commits on this branch not reachable from main, oldest-first.
FROM_COMMIT="${1:-}"
if [[ -n "$FROM_COMMIT" ]]; then
  COMMITS=($(git log --reverse --oneline "${FROM_COMMIT}..HEAD" | awk '{print $1}'))
  # Prepend the base commit itself so we have a "before" measurement
  COMMITS=("$FROM_COMMIT" "${COMMITS[@]}")
else
  COMMITS=($(git log --reverse --oneline "main..HEAD" | awk '{print $1}'))
  # Prepend main tip as baseline
  MAIN_TIP="$(git rev-parse main)"
  COMMITS=("$MAIN_TIP" "${COMMITS[@]}")
fi

ORIGINAL_BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD)"
ORIGINAL_HEAD="$(git rev-parse HEAD)"

echo "Auditing ${#COMMITS[@]} commits (including baseline)"
echo "Results → $AUDIT_CSV"
echo "timestamp,commit,commit_msg,benchmark,time_s" > "$AUDIT_CSV"

# ── helper: build + bench one commit ─────────────────────────────────────────
bench_commit() {
  local sha="$1"
  local msg
  msg="$(git log -1 --format='%s' "$sha" | sed 's/,/;/g')"

  echo ""
  echo "━━━ $sha  $msg"

  git checkout --quiet "$sha"

  echo -n "  building... "
  if ! cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" 2>/dev/null; then
    echo "BUILD FAILED — skipping"
    return
  fi
  echo "ok"

  local run_ts
  run_ts="$(date +%Y-%m-%dT%H:%M:%S)"

  for bench in "${BENCHMARKS[@]}"; do
    local src="$TEST_DIR/$bench"
    if [[ ! -f "$src" ]]; then continue; fi

    local START END ELAPSED_S
    START=$(date +%s%3N)
    "$EGGLOG" "$src" > /dev/null 2>&1
    END=$(date +%s%3N)
    ELAPSED_S=$(awk "BEGIN { printf \"%.3f\", $(( END - START )) / 1000 }")
    printf "    %-40s %s s\n" "$bench" "$ELAPSED_S"
    echo "$run_ts,$sha,$msg,$bench,$ELAPSED_S" >> "$AUDIT_CSV"
  done
}

# ── run ───────────────────────────────────────────────────────────────────────
trap "git checkout --quiet '$ORIGINAL_HEAD' && git checkout --quiet '$ORIGINAL_BRANCH' 2>/dev/null || true" EXIT

for sha in "${COMMITS[@]}"; do
  bench_commit "$sha"
done

# ── analysis ─────────────────────────────────────────────────────────────────
echo ""
echo "━━━ PAIRWISE ANALYSIS ━━━"
python3 - "$AUDIT_CSV" "${BENCHMARKS[@]}" <<'EOF'
import sys, csv
from collections import defaultdict

audit_csv = sys.argv[1]
benchmarks = sys.argv[2:]

# Load: {commit: {bench: time}}
rows_by_commit = defaultdict(dict)
commit_order = []
commit_msgs = {}

with open(audit_csv) as f:
    for row in csv.DictReader(f):
        sha = row["commit"]
        bench = row["benchmark"]
        t = float(row["time_s"])
        if sha not in rows_by_commit:
            commit_order.append(sha)
            commit_msgs[sha] = row["commit_msg"]
        rows_by_commit[sha][bench] = t

NOISE = 0.03   # 3% threshold — changes within this are "noise"

print()
print(f"  {'Commit':<10}  {'Message':<55}  {'Geomean Δ%':>11}  {'Verdict'}")
print("  " + "─" * 100)

flagged = []

for i in range(1, len(commit_order)):
    prev_sha = commit_order[i - 1]
    curr_sha = commit_order[i]
    prev = rows_by_commit[prev_sha]
    curr = rows_by_commit[curr_sha]

    ratios = []
    for b in benchmarks:
        if b in prev and b in curr and prev[b] > 0:
            ratios.append(curr[b] / prev[b])

    if not ratios:
        print(f"  {curr_sha[:8]}  {commit_msgs[curr_sha][:55]:<55}  {'N/A':>11}")
        continue

    import math
    geomean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    pct = (geomean - 1) * 100

    sign = "+" if pct >= 0 else ""

    if pct < -NOISE * 100:
        verdict = "✓ faster"
    elif pct > NOISE * 100:
        verdict = "✗ SLOWER"
        flagged.append((curr_sha, commit_msgs[curr_sha], pct))
    else:
        verdict = "~ noise / simplification"
        # Only flag if not labeled a simplification
        msg = commit_msgs[curr_sha].lower()
        is_simplification = any(w in msg for w in ["simplif", "fix", "refactor", "clean", "early-return", "trivial"])
        if not is_simplification:
            flagged.append((curr_sha, commit_msgs[curr_sha], pct))

    print(f"  {curr_sha[:8]}  {commit_msgs[curr_sha][:55]:<55}  {sign}{pct:>9.1f}%  {verdict}")

print()
if flagged:
    print(f"  ⚠  Flagged commits (≤{NOISE*100:.0f}% improvement, not labeled simplification):")
    for sha, msg, pct in flagged:
        sign = "+" if pct >= 0 else ""
        print(f"     {sha[:8]}  {sign}{pct:.1f}%  {msg}")
else:
    print("  All commits are either genuine improvements or labeled simplifications.")
print()
EOF
