#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Beginning POACH nightly script..."

###############################################################################
# This script generates the data for the nightly frontend
###############################################################################

export PATH=~/.cargo/bin:$PATH

# Ensure we start from a clean slate
rm -rf nightly
mkdir -p nightly/output nightly/tmp

# Standalone runs do their own setup (toolchain + benchmarks clone). When
# driven by the combined orchestrator, POACH_NIGHTLY_COMBINED=1 and the
# benchmarks dir is supplied via POACH_BENCHMARKS_DIR.
if [ ! -v POACH_NIGHTLY_COMBINED ]; then
  bash infra/setup.sh
fi

BENCHMARKS_DIR="${POACH_BENCHMARKS_DIR:-nightly/tmp/poach-benchmarks}"
if [ ! -d "$BENCHMARKS_DIR" ]; then
  echo "ERROR: benchmarks dir $BENCHMARKS_DIR not found" >&2
  exit 1
fi

# Build in release mode before running nightly.py
cargo build --release

# This script runs all of the benchmarks/experiments
python3 infra/nightly.py "$BENCHMARKS_DIR"

# Abort if nightly.py failed to produce data.json. Without this check,
# the nightly runner will report the nightly as successful even though the
# generated report is empty.
if [ ! -f nightly/output/data/data.json ]; then
  echo "ERROR: nightly/output/data/data.json was not generated."
  exit 1
fi

cp infra/nightly-resources/web/* nightly/output

# Uncomment for local development
# cd nightly/output && python3 -m http.server 8002
