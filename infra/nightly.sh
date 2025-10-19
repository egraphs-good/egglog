#!/bin/bash
# Nightly script for the egraph timeline

echo "Beginning egglog nightly script"

set -e

# determine physical directory of this script
src="${BASH_SOURCE[0]}"
while [ -L "$src" ]; do
  dir="$(cd -P "$(dirname "$src")" && pwd)"
  src="$(readlink "$src")"
  [[ $src != /* ]] && src="$dir/$src"
done
MYDIR="$(cd -P "$(dirname "$src")" && pwd)"

# Absolute directory paths
TOP_DIR="$MYDIR/.."
RESOURCE_DIR="$MYDIR/nightly-resources"
NIGHTLY_DIR="$TOP_DIR/nightly"

# Make sure we're in the right place
cd $MYDIR
echo "Switching to nighly script directory: $MYDIR"

# Clean previous nightly run
# CAREFUL using -f
rm -rf $NIGHTLY_DIR

# Prepare output directories
mkdir -p "$NIGHTLY_DIR/data" "$NIGHTLY_DIR/output"

# Run egglog files
pushd $TOP_DIR
cargo run --bin timeline -- "$TOP_DIR/tests" "$NIGHTLY_DIR/data"

# Annotate with time and command info
python3 timeline/transform.py "$NIGHTLY_DIR/data" "$NIGHTLY_DIR/output"

