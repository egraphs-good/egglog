#!/bin/bash
# Nightly script for the egraph timeline

echo "Beginning egglog nightly script"

set -e -x

export PATH=~/.cargo/bin:$PATH
rustup update

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
cargo run --bin timeline -- "$RESOURCE_DIR/test-files" "$NIGHTLY_DIR/data"

# Annotate with time and command info
python3 timeline/transform.py "$NIGHTLY_DIR/data" "$NIGHTLY_DIR/output/data"

# Plot run and extract time
python3 timeline/plot_run_vs_extract.py "$NIGHTLY_DIR/output/data" "Herbie: Hamming Benches"
popd

pushd $TOP_DIR
cargo run --release --bin egglog -- tests/
# Put the json data in a JS object for consumption by frontend
(echo "var data = "; cat "$TOP_DIR/summary.json") > "$NIGHTLY_DIR/data/summary.js"
popd


# Update HTML index page.
cp "$RESOURCE_DIR/web"/* "$NIGHTLY_DIR/output"

# This is the uploading part, copied directly from Herbie's nightly script.
DIR="$NIGHTLY_DIR/output"
B=$(git rev-parse --abbrev-ref HEAD)
C=$(git rev-parse HEAD | sed 's/\(..........\).*/\1/')
RDIR="$(date +%s):$(hostname):$B:$C"

# Upload the artifact!
nightly-results publish --name "$RDIR" "$DIR"

# For local dev
# cd "$NIGHTLY_DIR/output" && python3 -m http.server 8002