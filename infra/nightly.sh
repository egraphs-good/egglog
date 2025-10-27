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

pushd $TOP_DIR

for DIRPATH in infra/nightly-resources/test-files/*; do
  if [ -d $DIRPATH ]; then
    DIRNAME=$(basename $DIRPATH)

    mkdir "$NIGHTLY_DIR/data/$DIRNAME"

    # Run egglog files
    cargo run --bin timeline -- "$RESOURCE_DIR/test-files/$DIRNAME" "$NIGHTLY_DIR/data/$DIRNAME"

    # Annotate with time and command info
    python3 timeline/transform.py "$NIGHTLY_DIR/data/$DIRNAME" "$NIGHTLY_DIR/output/data/$DIRNAME"
  fi
done

popd

pushd $TOP_DIR
cargo run --release --bin egglog -- tests/
cp "$TOP_DIR/summary.json" "$NIGHTLY_DIR/output/data/summary.json"
popd


# Update HTML index page.
cp "$RESOURCE_DIR/web"/* "$NIGHTLY_DIR/output"

# No more uploading using nightly-results, that happens automatically by the nightly runner now.

# For local dev
# cd "$NIGHTLY_DIR/output" && python3 -m http.server 8002