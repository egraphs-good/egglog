#!/bin/sh
# Build the authors' CAV artifact, including its pinned egg version and benchmark inputs.
set -eu
if [ "$#" -ne 1 ]; then
    echo "usage: sh build-baseline.sh ARTIFACT_DIRECTORY" >&2
    exit 2
fi
script_dir=$(cd "$(dirname "$0")" && pwd)
artifact=$1
revision=ea34dcb7573fbdd2049c80aa6deafe1daed2d981

if [ ! -e "$artifact" ]; then
    git init -q "$artifact"
    git -C "$artifact" remote add origin https://github.com/luigirinaldi/CAV.git
    git -C "$artifact" fetch --depth 1 origin "$revision"
    git -C "$artifact" checkout -q --detach FETCH_HEAD
fi
cd "$artifact"
if [ "$(git rev-parse HEAD)" != "$revision" ]; then
    echo "Expected CAV commit $revision. Choose a new directory to fetch it." >&2
    exit 1
fi
if ! git diff --quiet HEAD -- .gitmodules || ! git diff --cached --quiet HEAD -- parabit benchmarks ||
   git submodule status -- parabit benchmarks | grep -Eq '^[+U]'; then
    echo "Submodule configuration or checkout differs from the pinned CAV revision. Choose a clean artifact directory." >&2
    exit 1
fi
# Upstream names public submodules with SSH URLs. HTTPS enables anonymous checkout.
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --depth 1
if [ "$(git -C parabit rev-parse HEAD)" != 221c2f05b2260377c3cd70b31b9ae6d0158fffab ] ||
   [ "$(git -C benchmarks rev-parse HEAD)" != d9a10b8731baa41a6582bf81b1db6b28f8eb6f5e ]; then
    echo "Solver or corpus differs from the pinned CAV submodules." >&2
    exit 1
fi
# HEAD identifies the release even when a user has edited tracked inputs or
# added an extra .bwlang file to a suite. The sweep discovers files by glob.
verify_inputs() {
    repo=$1
    shift
    if ! git -C "$repo" diff --quiet HEAD -- "$@" ||
       [ -n "$(git -C "$repo" ls-files --others -- "$@")" ]; then
        echo "Local changes to $repo benchmark inputs or solver sources. Choose a clean artifact directory." >&2
        exit 1
    fi
}
verify_inputs parabit Cargo.toml Cargo.lock build.rs ':(glob)src/**/*.rs' ':(glob)proofs/*_lemmas.thy'
verify_inputs benchmarks ':(glob)*/bwlang/*.bwlang'
if [ "$(uname -s)" = Darwin ]; then
    # Use the active Xcode toolchain's SDK. A shell-wide LIBRARY_PATH can point
    # to a different Command Line Tools SDK whose stubs the linker cannot read.
    unset LIBRARY_PATH
    SDKROOT=${SDKROOT:-$(xcrun --sdk macosx --show-sdk-path)}
    export SDKROOT
fi
metadata=$(mktemp)
trap 'rm -f "$metadata"' EXIT
cargo build --manifest-path parabit/Cargo.toml --target-dir parabit/target --release --locked --message-format json > "$metadata"
python3 - "$metadata" "$script_dir/timing.rs" <<'PYTHON'
import json, os, subprocess, sys
from pathlib import Path

libs = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    artifact = json.loads(line)
    if artifact["reason"] == "compiler-artifact":
        for filename in artifact["filenames"]:
            if filename.endswith(".rlib"):
                libs[artifact["target"]["name"]] = filename
deps = Path("parabit/target/release/deps").resolve()
cmd = [os.environ.get("RUSTC", "rustc"), "--edition=2021", "-C", "opt-level=3",
       "-L", f"dependency={deps}", sys.argv[2], "-o", "parabit/target/release/parabit-timed"]
for lib in ("parabit", "quanta", "serde_json"):
    cmd.extend(["--extern", f"{lib}={libs[lib]}"])
subprocess.run(cmd, check=True)
PYTHON
printf '%s\n' "$(pwd)/parabit/target/release/parabit" "$(pwd)/parabit/target/release/parabit-timed"
