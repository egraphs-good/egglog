#!/bin/bash
# One-time setup for the nightly run: rust toolchain update + benchmarks clone.
# Skipped when invoked by the combined-nightly orchestrator, which performs
# these steps once at the top level before driving each branch's nightly.sh.
set -euo pipefail

export PATH=~/.cargo/bin:$PATH

rustup update
cargo install rustfilt

mkdir -p nightly/tmp
git clone --depth 1 https://github.com/ajpal/poach-benchmarks.git nightly/tmp/poach-benchmarks
