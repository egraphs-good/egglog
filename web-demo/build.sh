#!/usr/bin/env bash

cd $(dirname "${BASH_SOURCE[0]}")

set -ev
wasm-pack build --target no-modules --no-typescript --out-dir www/pkg/