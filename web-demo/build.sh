#!/usr/bin/env bash

cd $(dirname "${BASH_SOURCE[0]}")

# make sure to remove the .gitignore introduces by wasm-pack
# so the deployment to github pages works

set -ev
wasm-pack build --target no-modules --no-typescript --out-dir www/pkg/
rm -f www/pkg/.gitignore