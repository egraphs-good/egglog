#!/bin/sh
set -eu
cd "$(dirname "$0")"
lake build Egglog lean-egglog Egglog.Examples.Boolean Egglog.Examples.StrengthReduction Egglog.Examples.ParaBitPass parabit-relational parabit-certified eggcc-certified luminal-certified
lake env lean --run CoreTests.lean
lake env lean --run LuminalTests.lean
lake env lean EggCCTests.lean
for suite in FrontendTests CompiledTests ProofTests CompilerTests ParaBitTests ParaBitSourceTests ParaBitPassTests parabit/RoverAssociativity; do
    lake env lean "$suite.lean"
done
bin=.lake/build/bin/lean-egglog
"$bin" --naturals ../tests/web-demo/naturals.egg
for input in ../tests/web-demo/eqsat-basic.egg ../tests/web-demo/bdd.egg \
             ../tests/web-demo/path.egg ../tests/until.egg ../tests/repro-define.egg \
             ../tests/merge-during-rebuild.egg ../tests/repro-empty-query.egg \
             ../tests/merge-saturates.egg ../tests/looking_up_global.egg \
             ../tests/set_sort_function.egg ../tests/repro-equal-constant.egg \
             ../tests/repro-equal-constant2.egg ../tests/repro-should-saturate.egg \
             ../tests/fibonacci-demand.egg ../tests/repro-stale-rows.egg \
             ../tests/test-combined.egg ../tests/combined-nested.egg \
             ../tests/term-encoding-subsume-rebuild-bug.egg tests/row-visibility.egg \
             tests/compiler-input.egg tests/scheduled-unions.egg tests/constructor-projection.egg tests/function-join.egg tests/function-deltas.egg tests/relational-joins.egg tests/multiway-join.egg; do
    "$bin" --raw "$input" >/dev/null
done

fixture=$(mktemp)
trap 'rm -f "$fixture"' EXIT
cat parabit/theory.egg parabit/analyses-tests.egg > "$fixture"
"$bin" --raw "$fixture" >/dev/null
python3 parabit/check-kernel.py smoke
