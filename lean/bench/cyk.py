"""CYK parsing on the repository's JSON grammar and token files."""
import json
from pathlib import Path

TESTS = Path(__file__).resolve().parents[2] / "tests"
TOKEN_FILE = {801: "small", 7821: "medium"}
TABLES = ("B", "End", "NT", "P", "Prod", "T", "getString")
# Reference cardinalities from the original length/start program in Rust.
REFERENCE = {
    801: (3218, 17, 120, 3218, 17, 11, 801),
    7821: (36023, 17, 300, 36023, 17, 11, 7821),
}


def cyk_workload(size):
    # The repository program identifies a span by (length, start). Represent
    # it by [start, end) so that the split position is an ordinary join key.
    # Input CSV rows are inserted before starting either schedule clock.
    declarations = (TESTS / "cykjson.egg").read_text().split("(rule", 1)[0]
    setup = [declarations, """
(rule ((End a token) (= token (getString i)))
      ((P i (+ i 1) a) (set (B i (+ i 1) a) (T a token))))
(rule ((Prod a b c) (P i j b) (P j k c)) ((P i k a)))
(rule ((Prod a b c) (= x (B i j b)) (= y (B j k c)))
      ((set (B i k a) (NT a x y))))
"""]
    for table in ("End", "Prod"):
        for row in (TESTS / f"cykjson_{table}.csv").read_text().splitlines():
            setup.append(f"({table} {' '.join(map(json.dumps, row.split(chr(9))))})")
    tokens = (TESTS / f"cykjson_{TOKEN_FILE[size]}_token.csv").read_text().splitlines()
    if len(tokens) != size:
        raise ValueError("token count differs from the benchmark size")
    for index, row in enumerate(tokens, 1):
        position, token = row.split("\t")
        if int(position) != index:
            raise ValueError("expected consecutive one-based positions")
        setup.append(f"(set (getString {index}) {json.dumps(token)})")
    root = f'(B 1 {size + 1} "VAL")'
    checks = [f'(check (P 1 {size + 1} "VAL"))', f"(check (= {root} {root}))"]
    checks.extend(f"(print-size {t})" for t in TABLES)
    return ("\n".join(setup), "(run 10000)", "\n".join(checks)), REFERENCE[size]
