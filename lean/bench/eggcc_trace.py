"""Compare source and emitted Bril print traces with the pinned eggcc interpreter."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import tempfile
from eggcc_import import CASES, Reader

HERE = Path(__file__).resolve().parent
SAMPLES = {"select_simple": [(2, 3), (3, 7), (4, 5), (-2, 12), (0, 8), (2**62, 4)],
           "if_dead_code": [(-3,), (0,), (5,)],
           "if_dead_code_nested": [(-3,), (0,), (1,), (3,), (4,), (2**63-1,)]}


def set_arguments(program, values, marker):
    lines = program.splitlines()
    if lines and lines[0].startswith(marker):
        lines.pop(0)
    return marker + " " + " ".join(map(str, values)) + "\n" + "\n".join(lines) + "\n"


def trace(binary, path):
    text = subprocess.check_output([binary, "--run-mode", "parse", "--interp", path], text=True, timeout=15)
    if not text.strip():
        raise ValueError("program produced an empty trace")
    return text.strip().splitlines()


def check(checkout, output):
    upstream = checkout / "target/release/eggcc"
    compiler = HERE.parent / ".lake/build/bin/eggcc-certified"
    with tempfile.TemporaryDirectory() as tmp, output.open("x", newline="") as csv_file:
        rows = csv.writer(csv_file, lineterminator="\n")
        rows.writerow(("source", "arguments", "source_trace", "compiled_trace"))
        for name in CASES:
            original = checkout / "tests/passing/small" / name
            pretty = subprocess.check_output([upstream, "--run-mode", "pretty-print", original],
                                             text=True, timeout=15)
            dag = HERE / "eggcc" / (original.stem + ".json")
            if Reader(pretty).read() != json.loads(dag.read_text()):
                raise ValueError(f"fixture differs from upstream {name}")
            lowered = subprocess.check_output([compiler, "--emit-bril", dag], text=True, timeout=60)
            if " = select " not in lowered or "  br " in lowered:
                raise ValueError(f"expected an if-converted Bril program for {name}")
            source = Path(tmp, name)
            compiled = Path(tmp, original.stem + "-selected.bril")
            for args in SAMPLES[original.stem]:
                source.write_text(set_arguments(original.read_text(), args, "// ARGS:" if source.suffix == ".rs" else "# ARGS:"))
                compiled.write_text(set_arguments(lowered, args, "# ARGS:"))
                want, got = trace(upstream, source), trace(upstream, compiled)
                if want != got:
                    raise ValueError(f"{name} {args}: {want!r} != {got!r}")
                rows.writerow([name, json.dumps(args), json.dumps(want), json.dumps(got)])
            csv_file.flush()
            print(f"{name}: all {len(SAMPLES[original.stem])} traces agree")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkout", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    check(args.checkout.resolve(), args.output)
