"""Tensor-kernel lowering and fusion on the complete checked-in Luminal Llama graph."""
import re
import subprocess
import tempfile
from pathlib import Path

from bench.egg_source import commands

SOURCE = Path(__file__).resolve().parents[2] / "tests/luminal-llama.egg"


def projection_rows(export):
    """Preparse the exported SSA equations and expected outputs for the Rust native API."""
    rows, roots, checks, extracts = [], {}, set(), set()

    def index(text, prefix):
        if not text.startswith(prefix) or not text[len(prefix):].isdecimal():
            raise ValueError(f"expected {prefix} and a decimal index, got {text}")
        return int(text[len(prefix):])

    def register(term, group):
        op, reg = term
        if op != f"Reg{group}" or not reg.isdecimal():
            raise ValueError(f"unexpected register {term} in group {group}")
        return int(reg)

    for op, *args in commands(export):
        if op == "union":
            result, (times, a, b) = args
            group = index(times, "Times")
            rows.append(("P", group, register(result, group), register(a, group), register(b, group)))
        elif op == "let":
            name, (add, src) = args
            group = index(add, "Sum")
            if name in roots:
                raise ValueError(f"duplicate root {name}")
            roots[name] = group, register(src, group)
        elif op == "check":
            (equal, name, (dot, a, b)), = args
            group, source = roots[name]
            if equal != "=" or dot != f"Dot{group}" or name in checks:
                raise ValueError(f"unexpected equality {args}")
            checks.add(name)
            rows.append(("R", group, index(name, "$root"), source, register(a, group), register(b, group)))
        elif op == "extract":
            name, = args
            if name in extracts:
                raise ValueError(f"duplicate extraction {name}")
            extracts.add(name)
    if len(roots) != 225 or checks != roots.keys() or extracts != checks:
        raise ValueError("incomplete projection workload")
    return "\n".join("\t".join(map(str, row)) for row in rows) + "\n"


def certified_projection_workload():
    """Export the same layout-specialized rewrite, SSA aliases and roots as the verified pass."""
    compiler = SOURCE.parents[1] / "lean/.lake/build/bin/luminal-certified"
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "projections.egg"
        subprocess.run([compiler, "--emit-projections", SOURCE, output], check=True,
                       stdout=subprocess.DEVNULL)
        setup, post = output.read_text().split("\n(run 1)\n")
    checks = [line for line in post.splitlines() if line.startswith("(check ")]
    if len(checks) != 225:
        raise ValueError("expected one equality per input projection")
    checks += [f"(print-size Dot{i})" for i in range(5)]
    return (setup, "(run 1)", "\n".join(checks)), (64, 64, 64, 32, 1)


# Select complete blocks from the upstream fixture. The pass retains the
# original rules and all tensor nodes, including every one of the 32 layers.
def luminal_workload():
    source = SOURCE.read_text()
    def part(start, stop):
        a = source.index(start)
        return source[a:source.index(stop, a + len(start))]
    setup = "\n".join([
        part("(ruleset expr)", "; ---- Algebraic rewrites"),
        part("(function len", "(function n_elements"),
        part("    (datatype*\n        (IR", "(rule\n            (\n                ; get mul"),
        part("(rule\n    (\n        (= ?a (Add ?out_shape", "    (rule\n        (\n            ; Ascending:"),
        part("(rule\n            (\n                ; get add", "        ; Cube mul - Tile sum"),
        part("(rule\n           ((= ?e (Input", "    (rule\n                ((= ?m (Exp"),
        part("(let t0 ", "(run-schedule"),
    ])
    schedule = "(run-schedule (saturate expr) (saturate (run)))"
    checks = [
        '(check (= (dtype t2412) (F32)))',
        '(check (= t52 (RowRMSNorm t13 t35 t9 t8 t51)))',
        '(check (= t2400 (RowRMSNorm t13 t2389 t9 t8 t2399)))',
        '(check (= t147 (RowSwishMul t13 t128 t102 t146 t102 t101)))',
        '(check (= t2383 (RowSwishMul t13 t2371 t102 t2382 t102 t101)))',
        '(check (= t42 (KernelMean t13 t8 t38 t9 t19 t20 (F32))))',
        '(check (= t2392 (KernelMean t13 t8 t2390 t9 t19 t20 (F32))))',
    ]
    tables = ("Input", "Output", "CustomOpHLIR", "KernelGather", "KernelIota",
              "KernelAdd", "KernelMul", "KernelSum", "KernelMean", "Exp",
              "Sigmoid", "RowAdd", "RowRMSNorm", "RowSwishMul", "dtype")
    checks.extend(f"(print-size {name})" for name in tables)
    expected = (293, 1, 32, 1, 2, 162, 614, 290, 65, 32, 32, 64, 65, 32, 1659)
    return (setup, schedule, "\n".join(checks)), expected


def projection_checks(setup):
    """Trace learned matrix products in the input SSA, independently of egglog."""
    bindings = {name: term for op, name, term in
                (form for form in commands(setup) if form[0] == "let")}
    checks = []
    for result, node in bindings.items():
        if not isinstance(node, list) or node[0] != "Sum":
            continue
        product = bindings[node[3]]
        if not isinstance(product, list) or product[0] != "Mul":
            continue
        activation, weight = product[2], product[4]
        parameter = bindings[weight]
        if isinstance(parameter, list) and parameter[0] == "Input":
            label = bindings[parameter[2]]
            if label.endswith('.weight"'):
                checks.append(f"(check (checked-matmul {result} {activation} {weight}))")
    if len(checks) != 225:
        raise ValueError("expected 32 layers with seven projections plus the final output head")
    return checks


def luminal_full_workload():
    """Original program split at its only schedule, with post-run structural checks."""
    source = SOURCE.read_text()
    setup, marker, schedule = source.rpartition("(run-schedule")
    if not marker or "(run-schedule" in setup:
        raise ValueError("expected one final Luminal schedule")
    snapshot = SOURCE.parent / "snapshots/files__shared_snapshot_luminal_llama.snap"
    rows = re.findall(r"\(([\w]+) (\d+)\)", snapshot.read_text())
    if len(rows) != 70:
        raise ValueError("unexpected Luminal reference snapshot")
    checks = [f"(print-size {table})" for table, _ in rows]
    checks += [
        '(check (= (dtype t2412) (F32)))',
        '(check (= t52 (RowRMSNorm t13 t35 t9 t8 t51)))',
        '(check (= t2400 (RowRMSNorm t13 t2389 t9 t8 t2399)))',
        '(check (= t147 (RowSwishMul t13 t128 t102 t146 t102 t101)))',
        '(check (= t2383 (RowSwishMul t13 t2371 t102 t2382 t102 t101)))',
        '(relation checked-matmul (IR IR IR))',
        '(ruleset check-matmuls)',
        '(rule ((= result (TileMatmul _ _ _ a _ _ _ b _ _ _ _ _ _))) '
        '((checked-matmul result a b)) :ruleset check-matmuls)',
        '(run check-matmuls 1)',
    ]
    checks.extend(projection_checks(setup))
    return (setup, marker + schedule, "\n".join(checks)), tuple(int(size) for _, size in rows)
