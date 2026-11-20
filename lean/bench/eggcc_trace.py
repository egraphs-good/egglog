"""Cross-check source, optimized Lean IR and emitted Bril on upstream eggcc programs."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import tempfile
from eggcc_import import CASES, Reader

HERE = Path(__file__).resolve().parent
SAMPLES = {"select_simple": [(2, 3), (3, 7), (4, 5), (-2, 12), (0, 8), (2**62, 4)],
           "if_dead_code": [(-3,), (0,), (5,)],
           "if_dead_code_nested": [(-3,), (0,), (1,), (3,), (4,), (2**63-1,)],
           "simple_loop": [()], "eliminate_loop": [()],
           "select": [(1,), (2,), (5,), (20,), (64,)],
           "if_in_loop": [(-3,), (0,), (1,)],
           "small-fib": [(0,), (1,), (2,), (6,), (10,)],
           "double-for": [(9,), (10,), (11,), (12,), (15,)],
           "flatten_loop": [(-1, 4), (0, 0), (3, 0), (1, 1), (2, 3), (4, 2)]}

# Pinned tests/passing/small programs with a DoWhile and operators in the
# imported scalar/tuple fragment. An import failure for any entry fails the run.
UPSTREAM_LOOP_ARGS = {
    "double-for.bril": (20,), "eliminate_loop.bril": (), "fib_shape.bril": (10,),
    "flatten_loop.bril": (10, 10), "gamma_condition_and.bril": (1,), "if_in_loop.bril": (1,), "ivt.rs": (20,),
    "jumping_loop.bril": (), "loop_hoist.bril": (1, 2, 3, 4), "loop_if.bril": (),
    "loop_pass_through.bril": (5,), "loop_with_mul_by_inv.bril": (4,), "peel_twice.bril": (),
    "peel_twice_precalc_pred.bril": (), "range_check.bril": (), "range_splitting.bril": (),
    "select.rs": (20,), "simple_loop.bril": (), "simplest_loop.bril": (), "small-fib.bril": (2,),
    "strong_loop.bril": (3,), "unroll_and_constant_fold.bril": (), "unroll_multiple_4.bril": (),
}
# The other DoWhile programs in this pinned directory require division,
# multiple functions, heap operations, or floating point.
OUTSIDE_FRAGMENT = {"collatz_redundant_computation.bril", "small-collatz.bril",
                    "implicit-return.bril", "mem_loop_store_forwarding.bril", "sqrt_small.bril"}
ELIMINATED = {"simple_loop", "eliminate_loop"}
NESTED = {"double-for", "flatten_loop"}
ACYCLIC = {"select_simple", "if_dead_code", "if_dead_code_nested"}


def expected_loop_trace(name, args):
    """An independent check that the interpreter actually used the supplied arguments."""
    if name == "simple_loop":
        return ["0"]
    if name == "eliminate_loop":
        return ["1"]
    if name == "select":
        n, = args
        return [str(n * (n + 1) // 2)]
    if name == "if_in_loop":
        return (["1", "true"] if args[0] <= 0 else ["0", "false"]) * 11
    if name == "small-fib":
        n, = args
        values = [1, 1]
        for _ in range(1, n):
            values.append(values[-1] + values[-2])
        return list(map(str, values[:n + 1]))
    if name == "double-for":
        return [str(sum(i + j for i in range(10, args[0]) for j in range(10, args[0])))]
    if name == "flatten_loop":
        n, m = args
        return [str(i * m + j) for i in range(n) for j in range(m)] + [str(max(n, 0))]
    return None


def loop_blocks(program):
    """Find natural loops in the emitted CFG using dominators, independently of label names."""
    blocks = {"": []}
    current = ""
    for raw in program.splitlines():
        line = raw.strip()
        if line.startswith(".") and line.endswith(":"):
            current = line[:-1]
            if current in blocks:
                raise ValueError(f"duplicate Bril label {current}")
            blocks[current] = []
        elif line.endswith(";"):
            blocks[current].append(line)
    order = list(blocks)
    succ = {}
    for index, (block, lines) in enumerate(blocks.items()):
        last = lines[-1] if lines else ""
        if last.startswith(("br ", "jmp ")):
            succ[block] = {word for word in last[:-1].split() if word.startswith(".")}
        elif last.startswith("ret"):
            succ[block] = set()
        else:
            succ[block] = set(order[index+1:index+2])
    pred = {block: set() for block in blocks}
    for source, targets in succ.items():
        for dest in targets:
            pred[dest].add(source)
    live, todo = set(), [""]
    while todo:
        source = todo.pop()
        if source not in live:
            live.add(source)
            todo.extend(succ[source])
    dom = {block: {""} if block == "" else live.copy() for block in live}
    while True:
        new = {b: ({b} | set.intersection(*(dom[p] for p in pred[b] & live)) if b else {""})
               for b in live}
        if new == dom:
            break
        dom = new
    loops = {}
    for block in live:
        for header in succ[block] & dom[block]:
            body = loops.setdefault(header, {header})
            todo = [block]
            while todo:
                node = todo.pop()
                if node not in body:
                    body.add(node)
                    todo.extend(pred[node] & live)
    return list(loops.values()), blocks


def check_control_flow(name, program):
    loops, blocks = loop_blocks(program)
    expected = 0 if name in ELIMINATED else input_loop_count(name)
    if len(loops) != expected:
        raise ValueError(f"{name}: expected {expected} loop headers, got {len(loops)}")
    if len(loops) == 2 and not (loops[0] < loops[1] or loops[1] < loops[0]):
        raise ValueError(f"{name}: expected nested loops")
    if name in {"select", "if_in_loop"}:
        if not any(" = select " in line for block in loops[0] for line in blocks[block]):
            raise ValueError(f"{name}: expected if-conversion inside the loop")
    if name in ACYCLIC:
        if " = select " not in program or re.search(r"^\s*br ", program, re.M):
            raise ValueError(f"{name}: expected an if-converted straight-line program")


def set_arguments(program, values, marker):
    lines = [line for line in program.splitlines() if not line.startswith(marker)]
    if values:
        lines.insert(0, marker + " " + " ".join(map(str, values)))
    return "\n".join(lines) + "\n"


def trace(binary, path):
    text = subprocess.check_output([binary, "--run-mode", "parse", "--interp", path], text=True, timeout=15)
    return text.strip().splitlines()


def input_loop_count(stem):
    if stem in ACYCLIC:
        return 0
    return 2 if stem in NESTED else 1


def pretty_print(upstream, source):
    return subprocess.check_output([upstream, "--run-mode", "pretty-print", source], text=True, timeout=15)


def check_loop_inventory(upstream, directory):
    names = set()
    for p in sorted(directory.iterdir()):
        if p.suffix in (".rs", ".bril") and re.search(r"\bdowhile\s*\(", pretty_print(upstream, p)):
            names.add(p.name)
    expected = set(UPSTREAM_LOOP_ARGS) | OUTSIDE_FRAGMENT
    if names != expected:
        raise ValueError(f"upstream loop inventory changed: missing={expected-names}, unlisted={names-expected}")


def import_fixture(upstream, original, tmp):
    pretty = pretty_print(upstream, original)
    raw = Reader(pretty).read()
    source_loops = sum(n["op"] == "dowhile" for n in raw["nodes"])
    expected = input_loop_count(original.stem)
    if source_loops != expected:
        raise ValueError(f"{original.name}: expected {expected} DoWhile nodes, got {source_loops}")
    if original.name in CASES:
        dag = HERE / "eggcc" / (original.stem + ".json")
        if raw != json.loads(dag.read_text()):
            raise ValueError(f"fixture differs from upstream {original.name}")
    else:
        dag = Path(tmp, original.stem + ".json")
        dag.write_text(json.dumps(raw))
    return dag


def sample_arguments(original, program, marker, all_small_loops):
    if not all_small_loops:
        return SAMPLES[original.stem]
    directives = [line[len(marker):] for line in program.splitlines() if line.startswith(marker)]
    args = tuple(map(int, directives[0].split())) if directives else ()
    if len(directives) > 1 or args != UPSTREAM_LOOP_ARGS[original.name]:
        raise ValueError(f"{original.name}: upstream argument directive changed")
    return [args]


def check(checkout, output, all_small_loops):
    upstream = checkout / "target/release/eggcc"
    compiler = HERE.parent / ".lake/build/bin/eggcc-certified"
    directory = checkout / "tests/passing/small"
    if all_small_loops:
        check_loop_inventory(upstream, directory)
    with tempfile.TemporaryDirectory() as tmp, output.open("x", newline="") as csv_file:
        rows = csv.writer(csv_file, lineterminator="\n")
        rows.writerow(("source", "arguments", "source_trace", "compiled_trace", "lean_input_trace", "lean_output_trace"))
        cases = UPSTREAM_LOOP_ARGS if all_small_loops else CASES
        for name in cases:
            original = directory / name
            program = original.read_text()
            marker = "// ARGS:" if original.suffix == ".rs" else "# ARGS:"
            samples = sample_arguments(original, program, marker, all_small_loops)
            dag = import_fixture(upstream, original, tmp)
            lowered = subprocess.check_output([compiler, "--emit-bril", dag], text=True, timeout=60)
            check_control_flow(original.stem, lowered)
            source = Path(tmp, name)
            compiled = Path(tmp, original.stem + "-selected.bril")
            for args in samples:
                source.write_text(set_arguments(program, args, marker))
                compiled.write_text(set_arguments(lowered, args, "# ARGS:"))
                want, got = trace(upstream, source), trace(upstream, compiled)
                known = expected_loop_trace(original.stem, args)
                if known is not None and known != want:
                    raise ValueError(f"{name} {args}: source interpreter returned {want!r}, expected {known!r}")
                if not want and original.stem not in {"jumping_loop", "simplest_loop"}:
                    raise ValueError(f"{name} {args}: unexpectedly empty print trace")
                native = subprocess.check_output([compiler, dag, *map(str, args)], text=True, timeout=60)
                traces = {}
                for line in native.splitlines():
                    key, _, value = line.partition("=")
                    if key in {"input_trace", "output_trace"}:
                        traces[key] = json.loads(value)
                before, after = traces["input_trace"], traces["output_trace"]
                if not want == got == before == after:
                    raise ValueError(f"{name} {args}: source={want!r}, emitted={got!r}, Lean={before!r}, {after!r}")
                rows.writerow([name, *map(json.dumps, (args, want, got, before, after))])
            csv_file.flush()
            print(f"{name}: all {len(samples)} traces agree")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkout", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--all-small-loops", action="store_true",
                   help="inventory the upstream loop suite, then check the 23 supported programs with their original arguments")
    args = p.parse_args()
    check(args.checkout.resolve(), args.output, args.all_small_loops)
