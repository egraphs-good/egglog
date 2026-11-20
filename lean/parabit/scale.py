#!/usr/bin/env python3
"""Generate scaled BWLang equivalences and time their solvers with the CAV harness."""
import argparse
import csv
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parabit.compare import ALL_MODES, FIELDS, MODES, check_load, measure


def bw(width, term):
    return ("bw", str(width), term)


def tree(lo, hi, leaf, branch):
    if hi - lo == 1:
        return leaf(lo)
    mid = (lo + hi) // 2
    return branch(lo, hi, tree(lo, mid, leaf, branch), tree(mid, hi, leaf, branch))


def mux(width, lo, hi, left, right):
    return bw(width, ("sel", bw(1, f"pick{lo}_{hi}"), left, right))


def mirror_add(n, negative):
    """Swap the two children of every adder. Cast width varies with its depth."""
    def side(reverse):
        def leaf(i):
            return bw("w0", "0" if reverse and negative and i == n - 1 else f"x{i}")

        def add(lo, hi, left, right):
            w = f"w{(hi - lo).bit_length() - 1}"
            return bw(w, ("+", right, left) if reverse else ("+", left, right))
        return tree(0, n, leaf, add)
    widths = {f"w{k}": 3 for k in range(n.bit_length())}
    return side(False), side(True), [], widths, f"x{n-1}"


def guarded_assoc(n, negative):
    """Select among n distinct ROVER add_assoc_4 sites under its four width guards."""
    def side(right):
        def triple(i):
            a = bw("p", "0" if right and negative and i == n - 1 else f"a{i}")
            b, c = bw("r", f"b{i}"), bw("s", f"c{i}")
            if right:
                return bw("t", ("+", a, bw("q", ("+", b, c))))
            return bw("t", ("+", bw("u", ("+", a, b)), c))
        return tree(0, n, triple, lambda lo, hi, a, b: mux("t", lo, hi, a, b))
    guards = [("<", "r", "q"), ("<", "s", "q"), ("<", "p", "u"), ("<", "r", "u")]
    widths = dict(p=2, r=3, s=2, q=4, u=4, t=5)
    return side(False), side(True), guards, widths, f"a{n-1}"


def mux_add(n, negative):
    """Factor one modular adder out of a full binary tree of selectors (ROVER sel_add)."""
    def a(i):
        return bw("w", f"a{i}")

    def b(i):
        return bw("w", f"b{i}")

    def add(x, y):
        return bw("w", ("+", x, y))

    def network(leaf):
        return tree(0, n, leaf, lambda lo, hi, x, y: mux("w", lo, hi, x, y))

    lhs = network(lambda i: add(a(i), b(i)))
    rhs = add(network(lambda i: bw("w", "0") if negative and i == n - 1 else a(i)), network(b))
    return lhs, rhs, [], {"w": 4}, f"a{n-1}"


FAMILIES = {"mirror-add": mirror_add, "guarded-assoc": guarded_assoc, "mux-add": mux_add}


def render(t):
    if isinstance(t, str):
        return t
    return "(" + " ".join([t[0], *map(render, t[1:])]) + ")"


def ast_nodes(t):
    yield t
    if isinstance(t, tuple):
        for x in t[1:]:
            yield from ast_nodes(x)


def evaluate(t, env):
    if isinstance(t, str):
        return int(t) if t.isdecimal() else env.get(t, 0)
    op, *xs = t
    vs = [evaluate(x, env) for x in xs]
    if op == "bw":
        return vs[1] % (1 << vs[0])
    if op == "+":
        return vs[0] + vs[1]
    if op == "<":
        return vs[0] < vs[1]
    if op == "sel":
        return vs[1] if vs[0] % 2 else vs[2]
    raise ValueError(op)


def generate(args):
    # Build each file only in a fresh corpus directory to avoid leaving stale cases.
    args.corpus.mkdir(parents=True, exist_ok=True)
    if any(args.corpus.iterdir()):
        raise SystemExit(f"choose an empty generated corpus directory: {args.corpus}")
    rows = []
    for family in args.family or FAMILIES:
        folder = args.corpus / family / "bwlang"
        folder.mkdir(parents=True)
        for n in sorted(set(args.sizes)):
            for negative in ([False, True] if args.controls else [False]):
                name = f"{family}-{n:04d}" + ("-negative" if negative else "")
                lhs, rhs, guards, widths, live = FAMILIES[family](n, negative)
                env = {**widths, live: 1}
                if not all(evaluate(g, env) for g in guards):
                    raise RuntimeError(f"{name}: witness violates a width guard")
                result = evaluate(lhs, env), evaluate(rhs, env)
                if result != (1, 0 if negative else 1):
                    raise RuntimeError(f"{name}: expected witness disagrees with {result}")
                file = folder / (name + ".bwlang")
                data = dict(name=name, preconditions=list(map(render, guards)),
                            lhs=render(lhs), rhs=render(rhs))
                file.write_text(json.dumps(data, indent=2) + "\n")
                terms = [*ast_nodes(lhs), *ast_nodes(rhs)]
                rows.append(dict(case=file.relative_to(args.corpus).as_posix(), family=family,
                                 size=n, expected="inequivalent" if negative else "solved",
                                 ast_nodes=len(terms), operators=sum(isinstance(t, tuple) for t in terms),
                                 unique_subterms=len(set(terms)), symbolic_widths=len(widths),
                                 bytes=file.stat().st_size,
                                 witness=json.dumps(env, sort_keys=True) if negative else ""))
    with (args.corpus / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cases to {args.corpus} (witnesses default unlisted variables to zero).")


def run(args):
    with (args.corpus / "manifest.csv").open() as f:
        rows = [r for r in csv.DictReader(f)
                if (not args.family or r["family"] in args.family)
                and (not args.sizes or int(r["size"]) in args.sizes)
                and (r["expected"] == "inequivalent" if args.only_controls
                     else args.controls or r["expected"] == "solved")]
    if not rows:
        raise SystemExit("No generated cases selected")
    modes = args.mode or list(MODES)
    rng = random.Random(42)
    check_load()
    # Warm the executable mappings with the smallest selected input. Each measured
    # process still creates a fresh solver and initializes the authors' clock.
    if not args.no_warmup:
        smallest = min(rows, key=lambda r: int(r["ast_nodes"]))
        for mode in modes:
            measure(mode, args.corpus / smallest["case"], args)
    with args.output.open("x", newline="") as f:
        writer = csv.DictWriter(f, FIELDS, lineterminator="\n")
        writer.writeheader()
        for trial in range(1, args.trials + 1):
            order = rows.copy()
            rng.shuffle(order)
            for row in order:
                check_load()
                sequence = modes.copy()
                rng.shuffle(sequence)
                for mode in sequence:
                    status, ms, solver, runner, detail = measure(mode, args.corpus / row["case"], args)
                    writer.writerow(dict(case=row["case"], mode=mode, trial=trial, status=status,
                                         wall_ms=ms, solver_ms=solver, runner_ms=runner, detail=detail))
                    f.flush()
                    if status == "error":
                        raise SystemExit(f"{mode} on {row['case']}: {detail}")
                    if row["expected"] == "inequivalent" and status == "solved":
                        raise SystemExit(f"unsound result for {mode} on {row['case']}")
                print(f"Finished trial {trial} of {row['case']}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_subparsers(dest="action", required=True)
    gen = actions.add_parser("generate")
    gen.add_argument("corpus", type=Path)
    gen.add_argument("--sizes", nargs="+", type=int, default=[2, 8, 32, 128])
    bench = actions.add_parser("run")
    bench.add_argument("artifact", type=Path, help="pinned CAV solver checkout")
    bench.add_argument("corpus", type=Path)
    bench.add_argument("output", type=Path)
    bench.add_argument("--sizes", nargs="+", type=int)
    for parser in [gen, bench]:
        parser.add_argument("--family", choices=FAMILIES, action="append")
    gen.add_argument("--controls", action="store_true", help="also generate inequivalent last-leaf mutations")
    controls = bench.add_mutually_exclusive_group()
    controls.add_argument("--controls", action="store_true", help="include both positives and mutations")
    controls.add_argument("--only-controls", action="store_true", help="run only inequivalent last-leaf mutations")
    bench.add_argument("--mode", choices=ALL_MODES, action="append")
    bench.add_argument("--no-warmup", action="store_true",
                       help="skip executable warm-up for single-case limit follow-ups")
    bench.add_argument("--trials", type=int, default=3)
    bench.add_argument("--seconds", type=float, default=10)
    bench.add_argument("--rounds", type=int, default=12, help="relational simplification round bound")
    bench.add_argument("--cert-rounds", type=int, default=12, help="certified saturation round bound")
    args = p.parse_args()
    if args.sizes and any(n < 2 or n & (n - 1) for n in args.sizes):
        p.error("sizes must be powers of two, at least 2")
    if args.action == "generate":
        generate(args)
    elif min(args.trials, args.seconds, args.rounds, args.cert_rounds) <= 0:
        p.error("all budgets must be positive")
    else:
        run(args)


if __name__ == "__main__":
    main()
