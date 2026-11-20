"""Run eggcc's body-containment and context propagation rules on the PolyBench 2mm Tree IR."""
import re
from pathlib import Path

from bench.egg_source import commands, render

SOURCE = Path(__file__).resolve().parents[2] / "tests/eggcc-2mm.egg"
TABLES = ("BodyContainsExpr", "BodyContainsListExpr", "ContextOf")
# Ordinary children evaluated in the current region. Branch and loop bodies
# instead introduce a new argument, so reachability must stop at that boundary.
CHILDREN = {"Top": (1, 2, 3), "Bop": (1, 2), "Uop": (1,), "Get": (0,),
            "Concat": (0, 1), "Single": (0,), "Switch": (0, 1), "If": (0, 1),
            "DoWhile": (0,), "Call": (1,), "Alloc": (1, 2)}
REGIONS = {"DoWhile": (1,), "If": (2, 3), "Function": (3,)}


def oracle(bindings):
    # Independently walk the original, unsaturated DAG. Intern nodes to mirror
    # structural equality, and retain a name from the input for every witness.
    parents = {}
    intern = {}
    bodies = set()
    contexts = {}
    nodes = {}
    for _, name, (op, *args) in bindings:
        args = tuple(parents.get(a, a) for a in args)
        key = op, args
        node = intern.setdefault(key, name)
        parents[name] = node
        nodes[node] = key
        for pos in REGIONS.get(op, ()):
            bodies.add(args[pos])
        if op in ("Arg", "Const", "Empty"):
            contexts[node] = {args[-1]}
        else:
            positions = (0,) if op == "Switch" else CHILDREN.get(op, ())
            contexts[node] = set().union(*(contexts.get(args[pos], set()) for pos in positions))
    pairs = set()
    for root in bodies:
        todo = [root]
        while todo:
            term = todo.pop()
            if (root, term) in pairs:
                continue
            pairs.add((root, term))
            op, args = nodes[term]
            todo.extend(args[pos] for pos in CHILDREN.get(op, ()))
    context_pairs = {(node, ctx) for node, ctxs in contexts.items() for ctx in ctxs}
    return [pairs, set(), context_pairs]


def eggcc_workload():
    program = commands(SOURCE.read_text())
    schema_end = next(i for i, cmd in enumerate(program) if cmd[:2] == ["ruleset", "always-run"])
    schema = program[:schema_end + 1]
    init, = (cmd for cmd in program if cmd[:2] == ["rule", []] and
             ":ruleset" in cmd and cmd[cmd.index(":ruleset")+1] == "initialization")
    bindings = [act for act in init[2] if act[0] == "let"]
    if len(bindings) != 2699 or any(act[0] not in ("let", "FunctionHasType") for act in init[2]):
        raise ValueError("unexpected eggcc initialization")
    rules = []
    for cmd in program:
        if cmd[0] == "relation" and cmd[1] in TABLES:
            rules.append(cmd)
        if cmd[0] == "rule" and cmd[2] and all(action[0] in TABLES for action in cmd[2]):
            if cmd[cmd.index(":ruleset")+1] == "always-run":
                rules.append(cmd)
    # The source's unconditional initialization holds the complete term DAG as
    # local lets. Lift these actions to top level so the oracle can refer to them.
    setup = "\n".join(map(render, schema + rules + init[2]))
    # Give Rust conventionally named globals to avoid warnings about non-$ lets.
    def qualify(text):
        return re.sub(r'\b__tmp(\d+)\b', r'$__tmp\1', text)
    facts = oracle(bindings)
    checks = [f"(check ({table} {a} {b}))" for table, pairs in zip(TABLES, facts)
              for a, b in sorted(pairs)]
    observed = (*TABLES, "Function", "If", "DoWhile")
    checks.extend(f"(print-size {name})" for name in observed)
    counts = tuple(map(len, facts)) + tuple(sum(term[0] == op for _, _, term in bindings)
                                             for op in observed[3:])
    return (qualify(setup), "(run-schedule (saturate always-run))",
            qualify("\n".join(checks))), counts
