"""Translate eggcc's pretty-printed straight-line/branch Tree IR into a shared DAG."""
import argparse
import json
from pathlib import Path
import re
import subprocess

CASES = ("select_simple.rs", "if_dead_code.bril", "if_dead_code_nested.bril")
TOKENS = re.compile(r'\s+|"(?:\\.|[^"\\])*"|[A-Za-z_][A-Za-z_0-9]*|-?\d+|[!()=;,.]')
IDENT = re.compile(r'[A-Za-z_]\w*')
OPS = {"add": 2, "sub": 2, "mul": 2, "less_than": 2, "less_eq": 2,
       "greater_than": 2, "greater_eq": 2, "eq": 2, "not": 1, "tif": 4,
       "tprint": 2, "ttrue": 0, "tfalse": 0, "single": 1}


class Reader:
    def __init__(self, text):
        self.tokens = []
        i = 0
        while i < len(text):
            token = TOKENS.match(text, i)
            if not token:
                raise ValueError(f"unsupported syntax at {text[i:i+35]!r}")
            if not token[0].isspace():
                self.tokens.append(token[0])
            i = token.end()
        self.pos = 0
        self.bindings = {}
        self.nodes = []

    def token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self, expected=None):
        word = self.token()
        if word is None or expected is not None and word != expected:
            raise ValueError(f"expected {expected}, got {word} at token {self.pos}")
        self.pos += 1
        return word

    def expr(self):
        op = self.take()
        if op.startswith('"'):
            return "string", json.loads(op)
        if re.fullmatch(r"-?\d+", op):
            return "number", int(op)
        if not IDENT.fullmatch(op):
            raise ValueError(f"expected expression, got {op!r}")
        if self.token() == ".":
            for word in (".", "clone", "(", ")"):
                self.take(word)
            if op not in self.bindings:
                raise ValueError(f"unknown binding {op}")
            return self.bindings[op]
        if self.token() == "!":
            self.take("!")
            if op not in ("tuplet", "parallel"):
                raise ValueError(f"unsupported macro {op}")
        self.take("(")
        args = []
        if self.token() != ")":
            args.append(self.expr())
            while self.token() == ",":
                self.take(",")
                args.append(self.expr())
        self.take(")")
        kinds = [kind for kind, _ in args]
        vs = [v for _, v in args]
        if op in ("intt", "statet", "boolt") and not args:
            return "type", op[:-1]
        if op == "tuplet" and all(k == "type" for k in kinds):
            return "tuple", vs
        if op == "function" and kinds == ["string", "tuple", "tuple", "node"]:
            if vs[0] != "main" or vs[2] != ["state"]:
                raise ValueError("expected main returning one state")
            return "function", {"inputs": vs[1], "root": vs[3]}
        node = {"op": op}
        if op in ("getat", "int") and kinds == ["number"]:
            if op == "getat" and vs[0] < 0:
                raise ValueError("negative projection")
            if op == "int" and not -(2**63) <= vs[0] < 2**63:
                raise ValueError("integer does not fit i64")
            node["n"] = vs[0]
        elif op == "get" and kinds == ["node", "number"] and vs[1] >= 0:
            node.update(args=[vs[0]], n=vs[1])
        elif (op == "parallel" or OPS.get(op) == len(args)) and all(k == "node" for k in kinds):
            node["args"] = vs
        else:
            raise ValueError(f"unsupported operation {op} with {kinds}")
        self.nodes.append(node)
        return "node", len(self.nodes) - 1

    def read(self):
        result = None
        for _ in range(self.tokens.count(";")):
            self.take("let")
            name = self.take()
            if not IDENT.fullmatch(name) or name in self.bindings:
                raise ValueError(f"invalid or duplicate let {name}")
            self.take("=")
            result = self.bindings[name] = self.expr()
            self.take(";")
        if self.token() is not None or not result or result[0] != "function":
            raise ValueError("expected a complete entry function as the final binding")
        return {**result[1], "nodes": self.nodes}


def upstream(project, output):
    binary = project / "target/release/eggcc"
    for name in CASES:
        case = project / "tests/passing/small" / name
        text = subprocess.check_output([binary, "--run-mode", "pretty-print", case], text=True)
        dag = Reader(text).read()
        path = output / (case.stem + ".json")
        path.write_text(json.dumps(dag, separators=(",", ":")) + "\n")
        print(f"{name}: {len(dag['nodes'])} nodes, {sum(n['op']=='tif' for n in dag['nodes'])} conditional regions")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkout", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    upstream(args.checkout.resolve(), args.output)
