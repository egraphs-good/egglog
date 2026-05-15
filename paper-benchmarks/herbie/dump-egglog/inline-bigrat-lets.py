#!/usr/bin/env python3
"""Inline Herbie's `(let $N (bigrat …))` bindings into their references.

The Herbie egglog dumps open with bindings like:
    (let $0 (bigrat (from-string "0") (from-string "1")))
    (let $1 (bigrat (from-string "1") (from-string "1")))
…and then reference `$0`, `$1` inside rule bodies and actions. Modern
egglog's term encoder + DuckDB backend stumble on the global-let
lifting for non-eq-sort outputs (BigRat). This script rewrites each
dump in place: it removes the `(let $N V)` declarations and
substitutes every standalone token `$N` with `V`. Then the dump can be
fed through `egglog-experimental --term-encoding` / `--duckdb` without
the term encoder needing to invent a `@$NView`.

Usage:
    python3 inline-bigrat-lets.py <dir-of-egg-files>
    # or in-place across the tarball's extracted contents:
    python3 inline-bigrat-lets.py /tmp/dump-egglog
"""

import os, re, sys

LET_RE = re.compile(r'^\(let\s+(\$\w+)\s+(.*)\)\s*$')


def parens_balanced(s: str) -> bool:
    depth = 0
    for c in s:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def extract_lets(text: str):
    """Walk the file line-by-line; collect leading `(let $N V)` lines.

    Returns (substitutions, rest_of_text). The lets are removed from
    the returned text. V is preserved verbatim — including the outer
    parens of `(bigrat …)`.
    """
    subs = {}
    out_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        m = LET_RE.match(stripped) if stripped.startswith('(let') else None
        if m and parens_balanced(m.group(2)):
            subs[m.group(1)] = m.group(2)
            continue
        out_lines.append(line)
    return subs, '\n'.join(out_lines)


def inline_refs(text: str, subs: dict) -> str:
    """Replace each `$N` token with its bound value.

    Uses a word-boundary regex so `$0` matches but `$001` doesn't.
    `$` is escaped to a literal in the pattern.
    """
    if not subs:
        return text
    keys = sorted(subs.keys(), key=lambda k: -len(k))
    pattern = re.compile('|'.join(re.escape(k) for k in keys) + r'(?![\w])')

    def repl(m):
        # The captured `$N` always sits as a sub-expression in S-expr
        # position; the bound value `subs[$N]` was the right-hand side
        # of `(let $N <RHS>)` which already includes its own parens
        # for `(bigrat …)` etc. — no wrapping needed. A bare-atom RHS
        # like `(let $0 7)` is theoretically possible but Herbie
        # doesn't emit it, and adding parens here would turn it into a
        # function call.
        return subs[m.group(0)]

    return pattern.sub(repl, text)


def process(path: str) -> bool:
    with open(path) as f:
        text = f.read()
    subs, rest = extract_lets(text)
    if not subs:
        return False
    new_text = inline_refs(rest, subs)
    with open(path, 'w') as f:
        f.write(new_text)
    return True


def main(root: str):
    n_changed = 0
    n_total = 0
    for fn in sorted(os.listdir(root)):
        if not fn.endswith('.egg'):
            continue
        n_total += 1
        if process(os.path.join(root, fn)):
            n_changed += 1
    print(f'inlined lets in {n_changed} / {n_total} .egg files in {root}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
