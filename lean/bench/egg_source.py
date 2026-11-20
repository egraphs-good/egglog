"""Read and re-emit the egglog syntax used in the compiler benchmark fixtures."""
import re

TOKEN = re.compile(r'\s+|;[^\n]*|"(?:[^"\\]|\\.)*"|[()]|[^\s();"\\]+')


def commands(text):
    stack = [[]]
    i = 0
    while i < len(text):
        m = TOKEN.match(text, i)
        if not m:
            raise ValueError(f"unrecognized input at {i}")
        word = m[0]
        i = m.end()
        if word.isspace() or word.startswith(";"):
            continue
        if word == "(":
            stack.append([])
        elif word == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced input")
            term = stack.pop()
            stack[-1].append(term)
        else:
            stack[-1].append(word)
    if len(stack) != 1:
        raise ValueError("unbalanced input")
    return stack[0]


def render(term):
    if isinstance(term, list):
        return "(" + " ".join(render(i) for i in term) + ")"
    return term
