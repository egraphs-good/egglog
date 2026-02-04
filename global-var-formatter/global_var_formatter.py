#!/usr/bin/env python3
import re
import sys
from typing import List, Tuple, Dict


def tokenize(code: str) -> List[Tuple[str, int]]:
    """Minimal tokenizer for S-expressions. Returns (token, position) tuples."""
    tokens = []
    for match in re.finditer(r'\(|\)|[^\s()]+', code):
        tokens.append((match.group(), match.start()))
    return tokens


def parse_top_level_lets(tokens: List[Tuple[str, int]]) -> Dict[str, int]:
    """Extract variable names and positions from top-level let bindings."""
    globals_without_prefix = {}
    i = 0
    depth = 0
    
    while i < len(tokens):
        token, _ = tokens[i]
        if token == '(':
            if depth == 0 and i + 1 < len(tokens) and tokens[i + 1][0] == 'let':
                # Found top-level let
                i += 2 # skip ( and let tokens
                if i < len(tokens):
                    var_name, var_pos = tokens[i]
                    if not var_name.startswith('$') and var_name not in ('(', ')'):
                        globals_without_prefix[var_name] = var_pos
                i -= 1  # Back up to process the opening paren
            depth += 1
        elif token == ')':
            depth -= 1
        i += 1
    
    return globals_without_prefix


def format_file(filepath: str) -> List[str]:
    """Format a Lisp file by prefixing global variables with $. Returns empty list (no conflicts)."""
    with open(filepath, 'r') as f:
        code = f.read()
    
    tokens = tokenize(code)
    globals_to_fix = parse_top_level_lets(tokens)
    
    if not globals_to_fix:
        return []
    
    for var, def_pos in globals_to_fix.items():
        pattern = r'(?<=[\s(]){0}(?=[\s)])'.format(re.escape(var))
        
        def replace_after_def(match):
            if match.start() >= def_pos:
                return f'${var}'
            return match.group()
        
        code = re.sub(pattern, replace_after_def, code)
    
    with open(filepath, 'w') as f:
        f.write(code)
    
    return []


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file.egg>")
        sys.exit(1)
    
    format_file(sys.argv[1])
    print("File formatted successfully!")
