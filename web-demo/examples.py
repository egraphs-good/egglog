#!/usr/bin/env python3

import json
import sys
from pathlib import Path

if len(sys.argv) <= 1:
    print("ERROR: give some files as input")
    sys.exit(1)

files = sorted(sys.argv[1:])

result = {}
for filename in files:
    with open(filename) as f:
        name = Path(filename).stem
        result[name] = f.read()

json.dump(result, sys.stdout, indent=2)