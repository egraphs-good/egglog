# egglog

egglog is a unified tool for program optimization and program analysis
 and can be conceptually thought of as Datalog + a union-find data structure.

## Project structure

- The main `egglog` crate has the source code under `./src/` and is concerned with parsing, desugaring, proofs, type checking, and lowering.
- `core-relations` is the backing database implementation. It supports both parallel and serial modes. By default, it runs in serial mode and uses only a single thread.
- `egglog-bridge` is the bridge between the main crate and the core-relations, and provides abstractions/utility for rules, union-find, rows, etc.
- The `tests` folder contains a list of example egglog programs.

## Best practices

When you edit the files, make sure to respect the following:

- Always run `make fixnits` (or `make nits`) before you stop.
- When running tests, always use the `--release` mode. Alternatively, you can also run `make test`.
- If your change is performance-critical, use `script/bench.py` as the ground truth to evaluate the performance impact.
- Keep your documentation concise and avoid duplicate information. The `tidy-diff-docs` skill (`.claude/skills/tidy-diff-docs/`) cleans the doc and code comments in a diff down to the caller-facing contract.
- Update CHANGELOG.md with a concise bullet when you make major changes (e.g., breaking changes or new features added) in the codebase.
