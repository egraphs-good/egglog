# egglog

egglog is a unified tool for program optimization and program analysis
 and can be conceptually thought of as Datalog + a union-find data structure.

## Project structure

- The main `egglog` crate has the source code under `./src/` and is concerned with parsing, desugaring, proofs, type checking, and lowering.
- `core-relations` is the backing database implementation. It supports parallel mode and serial mode. By default it runs the serial mode and use only a single thread.
- `egglog-bridge` is the bridge between the main crate the the core-relations, and provide abstractions/utility for rules, union-find, rows, etc.
- The `tests` folder contain a list of example egglog programs.

## Best practices

When you edit the files, make sure to respect the following:

- Always run `make fixnits` (or `make nits`) before you stop.
- When running test, always use the `--release` mode. Alternatively, you can also run `make test`.
- If your change is performance critical, use `script/bench.py` as the ground truth to evaluate the performance impact.
- Keep your documentation concise and avoids duplicate information.
