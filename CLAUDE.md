# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Changes should keep `cargo clippy` lints passing and should keep existing tests passing while also adding unit tests for new functionality.

## Project Overview

egglog is a language and engine that combines equality saturation with Datalog-style relational programming. It's the successor to the Rust library `egg`.

## Common Commands

### Building
```bash
# Ensure all lints pass *always do this*
# run cargo fmt regulalry to avoid lints
cargo fmt && cargo clippy --all

# Build in release mode (recommended for performance)
cargo build --release

# Build the profiling version with debug symbols
cargo build --profile profiling
```

### Running
```bash
# Run in REPL mode
cargo run --release

# Run an egglog file
cargo run --release -- tests/example.egg

# Run with parallel execution (N threads, or 0 for max parallelism)
cargo run --release -- -j N tests/example.egg

# Run with logging
RUST_LOG=info cargo run --release -- tests/example.egg

# Generate graphviz output
cargo run --release -- --to-dot tests/example.egg
cargo run --release -- --to-svg tests/example.egg  # requires Graphviz
```

### Testing
```bash
# Run all tests in current crate or workspace.
cargo nextest run

# For tests on the base egglog crate, *only* run in --release mode
cargo nextest run --release
```

### Documentation
```bash
# View documentation in browser
cargo doc --open
```

### Profiling
```bash
# Install samply
cargo install --locked samply

# Build profiling version
cargo build --profile profiling

# Profile an egglog file
samply record ./target/profiling/egglog tests/extract-vec-bench.egg

# Profile without logging/printing messages (for extraction profiling)
RUST_LOG=error samply record ./target/profiling/egglog --no-messages tests/extract-vec-bench.egg
```

## Architecture Overview

### Compilation Pipeline

The egglog system follows a progressive refinement pipeline:

```
User Input (.egg file text)
    ↓
[Parser] → AST (Command, Rule, Expr, Fact)
    ↓
[Desugarer] → NCommand (normalized, sugar removed)
    ↓
[Type Checker] → ResolvedNCommand (type-annotated)
    ↓
[Global Removal] → Core IR (CoreRule, Query, Actions)
    ↓
[Backend Compilation] → egglog_bridge::Rule
    ↓
[Execution] (core-relations database + union-find)
    ↓
Results (Values, Extractions, RunReports)
```

### Key Components

#### Frontend (egglog crate)
- **Parsing** (`src/ast/parse.rs`): Parses egglog syntax into AST
- **Desugaring** (`src/ast/desugar.rs`): Removes syntactic sugar
  - Datatypes → sort + constructors
  - Rewrites → bidirectional rules
  - BiRewrites → two rules
- **Type Checking** (`src/typechecking.rs`, `src/constraint.rs`): Constraint-based type inference
- **EGraph API** (`src/lib.rs`): Main public interface, manages state

#### Backend (egglog-bridge + core-relations)
- **egglog-bridge**: Translation layer between frontend IR and relational database
- **core-relations**: Low-level relation database with efficient join algorithms
- **union-find**: High-performance concurrent union-find structure for equivalence classes

### Core Data Structures

#### EGraph (src/lib.rs:210)
The central data structure:
```rust
pub struct EGraph {
    backend: egglog_bridge::EGraph,          // Low-level backend
    parser: Parser,                          // Syntax parser
    functions: IndexMap<String, Function>,   // Registered functions
    rulesets: IndexMap<String, Ruleset>,     // Rule collections
    type_info: TypeInfo,                     // Type system state
    overall_run_report: RunReport,           // Execution stats
    // ... push/pop stack, proof state, etc.
}
```

Key methods:
- `parse_and_run_program()` - Parse and execute egglog programs
- `eval_expr()` - Evaluate expressions
- `run_rules()` / `step_rules()` - Execute rules
- `extract_value()` - Extract optimal terms with cost model

#### Core IR (src/core.rs)
Low-level intermediate representation:
- **Query**: Conjunctive pattern matching (rule body)
- **CoreActions**: Side effects (unions, sets, deletes)
- **CoreRule**: Query + Actions (the complete rule)

#### TermDag (src/termdag.rs)
Hashconsed term representation for extraction:
```rust
pub enum Term {
    Lit(Literal),              // Constants
    Var(String),               // Variables
    App(String, Vec<TermId>),  // Function applications
}
```

### Module Organization

| Module | Purpose |
|--------|---------|
| `src/lib.rs` | Main EGraph API and execution |
| `src/core.rs` | Core IR (Query, CoreAction, CoreRule) |
| `src/ast/` | AST definitions, parsing, desugaring |
| `src/typechecking.rs` | Type checking and inference |
| `src/constraint.rs` | Constraint solving for type inference |
| `src/sort/` | Sort system (I64, String, Vec, Set, Map, etc.) |
| `src/extract.rs` | Term extraction with cost models |
| `src/term_encoding.rs` | Proof encoding pipeline |
| `src/scheduler.rs` | Rule scheduling and match filtering |
| `src/termdag.rs` | Term DAG representation |

### Workspace Crates

| Crate | Purpose |
|-------|---------|
| `egglog` | Main library and CLI |
| `egglog-bridge` | Translation layer to backend |
| `core-relations` | Low-level relational database |
| `union-find` | High-performance union-find |
| `egglog-ast` | Generic AST framework |
| `egglog-numeric-id` | Efficient numeric ID mapping |
| `egglog-concurrency` | Thread-safe data structures |
| `egglog-reports` | Execution statistics |

### Rule Execution Model

Rules follow a seminaive evaluation strategy:

1. **Desugaring**: High-level rules → CoreRules
2. **Compilation**: CoreRules → Backend queries + actions
3. **Query Execution**: Pattern matching via free-join algorithms
4. **Action Application**: Unions, sets, deletes applied to database
5. **Congruence Closure**: Union-find maintains equivalence classes
6. **Rebuild**: Functions rebuilt after unions

Parallelism is supported via `-j` flag using rayon's work-stealing thread pool.

### Sort System

Sorts are the type system of egglog:

**Base Sorts** (scalars):
- `I64Sort`, `F64Sort` - Numbers
- `StringSort`, `BoolSort` - Text and logic
- `BigIntSort`, `BigRatSort` - Arbitrary precision
- `UnitSort` - Unit type

**Container Sorts** (parameterized):
- `Vec<T>` - Vectors
- `Set<T>` - Sets
- `Map<K,V>` - Maps
- `MultiSet<T>` - Multisets
- `Function<Inputs, Output>` - First-class functions

Sorts are registered via the `Presort` trait for parameterized types.

### Type Checking

Uses constraint-based type inference:

1. Collect type constraints from primitives and function signatures
2. Solve constraints iteratively with bidirectional propagation
3. Report errors with span information for ambiguous or impossible constraints

Type constraints support:
- Simple unification
- Conjunction (AND)
- XOR (for overloaded primitives)

### Extraction

Extraction finds optimal terms according to a cost model:

- **Default**: `TreeAdditiveCostModel` - sum of node costs
- **Algorithm**: Bellman-Ford-style incremental cost computation
- **Customization**: Implement `CostModel` trait for custom strategies

### Proof Support

When enabled via `EGraph::new_with_term_encoding()`, instruments constructors with:
- Term tables (provisional term IDs)
- View tables (canonical ID to term mapping)
- Reason tables (proof steps)

This allows reconstructing equality proofs while preserving semantics.

## Development Workflow

### Adding a New Primitive
1. Implement `Primitive` trait
2. Use `add_primitive!` macro in `EGraph::default()` or custom setup
3. Define type constraints in `get_type_constraints()`

### Adding a New Sort
1. Create struct implementing `Sort` trait in `src/sort/`
2. Register in `EGraph::default()` via `add_base_sort()`
3. Implement `register_primitives()` for sort-specific operations
4. For parameterized sorts, implement `Presort` trait

### Debugging
- Set `RUST_LOG=info` or `RUST_LOG=debug` for detailed execution logs
- Use `(print-function name)` in egglog to inspect function contents
- Use `(print-stats)` to see execution statistics
- Check `RunReport` via `get_overall_run_report()` for rule firing counts

### Performance Considerations
- Benchmarks run via codspeed CI - check performance reports on PRs
- Benchmarks < 50ms are ignored in codspeed to reduce noise
- Use profiling build for investigating performance issues
- Avoid pessimizing parallel performance (don't add coarse-grained locks)

## Project-Specific Notes

### Main Branch
The main branch is `main`. PRs should target this branch.

### Testing
- Tests in `tests/*.egg` are automatically discovered and run
- Test failures should preserve error messages for debugging
- Use `(fail ...)` wrapper to test expected failures
- Type checking failures go in `tests/fail-typecheck/`

### Seminaive Evaluation
The default execution mode uses seminaive evaluation (incremental rule execution). This can be disabled via `EGraph.seminaive = false` but is recommended for most use cases.

### Parallel Execution
Parallelism is **disabled by default** (single-threaded) to ensure deterministic behavior. Enable with `-j` flag. The number of threads defaults to 1; use 0 for maximum parallelism.

### Global Variables
Global variables should be prefixed with `$` (e.g., `$global`). Enable strict mode via `set_strict_mode(true)` to enforce this.

### Value Representation
- **BaseValue**: Scalars stored directly in Value (i64, f64, etc.)
- **ContainerValue**: Collections stored in separate heap-allocated table
- All values are reference-counted via `Value` wrapper (8 bytes)
