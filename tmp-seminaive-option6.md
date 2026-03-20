# Option 6: Full Dependency Tracking for Query Primitives

## Overview

Option 6 is the general solution to the seminaive soundness problem for query
primitives.

Today, seminaive rule execution is driven by table deltas. That works as long as
the truth of a rule body only changes when a new row appears in some table. The
container bug showed a broader failure mode:

- a query primitive or matcher can fail at time `t`
- the underlying runtime state can change during rebuild
- the same query primitive call can succeed at time `t + 1`
- no new table row is emitted, so seminaive never revisits the old match

The container-specific same-id refresh fix is a targeted repair for one instance
of that pattern. Option 6 would make the runtime track primitive dependencies
directly, so invalidation is driven by what a primitive actually observed.

## Core Model

Every query-primitive evaluation would record the state it depended on.

That dependency record must cover both:

- successful evaluations, where the primitive returned values that can feed the
  rest of the rule
- failed evaluations, where the primitive returned `None` or otherwise filtered
  the match out

Tracking failures is essential. The container bug is exactly a case where a
previously failing query-side predicate becomes true later.

At a high level, the runtime would need something like:

```rust
struct PrimitiveDependencyRecord {
    primitive: PrimitiveId,
    invocation_key: InvocationKey,
    deps: DependencySet,
    outcome: PrimitiveOutcome,
}
```

where:

- `InvocationKey` identifies the primitive call within a specific partial or
  complete query match
- `DependencySet` records the pieces of runtime state that were read
- `PrimitiveOutcome` distinguishes success from failure and captures any cached
  output needed for incremental maintenance

## Dependency Kinds

The dependency set would likely need to support several classes of reads:

- union-find state
  - specific ids or eclasses
  - possibly canonical representatives or equivalence tests
- container state
  - specific container ids
  - possibly nested container ids if reads are recursive
- table state
  - specific rows
  - specific lookup keys
  - coarse table-level epochs as a fallback
- global runtime state
  - anything exposed through `ExecutionState`
  - any primitive-local mutable caches if they affect query answers

The more precise the dependency model, the more precise invalidation can be.

## Invalidation Granularity

There are several levels of invalidation Option 6 could support.

### 1. Coarse domain invalidation

Track only broad domains:

- any UF change
- any container rebuild
- any write to a given table

This is the simplest form of dependency tracking, but it is still more precise
than globally disabling seminaive for all query primitives. It would let the
runtime rerun only rules whose query primitives depend on a changed domain.

### 2. Per-object invalidation

Track concrete objects:

- specific eclasses
- specific container ids
- specific table keys or rows

This is the natural generalization of the container-specific refresh fix. It
reduces unnecessary reruns, but it needs reverse maps from each dependency kind
back to the primitive invocations that observed it.

### 3. Per-match invalidation

Track dependencies for each primitive call inside each active rule match, then
only invalidate the affected partial matches.

This is the most precise and potentially the fastest long-term approach, but it
requires substantially more runtime machinery. The system would need a stable
representation for partial query work items, not just final table deltas.

## Required Runtime Infrastructure

Compared with the current container-only fix, Option 6 needs several new pieces
of infrastructure.

### Primitive dependency reporting

The `Primitive` interface would need a query-time execution path that can report
what it read. A possible shape is:

```rust
fn apply_query(
    &self,
    values: &[Value],
    exec_state: &mut ExecutionState,
    deps: &mut DependencyRecorder,
) -> Option<Value>;
```

The key point is that the primitive must be able to log dependencies while it is
deciding both success and failure.

### Reverse dependency indexes

Once dependencies are recorded, the runtime needs reverse maps from a changed
piece of state to the primitive invocations that depend on it.

Examples:

- `Value/eclass -> primitive invocations`
- `container id -> primitive invocations`
- `table key -> primitive invocations`

These reverse indexes are the conceptual analogue of the per-table reverse index
in Option 3, but generalized across all stateful query primitives.

### Stable invocation identities

The runtime needs a stable notion of "this primitive call inside this rule
match". That is harder than it sounds because seminaive query execution is
currently optimized around table subsets and row timestamps, not explicit
materialized partial matches.

This means Option 6 likely needs either:

- materialized match-state records for stateful query primitives, or
- a way to reconstruct and invalidate the right partial work from table rows and
  variable bindings

### Cache lifecycle management

Because both successes and failures must be tracked, the system needs rules for:

- when dependency records are installed
- when they are invalidated
- when they are reclaimed
- how they interact with rebuild, push/pop, and snapshotting

Without that lifecycle management, the dependency cache will either leak or keep
rerunning work that is no longer relevant.

## Why Option 6 Is More Principled

Option 6 solves the real problem:

- query primitive truth can depend on evolving runtime state
- seminaive needs explicit invalidation when that state changes

Instead of encoding one special case for containers, Option 6 gives the runtime a
uniform story for:

- container-sensitive predicates
- UF-sensitive predicates
- table-sensitive predicates
- other stateful query primitives

That makes the semantics easier to explain and easier to extend.

## Why Option 6 Is More Invasive

Option 6 is materially more invasive than Option 3 for a few reasons.

### It changes the primitive contract

Today query primitives are opaque callbacks. Option 6 requires them to
participate in dependency reporting, or at least declare enough metadata for the
runtime to synthesize those dependencies.

### It changes the scheduler model

Current seminaive scheduling is based on table deltas. Option 6 introduces a new
source of invalidation: primitive dependency changes. That means the scheduler
must know how to wake up work even when no table row changed.

### It needs new indexes and caches

The runtime would need dependency indexes and invalidation state alongside its
existing row indexes. That is a broader architectural commitment than the
container-specific row-refresh path.

### It must handle failed matches

Successful derivations already fit naturally into table deltas. Failed primitive
matches do not. Option 6 only works if the runtime can remember failed
invocations and retry them when their dependencies change.

That is the biggest conceptual jump from the current implementation.

## Practical Recommendation

Option 6 is the right end-state if query primitives become an important and
performance-sensitive part of the engine.

It is probably too large for a short-term fix.

The pragmatic path is:

1. keep the targeted container repair in Option 3
2. optionally add coarse primitive metadata or domain invalidation if broader
   stateful cases become urgent
3. only build full dependency tracking if the broader class of stateful query
   primitives becomes important enough to justify the complexity

That keeps performance close to the fast container fix while leaving a clear
direction for a principled long-term design.
