---
name: tidy-diff-docs
description: Clean up the documentation and code comments introduced by a diff before merging. Strips implementation details, performance micro-claims, and restated or duplicate explanations from doc comments while keeping the caller-facing contract intact. Use when reviewing or preparing a PR's doc/regular comments, or when a reviewer asks to tidy up the docs and comments in a change.
---

# Tidy diff docs

A doc comment is a contract for the *caller*: what the item does, what it
guarantees, what the caller must uphold, and what can go wrong. It is not the
place to narrate *how* the thing is implemented or how fast it is — that
information rots, duplicates the code, and buries the contract.

This skill trims the comments **added or changed by a diff** down to that
contract and removes restated text. It matches the repo rule in `CLAUDE.md`:
"Keep your documentation concise and avoid duplicate information."

Two guardrails:

- Scope edits to the diff. Do not rewrite untouched docs elsewhere.
- Never drop information the caller needs to use the item correctly.

## Process

1. Collect the changed comments: `git diff main...HEAD` (or the working /
   staged diff under review). Look at every added or modified `///`, `//!`,
   and `//` line.
2. For each one, apply the rules below.
3. Re-run `make fixnits` and the relevant tests. Intra-doc links such as
   `` [`Foo`] `` must still resolve after the edits.

## Strip these from doc comments

- **Allocation / memory strategy** — "no per-row allocation", "stays off the
  heap", "one allocation, often pooled", "never touches the heap".
- **Internal data structures** — "backed by a `SmallVec<[_; 8]>`", "wraps a
  `TaggedRowBuffer`", "would force a boxed `dyn Iterator`" — unless the type
  actually appears in the public signature the caller sees.
- **Performance micro-claims** — "the monomorphized fast path", "as cheap as
  a direct streaming scan", "measurably slows the scan".
- **Rationale for an internal design choice** — "We deliberately do not
  implement `IntoIterator` because …". If it is worth recording at all, it is
  a plain `//` comment next to the code, not public rustdoc.
- **Restatement of what is already documented nearby** — if the type's doc
  already says its rows are `(inputs, output)`, a method returning that type
  need not repeat the shape; if a parameter's meaning is obvious from its name
  and type, do not spell it out again.

## Keep these

- What the item does and the shape of what it returns.
- Preconditions, invariants the caller must uphold, and the error / panic
  conditions.
- Units, ordering, and edge-case behavior (empty input, duplicates, etc.).
- Intra-doc links to related items.
- Safety requirements on `unsafe` items.

## Examples (from PR #901)

Before:

```rust
/// Call `f` on each [`Enode`] of a constructor / relation table. Rows
/// stream in batches, so there is no whole-table copy and no per-row
/// allocation — it is as cheap as a direct streaming backend scan.
/// Errors with `WrongSubtype` if `name` is a function table. To stop
/// part way through, use [`Read::constructor_enodes_while`].
```

After:

```rust
/// Call `f` on each [`Enode`] of a constructor / relation table.
/// Errors with `WrongSubtype` if `name` is a function. To stop early,
/// use [`Read::constructor_enodes_while`].
```

Before:

```rust
/// One enode from [`Read::constructor_enodes`]. Columns are raw
/// [`Value`]s borrowed from the streaming scan buffer, so there is no
/// per-row allocation. (We deliberately do *not* implement
/// `IntoIterator`: expressing its associated `IntoIter` type would
/// force a boxed `dyn Iterator`, whose per-row virtual dispatch slows
/// the scan.)
```

After:

```rust
/// One enode from [`Read::constructor_enodes`]. Columns are raw
/// [`Value`]s; convert with [`Core::value_to_base`].
```
