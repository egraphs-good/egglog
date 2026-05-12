# Souffle backend perf ideas

Captured 2026-05-12 while investigating math-microbench timeout.
Most efficient first.

## 1. Use souffle's native semi-naive (the one being tried now)

The wave column + IterCounter + rotation + canonical projection are all
re-implementing what souffle's @delta tracking already does well — at
a level above souffle's machinery, so souffle can't use its native
fast path. Plus rotation emits N clauses per rule, blowing up the
clause count.

Fix: modify the fork so each SCC subroutine call does **one** inner
semi-naive iteration (not a full fixpoint LOOP). Combined with
`outer-saturate=N`, this naturally gives N passes — matching
`(run N)` semantics. Restore `@snap_R := main_R` at SCC exit so
`@delta_R` at next entry is "new rows since last iter."

Then drop from the translator:
- The wave column.
- IterCounter atom injection.
- Rotation (semi-naive rotation becomes souffle's job).
- Canonical projection.
- Dedup subsumption.
- Wave preservation on drain/rebuild.

Estimated impact: should approach native egglog runtime (modulo
record-vs-int overhead, factor ~10x).

## 2. Drop buffer/drain (eliminate one relation per view)

Currently each view has a sibling `<view>_buffer` and a drain rule
copying buffer → view. The buffer existed to break a cycle in the
old strata design. With the wave column (or souffle's semi-naive)
doing cycle-breaking, the buffer is dead weight.

Translator-side rewrite:
- When a Set head targets `<view>_buffer`, redirect to `<view>`.
- Skip emitting drain rules (or just don't translate them).
- Skip declaring `<view>_buffer` (or declare but unused — harmless).
- Same for `_snap`, already unused since phase 60c.

Estimated impact: ~13 relations + ~13 clauses cut on math-microbench.
Modest. Mostly noise-reduction.

## 3. Integer term IDs instead of records (deepest win)

Every term is currently a souffle record `[tag, a, b, n]`. Each join
compares fields via `ord()`. Native egglog uses a single integer per
e-class — joins are plain int joins, an order of magnitude faster
on souffle's RAM evaluator.

Challenge: souffle doesn't have native "allocate fresh int"
semantics. Options:
- Stick with `ord()`-based interning but materialize the int once
  per term and reuse it (avoid repeated ord-of-record on hot paths).
- Pre-allocate IDs at encoding time (only works for bounded
  programs that the encoder fully enumerates).
- Hash-cons via a separate "term → id" relation, populated by a
  small initialization stratum.

This is a substantial rewrite. Worth doing if #1 isn't enough.

## 4. Inline let-create rules

`(let v (Add a b))` currently expands into:
- A separate souffle rule writing `AddView_buffer` for v.
- A self-loop UF init rule for v.
- A let-lookup atom added to every dependent rule's body.

For functional constructors, the create-rule is redundant — looking
up via the body atom directly would suffice. The encoder generates
these create-rules defensively for the proof-tracking case, but in
non-proof mode they're often unnecessary.

Translator could detect "let v = (F args)" where F is a functional
constructor and rewrite v → direct lookup via body atom. Cuts ~2
clauses per let, plus removes the lookup-atom overhead in
dependents.

## 5. Drop unused encoder scaffolding under souffle_compat_strata

The encoder emits relations + rules for:
- `to_delete_<F>`, `to_subsume_<F>` — drain-pattern helpers.
- Various proof helpers when proofs disabled.

Some of these are only useful in proof mode or in the strata
setup. Under souffle_compat_strata + non-proof mode, many are pure
overhead.

Cut emission at the encoder (gated on `souffle_compat_strata`) or
in the translator (filter out dead relations after analysis).

## 6. Use `(saturate)` semantics where bounded iter isn't needed

If the user explicitly uses `(saturate ...)` instead of `(run N)`,
the translator could skip the bounded-iter machinery entirely (no
IterCounter, no wave filtering, no per-iter @snap updates) and
let souffle's native fixpoint loop run. Much faster for terminating
rule systems. Just an opt-in escape hatch — doesn't help diverging
benchmarks like math-microbench.

## Order of attack

#1 first. If math-microbench is still slow after #1, do #3.
Everything else is incremental cleanup.
