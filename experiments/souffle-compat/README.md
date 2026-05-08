# Souffle compatibility experiments

Small Souffle Datalog programs that resolve the Phase 0 open questions in
[`../../souffle-backend-plan.md`](../../souffle-backend-plan.md). Each `.dl`
file is self-contained — run with `souffle <file>.dl` (Souffle 2.5+).

## Index of experiments

### Q1 — Subsumption timing
`q1-subsumption-timing.dl` — does Souffle subsumption fire continuously
during a stratum's fixpoint, or only at stratum end?

**Result:** ✅ Continuous. A same-stratum `Observer` rule that copies the
subsumed relation sees only the post-subsumption state — no intermediate
chain rows leak into other rules. Safe for the encoded rebuild rules.

### Q3 — Auto-indexer behavior
`q3-auto-indexer.dl` — does Souffle's auto-indexer build the btree indexes
our encoded program needs, without explicit hints?

**Result:** ✅ Yes. For the access patterns
  - `UF(c, ?)` — function-style first-column lookup
  - `AddView(a, b, ?)` — co-key prefix lookup for congruence

Souffle generates exactly the right prefix indexes (`t_btree_..._10` for
single-column prefix on UF, `t_btree_..._110` for 2-column prefix on
AddView). Inspect with `souffle -g out.cpp q3-auto-indexer.dl`.

### Q4 — Fresh ID generation (Skolemization)
- `q4-fresh-ids-1-records.dl` — do records hash-cons to distinct values
  when constructed in rule heads? **Result:** ✅ Yes.
- `q4-fresh-ids-2-recursive-rule.dl` — do recursive rules with record
  heads terminate when the structural space is finite? **Result:** ✅ Yes.
- `q4-fresh-ids-3-nested.dl` — do recursive record types (nested terms
  like `Add(Add(1,2), 3)`) work? **Result:** ✅ Yes.
- `q4-fresh-ids-4-egraph.dl` — full integration: build a tiny e-graph
  proving `Add(1,2) ~ Add(2,1)` using records as term IDs, subsumption
  for path compression, and `ord()` for deterministic union direction.
  **Result:** ✅ Both terms resolve to the same rep.

**Conclusion:** Skolemization via Souffle records is the answer.
`(let v5 (Add a b))` in egglog becomes `[Add_tag, a, b]` in Souffle —
same hash-cons mechanism, different syntax.

### Q5 — Cross-relation deletion
The encoded program uses `(__to_delete_<C>)` rows to trigger deletion of
matching `(__<C>View)` rows. Souffle subsumption is intra-relation only —
this pattern needs a different approach.

- `q5a-cross-deletion-naive.dl` — direct cross-relation subsumption
  attempt. **Result:** ❌ no-op (dominated and dominating must match).
- `q5b-cross-deletion-live-view.dl` — live-view via stratified negation:
  `AddViewLive(...) :- AddView(...), !ToDelete(...).`
  **Result:** ✅ works, requires queries to use the live view.
- `q5c-cross-deletion-via-self-subsume.dl` — inject a "tombstone" row from
  ToDelete into AddView itself, then self-subsume the real row. Filter
  tombstones in a live view. **Result:** ✅ works, physically deletes the
  real row.

**Recommendation:** start with (b) — simpler. Switch to (c) if memory
becomes an issue from retained-but-filtered tuples.

## Open Phase 0 questions still pending

- **Q2 — Souffle perf on egglog-shaped workloads.** Translate one encoded
  program (e.g., `tests/calc.egg`) by hand to Souffle, compare wall-clock
  against the current backend. Half-day investment.
