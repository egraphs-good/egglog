# Experiment Journal

## E3: Downcast bypass for for_each_col in SparseColumnIndex::new (2026-05-09)

**Hypothesis:** The E1 size==1 fast path still calls `for_each_col` through 6 dispatch layers (vtable → downcast → scan_generic → SubsetRef::offsets → RowId::range iterator → dyn FnMut). A direct downcast to `SortedWritesTable` with an inherent `read_value_at_row_unchecked` method eliminates most of this overhead without adding new trait methods (the narrower approach the E2 reviewer recommended).

**Approach:** Added `read_value_at_row_unchecked` inherent method on `SortedWritesTable` (mirrors scan_generic stale-row logic), added `inner_as_any` forwarder on `WrappedTableRef`, downcast at the call site in `SparseColumnIndex::new` with fallback to `for_each_col` for other Table impls.

**Results:**
- hardboiled_conv1d_32: 0.321s → 0.314s (-2.2%)
- hardboiled_conv1d_128: 0.863s → 0.832s (-3.5%)
- luminal-llama: 0.212s → 0.217s (+2.4%, noise on 0.21s)
- python_array_optimize: 0.519s → 0.512s (-1.3%)
- cykjson: 0.107s → 0.106s (-1.4%)
- eggcc-extraction: 0.460s → 0.458s (-0.5%)
- Overall average: -1.07%

**VERDICT: IMPROVEMENT** — Reviewer ACCEPTED. hardboiled_128 cleared ≥3% threshold. luminal-llama +2.4% is within noise. Reviewer confirmed this is the project-preferred pattern (inherent method + downcast at call site) vs. expanding the Table trait.

**Commit:** 16c83f12 — KEPT.

## E2: Direct read_value_at_row (rejected) (2026-05-09)

**Hypothesis:** The size==1 fast path in `SparseColumnIndex::new` (E1) still calls `for_each_col` through 6 layers of dispatch. A dedicated `read_value_at_row` trait method with a direct-array-index override on `SortedWritesTable` would cut this to 3 layers.

**Approach:** Added `read_value_at_row` to `Table` trait (with default impl), overrode on `SortedWritesTable`, propagated through `TableWrapper` and `WrappedTableRef` (~63 lines across 3 files).

**Results:**
- hardboiled_conv1d_128: -2.7% (best result, below 3% threshold)
- Overall average: -0.89%

**VERDICT: MODEST GAIN** — Reviewer REJECTED. 63 lines of trait boilerplate for -0.89% avg / -2.7% peak is exactly what the simplicity criterion is meant to reject. Reviewer suggested a narrower implementation (inherent method + downcast at call site, no trait additions) if re-attempted.

**Commit:** b60cf678 — DROPPED (git reset --hard).

## E1: Single-row fast path in SparseColumnIndex::new (2026-05-09)

**Hypothesis:** `SparseColumnIndex::new` is called on every small-subset column index lookup. For the common `subset.size() == 1` case (dominant after AA1 converts Sparse→Dense and GG1 returns Dense for single-row results), the general path wastes cycles: zero-initing an 8-pair stack array, running `sort_unstable` on 1 element, and running a grouping loop once. A fast path that skips all three should be measurably faster.

**Approach:** Add an early return at the top of `SparseColumnIndex::new` for `subset.size() == 1`. Extract the row id directly from the subset variant, call `for_each_col` with a one-element Dense subset to fetch the column value, then populate the result struct directly.

**Results:**
- hardboiled_conv1d_32: 0.324s → 0.326s (+0.8%, noise)
- hardboiled_conv1d_128: 0.873s → 0.861s (-1.3%)
- luminal-llama: 0.227s → 0.214s (-5.9%)
- python_array_optimize: 0.526s → 0.522s (-0.6%)
- cykjson: 0.118s → 0.101s (-14.6%)
- eggcc-extraction: 0.478s → 0.462s (-3.4%)
- Overall average: -4.18%

**VERDICT: IMPROVEMENT** — Reviewer ACCEPTED. Notable wins on cykjson (-14.6%) and luminal-llama (-5.9%). Reviewer noted a latent stale-row corner (if callback never fires, val stays sentinel and n_keys=1), currently masked by upstream refine_live invariant; flagged as follow-up.

**Commit:** eddd23e2 — KEPT.

## F1: Sort-based bulk ColumnIndex rebuild (2026-04-30)

**Hypothesis:** `ColumnIndex::add_row` is at 8.29% of cycles (per perf on hardboiled_conv1d_128)
and `memmove_avx512` is at 2.87%, both attributable to `SubsetBuffer::push_vec`'s doubling
reallocation during full-rebuild. On every major-version change (after union-find compaction),
the ColumnIndex is cleared and rebuilt row-by-row. For a group of n rows with the same key,
`add_row_sorted` triggers log2(n) doubling memmoves totaling O(n) copies — but with large
constant factor due to AVX512 copy overhead.

**Approach:** Override `IndexBase::rebuild_full` on `ColumnIndex` with a sort-based approach:
1. Collect all (Value, RowId) pairs via `for_each_col`
2. Sort by (Value, RowId) → groups same-key rows adjacently, in RowId order within each group
3. For each group: create a single pre-sized `SubsetBuffer::new_vec()` allocation (no memmoves)
4. Preserve Dense representation for contiguous RowId groups (matches `add_row_sorted` behavior)

The default `IndexBase::rebuild_full` (for TupleIndex) still uses the original scan_project+merge_rows
path. The parallel merge path is also unchanged.

**Correctness issue discovered:** Sorting by Value changes the HashMap insertion order, which
changes hashbrown's internal bucket layout, which changes the `for_each` iteration order used
in `execute.rs:1243` (and other places) to drive the outermost join scan. This alters the order
in which rules fire across 10 iterations, producing different e-graph states.

The semantic assertions all pass, but the `repro_unsound` snapshot needed updating: (Div 8654) →
(Div 9314) reflects different rule application order across 10 iterations. Both results are valid
egglog semantics for `(run 10)`.

**Results:**
- hardboiled_conv1d_32: 0.387s → 0.386s (-0.3%, essentially unchanged)
- hardboiled_conv1d_128: 1.102s → 1.100s (-0.2%, essentially unchanged)
- luminal-llama: 0.255s → 0.230s (-9.8%)
- python_array_optimize: 0.549s → 0.539s (-1.8%)
- cykjson: 0.142s → 0.118s (-16.9%)
- eggcc-extraction: 0.481s → 0.473s (-1.7%)

**Verdict:** KEPT. 4 benchmarks faster, 0 slower.

---

## G2: Sort-based bulk build in group_by_col (2026-04-30)

**Hypothesis:** `group_by_col` builds a ColumnIndex on-the-fly during join execution using the
same row-at-a-time `add_row` path as the pre-F1 full-rebuild. For hardboiled, the dynamic index
builds happen on large subsets (not just singletons), so the sort-based approach should help.

**Approach:** Change `group_by_col` (in `TableWrapper<T>`) to use `ColumnIndex::rebuild_full`
(the sort-based approach from F1) for subsets with size ≥ 32. For size < 32, keep the original
`add_row` path (sort overhead exceeds memmove savings for small subsets).

**Why the threshold matters:** The meeting notes mention "80~90% of indices build has size < 10"
for python_array_optimize. For these small subsets, the Vec allocation + sort overhead in
`rebuild_full` exceeds the doubling memmove savings. The threshold=32 preserves the original
path for small dynamic indexes.

**Snapshot change:** G2 does NOT change any snapshots, unlike F1. The hashmap insertion order
change from sort-based building doesn't affect the repro-unsound test (which was already
updated in F1).

**Results:**
- hardboiled_conv1d_32: 0.386s → 0.356s (-7.8%)
- hardboiled_conv1d_128: 1.100s → 0.977s (-11.2%)
- luminal-llama: 0.230s → 0.228s (-0.9%)
- python_array_optimize: 0.539s → 0.552s (+2.4% regression)
- cykjson: 0.118s → 0.117s (-0.8%)
- eggcc-extraction: 0.473s → 0.474s (+0.2%, essentially unchanged)

**Verdict:** KEPT. Large wins on hardboiled (-8 to -11%), small regression on python_array
(+2.4%). Per program.md: "huge speedup on one benchmark, mild slowdown on others → keep."

---

## AA1: Single-entry Sparse → Dense conversion in refine_atom_subset (2026-04-30)

**Hypothesis:** When `refine_subset` produces a 1-element Sparse subset, `refine_atom_subset`
still wraps it in `Arc::new(TrieNode::new(sub))` for the general Sparse case. A 1-element
Sparse subset `[r]` is semantically identical to `Dense(r..r+1)`, which can use the cheaper
`refine_atom_dense` path (no Arc allocation).

**Approach:** Add a special case before the general `sub => refine_atom(...)` arm:
```
Subset::Sparse(ref offsets) if offsets.slice().inner().len() == 1 => {
    let row = offsets.slice().inner()[0];
    self.refine_atom_dense(atom, OffsetRange::new(row, row.inc()));
}
```
This avoids `Arc::new(TrieNode::new(sub))` for single-row Sparse subsets, using the dense fast
path instead. Because `Dense(r..r+1)` is structurally different from `Sparse([r])`, downstream
`get_index` for this atom now enters the Dense branch (or SparseColumnIndex for small subsets),
potentially altering rule application order over 10 iterations.

**Snapshot change:** `repro_unsound` snapshot updated from `(Div 9314)` to `(Div 8654)`.
Both are valid egglog semantics for `(run 10)`.

**Results (post-G2 baseline ~0.977s hardboiled_128):**
- hardboiled_conv1d_32: essentially unchanged (~0.350s)
- hardboiled_conv1d_128: essentially unchanged (~0.977s)
- All others: essentially unchanged

**Verdict:** KEPT. Neutral on benchmarks; reduces Arc allocations for the common single-row
refine case, which should help under memory pressure.

---

## BB1: Raise SMALL_RESIDUAL from 8 to 32 (REVERTED) (2026-04-30)

**Hypothesis:** Sizes 9-31 in `get_index` fall through to `DynamicIndex::DynamicColumn` →
`get_cached_index` → `group_by_col` with add_row. Each call allocates a fresh ColumnIndex
HashMap plus per-key SubsetBuffers (~13 allocations). With SMALL_RESIDUAL=32, sizes 9-31
would use the stack-allocated `SparseColumnIndex` path (1 call, no heap allocation).

**Approach:** Change `const SMALL_RESIDUAL: usize = 8` → `32`, and expand the fixed arrays
in `SparseColumnIndex` to size 32.

**Problem discovered:** DynamicIndex is a Rust enum; its size is determined by its LARGEST
variant. SparseColumnIndex at SMALL_RESIDUAL=32 occupies:
- keys: [u32; 32] = 128 bytes
- offsets: [usize; 32] = 256 bytes  
- subset_ids: [u32; 32] = 128 bytes
- n_keys, n_subsets: 16 bytes
Total: ~528 bytes — 3× larger than the SMALL_RESIDUAL=8 variant (176 bytes).

This made every `Prober` (which embeds a `DynamicIndex`) grow proportionally, destroying cache
efficiency for all [a,b] and [rest] join paths, not just the SparseColumn path.

**Results:** hardboiled_conv1d_128: 0.977s → 1.078s (+10.3% regression). REVERTED.

**Verdict:** REVERTED. Root cause: inflating the DynamicIndex enum penalizes every path.
The fix would require indirection (Box<SparseColumnIndex32>) but that adds its own allocation
overhead. Left as future work.

---

## CC1: Adaptive LSB radix sort in ColumnIndex::rebuild_full (2026-04-30)

**Hypothesis:** Profiling (perf at -F 200) shows sort_unstable inside ColumnIndex::rebuild_full
accounts for ~10.4% of runtime (5.75% small_sort_network + 4.68% quicksort). This sort is
called for every group_by_col invocation with subset size ≥512 (G2's threshold). Value is a
u32 (egglog handle), and after union-find compaction, handles are small integers (typically
< 65,536). An adaptive LSB radix sort needs only 2 passes instead of O(N log N) comparisons,
beating comparison sort for medium N (512-10,000).

**Key correctness properties:**
1. **Stable within Value groups**: Since for_each_col scans rows in RowId order, all pairs
   with the same Value are already in RowId order in the input. LSB radix sort is stable →
   within-group RowId order is preserved → Dense detection still works correctly.
2. **Same HashMap insertion order**: Both sort_unstable and radix sort produce Value-ascending
   output for distinct keys. ColumnIndex's HashMap is populated in the same order → same
   for_each iteration order → no python_array_optimize regression risk.
3. **Multi-column fallback**: When cols.len() > 1, dedup() requires RowIds to be sorted
   within each Value group; we fall back to sort_unstable() + dedup() for correctness.
4. **Small-N fallback**: n < 64 uses sort_unstable (radix setup overhead not worth it).

**Approach:** Add `radix_sort_pairs_by_value` with adaptive 1-4 passes based on max(Value):
- 1 pass if max < 256, 2 if < 65536 (typical egglog), 3 if < 16M, 4 for full u32.
- Double-buffer with raw pointer swap; odd-pass-count copies buf back to pairs.
- Replace `pairs.sort_unstable()` with this in the single-column branch of rebuild_full.

**Results:**
- hardboiled_conv1d_32: 0.350s → 0.333s (-4.9%)
- hardboiled_conv1d_128: 0.977s → 0.909s (-7.0%)
- luminal-llama: essentially unchanged (~0.224s)
- python_array_optimize: essentially unchanged (~0.528s)
- cykjson: essentially unchanged (~0.113s)
- eggcc-extraction: essentially unchanged (~0.471s)

**Verdict:** KEPT. ≥5% speedup on hardboiled_128, no regressions anywhere.

---

## D3: Four get_index micro-optimizations (EE1+GG1+LL1+NN1) (2026-04-30)

**Hypothesis:** `get_index` and `get_column_index` have avoidable overhead on every call:
1. (EE1) The SparseColumnIndex fast path (subset.size() ≤ 8, single column) is checked AFTER
   computing `all_cacheable` (uncacheable_columns lookup) and `whole_table.all()` (table scan),
   wasting 2 memory accesses for the common small-subset path.
2. (GG1) SparseColumnIndex returns `SubsetRef::Sparse([r])` for single-row keys, causing a
   pool allocation + memcpy in `SubsetRef::to_owned`. A 1-row Sparse is identical to Dense(r..r+1).
3. (LL1) `group_by_col` for sizes 9-511 uses `add_row` into an unreserved HashMap, triggering
   hashbrown rehashing at ~1.15% of cycles.
4. (NN1) `get_column_index` delegates to `get_index` which first creates a 1-element
   SmallVec via `from_iter(iter::once(col))` and then checks `cols.len() == 1` multiple times.

**Approach:**
- EE1: Move SparseColumnIndex check to first position in get_index, before all_cacheable/whole_table
- GG1: Add `sparse_subset_ref()` helper returning Dense(r..r+1) for 1-element ranges in
  SparseColumnIndex::get_subset and SparseColumnIndex::for_each
- LL1: Add `ColumnIndex::reserve_for_n_rows(n)` (reserves n/n_shards+2 per shard) and call it
  in `group_by_col` before the add_row scan for sizes 9-511
- NN1: Expand `get_column_index` into a specialized single-column implementation — eliminates
  SmallVec creation, iter::once, `cols.len()` runtime checks, and unreachable multi-col branches

**Results (CC1 baseline: hardboiled_128 ≈ 0.904s):**
- hardboiled_conv1d_32: 0.333s → 0.322s (-3.3%)
- hardboiled_conv1d_128: 0.904s → 0.849s (-6.1%)
- luminal-llama: 0.224s → 0.219s (-2.2%)
- python_array_optimize: 0.528s → 0.525s (-0.6%)
- cykjson: 0.113s → 0.114s (+0.9%, within noise)
- eggcc-extraction: 0.471s → 0.471s (unchanged)

**Breakdown (incremental):**
- EE1+GG1: ~2.6% on hardboiled_128 (EE1 moves early exit before costly reads; GG1 avoids pool alloc)
- LL1: <0.5% (reserve eliminates hashbrown rehash overhead — measured noise level)
- NN1: ~3.7% on hardboiled_128 (eliminating SmallVec + iterator overhead per get_column_index call)

**Verdict:** KEPT. ≥5% speedup on hardboiled_128 (-6.1%), no regressions.

---
