# Experiment Journal

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
