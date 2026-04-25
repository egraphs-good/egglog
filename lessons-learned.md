# Lessons Learned — `reuse-indices` / `claude-experiment` branch

Branch goal: investigate why index reuse (`get_cached_trie_node`) did not improve performance,
and improve query execution efficiency in general.

---

## Final state vs main

| Benchmark | main (2026-04-24) | branch HEAD | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32 | 0.282s | ~0.250s | **-11%** |
| hardboiled_conv1d_128 | 0.920s | ~0.797s | **-13%** |
| luminal-llama | 0.099s | ~0.123s | ~noise |
| python_array_optimize | 0.903s | ~0.925s | ~+2% (noise) |
| cykjson | 0.083s | ~0.063s | **-24%** |
| eggcc-extraction | 0.263s | ~0.277s | ~+5% (noise) |

The branch is net faster, driven by the Sparse+Sparse intersection improvement (Exp 40/41)
and the serial drain optimization (Exp 69). The original `reuse-indices` overhead (~4-7% on
hardboiled) has been more than compensated.

---

## What worked

### 1. Sparse+Sparse intersection: eliminate binary search for common cases (Exp 40+41)
**The biggest win of the entire experiment.** The original code:

```rust
cur.retain(|rowid| match other.scan_for_offset(other_off, rowid) {
    Ok(found)    => { other_off = found + 1; true  }
    Err(next_off) => { other_off = next_off;  false }
})
```

`scan_for_offset` always calls `binary_search_from(other_off, rowid)` — O(log(N - other_off))
— even for the most common cases: exact match, or `rowid` already past `other[other_off]`.
Importantly, `other_off` does **not** always advance: when `rowid < other[other_off]`,
`binary_search_from` returns `Err(0)`, leaving `other_off` unchanged. So for M consecutive
elements in `cur` that all fall below a single element of `other`, you get M binary searches
over the same range — O(M log N) total.

The hybrid replacement (Exp 41) makes the two common cases O(1):
1. `other[other_off] == rowid` → advance and return true (1 comparison)
2. `other[other_off] > rowid` → return false (1 comparison, other_off unchanged)
3. `other[other_off] < rowid` → binary search in `other[other_off..]` to skip forward

Case 3 is the only one that incurs binary search cost, and it always advances `other_off`.
This gave **~13-16% speedup on hardboiled** (dense/overlapping sets hit cases 1 and 2 most).

The intermediate Exp 40 (pure two-pointer, linear scan for case 3) regressed
`python_array_optimize` because that workload has large gaps: linear scan is O(gap) per skip,
while the old binary search was O(log gap). The hybrid preserves binary search for case 3.

**Lesson:** Even when amortized complexity is similar, eliminating per-element overhead for
the common cases beats worrying about asymptotic behaviour. Profile which case dominates.

### 2. Early empty check before Dense+Sparse intersection alloc (Exp 36)
Before allocating a `SortedOffsetVector` from the pool, binary-search both bounds first.
If `l >= r`, the intersection is empty — return early, no allocation. Small but real: ~2%
on hardboiled, ~4% on luminal-llama. The benefit is proportional to how often Dense+Sparse
intersections are empty, which is common when a narrow dense range is intersected with a
sparse column index.

### 3. Skip ExecutionState::clone() for serial drain (Exp 69)
`drain_updates!` was routing through `drain_updates_parallel!` even for `InPlaceActionBuffer`
(which is single-threaded). This caused 2-3 `ExecutionState::clone()` calls per drain — each
allocating fresh `Box<Buffer>` handles for every table in the rule. Adding
`supports_parallel_drain() -> bool` to the `ActionBuffer` trait (false for
`InPlaceActionBuffer`) eliminates all these clones. Real improvement: **python ~-5%**,
cykjson further improved.

**Lesson:** Code paths that handle both parallel and serial execution often have overhead that
is only needed for the parallel case. Profiling or careful reading can find these.

### 4. Reuse Arc<TrieNode> via Arc::get_mut + reset() (Exp 68)
After `move_back` returns a TrieNode with refcount == 1, the next `insert_subset` for the
same atom can reuse the existing Arc in-place (calling `take()` on OnceLock fields to clear
caches) instead of allocating a new Arc. Real improvement: **cykjson ~-11%**.

**Lesson:** `Arc::get_mut` is a cheap exclusive-ownership check (~1 atomic load). When it
succeeds, in-place mutation avoids a heap allocation. Worth checking wherever Arc<T> values
cycle through "unique then shared" patterns.

### 5. Single Box<(col, Mutex<HashMap>)> per TrieNode (Exp 23, simplification)
Each TrieNode is always probed with exactly one column. The original code used
`IdVec<ColumnId, Mutex<HashMap>>` (one Mutex per column, only one ever used). Simplifying
to a single `Box<(ColumnId, Mutex<HashMap>)>` eliminated the pool entry for ChildrenMaps,
the `resize_with` initialization loop, and the array indexing by ColumnId. Performance
neutral, but simpler and more accurately models the access pattern.

### 6. Simplify ColumnIndex::for_each to direct nested loop (Exp 67, simplification)
Removed a `if shards.len() == 1` fast path plus multi-shard `flat_map` path, replacing both
with a single direct nested loop. The `flat_map` with closures was preventing the compiler
from seeing through the loop structure. Slightly faster on hardboiled_128, simpler code.

### 7. Dense+Sparse and Dense+Dense early exits (Exp 36, Exp 38)
Checking emptiness before pool allocation and propagating the Dense range through the index
as `Option<OffsetRange>` rather than a flag avoids unnecessary allocations and materializations.

---

## What did not work (and why)

### Cache removal / bypass — DO NOT attempt again
Removing `get_cached_trie_node` caused **+23% regression on hardboiled**. The cache's primary
value is **inter-bag deduplication in DecomposedPlan**: the planner calls `run_join_stages`
multiple times with the same `binding_info` (same `Arc<TrieNode>` instances) for different
bags. Bag 0 populates `cached_children`; bag 1 gets cache hits without re-running
`refine_subset`. Bypassing the cache at `cur == 0`, or for `SinglePlan`, or based on
`Arc::strong_count` — all caused similar +20% regressions. The cache is doing real work.

### ReadOptimizedLock — no better than Mutex here
Tried ReadOptimizedLock (ArcSwap-based) vs std::sync::Mutex for the ChildrenMap multiple
times (Exp 25, 26, 49 with RwLock variant). ReadOptimizedLock is no better than Mutex in
single-threaded mode because: (a) the read path still does an ArcSwap Acquire fence, (b)
the write path (miss) still does full locking. RwLock was actually +4.3% **SLOWER** because
upgrading from read → write requires re-acquiring, adding overhead on every cache miss.

### DashMap — much worse
DashMap uses parking_lot's sharded RwLock. For single-threaded uncontended access, this
is dramatically worse (+18-25%) due to shard index computation, more complex data structures,
and RwLock overhead per operation. std::sync::Mutex + HashMap is the right choice for
uncontended single-threaded access.

### SpinLock — neutral
Hand-rolled spinlock (AtomicBool + UnsafeCell) vs std::sync::Mutex: neutral. Modern Linux
pthread_mutex already uses an optimistic CAS (avoids futex syscall when uncontended), so
it's essentially already a spinlock for the uncontended case. No improvement from switching.

### Any extra step on the hot read path — +20% regression
The hot path for cache hits is: `OnceLock::get_or_init` → array index →
`Mutex::lock` → `HashMap::get` → `Arc::clone`. Each experiment that added one extra
step (extra OnceLock, extra branch, extra atomic) to this path caused ~20% regression.
Do not add anything to this path.

### Pool reuse for ChildrenMaps (Exp 7)
Preserving `ReadOptimizedLock` objects across pool reuses by implementing custom `Clear` that
calls `HashMap::clear()` (retaining capacity) instead of dropping: caused +25% regression.
Root cause: `HashMap::clear()` retains the backing array, but `bytes()` reported only
`len * 100` (severe underestimate). The pool stored far more data than intended, creating
memory pressure that hurt allocator performance. The principle is right but requires accurate
memory accounting, which requires unsafe code.

### Dynamic re-sort removal (Exp 2a)
Removing the `cur % 3 == 1` dynamic re-sort made things uniformly slower. The dynamic
re-sort corrects atom ordering based on actual runtime subset sizes, which is strictly better
than static plan-time estimation. Don't remove it.

### DecomposedPlan threshold change (Exp 2b)
Raising the atom threshold for DecomposedPlan from ≤2 to ≤4 caused +20% regression on
hardboiled. DecomposedPlan is essential for queries with many atoms.

### .sum() vs .max() in sort heuristic (Exp 3)
Using `.sum()` of `times_refined` instead of `.max()` in `sort_plan_by_size_inner` caused
+22% regression. `.max()` correctly prioritizes the single most-filtered dimension; `.sum()`
dilutes this signal.

### Dense retain optimization (Exp 39)
Dense `retain` with an all-pass fast path: neutral. Profiling insight: Dense `retain` is
rarely hit in the hot path because column indices store Sparse subsets. Optimizing it
does not help.

### Caching FusedIntersect probe results (Exp 27)
Adding `get_cached_trie_node` to the FusedIntersect probe path: worse. FusedIntersect probe
keys change with every cover row, so cache hit rates are too low to justify the lookup overhead.

### Exp 33 — Arc::get_mut reuse for RefineAtomDense drain (DISCARDED)
Adding `Arc::get_mut` check in drain was neutral because `Arc::get_mut` fails on the first
frame (slot is None), so the savings only accumulate across subsequent frames, which is
a tiny fraction of total allocation traffic in the tested benchmarks.

### Experiments 44–66, 70–72 — rolling baseline artifact
~23 experiments appeared to show improvements of 1–7% each but were actually within
measurement noise. The rolling baseline caused each experiment to be compared against the
previous (potentially fast) run rather than a fixed anchor. Fixed-baseline audit confirmed
all were ≤3% pairwise, indistinguishable from noise at 25% machine variance.
**See the rolling baseline section below.**

---

## Architecture understanding

### What DecomposedPlan does and why it matters
Queries with many atoms are decomposed into a tree of bags via tree decomposition. Each bag's
result is materialized before the next bag runs. The same root `Arc<TrieNode>` instances are
shared across all bags via `binding_info`. This is why `get_cached_trie_node` helps even in
the single-threaded case: bag N-1's cache populates `cached_children` on shared nodes, and
bag N gets cache hits.

### The true cost of Arc<TrieNode>
The branch wraps TrieNodes in `Arc` to enable sharing across bags. Cost breakdown per call:
- Cache miss: `Arc::new(TrieNode::new(...))` = one heap allocation (~30-50ns)
- Cache hit: `Arc::clone()` = one atomic increment (~5ns) + Mutex acquire/release (~20ns) + HashMap::get (~5-10ns)
- Total hot-path overhead vs main (no Arc): ~4-7% on hardboiled

Net effect: the cache saves ~23% by avoiding redundant `refine_subset` calls, but adds ~5%
overhead. Net benefit depends on workload:
- High inter-bag reuse (cykjson): **net +24% faster**
- Low inter-bag reuse (hardboiled): **net ~4% slower**

### Subset types and intersection costs
- `Dense + Dense`: pure arithmetic, O(1), no allocation
- `Dense + Sparse`: binary search for bounds, then copy — now properly early-exits on empty (Exp 36)
- `Sparse + Sparse`: was O(M log N), now O(M+N) with binary search fallback (Exp 40+41)
- Sparse sets in `python_array_optimize` have large gaps → O(M+N) is slower than O(M log N) for those; the hybrid addresses this

### Machine variance and measurement
Benchmark machine variance is approximately **±25%** on single runs. This means:
- Changes smaller than ~10% are unreliable on a single run
- Even 5% improvements require 2-3 runs with consistent direction to confirm
- Rolling baseline (comparing each run against the immediately preceding run) accumulates
  noise and causes false positives — see below

---

## Measurement methodology: the rolling baseline problem

**The core issue:** If experiment A shows a 2% improvement due to a fast noise run and is
kept, the new baseline is 2% lower. Experiment B is then measured against that lower baseline.
Even if B does nothing, it has a 2% "headroom" to appear improved before being flagged. Over
23 experiments, this compounded into falsely claiming ~30-40% cumulative improvement that
wasn't real.

**The fix:** `scripts/audit_commits.sh` replays every commit from a fixed anchor, rebuilds
and benchmarks each one from scratch, then compares pairwise (commit[i] vs commit[i-1]).
This eliminates run-to-run variance from the baseline. Run it periodically (every 5-10 kept
commits) to catch drift.

**Practical rules for future experiments:**
1. Always compare against a fixed archived baseline, not just the previous run
2. Run benchmarks at least twice before deciding to keep; require consistent direction
3. For improvements smaller than ~5%, run 3+ times and take medians
4. If many consecutive experiments all "pass," that is a signal of rolling baseline drift —
   re-anchor and re-audit
5. Consider changes that are "noise" as DISCARDED unless they are explicit simplifications

---

## Untried ideas (for next session)

### High potential
- **RefCell<HashMap> for single-threaded ChildrenMap:** For execution that is provably
  single-threaded (egglog default mode), replacing `Mutex<HashMap>` with `RefCell<HashMap>`
  eliminates the CAS on every cache lookup (~20-30ns per call). Requires threading a
  `is_parallel: bool` flag through the execution context or using a thread-local. This is
  the most promising remaining optimization — the Mutex is the single largest remaining
  overhead on the hot path.

- **Two TrieNode types (ThinNode vs FullNode):** Nodes that are never shared across bags
  (SinglePlan, shallow levels) don't need the caching machinery. A `ThinNode` (just a
  `Subset`, no OnceLock/Box overhead) could be used for non-shared contexts, falling back
  to `FullNode` when sharing is needed. Large refactor but could eliminate the Arc overhead
  for the majority of nodes.

- **Context-aware caching at query plan time:** The planner already knows which queries use
  `DecomposedPlan` (multi-bag) vs `SinglePlan`. For SinglePlan, the inter-bag cache never
  helps. Could tag the binding_info or pass a cache-enabled flag.

### Medium potential
- **Seqlock for cached_child:** Replace `Mutex<HashMap>` with a seqlock — readers retry if
  a write is in progress; writers are rare (only on cache miss). Unsafe but potentially
  faster than Mutex for read-heavy workloads. Risk: complex correctness argument.

- **AtomicPtr instead of OnceLock<Box<...>>:** `OnceLock` uses 8 bytes (pointer) + state.
  `AtomicPtr<T>` uses 8 bytes with a single Acquire load on the fast path. Could reduce
  TrieNode size by 8 bytes per OnceLock field.

- **Column index pre-sorting:** The `sort_plan_by_size` dynamic re-sort is effective but
  allocates a `DenseIdMap` from the pool per call. Investigate whether the pool contention
  (or allocation) is measurable.

- **Profiling with perf/flamegraph:** The experiments so far have been hypothesis-driven.
  Actual CPU profiling (`perf record -g`, `cargo flamegraph`) on `hardboiled_conv1d_128`
  (the largest benchmark) would reveal the actual hot functions by sampling, potentially
  revealing bottlenecks that aren't obvious from reading the code.

### Lower potential / architectural
- **HashMap capacity tracking for pool:** The pool underreports `bytes()` for ChildrenMaps
  because it can't query HashMap capacity from `&self`. Tracking capacity separately in a
  wrapper struct (at the cost of 8 bytes overhead) would make pool size limits accurate,
  potentially allowing better pool reuse. The benefit depends on allocation frequency.

- **plan.rs deeper exploration:** The planner (`plan_gj`, `topologically_sort_bags`,
  `loop_lifting`) has received no optimization. The audit showed plan.rs changes are high
  risk for regressions (DecomposedPlan threshold, sort heuristic) but there may be correct
  improvements in variable ordering or bag merging strategy.

---

## Summary table of all experiments

| Exp | Description | Outcome | Δ (geomean) |
|---|---|---|---|
| 1 | Remove get_cached_trie_node | DISCARDED | +20% hardboiled |
| 2a | Remove dynamic re-sort | DISCARDED | uniformly slower |
| 2b | DecomposedPlan threshold 2→4 | DISCARDED | +20% hardboiled |
| 3 | .sum() vs .max() sort heuristic | DISCARDED | +22% hardboiled |
| 4 | Skip cache at cur==0 | DISCARDED | +20% hardboiled |
| 5 | Lazy OnceLock<ROL> per column | DISCARDED | +21% hardboiled |
| 7 | Preserve ROL across pool reuses | DISCARDED | +25% hardboiled |
| 8 | entry() API in cache lookup | DISCARDED | ~+2% |
| 9 | cur % 4 sort frequency | DISCARDED | noise (high variance) |
| 10–13 | Various hot-path micro-opts | DISCARDED | neutral |
| 14 | Sort early-return (simplification) | KEPT | neutral |
| 15–17 | RwLock, threshold, empty-check | DISCARDED | neutral/worse |
| 18 | Skip cache for SinglePlan | DISCARDED | +2-6% |
| 19 | Merge duplicate [a] branches (simplification) | KEPT | neutral |
| 20 | Defer cover_atom Arc in FusedIntersect | DISCARDED | neutral |
| 21 | DashMap | DISCARDED | +18-25% |
| 22 | SpinLock | DISCARDED | neutral |
| 23 | Single Box<(col, Mutex<HashMap>)> (simplification) | KEPT | neutral |
| 24 | Store pool in JoinState | DISCARDED | neutral |
| 25–26 | ReadOptimizedLock single-map | DISCARDED | slightly worse |
| 27 | Cache FusedIntersect probe results | DISCARDED | ~+2% |
| 30 | RefineAtomDense (defer Arc for Dense cover rows) | KEPT | neutral (correct) |
| 31 | Fix prune_probers empty-subset bug | KEPT | correctness fix |
| 32a–c | Various micro-opts | DISCARDED | neutral |
| 33 | Arc::get_mut in RefineAtomDense drain | DISCARDED | neutral |
| 34 | Pre-check bounds in Prober::for_each | DISCARDED | slightly worse |
| 35 | Box cached_subsets | DISCARDED | slightly worse |
| 36 | Early empty check for Dense+Sparse alloc | KEPT | **~-2% overall** |
| 37 | to_owned_intersect_dense with runtime branch | DISCARDED | ~+2% |
| 38 | intersect_outer bool → Option<OffsetRange> (simplification) | KEPT | neutral |
| 39 | Dense retain optimization | DISCARDED | neutral |
| 40 | Two-pointer Sparse+Sparse intersection | KEPT | **~-15% hardboiled** |
| 41 | Hybrid intersect + binary search fallback | KEPT | **fixes Exp 40 regression** |
| 43 | Skip vtable call when constraints empty | KEPT | small positive |
| 44–66 | Various micro-opts (rolling baseline artifact) | DISCARDED | ≤±3% (noise) |
| 67 | Simplify ColumnIndex::for_each (simplification) | KEPT | neutral/slight win |
| 68 | Arc::get_mut reuse in insert_subset | KEPT | **cykjson -11%** |
| 69 | Skip ExecutionState::clone for serial drain | KEPT | **python -5%** |
| 70–72 | Various (rolling baseline artifact) | DISCARDED | ≤±3% (noise) |
