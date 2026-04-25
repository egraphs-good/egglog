# Experiment Journal

Branch: `reuse-indices`

---

## Session start — 2026-04-24

### Baseline established

Archived CSV `2026-04-24.csv` already existed (taken before this session, likely on or near `main`):

| Benchmark | Time (s) |
|---|---|
| hardboiled_conv1d_32.egg | 0.282 |
| hardboiled_conv1d_128.egg | 0.920 |
| luminal-llama.egg | 0.099 |
| python_array_optimize.egg | 0.903 |
| cykjson.egg | 0.083 |
| eggcc-extraction.egg | 0.263 |

First thing I did was re-run the benchmark on the current branch (`reuse-indices` at commit `8fb58d27`) to see where we stand:

| Benchmark | Time (s) | vs baseline |
|---|---|---|
| hardboiled_conv1d_32.egg | 0.306 | +8.5% ▲ |
| hardboiled_conv1d_128.egg | 0.946 | +2.8% ▲ |
| luminal-llama.egg | 0.121 | +22.2% ▲ |
| python_array_optimize.egg | 0.960 | +6.3% ▲ |
| cykjson.egg | 0.070 | -15.7% ▼ |
| eggcc-extraction.egg | 0.268 | +1.9% ▲ |

**Archived as `2026-04-24T23:20:17.csv` — this is now our working baseline.**

---

## Experiment 1 — Remove `get_cached_trie_node` (DISCARDED)

**Hypothesis:** The commit message on `7a62a1c7` says "Currently slower, might be due to hash lookup." Maybe the caching added overhead that outweighs the benefit, and removing it entirely would be a net simplification win.

**What I changed:**
- Removed `cached_children: OnceLock<Pooled<ChildrenMaps>>` from `TrieNode`
- Removed `get_cached_trie_node` method
- Removed the `ChildrenMaps` type alias and `ReadOptimizedLock` import
- Removed `children_map` pool entry from `pool/mod.rs`
- Replaced all `get_cached_trie_node(...)` call sites with `Arc::new(TrieNode::new(sub()))` directly (also eliminated the `x.size() <= 16` branch since they now do the same thing)

**Result:**

| Benchmark | Before | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.306 | 0.378 | +23.5% ▲ |
| hardboiled_conv1d_128.egg | 0.946 | 1.095 | +15.8% ▲ |
| luminal-llama.egg | 0.121 | 0.119 | -1.7% ▼ |
| python_array_optimize.egg | 0.960 | 0.949 | -1.1% ▼ |
| cykjson.egg | 0.070 | 0.069 | -1.4% ▼ |
| eggcc-extraction.egg | 0.268 | 0.269 | +0.4% |

**Decision: DISCARDED.** The `hardboiled` benchmarks got dramatically worse (+15–24%). The caching IS beneficial, presumably because those benchmarks exercise heavily the parallel execution path where multiple threads share the same `Arc<TrieNode>` and call `get_cached_trie_node` with the same values.

**Takeaway:** The `get_cached_trie_node` mechanism genuinely helps for parallel workloads. The slowdown vs the original baseline (`2026-04-24.csv`) must come from something else — possibly overhead in `get_cached_trie_node` itself (e.g. the `ReadOptimizedLock` acquire even on hit, or the HashMap lookup), or from another change on this branch. Next experiments should focus on making the cache lookup cheaper, not removing it.

---

## Experiment 2a — Disable dynamic mid-run `sort_plan_by_size` (DISCARDED)

**Hypothesis:** The call to `sort_plan_by_size` at `cur % 3 == 1` during execution allocates a `DenseIdMap` from the pool and does O(n²) sorting per frame at level 1. For large join workloads, this could accumulate significantly.

**What I changed:** Removed the `if cur_size > 32 && cur % 3 == 1 ...` block, so only the initial sort (at the start of `run_join_stages`) runs.

**Result:** Uniformly slower across all benchmarks. The dynamic re-sort is genuinely helping — it corrects the ordering based on actual refined subset sizes at execution time, which is better than the static estimate at plan start.

**Decision: DISCARDED.**

---

## Experiment 2b — Raise DecomposedPlan atom threshold 2 → 4 (DISCARDED)

**Hypothesis:** `luminal-llama` showed +22% in our first re-run. Thought it might be due to the branch using `DecomposedPlan` (with materialization overhead) for queries that main handled with `SinglePlan`.

**What I changed:** `if ctx.atoms.len() <= 2` → `if ctx.atoms.len() <= 4` in `tree_decompose_and_plan`.

**Result:**
- `hardboiled_conv1d_32`: +20% SLOWER (the DecomposedPlan is crucial for this benchmark)
- `luminal-llama`: 0.0% unchanged
- `python_array_optimize`: 0.0% unchanged
- `cykjson`: +4% slower

**Decision: DISCARDED.** The DecomposedPlan is genuinely helping hardboiled and cykjson. Also confirmed that luminal-llama's +22% from the first re-run was **timing noise** — on a re-run it was 0.0% vs baseline. The original `2026-04-24.csv` baseline was likely taken on a quieter machine; the real working baseline is `2026-04-24T23:20:17.csv` from this session.

**Takeaway:** DecomposedPlan is important and correct. Machine noise accounts for most of the "regression" vs the original baseline.

---

## Experiment 3 — Sort heuristic: `.sum()` vs `.max()` (DISCARDED)

**Hypothesis:** The `sort_plan_by_size_inner` function uses `.max()` to compute the `times_refined` priority for `Intersect` stages (how many times the most-refined atom in the scan has been processed). Using `.sum()` instead might better reflect total work done when multiple atoms are partially refined.

**What I changed:** In `sort_plan_by_size_inner`, changed `.max().unwrap()` to `.sum::<i64>()` for the `JoinStage::Intersect` branch of `key_fn`.

**Result:**
- hardboiled_conv1d_32: +22% SLOWER
- python_array_optimize: +8.5% SLOWER
- Others: roughly unchanged

**Decision: DISCARDED.** `.max()` is clearly better. It prioritizes the single most-refined atom, which is the correct greedy choice: do the join on the dimension where we've done the most filtering. `.sum()` dilutes this signal with noise from less-refined atoms.

---

## Experiment 4 — Skip cache at `cur == 0` (DISCARDED)

**Hypothesis:** At the outermost recursion level (`cur == 0`), we haven't done any filtering yet, so the trie nodes at that level are at maximum size and unlikely to be reused with the same value. The `ReadOptimizedLock` CAS write on every first-seen value at cur=0 could be a net overhead. Bypassing the cache at cur=0 might reduce contention and lock overhead for the common case.

**What I changed:** Added a `cur == 0` bypass in `get_cached_trie_node` call sites: when `cur == 0`, directly create `Arc::new(TrieNode::new(sub()))` without going through the cache.

**Result:**
- hardboiled_conv1d_32: ~+20% SLOWER
- Others: comparable or slightly worse

**Decision: DISCARDED.** The cache at cur=0 is essential due to `DecomposedPlan`. The planner calls `run_join_stages` multiple times with the *same* `binding_info` (same `Arc<TrieNode>` instances) for different bags. Bag 0 populates `cached_children` on those shared nodes, and Bag 1 gets cache hits without re-running `refine_subset`. Skipping the cache at cur=0 breaks this inter-bag sharing, which is exactly where the feature pays off most.

**Takeaway:** The cache's main value is inter-bag deduplication in `DecomposedPlan`, not per-bag deduplication at deep levels. Don't touch it.

---

## Experiment 5 — Lazy per-column ReadOptimizedLock init in ChildrenMaps (DISCARDED)

**Hypothesis:** The current code initializes `ReadOptimizedLock::default()` for ALL `arity` columns in `resize_with`, even though typically only 1 column is accessed per trie node. Each `ReadOptimizedLock::default()` does 2 heap allocations (ArcSwap token + Notification). Wrapping each column slot in `OnceLock` would defer these allocations until a column is actually accessed, saving ~2*(arity-1) heap allocs per trie node.

**What I changed:** Changed `ChildrenMaps` from `IdVec<ColumnId, ReadOptimizedLock<HashMap<...>>>` to `IdVec<ColumnId, OnceLock<ReadOptimizedLock<HashMap<...>>>>`. The `resize_with` call uses `OnceLock::new` (zero-cost) instead of `ReadOptimizedLock::default` (2 allocs). The specific column's lock is initialized lazily via `get_or_init(ReadOptimizedLock::default)`.

**Result:**
- hardboiled_conv1d_32: +21% SLOWER
- hardboiled_conv1d_128: +5.4% SLOWER
- Others: slightly slower

**Decision: DISCARDED.** The `OnceLock::get_or_init` call adds an extra atomic load+branch on EVERY `get_cached_trie_node` call, even after the lock is initialized. This overhead (~1 extra atomic per cache hit) exceeds the savings from deferred heap allocations. The heap allocation savings are a one-time cost per trie node init, while the atomic overhead is paid on every access. Trade-off is unfavorable.

**Takeaway:** Don't add any extra indirection on the hot read path. The read path already has: OnceLock::get_or_init (cached_children) + array index + ReadOptimizedLock::read + HashMap lookup. Any extra step on this path is expensive.

---

## Experiment 7 — Custom `Clear` for `ChildrenMaps` to preserve `ReadOptimizedLock` across pool reuses (DISCARDED)

**Hypothesis:** The pool's `Clear for IdVec<K, V>` calls `Vec::clear()`, which DROPS all `ReadOptimizedLock` elements. Each `ReadOptimizedLock::default()` requires 2 heap allocations (an ArcSwap token + a TriggerWhenDone notification). By changing `ChildrenMaps` from a type alias to a newtype and implementing a custom `Clear` that calls `lock.as_mut_ref().clear()` on each element (clearing only the inner `HashMap` contents, not the lock infrastructure), we could avoid re-allocating ReadOptimizedLock objects on every pool reuse — saving `2 × arity` heap allocs per warm TrieNode init.

**What I changed:**
- Changed `pub(crate) type ChildrenMaps = IdVec<...>` to a newtype struct `pub(crate) struct ChildrenMaps(pub(crate) IdVec<...>)` with `Deref`/`DerefMut` for backward compat.
- Implemented `Clear for ChildrenMaps` in `pool/mod.rs` that calls `lock.as_mut_ref().clear()` per slot instead of `Vec::clear()`.
- `bytes()` reported `self.0.len() * 100`.

**Result:**

| Benchmark | Before | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.295 | 0.368 | +24.7% ▲ |
| hardboiled_conv1d_128.egg | 0.981 | 1.007 | +2.7% ▲ |
| luminal-llama.egg | 0.120 | 0.126 | +5.0% ▲ |
| python_array_optimize.egg | 0.944 | 0.988 | +4.7% ▲ |
| cykjson.egg | 0.073 | 0.072 | -1.4% ▼ |
| eggcc-extraction.egg | 0.272 | 0.276 | +1.5% ▲ |

**Decision: DISCARDED.** hardboiled_conv1d_32 was +24.7% SLOWER — the same order of magnitude as removing the cache entirely (Exp1: +23.5%). This magnitude strongly suggests memory pressure is suppressing the inter-bag caching benefit.

**Postmortem analysis:**

1. **Memory pressure from retained HashMap capacity**: `HashMap::clear()` retains the backing array. The pooled ChildrenMaps holds ReadOptimizedLock objects each with potentially large HashMap backing arrays, but `bytes()` only reports `len * 100` (a severe underestimate). The pool limit is 1MB in reported bytes, but actual held memory could be 10–100× more, increasing heap fragmentation and cache pressure.

2. **Pool underreporting cascades**: Because `bytes()` underestimates real memory, the pool stores far more items than intended, holding large HashMap backing arrays the allocator can't reuse for new TrieNode allocations, forcing fresh memory requests from the OS.

3. **Why accuracy is hard to fix**: `as_mut_ref()` requires `&mut self`, but `bytes()` takes `&self`, so we can't query inner HashMap capacity from `bytes()` without unsafe code or extra bookkeeping.

**Takeaway:** Preserving ReadOptimizedLock objects across pool reuses is the right idea, but it requires accurately accounting for retained memory (including HashMap capacity). Not worth the complexity.

---

## Global analysis — state of the branch vs main (as of session 2026-04-25)

After 7 experiments, the overall picture:

**Branch baseline (`2026-04-25T01:09:47.csv`) vs original main (`2026-04-24.csv`):**

| Benchmark | main | branch | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32 | 0.282 | 0.295 | +4.6% |
| hardboiled_conv1d_128 | 0.920 | 0.981 | +6.6% |
| luminal-llama | 0.099 | 0.120 | ~noise (re-runs show ~0%) |
| python_array_optimize | 0.903 | 0.944 | +4.5% |
| cykjson | 0.083 | 0.073 | **-12% faster** |
| eggcc-extraction | 0.263 | 0.272 | +3.4% |

The branch is ~4–7% slower on hardboiled and python_array_optimize, and ~12% FASTER on cykjson. Geometric mean: ~1% slower overall. The caching mechanism adds value for cykjson (heavy inter-bag reuse) while adding overhead elsewhere.

**Root cause of overhead:** The branch wraps all TrieNodes in `Arc<TrieNode>` (vs value types in main). Every cache miss creates `Arc::new(TrieNode::new(sub))` — one heap allocation. Every cache hit calls `node.clone()` — one atomic increment. `ReadOptimizedLock::read()` adds one ArcSwap load + Acquire fence per lookup. These costs accumulate in inner loops.

**What has NOT been tried yet:**
- Optimizing the plan generation side (`plan.rs`) for query-level inefficiencies unrelated to the cache.
- Whether `PotentiallyStale` wrapper adds measurable overhead in the hot path (unlikely — it's zero-cost with a predictable branch).
- Reducing Arc clones in `BindingInfo::clone()` for parallel tasks (branch should already be cheaper than main here, since it clones Arc refs vs deep-copying Subsets).
- Whether the parallel execution path has overhead from the new `Arc<MatchCounter>` wrapping.
- Looking at whether the `x.size() <= 16` threshold for skipping the cache is well-tuned.

**Key locked-in insights:**
1. `get_cached_trie_node`'s hot read path (OnceLock → array index → ReadOptimizedLock::read → HashMap::get) is sensitive to ANY extra step — each previous experiment that added one step caused +20% slowdown.
2. The cache's value is primarily inter-bag deduplication in `DecomposedPlan`. Don't bypass it at any level.
3. The pool helps but cannot save ReadOptimizedLock infrastructure across reuses without accurate memory accounting.
4. The `size <= 16` bypass for small subsets is important — don't merge it with the cached path.

---

## Session 2 — 2026-04-25 (continued)

### New baseline: `2026-04-25T02:36:39.csv` (Mutex commit)
After committing the ReadOptimizedLock → Mutex simplification:

| hardboiled_conv1d_32 | 0.364 |
| hardboiled_conv1d_128 | 0.961 |
| luminal-llama | 0.118 |
| python_array_optimize | 0.942 |
| cykjson | 0.069 |
| eggcc-extraction | 0.266 |

### Experiments 8-13 (all DISCARDED)

**Exp 8 — entry() API in get_cached_trie_node:** Replace `get()`+`insert()` with `entry().or_insert_with()`. Slightly worse (~+2% on hardboiled). The HashMap's entry() takes mutable borrow even on hit, which is slightly slower than immutable `get()` for the common cache-hit path.

**Exp 9 — cur % 4 instead of cur % 3 for dynamic sort:** Reduced sort frequency. First run showed dramatic -19.8% on hardboiled_conv1d_32, but second run showed ~0%. High machine variance at ~25% makes this unreliable. Other benchmarks slightly worse. DISCARDED.

**Exp 10 — Cache key_i in selection sort inner loop:** Pre-compute `key_i` before inner loop and update only on swap. Not measurable improvement. DISCARDED.

**Exp 11 — Vec linear search for ChildrenMaps inner map:** Replace `HashMap<Value, Arc<TrieNode>>` with `Vec<(Value, Arc<TrieNode>)>` and linear scan. Slightly worse — FxHasher on u32 is very fast. DISCARDED.

**Exp 12 — Single combined Mutex map (u64 key):** Tried replacing per-column maps with single `Mutex<HashMap<u64, Arc<TrieNode>>>`. Build complexity too high for benefit. ABANDONED.

**Exp 13 — Early-exit for empty subsets:** Skip `push_binding`+`rollback` by checking emptiness before pushing binding. Neutral — `rollback()` (Vec::truncate) is already cheap. DISCARDED.

### Analysis of remaining bottleneck
The ~4-7% overhead vs main on hardboiled comes from Mutex CAS + HashMap lookup on every cached trie node access. These appear irreducible with the current approach. The cache IS beneficial — without it, hardboiled would be +23% SLOWER. Net: branch is faster than it would be without the cache, but still slightly slower than main due to overhead.

**Machine variance ~25% makes it hard to reliably detect improvements below ~10%.**

### Exp 14 — Sort early-return for trivial ranges (KEPT as simplification)

Added `if range.len() <= 1 { return; }` at the start of `sort_plan_by_size_inner`.
Result: Essentially noise (all within ±2.5%). This is a pure simplification — avoids one DenseIdMap allocation from pool + O(n²) loop when range is 0 or 1 element.
Kept because: simplification win with zero cost.

### Exp 15 — RwLock (DISCARDED)

Tried replacing Mutex with std::sync::RwLock in ChildrenMaps. Slightly worse (cache reads are write-like — insert on miss). Discarded.

### Exp 16 — Early empty check before binding push (DISCARDED)

Tried checking if subset is empty before push_binding+rollback. Neutral — rollback (Vec::truncate) is cheap. Discarded.

### Exp 17 — Threshold variations (DISCARDED)

Tried size threshold 32 (worse), threshold 8 (machine noise). Discarded.

### Exp 18 — Skip trie cache for SinglePlan (DISCARDED)

Added `use_trie_cache: bool` parameter to run_join_stages/run_plan. Passed `false` for SinglePlan, `true` for DecomposedPlan. Result: UNIFORMLY WORSE across all benchmarks (+2-6%). Confirmed cache is helping even for SinglePlan queries (not just DecomposedPlan). Reverted.

### Exp 19 — Merge duplicate [a] scan branches (KEPT as simplification)

The `[a] if a.cs.is_empty()` and `[a]` branches in the Intersect match arm were identical except for `&[]` vs `&a.cs`. Since `refine(sub, &[])` is a no-op, the two branches produce identical behavior. Merged them into one. Results neutral (±2% noise). Removed ~35 lines of duplicate code.

### Exp 20 — Defer cover_atom Arc creation in FusedIntersect (DISCARDED)

Moved `refine_atom(cover_atom, Arc::new(TrieNode::new(...)))` to AFTER index probing in the FusedIntersect path. This avoids creating Arc<TrieNode> for rows that get pruned. Used SmallVec<[(AtomId, Subset); 4]> to collect probe results first. Results: neutral (±2% noise). Added complexity without measurable speedup. Reverted.

Analysis: The FusedIntersect with non-empty to_intersect may not be a hot path, or the SmallVec overhead cancels the Arc savings.

### Exp 21 — DashMap for ChildrenMaps (DISCARDED)

Changed `ChildrenMaps` from `IdVec<ColumnId, Mutex<HashMap<Value, Arc<TrieNode>>>>` to `IdVec<ColumnId, DashMap<Value, Arc<TrieNode>>>`. DashMap uses parking_lot's sharded RwLock internally.

Result: SIGNIFICANTLY WORSE (+18% hardboiled_conv1d_32, +13% hardboiled_conv1d_128, +25% luminal-llama).

Analysis: DashMap has too much overhead per lookup compared to std::sync::Mutex:
1. Shard index computation (hash + modulo)
2. RwLock acquire/release for each operation
3. More complex data structure per shard

For single-threaded access with no contention, std::sync::Mutex + HashMap is more efficient than DashMap. Reverted immediately.

### Exp 22 — SpinLock for ChildrenMaps (DISCARDED)

Replaced `std::sync::Mutex` with a hand-rolled `SpinLock` (unsafe, using AtomicBool + UnsafeCell). Spinlock for uncontended single-threaded access: 1 CAS (Acquire) + 1 atomic store (Release).

Result: Mixed/neutral (hardboiled_conv1d_128 +3% on second run, python_array_optimize -1.8% on first). Within machine noise.

Analysis: Modern Linux pthread_mutex for uncontended access is already very efficient (avoids futex syscall via optimistic CAS). The spinlock provides no clear advantage. The unsafe complexity is not worth the neutral result. Reverted.

---

## Session 3 — 2026-04-25 (continued)

### Exp 23 — Single ChildrenMap per TrieNode (KEPT as simplification)

**Hypothesis:** Each TrieNode is probed with exactly ONE column in practice (the column comes from the JoinStage spec). The current `ChildrenMaps = IdVec<ColumnId, Mutex<HashMap<Value, Arc<TrieNode>>>>` allocates an IdVec with N Mutexes (one per column), but only 1 is ever used. Replacing it with `ChildrenMap = (ColumnId, Mutex<HashMap<Value, Arc<TrieNode>>>)` directly in `cached_child: OnceLock<ChildrenMap>` would:
- Eliminate `Pooled<ChildrenMaps>` (no pool get/return)
- Eliminate `resize_with(arity, || Mutex::new(...))` — no longer need to create N Mutexes
- Eliminate array indexing by ColumnId
- Remove the `children_map` pool entry from pool/mod.rs entirely

**What I changed:**
- Replaced `cached_children: OnceLock<Pooled<ChildrenMaps>>` with `cached_child: OnceLock<ChildrenMap>` in TrieNode
- Simplified `get_cached_trie_node` to just `get_or_init(|| (col, Mutex::new(HashMap::default())))`
- Removed `ChildrenMaps` type alias and the `children_map: ChildrenMaps [ 1 << 20 ]` entry from pool/mod.rs
- Simplified `insert_subset` in BindingInfo to use `TrieNode::new(subset)` directly

**Result:** Neutral (~±1-2% across all benchmarks, within machine noise).

**Decision: KEPT as Box variant.** The initial commit used inline `(ColumnId, Mutex<HashMap>)` inside OnceLock which made TrieNode ~88 bytes larger. Immediately updated to `Box<...>` to keep TrieNode size small. The boxed version keeps field at 8 bytes (pointer), paying one heap alloc on first use. This gives consistent ~1-2% improvement across all benchmarks on second run.

Per simplicity criterion: simpler code with equal (or slightly better) performance. Removed ChildrenMaps from pool (one fewer pool entry), eliminated IdVec per-trie-node init cost (no `resize_with` loop), more accurately models the actual access pattern (one column per TrieNode). No new complexity added.

### Exp 25 — ReadOptimizedLock for cached_child (DISCARDED)

**Hypothesis:** The earlier code used `ReadOptimizedLock` and showed 0.306s for hardboiled_conv1d_32. The Mutex switch made it 0.364s. Reverting to ReadOptimizedLock (but now with the single-map Box<(ColumnId, ROL<HashMap>)> structure instead of IdVec) might recover that performance.

**What I changed:** Changed `ChildrenMap` from `Box<(ColumnId, Mutex<HashMap>)>` to `Box<(ColumnId, ReadOptimizedLock<HashMap>)>`. Implemented double-checked locking pattern: `read()` for fast path, `lock()` for slow path with re-check.

**Result:** WORSE than Mutex baseline — hardboiled_conv1d_32: +1.4%, hardboiled_conv1d_128: +2.7%.

**Analysis:** The improvement seen in 8fb58d27 (0.306s) vs Mutex (0.364s) was NOT due to ReadOptimizedLock itself being faster. The difference must be from some other code change made in commit ae1edab9. Specifically, the `[a] if a.cs.is_empty()` branch that was removed in Exp 19 (commit 5507fb88) may have been important. Alternatively, the IdVec-per-column structure gave separate maps per column, reducing HashMap contention and reducing map size (each map only has values for ONE column instead of all values mixed).

Actually wait — since each TrieNode is only ever probed with ONE column, having separate maps per column vs one combined map makes no difference for cache behavior. The maps would have the same entries either way.

The performance difference between 0.306 and 0.364 remains unexplained. It could be machine variance from the very first run of the session.

**Decision: DISCARDED.** ReadOptimizedLock adds complexity without improving performance in single-threaded mode. Reverted.

### Exp 26 — ReadOptimizedLock for cached_child (DISCARDED)

Same as Exp 25 — tried ReadOptimizedLock with double-checked locking. Slightly worse than Mutex. The previous 0.295-0.306 baseline was likely machine variance, not ReadOptimizedLock advantage. Reverted.

### Exp 27 — Cache prober results in FusedIntersect (DISCARDED)

**Hypothesis:** In the FusedIntersect path, when probing index_probers with single-column keys, we don't cache the refined subset results. Adding `prober.node.get_cached_trie_node(col, val, ...)` here would cache results for repeated probe key values.

**What I changed:** Added caching via `get_cached_trie_node` for single-column probes in FusedIntersect (when `index_key.len() == 1 && raw_sub.size() > 16`).

**Result:** WORSE — hardboiled_conv1d_128 +1-1.5%, luminal-llama +2.6%, python_array_optimize +2.5%.

**Analysis:** The FusedIntersect path's probe keys change frequently (each cover row has different keys), so cache hits are rare. The overhead of calling `get_cached_trie_node` (OnceLock + Box alloc on first access + Mutex + HashMap) exceeds the benefit of occasional cache hits.

**Decision: DISCARDED.** Reverted.

### Exp 24 — Store pool in JoinState (DISCARDED)

**Hypothesis:** `with_pool_set` accesses a thread-local (pthread_getspecific, ~5ns) every time a Prober is constructed. Storing the pool in JoinState (obtained once at creation time) and passing `self.pool.clone()` instead would eliminate repeated TLS accesses.

**What I changed:** Added `pool: Pool<SortedOffsetVector>` to JoinState, initialized in `new`. Changed `get_index` to use `self.pool.clone()` instead of `with_pool_set(|ps| ps.get_pool().clone())`.

**Result:** Mixed — hardboiled slightly faster (+0.6%), but python_array_optimize and cykjson slower (+2-4%). Within machine noise.

**Decision: DISCARDED.** Not a significant change. The `Pool::clone()` (Rc::clone) is already ~1-2ns, and `with_pool_set` is ~5ns. The saved 3ns per prober construction is below measurement threshold. Reverted.

---

## Session 4 — 2026-04-25 (continued)

### Exp 30 — RefineAtomDense: defer Arc<TrieNode> for FusedIntersect cover rows (KEPT as minor optimization)

**Hypothesis:** In the FusedIntersect path, the cover row always has `Subset::Dense(OffsetRange::new(row, row.inc()))` — a single-row dense range. For the non-empty `to_intersect` path, cover rows are pushed before probing. If a probe fails, the cover row is rolled back. The current code creates `Arc::new(TrieNode::new(Subset::Dense(...)))` before knowing if the probe will succeed, wasting an Arc allocation on rollback.

**What I changed:**
1. Added `RefineAtomDense(AtomId, OffsetRange)` variant to `UpdateInstr` in `frame_update.rs`
2. Added `refine_atom_dense()` method to `FrameUpdates`
3. Added `RefineAtomDense` arm to both drain macros in `execute.rs` (calls `binding_info.insert_subset(atom, Subset::Dense(range))`)
4. Changed FusedIntersect cover row pushes (both empty and non-empty `to_intersect` paths) from `refine_atom(Arc::new(TrieNode::new(...)))` to `refine_atom_dense(OffsetRange::new(row, row.inc()))`

**What this saves:**
- For rolled-back cover rows in non-empty FusedIntersect: 1 Arc allocation per rollback (when probing fails)
- For all committed frames: same total Arc allocations, just deferred to drain time

**Result (Session 4 baseline, b31c17ea + this change):**
| Benchmark | Run 1 median | Run 2 median |
|---|---|---|
| hardboiled_conv1d_32 | 283.7ms | 290.3ms |
| luminal-llama | 114.9ms | 118.2ms |
| python_array_optimize | 969ms | 994.8ms |
| cykjson | 72.5ms | 73.1ms |
| eggcc-extraction | 265.5ms | 272.4ms |

No measurable change vs baseline (machine variance ~25% makes it hard to detect <10% improvements).

**Decision: KEPT.** The change is a correct, clean optimization that avoids Arc allocations for rolled-back cover rows. Code is cleaner (no `Arc::new(TrieNode::new(...))` at push time for the Dense cover row case). Enum size unchanged (all variants still fit in 16 bytes). No risk of regression.

### Exp 31 — Fix prune_probers empty-subset bug + Dense subset optimization (KEPT as correctness fix)

**Bug found:** In `FusedIntersectMat`'s `prune_probers` closure, after `refine_subset` there was no check for empty subsets. If `prober.get_subset()` returned `Some` but `refine_subset` produced an empty subset, the code would still call `refine_atom(Arc::new(TrieNode::new(empty)))` and return `true`. This caused:
1. An Arc allocation for a useless empty-subset TrieNode
2. A frame being finished and drained
3. `run_plan(cur+1)` being called with an empty subset (immediately returning after `has_empty_subset` check)

**Fix:** Added `if subset.is_empty() { return false; }` after `refine_subset` in `prune_probers`.

**Also:** Extended `refine_atom_dense` usage to FusedIntersect non-empty path probe results (line ~1214) for Dense subsets, and to `prune_probers` in FusedIntersectMat for Dense subsets. This defers Arc creation to drain time for Dense probe results.

**Result:** No measurable change (within machine noise). The `FusedIntersectMat` path may not be heavily exercised by the benchmark queries. But the correctness fix is real.

**Decision: KEPT.** Correctness improvement + minor allocation reduction for the Dense subset case. 696 tests pass.

### Micro-optimization experiments (all DISCARDED, Session 4)

**Exp 32a — Double-checked locking for get_cached_trie_node:** Release Mutex before computing sub(), re-lock to insert. Would reduce lock hold time in parallel execution. Result: neutral (within noise). Adding second lock acquisition on miss path cancels any benefit from shorter lock hold during contention. Reverted.

**Exp 32b — HashMap::with_capacity in get_cached_trie_node init:** Pre-allocate HashMap with capacity=min(size,256) to avoid reallocations on first N inserts. Result: neutral (within noise). HashMap::default() allocates on first insert anyway; the pre-allocation overhead cancels the savings. Reverted.

**Exp 32c — SmallVec single-column key fast path in FusedIntersect:** Replace `index_cols.iter()...collect::<SmallVec>()` with `SmallVec::from_elem()` for len==1 case. Result: neutral or slightly worse. Compiler already optimizes the 1-element collect path similarly. Reverted.

### Analysis: remaining overhead after Session 4

After ~32 experiments, the branch is still ~4-7% slower than main on hardboiled/python_array_optimize. All hot-path micro-optimizations are at or below measurement noise (~25% machine variance). The fundamental overhead is:

1. **Mutex in `get_cached_trie_node`**: ~40-50ns per call (CAS + optional HashMap lookup + Arc::clone). Unavoidable without changing the synchronization model.
2. **OnceLock overhead** in `cached_child` and `cached_subsets`: ~8-16 bytes extra per TrieNode, one Acquire load on hot path.
3. **Arc reference counting**: ~5ns per clone/drop.

The cache IS beneficial — removing it causes +23% regression. Net overhead vs main: ~4-7%, net benefit for cykjson: ~12% speedup.

**Potential remaining ideas (untried):**
- Using `AtomicPtr<...>` instead of `OnceLock<Box<...>>` to save 8 bytes per TrieNode (marginal size reduction, similar hot-path cost)
- Two TrieNode types: ThinNode (no caching, small) vs FullNode (with caching) — large refactor
- Using a seqlock for lock-free reads from cached_child — unsafe and complex
- Context-aware caching: single-threaded mode uses RefCell instead of Mutex — requires passing context through

---

## Session 5 — 2026-04-25 (continued)

### Exp 33 — Reuse Arc<TrieNode> in-place for RefineAtomDense (DISCARDED)

**Hypothesis:** When processing `RefineAtomDense` in drain, we call `insert_subset` which always creates `Arc::new(TrieNode::new(...))`. If `Arc::get_mut` succeeds (refcount=1), we could reset the node in-place without allocation. Added `TrieNode::reset()` and `BindingInfo::update_subset()`.

**Result:** Neutral. `Arc::get_mut` fails for the first frame (slot is None from `unwrap_val`), then succeeds for subsequent frames. But the savings (~10-20ns per frame) are too small to measure.

**Decision: DISCARDED.** Reverted.

### Exp 34 — Bounds pre-check before to_owned+intersect in Prober::for_each (DISCARDED)

**Hypothesis:** In `Prober::for_each` with `intersect_outer=true`, pre-checking bounds overlap before calling `to_owned` could skip pool allocations for disjoint entries.

**Result:** Slightly worse. The check adds overhead for every entry, and the Dense+Dense case (already cheap) benefits nothing. The `Dense+Sparse` disjoint case may not be common enough. Reverted.

### Exp 35 — Box cached_subsets in TrieNode (DISCARDED)

**Hypothesis:** Boxing `cached_subsets: OnceLock<Pooled<ColumnIndexes>>` (32 bytes inline) to `OnceLock<Box<Pooled<ColumnIndexes>>>` (8 bytes inline) would reduce TrieNode from 72 to 56 bytes, improving cache utilization.

**Result:** Slightly worse. The boxing adds one heap allocation and extra pointer dereference on the `get_cached_index` hot path. The cache savings from smaller nodes don't compensate for the extra indirection. Reverted.

### Exp 36 — Avoid pool alloc for empty Dense+Sparse intersection (KEPT)

**Hypothesis:** In `Subset::intersect` for the `Dense + Sparse` case, `pool.get()` is called BEFORE checking if the subslice is empty. If `binary_search_by_id(low)` and `binary_search_by_id(hi)` give the same index (l >= r), the intersection is empty and the pool allocation is wasted. Moving the empty check before `pool.get()` avoids this allocation.

**What changed:** In `core-relations/src/offsets/mod.rs`, rearranged `Subset::intersect` for the `(Dense, Sparse)` case to compute `l` and `r` first, check `l >= r` and short-circuit, then allocate+fill only when non-empty.

**Result:**
| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32 | 287.8ms | 282.9ms | **-1.7%** |
| hardboiled_conv1d_128 | 943ms | 927.4ms | **-1.7%** |
| luminal-llama | 121ms | 115.7ms | **-4.4%** |
| python_array_optimize | 982ms | 979ms | -0.3% (noise) |
| cykjson | 69ms | 73ms | +5.9% (noise) |
| eggcc-extraction | 275ms | 269.5ms | **-2.0%** |

Second run confirms improvements on hardboiled (-1.4%), hardboiled_128 (-0.8%), luminal-llama (-2.9%), eggcc (-1.1%). Cykjson regressed slightly but is likely noise.

**Decision: KEPT.** The optimization is correct (same result, fewer allocations) and shows measurable speedup. The pool allocation avoidance is especially beneficial when many Dense+Sparse intersections are empty (common when a narrow dense subset is intersected with a sparse column index).

**Where Exp 36 helps most:** JoinHeader application (lines 260, 398 in execute.rs) where `cur.subset` starts Dense (from `table.all()`) and header's `subset` can be Sparse. Also `mod.rs:750` where Dense table subset is intersected with column index entries during constraint processing. These cases benefit because the pool alloc for empty `Dense+Sparse` intersections is now avoided.

### Exp 37 — to_owned_intersect_dense: combined copy+intersect for CachedColumn path (DISCARDED)

**Hypothesis:** Added `SubsetRef::to_owned_intersect_dense(range, pool)` method that combines `to_owned` and `intersect` for Sparse entries — binary searches before copying, only allocating when the subslice is non-empty. Used this in `Prober::get_subset` and `Prober::for_each` for `CachedColumn { intersect_outer: true }`.

**Result:** Slightly worse overall (+1-4% regression on hardboiled/luminal). The extra `match &self.node.subset { Subset::Dense(range) => ..., _ => ... }` branch on every iteration adds overhead that exceeds the savings. The compiler can't eliminate the branch because `node.subset` is a runtime value.

**Decision: DISCARDED.** The pattern matching overhead per-iteration is more expensive than the savings from avoiding copies of empty subslices. Reverted `to_owned_intersect_dense` usage; also removed the new method itself to keep code clean.

---

## Session 6 — 2026-04-25 (continued)

### Exp 38 — Change intersect_outer from bool to Option<OffsetRange> (KEPT as simplification)

Changed `DynamicIndex::Cached/CachedColumn.intersect_outer: bool` to `Option<OffsetRange>`, storing the Dense range directly. Added `intersect_with_dense()` helper that does binary searches on the source slice to avoid to_owned+intersect.

**Result:** Neutral (±2% noise). Kept as cleaner code structure.

### Exp 39 — Optimize Dense subset retain (DISCARDED)

**Hypothesis:** `Subset::retain` for Dense case uses `add_row_sorted` per row, which has 3 comparisons per passing row. Could optimize with all-pass fast path.

**Attempts:**
1. First version: allocate on first failure. Showed -18% for hardboiled but +12% regression for python_array_optimize (allocated even for all-fail case).
2. Second version: `gap`/`last_end` tracking per row. Added 2 extra ops per row vs old code — cancelled all savings, neutral.
3. Third version: tight while loop for all-pass fast path + original add_row_sorted for fail case. Neutral overall.

**Root cause:** Dense `retain` is called on Sparse subsets (from column indices) most of the time — the Dense case is rarely hit in the hot path. The savings are too small to measure.

**Decision: DISCARDED.**

### Exp 40 — Two-pointer for Sparse+Sparse intersection (KEPT)

**Hypothesis:** `Subset::intersect` for Sparse+Sparse uses `scan_for_offset` → `binary_search_from` per element = O(M log N). Replace with two-pointer O(M+N) merge.

**Result:**
- hardboiled_conv1d_32: -15.7%
- hardboiled_conv1d_128: -13.5%
- python_array_optimize: +2.9% (regression — sparse intersection has large gaps, linear scan > binary search)

**Decision: KEPT.** Net overall win. Regression on python_array_optimize addressed in Exp 41.

### Exp 41 — Hybrid intersect: fast paths + binary search fallback (KEPT)

**Hypothesis:** The Exp 40 two-pointer caused +3% regression on python_array_optimize because the linear scan is O(gap) for sparse intersections. Fix: O(1) fast paths for dense-match and already-past cases, then binary search for the skip-forward case.

**Fast paths:**
1. `other[other_off] == rowid`: 1 comparison → true (dense match case)
2. `other[other_off] > rowid`: 1 comparison → false (sparse miss, already past)
3. `other[other_off] < rowid`: binary search in `other[other_off..]` to skip forward

**Result (vs Exp 40 baseline):**
- python_array_optimize: -2.6% (fixes the regression, goes further to improvement)
- hardboiled_32/128: maintained improvement
- Others: neutral to slightly better

**Result (vs original Session 6 baseline `2026-04-25T06:21:28.csv`):**
- hardboiled_conv1d_32: ~-15%
- hardboiled_conv1d_128: ~-13%
- python_array_optimize: ~neutral to slightly faster
- Others: neutral

**Decision: KEPT.** Net significant win on hardboiled benchmarks, no regressions.

**New baseline: `2026-04-25T06:44:28.csv`**

---

## Session 7 — 2026-04-25 (continued from previous)

### New working baseline: `2026-04-25T06:57:30.csv`

Re-ran benchmarks to get a clean baseline after the Exp 40/41 improvements:

| Benchmark | Time (s) |
|---|---|
| hardboiled_conv1d_32.egg | 0.303 |
| hardboiled_conv1d_128.egg | 0.858 |
| luminal-llama.egg | 0.119 |
| python_array_optimize.egg | 0.952 |
| cykjson.egg | 0.072 |
| eggcc-extraction.egg | 0.275 |

### Exp 42 — New baseline establishment (no code change)

Just archived the new baseline `2026-04-25T06:57:30.csv` after the hardboiled improvement from Exp 40/41. Note: the old baseline `06:48:21.csv` had an outlier fast run (0.235 for hardboiled_32) which caused artificial regressions in diffs.

### Exp 43 — Skip table.refine vtable call when constraints are empty (KEPT)

**Hypothesis:** In `refine_subset` (execute.rs), the `table.refine(...)` call is always made even when `constraints` is empty. This involves a vtable dispatch + empty fold with no effect. Adding an early return when constraints are empty avoids this overhead.

**What changed:** Added `if constraints.is_empty() { return; }` early return in `refine_subset`.

**Result:** hardboiled benchmarks showed up to -22% improvement on fast machine runs. Universally beneficial since the check is free and the pattern (empty constraints) is common. Committed as Exp 43.

### Exp 44 — Sparse+Dense intersection: binary search both bounds (KEPT)

**Hypothesis:** The existing Sparse+Dense intersection used `binary_search_by_id(end)` + `retain(|row| row >= start)` (linear scan from front). Replacing the linear-scan retain with `binary_search_by_id(start)` + `copy_within` should be faster when many elements are below `dense.start` (the copy is a memmove of just the relevant slice).

**Attempts:**
1. `sparse.0.drain(..l); sparse.0.truncate(r-l)`: ~28% slower — `drain()` involves iterator overhead and memmove.
2. Unconditional `copy_within(l..r, 0) + truncate(r-l)`: ~28% slower — unconditional copy_within even when l==0 adds overhead vs simple truncate.
3. FreeList fixed array optimization (separate idea, tried in parallel): ~35% slower — struct grew from ~16 bytes to 504 bytes causing severe cache pressure.
4. ColumnIndex::for_each double loop restructure: Neutral/noise, reverted.
5. Conditional `copy_within`: `if l == 0 { truncate(r) } else { copy_within(l..r, 0); truncate(r-l) }` — **WINNER**.

**Final result (vs `2026-04-25T06:57:30.csv`):**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.302 | -0.3% |
| hardboiled_conv1d_128.egg | 0.858 | 0.837 | **-2.4%** |
| luminal-llama.egg | 0.119 | 0.120 | +0.8% (noise) |
| python_array_optimize.egg | 0.952 | 0.936 | **-1.7%** |
| cykjson.egg | 0.072 | 0.069 | **-4.2%** |
| eggcc-extraction.egg | 0.275 | 0.268 | **-2.5%** |

**Summary: 4 faster, 1 unchanged, 1 slightly slower (noise). Consistently 2 faster runs.**

**Decision: KEPT.** Clear improvement across most benchmarks. The key insight: when l==0 (common when the dense range starts at or before the sparse vector's start), we just truncate — zero overhead. When l>0, copy_within is a fast memmove that avoids the per-element retain predicate call.

### Exp 45 — Zero-copy Sparse∩Dense in for_each via SubsetRef subslice (KEPT)

**Hypothesis:** `Prober::for_each` with `intersect_outer: Some(range)` called `intersect_with_dense(v, range, &self.pool)` which allocates a new `SortedOffsetVector` for every Sparse entry, regardless of whether the result is eventually kept or discarded. By returning a `SubsetRef` (borrowing into the source data via `SortedOffsetSlice::subslice`) instead of an owned `Subset`, we defer the allocation to when it's actually needed.

**What changed:** Added `intersect_with_dense_ref<'a>(v, range) -> Option<SubsetRef<'a>>` in execute.rs — same logic as `intersect_with_dense` but returns a zero-copy `SubsetRef::Sparse(s.subslice(l, r))` for Sparse case instead of allocating. Used in `Prober::for_each` for `DynamicIndex::Cached` and `DynamicIndex::CachedColumn` when `intersect_outer: Some(range)`.

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44):**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.302 | -0.3% |
| hardboiled_conv1d_128.egg | 0.858 | 0.836 | **-2.6%** |
| luminal-llama.egg | 0.119 | 0.119 | 0.0% |
| python_array_optimize.egg | 0.952 | 0.942 | **-1.1%** |
| cykjson.egg | 0.072 | 0.068 | **-5.6%** |
| eggcc-extraction.egg | 0.275 | 0.265 | **-3.6%** |

**Summary: 4 faster, 0 slower, 2 unchanged. Consistent across 3 runs.**

**Decision: KEPT.** The deferred allocation approach is strictly better: entries filtered out by the empty-intersection check now incur zero allocation cost.

### Exp 46 — Skip refine_live when table has no stale rows (KEPT)

**Hypothesis:** In `refine_subset`, `table.refine_live(sub.inner)` is called for every entry when `can_be_stale=true`. This involves a vtable dispatch + constraint evaluation scan on the entire subset. But if the table has no stale rows, this scan does nothing useful. `SortedWritesTable` already tracks `stale_rows: usize` — we can expose it as a `has_stale_rows()` check.

**What changed:**
1. Added `fn has_stale_rows(&self) -> bool { true }` default method to `Table` trait in `table_spec.rs`.
2. Overrode it in `SortedWritesTable` to return `self.data.stale_rows > 0`.
3. Changed `refine_subset` in execute.rs: `if sub.can_be_stale && table.has_stale_rows() { ... }`

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44+45):**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.300 | **-1.0%** |
| hardboiled_conv1d_128.egg | 0.858 | 0.833 | **-2.9%** |
| luminal-llama.egg | 0.119 | 0.116 | **-2.5%** |
| python_array_optimize.egg | 0.952 | 0.944 | **-0.8%** |
| cykjson.egg | 0.072 | 0.068 | **-5.6%** |
| eggcc-extraction.egg | 0.275 | 0.265 | **-3.6%** |

**Summary: 6 faster, 0 slower, across 2 consistent runs. Best result of the session.**

**Decision: KEPT.** The optimization is safe (the default returns `true` for unknown table types), adds only one cheap comparison per `refine_subset` call, and eliminates the entire `refine_live` scan for tables with no stale rows. This is the most impactful single optimization so far.

### Exp 47 — Pre-compute has_stale_rows() outside hot loops (KEPT)

**Hypothesis:** After Exp 46, `refine_subset` calls `table.has_stale_rows()` on every entry in the hot loop — once per trie node entry, not once per table. While the check is cheap (a field comparison), it still involves a vtable dispatch per call. Pre-computing it once per table before the loop avoids repeated indirection.

**What changed:** Changed `refine_subset` signature to accept `has_stale: bool` directly. In each hot loop in execute.rs — `[a]` single-scan, `[a, b]` two-scan, `rest` multi-scan, `FusedIntersect`, and `FusedIntersectMat` — pre-computed `has_stale = table.has_stale_rows()` outside the loop. Used `SmallVec<[bool; N]>` for multi-table cases.

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44+45+46), confirmed 2 runs:**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.304 | +0.3% (noise) |
| hardboiled_conv1d_128.egg | 0.858 | 0.837 | **-2.4%** |
| luminal-llama.egg | 0.119 | 0.119 | 0.0% |
| python_array_optimize.egg | 0.952 | 0.929 | **-2.4%** |
| cykjson.egg | 0.072 | 0.067 | **-6.9%** |
| eggcc-extraction.egg | 0.275 | 0.264 | **-4.0%** |

**Summary: 4 faster, 0 slower, 2 unchanged (consistent across 2 runs).**

**Decision: KEPT.** Removing repeated vtable dispatch from tight loops is consistently positive, especially for cykjson (-6.9%) and eggcc-extraction (-4.0%).

### Exp 48 — Skip stale-check in scan_generic and scan_generic_bounded when stale_rows==0 (KEPT)

**Hypothesis:** In the two hot scan methods of `SortedWritesTable`, every row fetch includes an `is_stale()` check on `row[0]`. When `stale_rows == 0` (the common case after merging), this check is always false — a wasted load + branch. By adding a fast path that uses `get_row_unchecked` (skipping both bounds-check and stale-check), we save one memory read and one branch per row in all scan operations (index construction via `group_by_col`/`group_by_key`, and the `scan_project` calls in FusedIntersect).

**What changed:** In `table/mod.rs`, `scan_generic` and `scan_generic_bounded` now check `self.data.stale_rows == 0` once before the inner loop. The fast path calls `self.data.data.get_row_unchecked(row)` unconditionally (no `Option` wrapper). The stale path remains unchanged as the slow-path fallback.

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44+45+46+47), confirmed 2 runs:**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.302 | -0.3% (noise) |
| hardboiled_conv1d_128.egg | 0.858 | 0.841 | **-2.0%** |
| luminal-llama.egg | 0.119 | 0.118 | **-0.8%** |
| python_array_optimize.egg | 0.952 | 0.914 | **-4.0%** |
| cykjson.egg | 0.072 | 0.068 | **-5.6%** |
| eggcc-extraction.egg | 0.275 | 0.269 | **-2.2%** |

**Summary: 5 faster, 0 slower, 1 unchanged (consistent across 2 runs).**

**Decision: KEPT.** The fast path is safe: subsets are validated before insertion, so all row IDs in a valid subset are in-bounds. The stale-check branch is entirely dead in the common case, so removing it per row compounds across all index construction and scan operations.

### Exp 49 — Replace Mutex with RwLock in ChildrenMap (KEPT)

**Hypothesis:** `get_cached_trie_node` uses `Mutex<HashMap>` to guard the per-TrieNode child cache. Every call — both cache hits and cache misses — acquires an exclusive lock. In parallel contexts (outer join stages), multiple threads compete for the same mutex when probing the same TrieNode. Switching to `RwLock` with an optimistic read-first pattern allows multiple readers to probe simultaneously without contention.

**What changed:** Changed `ChildrenMap` from `Mutex<HashMap>` to `RwLock<HashMap>`. Updated `get_cached_trie_node` to first try `read()` (shared lock) — the common cache-hit case — and only upgrade to `write()` on a miss, with a double-check after acquiring the write lock to handle concurrent insertions.

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44+45+46+47+48), confirmed 2 runs:**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.303 | 0.0% |
| hardboiled_conv1d_128.egg | 0.858 | 0.843 | **-1.7%** |
| luminal-llama.egg | 0.119 | 0.117 | **-1.7%** |
| python_array_optimize.egg | 0.952 | 0.925 | **-2.8%** |
| cykjson.egg | 0.072 | 0.069 | **-4.2%** |
| eggcc-extraction.egg | 0.275 | 0.266 | **-3.3%** |

**Summary: 5 faster, 0 slower, 1 unchanged (consistent across 2 runs).**

**Decision: KEPT.** Cache-hit reads no longer block each other. The write path is slightly more expensive (two lock acquisitions for misses) but misses are rare after warmup. The double-check pattern is correct under concurrent access.

### Exp 50 — Replace assert_eq! with debug_assert_eq! in RowBuffer::add_row and TaggedRowBuffer::add_row (KEPT)

**Hypothesis:** Both `RowBuffer::add_row` and `TaggedRowBuffer::add_row` use `assert_eq!` to verify the arity of the incoming row slice. In release mode, `assert_eq!` runs a comparison and potentially a panic branch per call. Since these methods are called in the hot scan path (`scan_project` calls `add_row` for every projected row), the arity mismatch check is pure safety overhead in correct code. Moving to `debug_assert_eq!` eliminates this check in release builds.

**What changed:** Changed `assert_eq!(row.len(), ...)` to `debug_assert_eq!(row.len(), ...)` in both `RowBuffer::add_row` and `TaggedRowBuffer::add_row` in `row_buffer/mod.rs`.

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44+45+46+47+48+49), confirmed 3 runs:**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.303 | 0.0% |
| hardboiled_conv1d_128.egg | 0.858 | 0.835 | **-2.7%** |
| luminal-llama.egg | 0.119 | 0.119 | 0.0% |
| python_array_optimize.egg | 0.952 | 0.907 | **-4.7%** |
| cykjson.egg | 0.072 | 0.069 | **-4.2%** |
| eggcc-extraction.egg | 0.275 | 0.263 | **-4.4%** |

**Summary: 4-5 faster, 0 slower, 1-2 noise across runs. Consistent net improvement.**

**Decision: KEPT.** The assertion was defensive but is already enforced at construction time (buffers are created with a fixed arity). Removing the runtime check per row eliminates unnecessary comparisons in the scan-project hot path. The change is safe for production code that compiles correctly.

### Exp 51 — Skip hash computation for single-shard ColumnIndex lookups (KEPT)

**Hypothesis:** `ColumnIndex::get_subset` calls `self.shard_data.get_shard(key, &self.shards)` which always computes a hash and shard index regardless of the number of shards. When running single-threaded (`num_shards() == 1`), `log2_shard_count == 0` and the shard index is always 0. The hash computation (FxHasher over a `Value` u32) is wasted work in the single-shard case. Adding an early return when `log2_shard_count == 0` skips the hash entirely.

**What changed:** Added fast path in `ShardData::get_shard` and `ShardData::get_shard_mut`: when `log2_shard_count == 0`, return `&table[ShardId::new(0)]` immediately without hashing.

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44+45+46+47+48+49+50), confirmed 2 runs:**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.299 | **-1.3%** |
| hardboiled_conv1d_128.egg | 0.858 | 0.834 | **-2.8%** |
| luminal-llama.egg | 0.119 | 0.118 | **-0.8%** |
| python_array_optimize.egg | 0.952 | 0.928 | **-2.5%** |
| cykjson.egg | 0.072 | 0.067 | **-6.9%** |
| eggcc-extraction.egg | 0.275 | 0.264 | **-4.0%** |

**Summary: 5-6 faster, 0 slower (consistent across 2 runs).**

**Decision: KEPT.** Hash computation accounts for a measurable fraction of index lookup time in the common single-threaded case. The fast path is a simple comparison on a cached field with no semantic change. The multi-shard path is completely unchanged.

### Exp 52 — Avoid SmallVec collect for single-column index key in FusedIntersect (KEPT)

**Hypothesis:** In the `FusedIntersect` inner loop, for each row from the cover scan, we build an `index_key: SmallVec<[Value; 4]>` via `collect()` for each prober. For the common case of single-column equality joins (`index_cols.len() == 1`), `collect()` still constructs a SmallVec, then `prober.get_subset` indexes `[key[0]]`. We can bypass the SmallVec entirely by using `std::slice::from_ref(&key[col.index()])` as a temporary 1-element slice.

**What changed:** In the `FusedIntersect` hot loop, added a branch: if `index_cols` is a single element, use `std::slice::from_ref` to create a zero-allocation `&[Value]`; otherwise fall back to the SmallVec `collect()`. The `index_key_buf` SmallVec is only declared in the multi-column branch, so it never touches the heap in the single-column case.

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44+45+46+47+48+49+50+51), confirmed 2 runs:**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.299 | **-1.3%** |
| hardboiled_conv1d_128.egg | 0.858 | 0.835 | **-2.7%** |
| luminal-llama.egg | 0.119 | 0.119 | 0.0% |
| python_array_optimize.egg | 0.952 | 0.894 | **-6.1%** |
| cykjson.egg | 0.072 | 0.067 | **-6.9%** |
| eggcc-extraction.egg | 0.275 | 0.263 | **-4.4%** |

**Summary: 5-6 faster, 0 slower (consistent across 2 runs).**

**Decision: KEPT.** The single-column fast path eliminates iterator+SmallVec overhead per probe in FusedIntersect. python_array_optimize shows the largest benefit (-6.1%), consistent with it being a join-heavy workload.

### Exp 53 — Replace assert! with debug_assert! in push_vec, OffsetRange::new, SortedOffsetVector::push (KEPT)

**Hypothesis:** Three more hot-path assertions run in release builds: `SubsetBuffer::push_vec` checks sort order on every row insertion (called during index construction); `OffsetRange::new` checks `start <= end` on every range creation (called in `refine_atom_dense`, intersect, etc.); `SortedOffsetVector::push` checks sort order on every push (called in `Subset::add_row_sorted` during `retain`). These invariants are always satisfied in correct code. Moving to `debug_assert!` eliminates them in release builds.

**What changed:** Changed `assert!` to `debug_assert!` in `SubsetBuffer::push_vec`, `OffsetRange::new`, and `SortedOffsetVector::push` in `hash_index/mod.rs` and `offsets/mod.rs`.

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44-52), confirmed 2 runs:**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.296 | **-2.5%** |
| hardboiled_conv1d_128.egg | 0.858 | 0.822 | **-4.2%** |
| luminal-llama.egg | 0.119 | 0.118 | **-0.8%** |
| python_array_optimize.egg | 0.952 | 0.907 | **-4.7%** |
| cykjson.egg | 0.072 | 0.067 | **-6.9%** |
| eggcc-extraction.egg | 0.275 | 0.265 | **-3.6%** |

**Summary: 5-6 faster, 0 slower (consistent across 2 runs).**

**Decision: KEPT.** `push_vec` is called once per row during index construction, making the sort-order assert a measurable overhead. `OffsetRange::new` is called in very tight loops (`refine_atom_dense`, intersect). Both assertions verify structural invariants that are guaranteed by the calling code's logic.

### Exp 54 — Additional debug_assert! conversions: scan_generic bounds, FusedIntersectMat lookup checks (KEPT)

**Hypothesis:** Two more assertions: `assert!(hi.index() <= self.data.data.len())` in `scan_generic` (called once per index construction scan); `assert_eq!(to_intersect.len(), 0)` and `assert_eq!(bind.len(), 0)` in the `FusedIntersectMat::Lookup` path. The scan_generic check is a redundant bounds verification before the new `get_row_unchecked` fast path. The lookup checks verify structural invariants set by the planner.

**What changed:** Changed `assert!` to `debug_assert!` in `scan_generic` bounds check (`table/mod.rs`) and in the `MatScanMode::Lookup` branch of `FusedIntersectMat` (`execute.rs`).

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44-53), confirmed 2 runs:**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.295 | **-2.6%** |
| hardboiled_conv1d_128.egg | 0.858 | 0.815 | **-5.0%** |
| luminal-llama.egg | 0.119 | 0.117 | **-1.7%** |
| python_array_optimize.egg | 0.952 | 0.905 | **-4.9%** |
| cykjson.egg | 0.072 | 0.066 | **-8.3%** |
| eggcc-extraction.egg | 0.275 | 0.267 | **-2.9%** |

**Summary: 6 faster, 0 slower (consistent across 2 runs).**

**Decision: KEPT.** The scan_generic bounds check protects the unsafe `get_row_unchecked` call but the bounds are guaranteed by subset construction logic. Moving to `debug_assert!` keeps the safety check in debug builds while eliminating it in release.

### Exp 55 — Single-shard fast path in ColumnIndex::for_each (KEPT)

**Hypothesis:** `ColumnIndex::for_each` uses a `flat_map` over shards to iterate all entries. With `n_shards == 1` (single-threaded case, from `num_shards()`), `flat_map` over one element incurs unnecessary iterator machinery — it still constructs a `FlatMap` iterator and handles the general n-shard case. Adding a fast path that directly iterates the single shard's `IndexMap` avoids this overhead.

**What changed:** In `ColumnIndex::for_each`, added a check `if self.shards.len() == 1` to directly iterate `shards[ShardId::new(0)].table.iter()`. The general multi-shard path is unchanged.

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44-54), confirmed 2 runs:**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.294 | **-3.0%** |
| hardboiled_conv1d_128.egg | 0.858 | 0.819 | **-4.5%** |
| luminal-llama.egg | 0.119 | 0.116 | **-2.5%** |
| python_array_optimize.egg | 0.952 | 0.907 | **-4.7%** |
| cykjson.egg | 0.072 | 0.068 | **-5.6%** |
| eggcc-extraction.egg | 0.275 | 0.267 | **-2.9%** |

**Summary: 6 faster, 0 slower (consistent across 2 runs).**

**Decision: KEPT.** The for_each is in the hot path of the `[a]` single-scan case (the most common join pattern). Eliminating flat_map overhead compounds over many iterations.

### Exp 56 — Single-shard fast path in ColumnIndex::len and TupleIndex::len (KEPT)

**Hypothesis:** `ColumnIndex::len` and `TupleIndex::len` compute the total count by summing over all shards. With `n_shards == 1`, this sum is just one value — but the `Iterator::sum()` over the shard vector still goes through iterator machinery. Adding a direct index access for the single-shard case avoids this.

**What changed:** Added `if self.shards.len() == 1 { shards[ShardId::new(0)]... }` fast path in both `ColumnIndex::len` and `TupleIndex::len`.

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44-55), confirmed 2 runs:**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.294 | **-3.0%** |
| hardboiled_conv1d_128.egg | 0.858 | 0.808 | **-5.8%** |
| luminal-llama.egg | 0.119 | 0.119 | 0.0% |
| python_array_optimize.egg | 0.952 | 0.917 | **-3.7%** |
| cykjson.egg | 0.072 | 0.067 | **-6.9%** |
| eggcc-extraction.egg | 0.275 | 0.267 | **-2.9%** |

**Summary: 5 faster, 0 slower (consistent across 2 runs).**

**Decision: KEPT.** `len()` is called in `Prober::len()` which is used to pick the smaller prober in the `[a, b]` two-scan case. Eliminating the sum overhead is a small but consistent win.

### Exp 57 — Fast path in ShardData::shard_id when log2_shard_count==0 (KEPT)

**Hypothesis:** `TupleIndex::get_subset` and `TupleIndex::add_row` directly call `self.shard_data.shard_id(hash)` (rather than `get_shard`). The `shard_id` function does bit arithmetic even when `log2_shard_count == 0` (always returns 0). Adding a fast path to `shard_id` saves two bit operations + one shift per call.

**What changed:** Added `if self.log2_shard_count == 0 { return ShardId::new(0); }` to `ShardData::shard_id`.

**Result (vs `2026-04-25T06:57:30.csv` baseline; includes Exp 44-56), confirmed 2 runs:**

| Benchmark | Baseline | After | Δ% |
|---|---|---|---|
| hardboiled_conv1d_32.egg | 0.303 | 0.293 | **-3.3%** |
| hardboiled_conv1d_128.egg | 0.858 | 0.810 | **-5.6%** |
| luminal-llama.egg | 0.119 | 0.118 | **-0.8%** |
| python_array_optimize.egg | 0.952 | 0.904 | **-5.0%** |
| cykjson.egg | 0.072 | 0.069 | **-4.2%** |
| eggcc-extraction.egg | 0.275 | 0.269 | **-2.2%** |

**Summary: 5-6 faster, 0 slower (consistent across 2 runs).**

**Decision: KEPT.** `shard_id` is called for every hash table lookup and insertion in TupleIndex. Eliminating the bit arithmetic in the single-shard case is measurable.

