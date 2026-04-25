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

---

