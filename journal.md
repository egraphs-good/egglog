# Experiment Journal

## Q11: SmallVec for keys in FusedIntersectMat::Lookup/Value (2026-05-09) — REVERTED

**Hypothesis:** Per-lookup `Vec<Value>` alloc replaced with `SmallVec<[Value; 4]>`. Cykjson doesn't reach FusedIntersectMat (no multi-arg trees), should be safe.

**Approach:** Replace `index_vars.iter().map(...).collect::<Vec<_>>()` with a SmallVec push loop and `cover_mat.get(keys.as_slice())`.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs Q9-revert baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.293 → 0.291 (-0.5%, noise)
- hardboiled_conv1d_128: 0.759 → 0.765 (+0.8%)
- luminal-llama: 0.213 → 0.218 (**+2.4%**)
- python_array_optimize: 0.508 → 0.509 (+0.1%, noise)
- cykjson: 0.103 → 0.104 (+1.0%)
- eggcc-extraction: 0.436 → 0.437 (+0.3%, noise)
- Overall average: +0.69%

**VERDICT: REGRESSION** — Reverted. luminal-llama +2.4% (uses tree-decomposed plans). Net +0.69%.

**Lesson:** Even this contained micro-alloc-removal in a path that should not affect cykjson regressed multiple benchmarks. **Run_plan's stack frame is so densely packed that ANY new local variable shifts register allocation enough to perturb perf elsewhere.** This effectively confirms we've hit the ceiling on incremental optimization of run_plan via the analyst-driven loop. Future work would need: (a) profile-guided optimization, (b) separating run_plan into smaller specialized functions, or (c) targeting code paths that aren't called from run_plan at all (e.g. setup, planning, action execution).

## Q9: get_unchecked in UnionFind::find_naive (2026-05-09) — REVERTED

**Hypothesis:** Hoisting bounds checks out of the parent-walk loop in `union-find/src/lib.rs::find_naive` should help rebuild-heavy benchmarks; cykjson doesn't enter rebuild paths so should be untouched. Invariant: every value in `parents[]` is itself a valid index.

**Approach:** `parents.get_unchecked(cur.index())` with SAFETY comment. 7-line change in a function in a separate crate.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs Q8-revert baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.294 → 0.290 (-1.3%)
- hardboiled_conv1d_128: 0.765 → 0.758 (-0.9%)
- luminal-llama: 0.216 → 0.217 (+0.4%, noise)
- python_array_optimize: 0.511 → 0.511 (-0.0%, noise)
- cykjson: 0.101 → 0.106 (**+4.7% regression**)
- eggcc-extraction: 0.432 → 0.443 (**+2.5% regression**)
- Overall average: +0.89%

**VERDICT: REGRESSION** — Reverted. Cykjson +4.7% despite cross-crate isolation; eggcc +2.5% unexpected (eggcc is rebuild-heavy and should have benefited).

**Lesson:** Cykjson's hyper-sensitivity reaches even cross-crate changes. Possible mechanisms: (a) `find_naive` IS called occasionally for cykjson (via `DisplacedTable::get_row_column` which the analyst may have undercounted), (b) the unsafe `get_unchecked` codegen somehow perturbs LTO inlining decisions across crate boundaries. Eggcc regression is harder to explain. **Implication: the cykjson canary is essentially impossible to bypass at this point. Further iteration risks net regressions.**

## Q8: Hoist FusedIntersectMat bind partition to per-stage (2026-05-09) — REVERTED

**Hypothesis:** Q7's per-group partition can be done once per stage (group_key_len is invariant within a materialization). Should net a small additional gain.

**Approach:** Peek first group for group_key_len, partition `bind_key`/`bind_nonkey` once before the per-group loop. Added `debug_assert_eq!(group_key.len(), group_key_len)` inside the loop. Teardown unchanged.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs Q7 baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.293 → 0.289 (-1.7%)
- hardboiled_conv1d_128: 0.763 → 0.758 (-0.5%)
- luminal-llama: 0.215 → 0.214 (-0.5%, noise)
- python_array_optimize: 0.511 → 0.513 (+0.4%, noise)
- cykjson: 0.103 → 0.104 (+1.8%) — small but real regression
- eggcc-extraction: 0.438 → 0.440 (+0.5%, noise)
- Overall average: +0.01%

**VERDICT: REGRESSION** — Reverted. Cykjson +1.8% breaks the cykjson-must-not-regress rule despite hardboiled gains.

**Lesson:** Even a code-restructure that doesn't touch cykjson's hot path can regress it via stack frame layout / register pressure / instruction cache effects. The hoist allocates the SmallVecs at a higher scope, slightly perturbing the function's stack frame even when unused. Cykjson's hyper-sensitivity extends to layout-only changes in shared functions. **Implication: stop trying to incrementally improve Q7. Look for completely different code paths.**

## Q7: Pre-partition bind in FusedIntersectMat (2026-05-09) — KEPT

**Hypothesis:** In `JoinStage::FusedIntersectMat`'s `MatScanMode::Full` block, the per-`non_keys`-row inner loop scanned `bind` TWICE with branch filters (col.index() < group_key_len vs >= group_key_len). For groups with many non_keys rows, this is quadratic in `bind.len()`.

**Approach:** Partition `bind` into `bind_key`/`bind_nonkey: SmallVec<[(ColumnId, Variable); 2]>` ONCE per group (just inside the outer scan loop). Then each per-row inner loop iterates only the relevant SmallVec, with no branching. Same change applied to `MatScanMode::KeyOnly`.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs Q6-revert baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.293 → 0.290 (-1.3%)
- hardboiled_conv1d_128: 0.760 → 0.757 (-0.3%, noise)
- luminal-llama: 0.216 → 0.212 (-1.9%)
- python_array_optimize: 0.513 → 0.514 (+0.3%, noise)
- cykjson: 0.105 → 0.104 (-0.8%) — slight improvement, not regressed
- eggcc-extraction: 0.442 → 0.437 (-1.0%)
- Overall average: −0.82%

**VERDICT: MODEST GAIN** — Accepted. 4 benchmarks improved, no regressions ≥1%, even cykjson slipped slightly faster (likely cache/code-layout improvement from removing branch density). Predicted "cykjson untouched" held.

**Commit:** 9ea072ba — KEPT.

**Lessons:**
- The "pick a code path cykjson doesn't exercise" strategy worked. FusedIntersectMat is gated behind tree decomposition (multi-arg constructors), which CYK grammars don't trigger.
- Branch-density-removal in tight inner loops paid off across multiple benchmarks even at modest magnitudes.
- The partition is per-group (could be hoisted further to per-stage since `group_key_len` is constant within a materialization) — possible follow-up.

## Q6: extend_from_slice in SortedOffsetVector (2026-05-09) — REVERTED

**Hypothesis:** `SortedOffsetVector::extend_nonoverlapping` calls `self.0.extend(other.iter())` where `iter()` returns `Copied<slice::Iter<RowId>>` wrapped in opaque impl-trait. The opaque return type may block Vec::extend's memcpy specialization. Switching to `extend_from_slice(other.inner())` should give direct memcpy.

**Approach:** Two-line change in `core-relations/src/offsets/mod.rs::extend_nonoverlapping`.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs Q5-revert baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.295 → 0.295 (-0.0%, noise)
- hardboiled_conv1d_128: 0.758 → 0.758 (-0.1%, noise)
- luminal-llama: 0.214 → 0.213 (-0.2%, noise)
- python_array_optimize: 0.514 → 0.513 (-0.2%, noise)
- cykjson: 0.103 → 0.108 (**+4.7% regression**)
- eggcc-extraction: 0.443 → 0.440 (-0.7%)
- Overall average: +0.58%

**VERDICT: REGRESSION** — Reverted via `git reset --hard HEAD~1`. cykjson again.

**Lesson:** `extend_from_slice` calls `ptr::copy_nonoverlapping` (an out-of-line memcpy via libc). For the small slices that cykjson processes in `to_owned`, the function-call overhead exceeds the iteration cost. The opaque-impl-trait `iter().copied()` path was likely fully inlined and unrolled by LLVM for these tiny lengths. **General rule: `extend_from_slice` is faster for medium/large slices but slower for tiny ones because of the libc-call overhead.** Cykjson's hot path uses this on consistently-small slices.

## Q5: In-place Arc::get_mut for plan-header intersect (2026-05-09) — REVERTED

**Hypothesis:** The header-intersect loop in `core_run_impl` (sequential at ~line 551, parallel at ~line 413) does `Arc::try_unwrap(binding_info.unwrap_val(*atom)).unwrap()` then `binding_info.move_back_node(*atom, Arc::new(cur))` per atom per rule eval. The `try_unwrap.unwrap()` proves unique ownership; `Arc::get_mut` would let us mutate in place, eliminating one alloc + free per call.

**Approach:** Added `BindingInfo::header_intersect(atom, subset, pool)` that calls `Arc::get_mut(slot).expect(...)`, mutates the TrieNode's subset in place. Replaced both call sites' 5-line block with two lines (`header_intersect` + `has_empty_subset` check).

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs Q4-revert baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.294 → 0.293 (-0.4%, noise)
- hardboiled_conv1d_128: 0.759 → 0.762 (+0.4%, noise)
- luminal-llama: 0.215 → 0.217 (+0.5%, noise)
- python_array_optimize: 0.511 → 0.512 (+0.1%, noise)
- cykjson: 0.102 → 0.108 (**+5.3% regression**)
- eggcc-extraction: 0.445 → 0.439 (-1.2%)
- Overall average: +0.78%

**VERDICT: REGRESSION** — Reverted via `git reset --hard HEAD~1`. cykjson hit again.

**Lesson:** Same pattern as Q4 — saving an Arc alloc/free pair regressed cykjson. Likely reasons:
1. The alloc+drop pair via mimalloc's per-thread heap is very fast (~10ns for ArcInner), and the bytes touched serve as cache warmup.
2. `Arc::get_mut` introduces a refcount load + compare per call; `try_unwrap` may have been better-optimized in mimalloc's hot path.
3. The function-call boundary of the new helper may have prevented some inlining LLVM was doing on the original.

**General rule for cykjson sensitivity: removing micro-allocations from short-rule benchmarks (cykjson) often regresses because (a) mimalloc's hot path makes them essentially free, (b) the implicit cache warming is real.** Cykjson's hot path is dominated by ColumnIndex's add_row pattern (which Q3 helped) and ColumnIndex's rebuild_full (which H1 helped). Don't optimize Arc churn for cykjson — it isn't the bottleneck and the changes carry hidden costs.

## Q4: Skip placeholder fill in SubsetBuffer growth (2026-05-09) — REVERTED

**Hypothesis:** `SubsetBuffer`'s 3 growth paths call `Vec::resize(new_len, RowId::new(u32::MAX))`. The placeholders are immediately overwritten by `fill_at` for the prefix, and the slack is never read (verified). Replacing with `reserve` + `unsafe set_len` should skip a wasted memset.

**Approach:** All 3 growth sites (`new_vec`, `new_vec_with_extra`, `push_vec` else-branch) changed from `resize(.., placeholder)` to `reserve(additional) + unsafe { set_len(new_len) }` with a SAFETY comment. Removed the now-meaningless `assert_ne!(x.rep(), u32::MAX)` debug check in `make_ref`.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs Q3 baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.292 → 0.292 (-0.1%, noise)
- hardboiled_conv1d_128: 0.757 → 0.757 (-0.0%, noise)
- luminal-llama: 0.213 → 0.215 (+0.7%, noise)
- python_array_optimize: 0.517 → 0.512 (-0.9%)
- cykjson: 0.102 → 0.106 (**+3.6%**)
- eggcc-extraction: 0.440 → 0.441 (+0.2%, noise)
- Overall average: +0.60%

**VERDICT: REGRESSION** — Reverted via `git reset --hard HEAD~1`. cykjson regressed exactly the bench Q3 just won big on.

**Lesson:** Removing the memset placeholder fill counterintuitively HURT cykjson by +3.6%. Likely reasons:
1. `Vec::resize` with a constant value uses highly-optimized SIMD memset that warms up cache lines (those bytes are about to be touched by `fill_at`). Skipping the memset means the cache lines arrive cold during fill_at.
2. With placeholders gone, the slack contains random recycled bytes; if any subsequent code path observes them via vectorized loads (even speculatively), it might trigger branch mispredicts.

**General rule: a "wasted" memset of bytes that are about to be touched is NOT necessarily wasted — modern CPUs benefit from prefetch-via-write.** Skip-the-memset optimizations need to be paired with explicit prefetch hints or alternative cache warming.

## Q3: Skip Dense→Sparse transition copy in SubsetBuffer (2026-05-09) — KEPT

**Hypothesis:** Investigating the 4.4% `memmove` profile entry by reading code (lessons-learned ruled out the 64–511 group_by_col range), found a concrete source: `SubsetBuffer::push_vec` (`core-relations/src/hash_index/mod.rs`) does `copy_within` on every power-of-2 growth. The hottest call site is `BufferedSubset::add_row_sorted`'s Dense→Sparse transition (`buf.new_vec(range)` then `buf.push_vec(v, row)`). When `range.len()` is a power of 2, the `next_power_of_two` allocation in `new_vec` is already filled, so the immediate `push_vec` triggers a wasted copy.

**Approach:** Added `SubsetBuffer::new_vec_with_extra(rows, extra)` that allocates `(rows.len() + 1).next_power_of_two()` slots in one shot, fills them with `rows`, then writes `extra` to the next slot. Replaced the `new_vec + push_vec` pair in `BufferedSubset::add_row_sorted`'s Dense→Sparse arm with a single call.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs Q2-revert baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.293 → 0.293 (+0.1%, noise)
- hardboiled_conv1d_128: 0.759 → 0.760 (+0.1%, noise)
- luminal-llama: 0.214 → 0.214 (-0.1%, noise)
- python_array_optimize: 0.518 → 0.507 (**-2.1%**)
- cykjson: 0.106 → 0.101 (**-5.3%**)
- eggcc-extraction: 0.440 → 0.444 (+0.8%, noise)
- Overall average: −1.09%

**VERDICT: IMPROVEMENT** — Clean accept (no reviewer needed). cykjson cleared 3% threshold. Hardboiled essentially unchanged (its add_row_sorted hot path is dominated by other patterns).

**Commit:** 51848939 — KEPT.

**Lessons:**
- The 4.4% memmove was largely the SubsetBuffer power-of-2 doubling pattern, hit hardest by cykjson and python_array_optimize whose Dense→Sparse transitions happen frequently with power-of-2 range lengths.
- Direct code reading (no perf profiling needed) successfully identified the bottleneck after lessons-learned mis-diagnosed it as the 64–511 group_by_col range. The actual source was a much narrower transition path.
- This was a true work-removal change (saves a copy_within) rather than work-hoisting — different from the P3 vein but equally valid.

## Q2: Replace Prober::can_be_stale field with method (2026-05-09) — REVERTED

**Hypothesis (P3 vein):** `Prober::can_be_stale: bool` is pure derived state from `self.ix`'s variant. Replace the field with an `#[inline] fn can_be_stale(&self) -> bool` that reads the discriminant via `matches!(self.ix, DynamicIndex::Cached{..} | DynamicIndex::CachedColumn{..})`. Removes the field, the two construction-site computations, and may shrink Prober.

**Approach:** Deleted the `can_be_stale: bool` field, added the inline method, removed the two `let can_be_stale = matches!(...)` blocks at construction, updated 7 read sites to call `.can_be_stale()`. Net -3 LOC.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs Q1 baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.292 → 0.290 (-0.7%)
- hardboiled_conv1d_128: 0.761 → 0.753 (-1.0%)
- luminal-llama: 0.213 → 0.217 (+1.9%)
- python_array_optimize: 0.522 → 0.518 (-0.8%)
- cykjson: 0.102 → 0.104 (+2.5%)
- eggcc-extraction: 0.444 → 0.439 (-1.1%)
- Overall average: +0.13%

**VERDICT: REGRESSION** — Just barely (4 faster, 2 slower). Reverted via `git reset --hard HEAD~1`.

**Lesson:** Field-read vs discriminant-check tradeoff went the wrong way. The bool field was a single byte load (already in cache from the Arc<TrieNode> deref + variant payload). The new `matches!` does a discriminant load + compare-with-immediate, which is at minimum the same cost and at worst slightly more — and at 7 hot read sites, even a tiny per-site delta accumulates. **Lesson: when the field-vs-method tradeoff has equal per-call cost, the FIELD wins because the field-write happens once at construction (cold) while the method-call cost is paid on every read.** P3's win came from removing per-VALUE state (every PotentiallyStale instance had the bool); this would have removed per-PROBER state (much rarer write/read ratio is unfavorable).

## Q1: Cleanup — delete dead FrameUpdates::with_capacity (2026-05-09) — KEPT

**Hypothesis:** None — pure cleanup. `FrameUpdates::with_capacity` has been unused since G3 introduced `from_pooled_vec` at every call site. Pre-existing dead_code warning had been cluttering build output for several iterations.

**Approach:** Deleted the 7-LOC method.

**Build/tests:** Clean release build (no more dead_code warning), all `make test` suites pass.

**Results vs P3 baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.296 → 0.290 (-1.9%)
- hardboiled_conv1d_128: 0.761 → 0.761 (-0.0%)
- luminal-llama: 0.217 → 0.214 (-1.3%)
- python_array_optimize: 0.522 → 0.523 (+0.2%, noise)
- cykjson: 0.104 → 0.103 (-1.0%)
- eggcc-extraction: 0.442 → 0.446 (+0.9%, noise)
- Overall average: −0.53%

**VERDICT: MODEST GAIN** — Accepted (this isn't a perf change, the deltas are within noise of P3's baseline; the build cleanup justifies it). The dice rolled favorably.

**Commit:** 0fe115b8 — KEPT.

## P3: Remove PotentiallyStale<T> wrapper, hoist refine_live decision (2026-05-09) — KEPT

**Hypothesis:** `struct PotentiallyStale<T> { inner: T, can_be_stale: bool }` wraps every value returned by `Prober::get_subset`/`for_each`. The `can_be_stale` field is **purely a function of which `DynamicIndex` variant produced it** (`Cached*` → always true; `Dynamic*`/`SparseColumn` → always false). So the per-value bool is redundant metadata, and `refine_subset`'s `sub.can_be_stale && has_stale` AND can be hoisted to a single `must_refine_live` constant per Prober.

**Approach:** Delete `struct PotentiallyStale<T>` and its 4 impls. Add `can_be_stale: bool` to `Prober`, set in `get_index`/`get_column_index` via `matches!(dyn_index, DynamicIndex::Cached{..} | DynamicIndex::CachedColumn{..})`. Change `refine_subset` signature from `(PotentiallyStale<Subset>, ..., has_stale)` to `(Subset, ..., must_refine_live)`. At each of the 5 Intersect call sites, compute `let must_refine_live = prober.can_be_stale && table.has_stale_rows();` once before the closure and pass that to `refine_subset`. The closure no longer carries the wrapper, and LLVM can constant-fold-away `refine_subset`'s first branch when `must_refine_live` is statically false.

**Build/tests:** Clean release build, all `make test` suites pass. Net -6 LOC (103 added, 109 removed).

**Results vs P2-revert baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.293 → 0.295 (+0.7%, within noise)
- hardboiled_conv1d_128: 0.768 → 0.757 (-1.4%)
- luminal-llama: 0.215 → 0.214 (-0.6%)
- python_array_optimize: 0.518 → 0.521 (+0.4%, noise)
- cykjson: 0.105 → 0.104 (-1.1%)
- eggcc-extraction: 0.446 → 0.443 (-0.5%)
- Overall average: −0.42%

**VERDICT: MODEST GAIN** — Accepted. No single bench cleared 3%, but net is negative, every meaningful benchmark improved, NO regressions ≥1%, and the change REMOVES code (-6 LOC) so the complexity cost is negative. Per program.md's simplicity criterion: "removing code and getting equal or better results is a great outcome."

**Commit:** adb35eb0 — KEPT.

**Lessons:**
- After 4 consecutive add-code reverts (H3, PP1, P1, P2), a code-removal refactor finally landed cleanly with broad-spectrum modest wins and no regressions. Confirms the lesson: **prefer changes that remove redundancy or hoist work outside hot loops; avoid changes that add per-iteration branches or write-amplification.**
- The "redundant per-value metadata derivable from variant" pattern is worth looking for elsewhere — bools or small fields that are constant across a Prober/scan lifetime.

## P2: Bypass to_owned + refine_subset round-trip when refinement is no-op (2026-05-09) — REVERTED

**Hypothesis:** In `JoinStage::Intersect` `[a]` and `[a,b]` arms, when `cs.is_empty() && !has_stale`, `refine_subset` is the identity. The to_owned + Pool alloc + memcpy + Arc<TrieNode>::new round-trip is wasted. Hoisting a `no_refine` flag once and dispatching directly to `refine_atom_dense` for Dense (and 1-row Sparse) subsets should save ~1.5% (combination of `SubsetRef::to_owned` 1.5% and parts of `refine_atom_subset` 2.1%).

**Approach:** Hoist `let no_refine = a.cs.is_empty() && !has_stale;` once per scan. Inside the closure's `if x.size() <= 16` branch, bypass when `no_refine` is true via a 3-way match: `SubsetRef::Dense(r)` → `refine_atom_dense(r)`, `SubsetRef::Sparse(s) if s.inner().len() == 1` → `refine_atom_dense(OffsetRange::new(row, row.inc()))`, multi-row Sparse → still call `to_owned` but skip `refine_subset`. Apply to both `[a]` and `[a,b]` arms.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs P1-revert baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.295 → 0.296 (+0.5%, noise)
- hardboiled_conv1d_128: 0.767 → 0.763 (-0.5%, noise)
- luminal-llama: 0.213 → 0.216 (+1.2%)
- python_array_optimize: 0.520 → 0.518 (-0.4%, noise)
- cykjson: 0.104 → 0.111 (**+7.6%**)
- eggcc-extraction: 0.438 → 0.441 (+0.6%, noise)
- Overall average: +1.50%

**VERDICT: REGRESSION** — Reverted via `git reset --hard HEAD~1`. cykjson took the worst hit again.

**Lesson:** The added branching (`if no_refine` then a 3-way match) in the hot per-iteration loop costs more than the saved alloc. cykjson has small subsets and shallow recursion → more iterations of the closure → branch overhead amortizes badly. Also: `refine_atom_subset(atom, Subset::Dense(r))` is likely already efficient because `Subset::Dense` matches early in `refine_atom_subset`'s switch and the Arc<TrieNode>::new is hot in cache after the prior FrameUpdates push. **General rule: adding branches to a hot inner loop to skip occasional work is a losing trade unless the skip is dramatic AND the branch is very predictable.**

## P1: #[inline] on Offsets trait impls (2026-05-09) — REVERTED

**Hypothesis:** `SubsetRef::offsets` shows 5.4% self-time in profile despite being a 2-4-instruction loop body, suggesting the compiler is materializing it out-of-line. Adding `#[inline]` to all `Offsets::offsets` and `Offsets::bounds` impls (`OffsetRange`, `SortedOffsetVector`, `SortedOffsetSlice`, `&SortedOffsetSlice`, `SubsetRef`, `Subset` — 12 methods total) should let LLVM fuse the closure body into the per-row loop.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs PP1-revert baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.293 → 0.296 (+1.0%)
- hardboiled_conv1d_128: 0.776 → 0.769 (-0.8%)
- luminal-llama: 0.212 → 0.215 (+1.3%)
- python_array_optimize: 0.523 → 0.517 (-1.0%)
- cykjson: 0.100 → 0.107 (**+6.7%**)
- eggcc-extraction: 0.454 → 0.442 (-2.8%)
- Overall average: +0.74%

**VERDICT: REGRESSION** — Reverted via `git reset --hard HEAD~1`. Mixed: 3 faster (hardboiled_128, python, eggcc) and 3 slower (hardboiled_32, luminal, cykjson). Cykjson took the biggest hit (+6.7%).

**Lesson:** `#[inline]` is not always free even though it's "purely a hint." Forcing inlining of these closure-taking trait methods at every call site bloats code, increases instruction-cache pressure, and hurts paths where the call already devirtualized cleanly via monomorphization. The compiler's default heuristic for these methods was apparently better than blanket inlining. **General rule: prefer per-callsite or per-method inlining decisions; blanket-inlining a hot trait is risky.** A more nuanced version would benchmark each `#[inline]` separately or use `#[inline(always)]` only on the body of the truly tightest loop.

## PP1: Cache subset sizes in BindingInfo (2026-05-09) — REVERTED

**Hypothesis (lessons-learned #7):** `estimate_size` walks Arc<TrieNode> → Subset → Pooled<SortedOffsetVector> → Vec::len for every scan in every `run_plan` recursion frame (~1.2% profile). Caching the size as a parallel `DenseIdMap<AtomId, u32>` next to `subsets` should skip this pointer chain on the hot reduction.

**Approach:** Added `subset_sizes: DenseIdMap<AtomId, u32>` to `BindingInfo`. Maintained sync at all 5 mutation sites: `insert_subset`, `insert_node`, `move_back`, `move_back_node`, `unwrap_val`. Added `BindingInfo::size(atom)` method with `debug_assert_eq!` against the underlying `subsets[atom].size()`. Redirected the two `estimate_size` call sites to `binding_info.size(atom)`.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs H2 baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.296 → 0.303 (+2.3%)
- hardboiled_conv1d_128: 0.768 → 0.781 (+1.7%)
- luminal-llama: 0.218 → 0.216 (-1.2%)
- python_array_optimize: 0.523 → 0.519 (-0.6%)
- cykjson: 0.104 → 0.109 (+4.5%)
- eggcc-extraction: 0.442 → 0.447 (+1.0%)
- Overall average: +1.26%

**VERDICT: REGRESSION** — Reverted via `git reset --hard HEAD~1`.

**Lesson:** The extra writes at the 5 mutation sites cost more than the saved reads at `estimate_size`. Mutation sites fire on every binding update (very hot in `run_plan`'s recursion); `estimate_size` fires once per `Intersect` stage entry plus periodically inside the loop. Mutations >> consumer reads, so write-side amortization fails. Also: the just-installed `Arc<TrieNode>` is hot in cache for the next `estimate_size` read, so the "deref chain" wasn't actually paying cache-miss costs in practice. The 1.2% profile attribution was overstated. **General rule: only cache derived values when reads ≫ writes AND the derivation cost is genuine (uncached).**

## H3: Direct-read SortedWritesTable in rebuild_incremental (2026-05-09) — REVERTED

**Hypothesis (H1/H2 generalization):** Apply the SortedWritesTable downcast pattern to the `table.scan_project(...)` call in `SortedWritesTable::rebuild_incremental` (`core-relations/src/table/rebuild.rs`). Note: an initial proposal mistakenly assumed `table` and `self` were the same SortedWritesTable; the implementer correctly refused that version. The corrected version downcasts the `table` argument (which is a *different* SWT — the upstream relation being scanned during rebuild).

**Approach:** Wrap the `table.scan_project(to_scan.as_ref(), &[search_col], Offset::new(0), usize::MAX, &[], &mut buf)` call in `if let Some(swt) = table.as_ref().inner_as_any().downcast_ref::<SortedWritesTable>()`. SWT branch: walk Dense range / Sparse `inner()` calling `swt.read_value_at_row_unchecked(row, search_col)` and pushing `(row, [val])` into `buf`. Fallback to original `scan_project` for non-SWT tables.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs H2 baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.295 → 0.297 (+0.8%)
- hardboiled_conv1d_128: 0.762 → 0.766 (+0.5%)
- luminal-llama: 0.218 → 0.213 (-1.9%)
- python_array_optimize: 0.522 → 0.520 (-0.3%)
- cykjson: 0.101 → 0.107 (**+6.3% regression**)
- eggcc-extraction: 0.441 → 0.442 (+0.2%)
- Overall average: +0.93%

**VERDICT: REGRESSION** — Reverted via `git reset --hard HEAD~1`. cykjson (which H2 just won big on) regressed by exactly the amount that H2 won. Hypothesis: rebuild_incremental in cykjson's hot path runs over many small `to_scan` subsets, where the inline Dense/Sparse loop has higher per-call overhead (longer inlined codegen, instruction cache pressure) than `scan_project`'s tight inner loop. luminal-llama wins because it has fewer-but-larger rebuild_incremental subsets.

**Lesson:** The H1/H2 pattern is NOT universally applicable. It pays off when the per-row dispatch chain dominates per-call setup (large subsets, many rows, full rebuild paths called once per epoch). It HURTS when the call site fires many times on small subsets (incremental rebuilds with frequent updates), because then `scan_project`'s amortized per-row cost is below the new inline loop's per-call setup. **For incremental/streaming hot paths, prefer the existing batched dispatch.**

## H2: Direct-read SortedWritesTable in TupleIndex::rebuild_full (2026-05-09) — KEPT

**Hypothesis (H1 generalization):** H1's SortedWritesTable downcast pattern won at the single-column `ColumnIndex::rebuild_full`. Apply the SAME pattern to the multi-column `TupleIndex::rebuild_full`, which currently uses the *default* `IndexBase::rebuild_full` (a `scan_project` → `TaggedRowBuffer` → `merge_rows` chain in 1024-row batches). TupleIndex backs every multi-key cached/HashIndex, including the multi-arg constructor joins prevalent in hardboiled (Bop, Cast, Select, Load, Ramp, Call) and cykjson.

**Approach:** In `core-relations/src/hash_index/mod.rs`, override `IndexBase::rebuild_full` for `TupleIndex`. SWT fast path: iterate Dense range or Sparse `inner()` rows; for each row, build the key tuple in a stack-local `Vec<Value>` scratch by calling `swt.read_value_at_row_unchecked(row, col)` per col; if any column reads `None`, set a `stale` flag and skip the whole row (matching `scan_generic` semantics); otherwise call `self.add_row(&scratch, row)` directly. Fallback else-branch is the verbatim default `scan_project` + `merge_rows` loop. Borrow-checker: the `&mut self` (for `add_row`) vs `swt` borrows would conflict in a closure, so the per-row body is inlined in both Dense and Sparse arms.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs H1 baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.296 → 0.295 (-0.5%, noise)
- hardboiled_conv1d_128: 0.768 → 0.764 (-0.4%, noise)
- luminal-llama: 0.217 → 0.217 (+0.3%, noise)
- python_array_optimize: 0.521 → 0.522 (+0.2%, noise)
- cykjson: 0.109 → 0.102 (**-6.3%**)
- eggcc-extraction: 0.442 → 0.439 (-0.8%)
- Overall average: −1.25%

**VERDICT: IMPROVEMENT** — Clean accept (no reviewer needed). cykjson ≥3% threshold cleared. No regressions. cykjson now firmly below pre-G3 levels (0.102 < 0.108).

**Commit:** c8ee3469 — KEPT.

**Lessons:**
- The H1 generalization paid off — TupleIndex's `add_row` per-key path still has the per-row dispatch chain that direct read eliminates, and cykjson's multi-arg constructor joins exercise this heavily.
- Hardboiled barely benefits because its TupleIndex-backed joins were less of its hot path than its single-column `ColumnIndex::rebuild_full` (which H1 already shaved).
- The pattern is now applied to all three known SortedWritesTable hot consumers: `SparseColumnIndex::new` (E3, E4), `ColumnIndex::rebuild_full` (H1), `TupleIndex::rebuild_full` (H2). Next candidates would be elsewhere (e.g. `merge_parallel`, `scan_project` consumers outside rebuild paths).

## H1: Direct-read SortedWritesTable in ColumnIndex::rebuild_full (2026-05-09) — KEPT

**Hypothesis:** The post-D3 9.4% "rebuild_full + closure + for_each_col" cluster is the same per-row dispatch chain (vtable → downcast → scan_generic → SubsetRef::offsets → RowId::range iter → dyn FnMut) that E3 (commit 16c83f12) and E4 (c173a8cf) eliminated for `SparseColumnIndex::new`. Applying the identical SortedWritesTable downcast + `read_value_at_row_unchecked` pattern to `rebuild_full`'s pair-collection loop should win at a hotter call site.

**Approach:** In `core-relations/src/hash_index/mod.rs::ColumnIndex::rebuild_full`, wrap the `for &col in cols { table.for_each_col(...) }` block in an `if let Some(swt) = table.inner_as_any().downcast_ref::<SortedWritesTable>()` fast path. On the SWT branch, iterate `SubsetRef::Dense` (range loop) or `SubsetRef::Sparse` (slice iter via `s.inner()`) and call `swt.read_value_at_row_unchecked(row, col)`, pushing `(val, row)` to `pairs` only when it returns `Some` (stale rows are silently skipped — same semantics as `for_each_col`/`scan_generic`). Fall back to `for_each_col` for non-SWT impls.

**Build/tests:** Clean release build, all `make test` suites pass.

**Results vs G3 baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.300 → 0.297 (-1.2%)
- hardboiled_conv1d_128: 0.787 → 0.771 (-2.0%)
- luminal-llama: 0.216 → 0.215 (-0.6%)
- python_array_optimize: 0.516 → 0.521 (+0.9%, noise)
- cykjson: 0.114 → 0.110 (**-3.5%**)
- eggcc-extraction: 0.440 → 0.441 (+0.2%, noise)
- Overall average: −1.03%

**VERDICT: IMPROVEMENT** — Clean accept (no reviewer needed). cykjson cleared ≥3% threshold. As a bonus, this nearly cancels the G3 cykjson regression (G3 baseline 0.114 → now 0.110, vs pre-G3 0.108).

**Commit:** 9226bd76 — KEPT.

**Lessons:**
- Mechanical replication of E3/E4 at a hotter call site paid off — pattern is now battle-tested.
- The downcast-bypass approach is generically applicable wherever `for_each_col` is called on a `SortedWritesTable`. Worth checking if there are other call sites that would benefit (e.g. `SubsetIndex::rebuild_full`, `scan_project` consumers).

## G4: Bypass FrameUpdates pool for cap < 32 (2026-05-09) — REVERTED

**Hypothesis (follow-up to G3):** cykjson's +5.9% under G3 is from the pool overhead (Rc::clone of self.pool, branch + push/pop) being charged on small-`cap` calls where the saved alloc is tiny. Skipping the pool path when `cap < 32` should restore cykjson without losing hardboiled.

**Approach:** Added a `SMALL_CAP_BYPASS = 32` early-return at the top of both `take_update_buf` and `return_update_buf` in `core-relations/src/free_join/execute.rs`. When `cap < 32` (take) or `v.capacity() < 32` (return), the pool is bypassed and we fall back to a fresh `Vec::with_capacity(cap)` / drop.

**Build/tests:** Clean, all `make test` suites pass.

**Results vs G3 baseline (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.301 → 0.313 (**+4.2%**)
- hardboiled_conv1d_128: 0.788 → 0.830 (**+5.4%**)
- luminal-llama: 0.213 → 0.214 (+0.4%, noise)
- python_array_optimize: 0.515 → 0.516 (+0.3%, noise)
- cykjson: 0.116 → 0.108 (**-6.7%**) — fully recovered
- eggcc-extraction: 0.445 → 0.458 (+2.9%)
- Overall average: +1.09%

**VERDICT: MIXED** — Reverted via `git reset --hard HEAD~1`. The bypass also kicks in on hardboiled's deep-recursion paths because `cap = cmp::min(chunk_size, cur_size)` is often small there too — the bypass essentially undoes G3 for hardboiled.

**Lesson:** "Small cap" and "shallow recursion" are not the same axis. Hardboiled's deep recursion produces many calls with small `cur_size` (refined-down subsets), so the pool actually does pay off there. To fix cykjson without hurting hardboiled, we'd need to discriminate by recursion depth or by whether previous take/returns observed a non-empty pool — neither is cleanly available in this code path. **G3 stays as-is; cykjson +5.9% is the price of the hardboiled +5.6%.**

## G3: Pool FrameUpdates backing Vec on JoinState (2026-05-09) — KEPT

**Hypothesis:** Repeated `Vec::with_capacity(cmp::min(chunk_size, cur_size))` allocation/deallocation for `FrameUpdates.updates` at the 6 `Intersect`/`UnboundCover`/`BoundCover`/`MaterializedIntersect` call sites in `run_plan` accounts for ~3–5% on hardboiled_128 (FrameUpdates::drain 3.3% + Drain drop 1.2% + Vec drop 1.2% + part of mi_heap_malloc 1.3%). Pooling backing Vecs across recursive `run_plan` invocations should let the drained-empty Vecs drop become a true no-op.

**Approach:** Per-`JoinState` `update_buf_pool: Vec<Vec<UpdateInstr>>` stack. New `FrameUpdates::from_pooled_vec(buf, cap)` (clears + reserves) and `into_inner()` return the backing Vec. JoinState helpers `take_update_buf(cap)` (pop or alloc fresh) and `return_update_buf(v)` (debug_assert empty, drop if cap > 4096 or pool depth ≥ 16). Required `&self` → `&mut self` on `run_plan` and `run_join_stages`, plus `let pool = self.pool.clone()` (cheap Rc::clone) to release the borrow on `self` for the take/return calls.

This applies the F2 lesson explicitly: scratch on `&mut self`, not TLS+RefCell. Parallel `JoinState`s spawned in `drain_updates_parallel` get a fresh empty pool.

**Build/tests:** Clean release build (one `dead_code` warning on the now-unused `with_capacity`), all `make test` suites pass.

**Results (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.314 → 0.303 (-3.7%)
- hardboiled_conv1d_128: 0.832 → 0.786 (**-5.6%**)
- luminal-llama: 0.214 → 0.215 (+0.3%, noise)
- python_array_optimize: 0.517 → 0.520 (+0.7%, noise)
- cykjson: 0.108 → 0.114 (**+5.9% regression**)
- eggcc-extraction: 0.461 → 0.445 (-3.5%)
- Overall average: −0.97%

**VERDICT: IMPROVEMENT** — Reviewer ACCEPTED. Hardboiled wins (primary benchmark) outweigh cykjson regression. cykjson is shallow-recursion: pool overhead (Rc::clone of self.pool, branch + push/pop) likely outweighs the saved alloc on small `cap`. Reviewer noted a follow-up could add a small-`cap` bypass to recover cykjson.

**Commit:** e5fbe944 — KEPT.

**Lessons:**
- Apply the F2 lesson — scratch on `&mut self` over TLS+RefCell — and the same hot-path saving pays off.
- One-bench regression of comparable magnitude to the primary-bench win is acceptable when the primary is the heavier workload.
- The `&self` → `&mut self` propagation cost was just one `Rc::clone` and a few `&pool` substitutions — not invasive enough to fail the simplicity test.

**Follow-up idea:** Skip pooling when `cap < 32` (use a fresh `Vec::with_capacity`). Should recover cykjson.

## F2: Pool pairs scratch Vec in ColumnIndex::rebuild_full (2026-05-09) — REVERTED

**Hypothesis:** `Vec::with_capacity(n)` at the start of every `ColumnIndex::rebuild_full` call (and its drop at the end) costs ~1–2% on hardboiled_128 via mi_heap alloc/free pairs. Replacing with a thread-local reused scratch Vec should remove that cost. Targets `_mi_heap_realloc_zero` (2.1%) + `mi_heap_malloc_aligned_at` (1.3%) in the post-D3 profile.

**Approach:** Wrap `rebuild_full`'s body in a `RADIX_PAIRS_SCRATCH.with(|cell| { ... })` closure backed by `thread_local! { static RADIX_PAIRS_SCRATCH: RefCell<Vec<(Value, RowId)>> }`. `pairs.clear(); pairs.reserve(n);` reuses prior capacity. Sort/dedup branches and grouping loop preserved exactly.

**Build/tests:** Clean build, all `make test` suites pass.

**Results (hyperfine, 15 runs):**
- hardboiled_conv1d_32: 0.308 → 0.310 (+0.6%)
- hardboiled_conv1d_128: 0.817 → 0.826 (+1.0%)
- luminal-llama: 0.213 → 0.219 (+2.7%)
- python_array_optimize: 0.515 → 0.523 (+1.5%)
- cykjson: 0.107 → 0.105 (-1.9%)
- eggcc-extraction: 0.462 → 0.461 (-0.3%)
- Overall average: +0.60%

**VERDICT: REGRESSION** — 4 benchmarks slower, no benchmark improved ≥3%. Reverted via `git reset --hard HEAD~1`. Reviewer step skipped because verdict is unambiguous.

**Lesson:** Thread-local indirection (`thread_local!.with()` + `RefCell::borrow_mut`) is not free. The savings from avoiding one `Vec::with_capacity` per call were dominated by the per-call TLS access + RefCell borrow tracking + closure overhead. A scratch buffer that lives across calls is also less cache-friendly than a fresh allocation when the next caller wanted a smaller size — the warm scratch may force the data through L2/L3 round trips. If we want to revisit this, candidate fixes: (a) put the scratch on `&mut self` (per-ColumnIndex) so no TLS, or (b) thread an explicit `&mut Vec` arg through the call chain. Both are more invasive and may not pay off — the F1 finding (rebuild_full alloc rarely shows up) suggests this hotspot is overstated by the profile.

## Bugfix: Stale-row correctness in SparseColumnIndex::new size==1 fast path (2026-05-09)

**Bug:** E1's fast path unconditionally returned `n_keys: 1` even when no value was found (stale row where `for_each_col` / `read_value_at_row_unchecked` produces nothing). `keys[0]` stayed at `Value::new_const(0)`, making stale rows appear as valid join matches. Tests `fusion` (panics in set-union primitive) and `fail_wrong_assertion` (deleted row still matched) were broken since E1.

**Root cause of silent failure:** `make test 2>&1 | tail -5` masks the exit code — `tail` always exits 0.

**Fix (commit 1e237463):** In the `None` arm: SortedWritesTable path returns `None` directly (no fallback); non-SortedWritesTable wraps `for_each_col` in `Option`. Either `None` → empty index (`n_keys: 0, n_subsets: 0`).

**Lesson:** Check test exit code explicitly; never pipe through `tail` without `${PIPESTATUS[0]}`.

## E4: Direct-read loop for SparseColumnIndex::new general path (2026-05-09)

**Hypothesis:** E3 applied the `SortedWritesTable` downcast trick only to the size==1 fast path. The 2..=8-row general path still uses `for_each_col` (vtable → downcast → scan_generic → iterator → dyn-FnMut per row). Applying the same direct-read pattern there should save the same per-row overhead.

**Approach:** In the general path, try `inner_as_any().downcast_ref::<SortedWritesTable>()` and loop directly over Dense/Sparse rows calling `read_value_at_row_unchecked`. Fallback to `for_each_col` for other Table impls.

**Results:**
- hardboiled_conv1d_32: 0.314s → 0.310s (-1.4%)
- hardboiled_conv1d_128: 0.830s → 0.818s (-1.5%)
- luminal-llama: unchanged (-0.1%)
- python_array_optimize: +0.5% (noise)
- cykjson: 0.110s → 0.103s (-6.3%)
- eggcc-extraction: 0.458s → 0.464s (+1.4%, likely noise)
- Overall average: -1.25%

**VERDICT: IMPROVEMENT** — Reviewer ACCEPTED. cykjson cleared ≥3% threshold. eggcc-extraction +1.4% within noise. Pattern mirrors E3 exactly — no new infrastructure.

**Commit:** c173a8cf — KEPT.

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
