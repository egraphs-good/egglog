# Greedy DAG Extractor Performance Experiments

## Setup

- Date: 2026-06-22
- Host: macOS 26.3.1 on Apple M4
- Branch: `codex/greedy-dag-extractor`
- Benchmark canaries:
  - Tree: `tests[extract-vec-bench]`
  - Greedy DAG: `tests[greedy-dag-vec-extract]`
- CodSpeed mode: `walltime`; simulation is unavailable on this host because the Valgrind executor is unsupported.
- Profiling: `codspeed samply record --save-only --presymbolicate` against the built `ci_benchmarking` binary.

## Experiment 0: baseline and hotspot shape

- Status: complete
- Smallest repro: copied `extract-vec-bench.egg` with only the final command changed to `:extractor greedy-dag`.
- Question: why is greedy-DAG extraction slower than tree extraction on the copied workload?
- Current hypothesis: greedy-DAG is dominated by `DagCostSet` construction and merging rather than backend table lookup.
- Confirming prediction: profiles should show time under `GreedyDagExtractor::compute_cost_hyperedge`, `merge_cost_set`, or `insert_cost`; forcing more indexed lookup should not improve the benchmark.
- Disconfirming prediction: profiles should instead show backend scans as the dominant self cost, and forcing indexed lookup should improve the benchmark.
- Exact commands:
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking extract-vec-bench`
  - `codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `codspeed samply record --save-only --presymbolicate -o .tmp/<profile>.json.gz -- target/release/deps/ci_benchmarking-* greedy-dag-vec-extract --sample-count 10`
- Observed result:
  - `extract-vec-bench`: best `63.95 ms`, report `6a3949aad11d3576c6021524`.
  - `greedy-dag-vec-extract`: best `165.43 ms`, report `6a3949bad11d3576c6021526`.
  - `cargo codspeed run` accepts a single benchmark filter here; the attempted two-filter run failed and was replaced with serial runs.
  - Running two `codspeed run` commands concurrently collided on CodSpeed local state (`EEXIST`), so future runs should be serial.
  - Current `samply` profile: 536 main-thread samples; extractor path 158 samples, `compute_cost_hyperedge` 136, `merge_cost_set` 132, `insert_cost` 76. Hash table reserve/rehash under merge accounted for 47 inclusive samples.
- Decision: baseline supports a cost-set construction hypothesis.

## Experiment 1: reserve `DagCostSet` capacity before merging

- Status: complete
- Smallest repro: `greedy-dag-vec-extract`.
- Current hypothesis: repeated growth/rehash of each newly built `DagCostSet` is a material part of greedy-DAG overhead.
- Confirming prediction: precomputing child set sizes and reserving one map capacity before merges should improve `greedy-dag-vec-extract` walltime.
- Disconfirming prediction: walltime should stay within noise or regress, meaning rehash samples are incidental or the extra pre-pass/temporary vector costs more.
- Exact commands:
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `cargo test --test files greedy_dag_vec_extract`
- Last attempted fix: `GreedyDagExtractor::merge_cost_set` calls `target.costs.reserve(source.costs.len())` before inserting source entries.
- Observed result:
  - First run: best `123.11 ms`, but high variance, report `6a394a680090450daf1fae66`.
  - Confirmation run: best `119.68 ms`, relative stddev `0.67%`, report `6a394a76d11d3576c6021546`.
  - Baseline was `165.43 ms`, so the stable confirmation run is about 28% faster.
- Decision: keep the change; the reserve/rehash hypothesis is confirmed.

## Experiment 2: next hotspot after reserved merges

- Status: complete
- Smallest repro: `greedy-dag-vec-extract`.
- Current hypothesis: after avoiding repeated rehashes, remaining overhead is either duplicate cost-set insertion work or expensive `(String, Value)` keys cloned during merges.
- Confirming prediction: a new profile should show lower reserve/rehash samples and remaining time under `insert_cost`, `HashMap::entry`, string clone/hash, or table traversal.
- Disconfirming prediction: the next profile should move most time outside the greedy-DAG extractor.
- Exact commands:
  - `codspeed samply record --save-only --presymbolicate -o .tmp/greedy-dag-reserve-profile.json.gz -- target/release/deps/ci_benchmarking-51041ae6619f6d69 greedy-dag-vec-extract --sample-count 20`
- Observed result:
  - Post-reserve profile: extractor path 115 samples, `compute_cost_hyperedge` 102, `merge_cost_set` 93, `insert_cost` 24.
  - Within the `merge_cost_set` subtree, 58 of 93 samples include `String::clone`; 52 include `Vec::clone`/`to_vec_in`; reserve/rehash under merge is down to 6 samples.
- Decision: test compact DAG cost-set keys that do not allocate sort-name strings during merge.

## Experiment 3: compact `(sort_id, value)` DAG cost-set keys

- Status: complete
- Smallest repro: `greedy-dag-vec-extract`.
- Current hypothesis: cloning `(String, Value)` keys dominates the remaining `merge_cost_set` cost.
- Confirming prediction: replacing `DagCostSet` keys with copyable `(usize, Value)` keys should improve walltime and reduce `String::clone` samples under `merge_cost_set`.
- Disconfirming prediction: walltime should stay within noise or regress, meaning the key representation was not important enough to justify extra sort-id plumbing.
- Exact commands:
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `codspeed samply record --save-only --presymbolicate -o .tmp/greedy-dag-compact-key-profile.json.gz -- target/release/deps/ci_benchmarking-51041ae6619f6d69 greedy-dag-vec-extract --sample-count 20`
  - `cargo test --test files greedy_dag_vec_extract`
- Last attempted fix: `DagCostSet` keys changed from `(String, Value)` to a copyable internal `DagCostKey { sort_id, value }`; `GreedyDagExtractor` builds a sort-name to sort-id map during `prepare`.
- Observed result:
  - CodSpeed walltime: best `51.09 ms`, relative stddev `0.68%`, report `6a394b10d11d3576c6021563`.
  - Previous accepted baseline was `119.68 ms`, so this is about 57% faster than Experiment 1 and about 69% faster than the original `165.43 ms` baseline.
  - Focused correctness canary passed: `cargo test --test files greedy_dag_vec_extract`.
  - Post-change profile: extractor path 44 samples, `compute_cost_hyperedge` 41, `merge_cost_set` 35, `insert_cost` 30. `String::clone` no longer appears under `merge_cost_set`.
- Decision: keep the compact key change; the key-clone hypothesis is confirmed.

## Experiment 4: revalidate the reserve call after compact keys

- Status: complete
- Smallest repro: `greedy-dag-vec-extract`.
- Current hypothesis: reserving before each merge is still useful after compact keys, but its benefit may be smaller now that key cloning is gone.
- Confirming prediction: removing the reserve call should regress `greedy-dag-vec-extract`.
- Disconfirming prediction: removing the reserve call should be neutral or faster, in which case the reserve call should be removed to simplify the code.
- Exact commands:
  - Temporarily remove `target.costs.reserve(source.costs.len())` from `merge_cost_set`.
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - Without the reserve call but keeping compact keys: best `89.89 ms`, relative stddev `0.79%`, report `6a394b97d11d3576c602156e`.
  - Compact-key baseline with reserve was `51.09 ms`.
- Decision: restore and keep the reserve call; it remains a validated performance improvement after compact keys.

## Greedy-DAG validation before tree-regression fix

- Status: superseded by Experiment 5 for tree comparison; still useful for greedy-DAG before/after numbers.
- CodSpeed walltime at this point:
  - `greedy-dag-vec-extract`: best `51.03 ms`, relative stddev `1.14%`, report `6a394bd2d11d3576c6021576`.
  - `extract-vec-bench`: best `63.92 ms`, relative stddev `1.56%`, report `6a394be1d11d3576c6021579`.
- Comparison at this point:
  - Greedy-DAG started at `165.43 ms` on this workload.
  - The final greedy-DAG implementation is about 69% faster than that starting point.
  - The tree number here was later found to be regressed by per-target indexed probes; see Experiment 5 for the corrected tree comparison.

## Experiment 5: tree extractor before/after PR comparison

- Status: complete
- Smallest repro: `extract-vec-bench`.
- Question: did the PR regress the default tree extractor on the copied workload?
- Before-PR source: detached worktree at fetched `upstream/main` commit `5294cdc66a7b90a9a1480cb2d930f2ee5785c8dd`, path `/private/tmp/egg-smol-tree-before-pr-codex`.
- Initial observation:
  - Upstream-main tree run: best `10.50 ms`, report `6a394dbad11d3576c6021599`.
  - PR tree run before this experiment: best `63.22 ms`, report `6a394d6f0090450daf1fae96`.
- Hypothesis: the PR regressed tree extraction by changing Bellman-Ford from full table scans to one indexed probe per reachable target value.
- Confirming prediction: profile should show PR time under `Extractor::bellman_ford` and `egglog_bridge::EGraph::for_each_matching_col`; restoring full table scans should bring the benchmark back near upstream-main.
- Disconfirming prediction: profile should show the time elsewhere, or full scans should not improve walltime.
- Probe:
  - Captured upstream-main and PR profiles with `codspeed samply record --save-only --presymbolicate`.
  - The PR profile showed `extract_best`/`prepare`/`bellman_ford` at 58/57/56 main-thread samples and `for_each_matching_col` at 57 samples.
  - The upstream-main profile showed tree extraction at only 4 main-thread samples.
- Fix:
  - Restored the tree extractor's Bellman-Ford and parent-edge pass to full `for_each` scans.
  - Removed the tree extractor's target-value field.
  - Removed tree-internal canonicalization in recursive cost/topo/reconstruction helpers to match upstream-main behavior; root canonicalization remains.
- Observed result:
  - After restoring full scans: best `11.56 ms`, report `6a394e8bd11d3576c60215b8`.
  - PR reruns after the fix: best `11.43 ms`, report `6a394ea8329541431b366b0c`; best `11.51 ms`, report `6a394edc329541431b366b17`.
  - Upstream-main reruns: best `10.82 ms`, report `6a394e9bd11d3576c60215bb`; best `10.61 ms`, report `6a394ee80090450daf1faebe`.
  - Removing tree-internal canonicalization: best `11.37 ms`, report `6a394f360090450daf1faed0`.
  - Post-fix profile no longer showed tree extraction as a hot path; extractor/backend samples were back to 4 inclusive samples, matching upstream-main's profile shape.
  - Greedy-DAG remained in the improved range after the tree fix: best `49.48 ms`, report `6a394f47d11d3576c60215c8`.
- Decision:
  - Keep the tree full-scan restoration. The large regression was real and fixed.
  - The remaining end-to-end gap is about 0.6-0.9 ms in local walltime runs; profiles no longer attribute it to tree extraction.

## Experiment 6: tree extractor ratio-of-means CI

- Status: complete
- Question: using the `../egglog_repro` percent-change algorithm, is the remaining tree extractor before/after difference plausibly zero?
- Algorithm source:
  - `../egglog_repro/scripts/compute_timing_percent_change.mjs`.
  - Uses ratio of execution-time means, sample variance with `n - 1`, Welch degrees of freedom, interpolated 95% t critical value, and a Fieller-style CI for the ratio of means.
  - Percent columns are `(ratio - 1) * 100`.
- Input samples:
  - One independent CodSpeed walltime run mean per row.
  - Interleaved 5 upstream-main runs and 5 PR runs.
  - CSV input was written temporarily to `.tmp/tree_extract_codspeed_means.csv` and then removed after recording the results here.
- Exact commands:
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking extract-vec-bench`
  - `node ../egglog_repro/scripts/compute_timing_mean_ci.mjs .tmp/tree_extract_codspeed_means.csv .tmp/tree_extract_mean_ci.csv`
  - `node ../egglog_repro/scripts/compute_timing_percent_change.mjs .tmp/tree_extract_codspeed_means.csv .tmp/tree_extract_percent_change.csv`
- Observed samples:
  - upstream-main means: `10.595`, `11.361`, `10.897`, `10.403`, `10.512` ms.
  - PR means: `11.440`, `11.118`, `11.445`, `11.391`, `11.304` ms.
- Mean and standard deviation:
  - upstream-main: mean `10.7536 ms`, sample standard deviation `0.3860 ms`.
  - PR: mean `11.3396 ms`, sample standard deviation `0.1362 ms`.
- Ratio-of-means result:
  - Ratio `1.0544933789614641`.
  - Percent change `+5.449337896146411%`.
  - Fieller 95% CI: ratio `1.01026243222885` to `1.1023334971249965`.
  - Percent CI: `+1.0262432228850038%` to `+10.23334971249965%`.
- Decision:
  - On this 5x interleaved CodSpeed walltime sample set, the remaining tree difference excludes zero and still indicates a small slowdown.

## Experiment 7: profile the remaining tree slowdown

- Status: complete
- Question: where does the remaining `extract-vec-bench` slowdown come from after restoring tree full-table scans?
- Alternative hypotheses:
  - H1: tree extraction is still slower because of new extractor setup or cost-model dispatch.
  - H2: the benchmark is slower outside extraction, for example parse/typecheck or benchmark harness setup.
  - H3: the observed percent-change is a walltime artifact not visible in CPU profiles.
- Confirming predictions:
  - H1: high-sample PR profiles should show extra inclusive/self samples under `egglog::extract`.
  - H2: high-sample PR profiles should show extra samples in non-extraction stacks.
  - H3: high-sample profiles should have no stable hot-path difference.
- Exact probes:
  - `codspeed samply record --save-only --presymbolicate -o /private/tmp/tree-old-high-profile.json.gz -- /private/tmp/egg-smol-tree-before-pr-codex/target/release/deps/ci_benchmarking-51041ae6619f6d69 extract-vec-bench --sample-count 200`
  - `codspeed samply record --save-only --presymbolicate -o /private/tmp/tree-pr-high-profile.json.gz -- /Users/saul/p/egg-smol/target/release/deps/ci_benchmarking-51041ae6619f6d69 extract-vec-bench --sample-count 200`
- Observed result:
  - High-sample profiles captured 385 upstream-main main-thread samples and 390 PR main-thread samples.
  - Extraction is not the source of the remaining gap: upstream-main had 3 `extract` category samples (0.78%), while the PR had 4 (1.03%).
  - The PR had slightly more samples outside extraction: typecheck/constraint went from 61.82% to 62.82%, parse/AST from 17.40% to 18.21%, and runtime-other from 5.19% to 5.90%.
  - The largest inclusive PR-minus-upstream deltas were in resolve/typecheck/constraint paths: `egglog::EGraph::resolve_program` +4.64 percentage points, `egglog::typechecking::TypeInfo::typecheck_standalone_actions` +3.86 percentage points, `egglog::typechecking::<impl egglog::EGraph>::typecheck_command` +3.39 percentage points, and `im_rc::hash::map::HashMap<K,V,S>::get` +5.81 percentage points.
- Decision:
  - Reject H1 as the primary cause. After the Experiment 5 tree full-scan restoration, no tree-extraction hot path remains in the profile.
  - H2 needs a narrower interpretation: this direct `samply` profile is dominated by benchmark discovery/setup, not by the measured benchmark body.
  - Do not add extractor performance complexity for this residual. The next useful probe would be a typechecking-specific profile/benchmark or an extractor-only harness, depending on whether we want to chase the end-to-end benchmark gap or isolate extractor cost.

## Experiment 8: explain the typecheck samples in Experiment 7

- Status: complete
- Question: why does the Experiment 7 profile show most samples in resolve/typecheck when the intended PR changes are extraction-focused?
- Hypothesis: the direct `samply` command profiles the whole `ci_benchmarking` process, including eager Divan benchmark argument construction, before the filtered `extract-vec-bench` body runs.
- Confirming prediction: most resolve/typecheck samples should be under `bench_cases_proof_testing` / `file_supports_proofs`, not under `run_example` / `parse_and_run_program` for the selected benchmark.
- Exact probe:
  - Re-aggregated the same `/private/tmp/tree-old-high-profile.json.gz` and `/private/tmp/tree-pr-high-profile.json.gz` profiles by stack ancestry.
- Observed result:
  - Upstream-main: `file_supports_proofs` accounted for 327 / 385 main-thread samples (84.94%); resolve/typecheck under `file_supports_proofs` accounted for 312 samples (81.04%).
  - PR: `file_supports_proofs` accounted for 340 / 390 main-thread samples (87.18%); resolve/typecheck under `file_supports_proofs` accounted for 329 samples (84.36%).
  - Resolve/typecheck under the selected benchmark's `run_example` body was tiny in both profiles: 4 samples upstream-main and 3 samples in the PR.
- Decision:
  - The Experiment 7 direct profile explains process-level sampling, not the CodSpeed benchmark body timing.
  - The typecheck/constraint samples are expected from `benches/common.rs`: `bench_cases("tests/**/*.egg")` eagerly calls `bench_cases_proof_testing`, and `file_supports_proofs` resolves programs to decide whether each file supports proofs. Adding `tests/greedy-dag-vec-extract.egg` adds another file to that setup path.
  - Do not treat these samples as evidence that extraction changes made typechecking slower. To explain a CodSpeed body-time delta, use CodSpeed's benchmark timing or add a narrower extractor-only benchmark.

## Experiment 9: direct CLI timing without Divan benchmark discovery

- Status: complete
- Question: does the default tree `extract-vec-bench.egg` slowdown reproduce when running a release-built egglog harness directly, without Divan benchmark discovery and proof-support setup?
- Hypothesis: if the slowdown is only from Divan argument discovery, direct `parse_and_run_program` timing should be near zero-change; if the default command path itself regressed, the direct harness should still show a slowdown.
- Harness:
  - Extended `../egglog_repro/scripts/run_benchmark_matrix.mjs` so a commit entry can use `{ "path": "/Users/saul/p/egg-smol" }` as a local path dependency.
  - The local path cache key includes the checkout HEAD, tracked diff, and untracked file contents.
  - Increased generated-harness timing precision from milliseconds to microseconds of seconds because this case is only about 15 ms.
- Exact config:
  - `../egglog_repro/.tmp/egg-smol-extract-vec-working-tree.json`
  - Baseline: `upstream/main` commit `5294cdc66a7b90a9a1480cb2d930f2ee5785c8dd`.
  - Candidate: local `/Users/saul/p/egg-smol` path dependency, fingerprint `e1ba95e10ae3`.
  - File: `/Users/saul/p/egg-smol/tests/extract-vec-bench.egg`.
  - Runs: 20; query decomposition on; concurrency off / 1 thread; timeout 30 s; no rendering.
- Exact command:
  - `node scripts/run_benchmark_matrix.mjs --config .tmp/egg-smol-extract-vec-working-tree.json --no-render`
- Outputs:
  - `../egglog_repro/.tmp/egg-smol-extract-vec-working-tree_raw.csv`
  - `../egglog_repro/.tmp/egg-smol-extract-vec-working-tree_mean_ci.csv`
  - `../egglog_repro/.tmp/egg-smol-extract-vec-working-tree_percent_change.csv`
- Observed result:
  - Upstream-main samples: mean `14.0957 ms`, sample standard deviation `0.6296 ms`, min `13.499 ms`, max `15.578 ms`.
  - Working-tree samples: mean `15.42795 ms`, sample standard deviation `0.6439 ms`, min `14.681 ms`, max `17.076 ms`.
  - Ratio-of-means result: `+9.451463921621507%`.
  - Fieller 95% CI: `+6.56033608913591%` to `+12.42649856133109%`.
  - Tuple counts matched: both variants produced `1248` tuples in every run.
- Order-effect check:
  - Ran a second matrix with the commit order reversed and `top_n = 1` only to force a fresh timing cache key; the generated harness ignores `top_n`, so the measured work is unchanged.
  - Config: `../egglog_repro/.tmp/egg-smol-extract-vec-working-tree-reversed.json`.
  - Output: `../egglog_repro/.tmp/egg-smol-extract-vec-working-tree-reversed_percent_change.csv`.
  - Working-tree-first samples: working tree mean `15.9057 ms`, upstream-main mean `14.20295 ms`.
  - In that reversed baseline view, upstream-main is `-10.705281754339623%` relative to the working tree, with Fieller 95% CI `-17.444700174312587%` to `-2.9374176527587226%`.
  - This reversed run had one working-tree outlier (`26.741 ms`), but it still supports the same direction: upstream-main is faster than the working tree on this direct CLI harness.
- Decision:
  - Direct CLI timing reproduces a tree-default slowdown outside Divan benchmark discovery. The previous `samply` profile was dominated by discovery setup, but that does not explain this direct harness result.
  - The next useful probe is a direct profile of the generated upstream and working-tree harness binaries, or an extractor-only benchmark if we want to separate parse/typecheck/run overhead from extraction overhead.

## Experiment 10: profile and fix direct tree slowdown

- Status: complete
- Question: what slowed down in the direct `extract-vec-bench.egg` CLI path, and can a targeted change remove it?
- Alternative hypotheses:
  - H1: default tree extraction now does extra setup work before Bellman-Ford.
  - H2: parse/typecheck/run outside extraction is slower.
  - H3: the single-run direct slowdown is a one-shot process or code-layout artifact that disappears when the same process repeats the workload.
- Probe 1: repeated direct timing.
  - Added `inner_repeats` support to `../egglog_repro/scripts/run_benchmark_matrix.mjs`; generated harnesses loop over a fresh `EGraph::default()` and `parse_and_run_program` for the same input, then report per-iteration time.
  - Config: `../egglog_repro/.tmp/egg-smol-extract-vec-inner-repeats.json`.
  - Command: `node scripts/run_benchmark_matrix.mjs --config .tmp/egg-smol-extract-vec-inner-repeats.json --no-render`.
  - Runs: 5 process-level samples per variant, `inner_repeats = 100`, serial.
  - Before fix: upstream-main mean `15.4892 ms`; working tree mean `19.501 ms`.
  - Before-fix percent change: `+25.900627534023712%`, Fieller 95% CI `+10.377495720796471%` to `+41.77089432208214%`.
- Probe 2: direct repeated profiles.
  - Upstream profile: `/private/tmp/egg-smol-repeat-upstream-profile.json.gz`.
  - Working-tree before-fix profile: `/private/tmp/egg-smol-repeat-working-profile.json.gz`.
  - Exact profile shape: `codspeed samply record --save-only --presymbolicate ... bench /Users/saul/p/egg-smol/tests/extract-vec-bench.egg 0 1 300`.
  - Upstream profile had 4042 main-thread samples; working-tree profile had 4381.
  - The extra samples were in default tree extraction setup: before-fix working tree had `collect_reachable_extraction_nodes` at 276 samples (6.30%), `ReachableExtractionBuilder::discover_node` at 272 samples (6.21%), and `egglog_bridge::EGraph::for_each_matching_col` at 167 samples (3.81%). Upstream had no value-level producer walk.
  - Upstream's tree setup was the old sort/function setup path, `Extractor<C>::compute_costs_from_rootsorts` at 838 samples (20.73%); before-fix working tree used `Extractor<C>::prepare` at 1104 samples (25.20%), which included the new value-level reachability pass.
- Fix:
  - Keep greedy-DAG's value-level reachability, because it needs reachable values for DAG cost sets.
  - Change the default tree extractor setup to collect extractable functions by output sort, initialize output cost maps, and let Bellman-Ford scan constructor tables as before.
  - Added a comment in `src/extract.rs` explaining that profiles showed greedy-DAG's value-level producer walk regressed default tree extraction.
- Observed result after fix:
  - Repeated direct timing: upstream-main mean `15.4892 ms`; working tree mean `15.4856 ms`.
  - Repeated direct percent change: `-0.023242000878032076%`, Fieller 95% CI `-3.755251061039011%` to `+3.936191713576398%`.
  - Original single-iteration direct matrix also recovered: upstream-main mean `14.41925 ms`; working tree mean `14.37045 ms`; percent change `-0.338436465142089%`, Fieller 95% CI `-2.061673809714537%` to `+1.4165313296456983%`.
  - Fixed working-tree profile: `/private/tmp/egg-smol-repeat-working-fixed-profile.json.gz`; stack shape no longer contains `collect_reachable_extraction_nodes`, `ReachableExtractionBuilder::discover_node`, or `for_each_matching_col` under default tree setup. The profiled fixed run itself was noisy, so use it only for stack-shape confirmation.
- Validation:
  - `cargo fmt --check`
  - `cargo test --test files extract_vec_bench`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test files proof_support_snapshot`
  - In `../egglog_repro`: `node --check scripts/run_benchmark_matrix.mjs`, `cargo fmt --check`, `cargo check --quiet --features latest_main`.
- Decision:
  - H1 is confirmed. The slowdown came from using greedy-DAG's value-level backwards producer traversal in the default tree extractor setup.
  - Keep the tree setup fix. It removes the measured slowdown while leaving greedy-DAG's value-reachable setup intact.

## Experiment 11: avoid `Arc` allocation for rejected hyperedge candidates

- Status: rejected
- Question: after the cleanup pass, is greedy-DAG spending enough time allocating candidate `Arc<DagCostSet<_>>` values that returning an owned candidate and wrapping only accepted updates is worthwhile?
- Hypothesis: if transient candidate `Arc` allocation is material, changing `compute_cost_hyperedge` to return `DagCostSet<C>` should improve `greedy-dag-vec-extract`.
- Exact commands:
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - Owned-candidate run: best `50.17 ms`, report `6a395e3a208a6fcce26cb48c`.
  - Temporary old-`Arc` A/B run: best `50.49 ms`, report `6a395e610090450daf1fb08a`.
- Decision:
  - The difference is within local walltime noise. Revert the owned-candidate change for now and keep looking at the larger cost-set union path.

## Experiment 12: stage child sets and allocate candidate entries once

- Status: complete
- Question: can we reduce the remaining `HashMap` rehash/copy overhead in greedy-DAG cost-set merging?
- Hypothesis: staging child cost sets, computing the combined entry count, and allocating the candidate map once should reduce `reserve_rehash` and improve `greedy-dag-vec-extract`.
- Exact commands:
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `codspeed samply record --save-only --presymbolicate -o /private/tmp/egg-smol-greedy-dag-reserve-once-profile.json.gz -- ../egglog_repro/.egglog_repro_cache/builds/799279f743b0c7e4/target/release/bench /Users/saul/p/egg-smol/tests/greedy-dag-vec-extract.egg 0 1 100`
- Observed result:
  - Staged allocation run: best `44.66 ms`, report `6a395ed0fa4fbf27dcacfa8b`.
  - Confirmation run was noisy but stayed in the improved best-time range: best `45.93 ms`, report `6a395edb0090450daf1fb0af`.
  - Previous old-`Arc` A/B baseline was best `50.49 ms`, so this is about 9-12% faster by best time.
  - Profile check: `reserve_rehash` dropped from `11.38%` inclusive to `0.86%`; `resize_inner` dropped from `11.10%` to `0.84%`.
  - Removing the now-redundant inner `merge_cost_set` reserve produced best `45.64 ms`, report `6a395f47fa4fbf27dcacfa95`; this was neutral/noisy, but the reserve call was no longer needed because both merge call sites pre-allocate the final candidate capacity.
- Decision:
  - Keep staged child collection and one-shot candidate allocation. Keep the inner reserve removed because the new pre-allocation invariant makes it redundant.

## Experiment 13: dense cost-key ids plus bitset-backed cost sets

- Status: complete
- Question: can the greedy-DAG extractor get substantially closer to tree extraction by replacing `HashMap`-backed set union in `DagCostSet`?
- Algorithmic model:
  - The current greedy-DAG algorithm still has to account for sharing, so candidate scoring is roughly `O(candidate_edges * selected_dag_size)`.
  - Tree extraction scores each candidate with scalar child costs, roughly `O(candidate_edges * arity)`.
  - Without requiring subtraction on arbitrary cost types, we still need to visit newly selected entries when unioning child DAGs. The practical target is therefore to make that union much cheaper, not to make it scalar like tree extraction.
- Hypothesis: assigning dense ids to reachable `(sort, value)` cost keys and storing each `DagCostSet` as `Vec<(key_id, cost)> + FixedBitSet` should remove most hash-table probing/allocation from the hot union loop while preserving no-subtraction semantics.
- Exact commands:
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking extract-vec-bench`
  - `codspeed samply record --save-only --presymbolicate -o /private/tmp/egg-smol-greedy-dag-bitset-profile.json.gz -- ../egglog_repro/.egglog_repro_cache/builds/799279f743b0c7e4/target/release/bench /Users/saul/p/egg-smol/tests/greedy-dag-vec-extract.egg 0 1 150`
- Observed result:
  - First bitset run: best `20.17 ms`, relative stddev `2.81%`, report `6a396006836494ceadaff62a`.
  - Confirmation run: best `19.97 ms`, relative stddev `2.03%`, report `6a396013c93a1577f817bfb1`.
  - Previous staged-allocation best range was `44.66-45.93 ms`, so the bitset representation is about 55% faster than that accepted baseline.
  - Tree canary stayed in the expected range: `extract-vec-bench` best `10.37 ms`, report `6a39601d836494ceadaff62e`.
  - Profile check: `HashMap` inclusive samples in the greedy-DAG profile dropped from `38.34%` to `12.77%`; `RawTable` from `57.41%` to `9.07%`; `copy_nonoverlapping` from `18.05%` to `3.84%`; `merge_cost_set` from `54.66%` to `20.45%`.
- Decision:
  - Keep the dense-id + bitset representation. This is the first optimization that changes the constant factor enough to make greedy-DAG much closer to tree on the copied workload, while preserving arbitrary cost types and avoiding subtraction.

## Experiment 14: retry owned candidates after bitset representation

- Status: rejected
- Question: after the bitset change shifts the profile toward allocation/refcounting, is owned candidate return now worthwhile?
- Hypothesis: returning `DagCostSet<C>` from `compute_cost_hyperedge` and wrapping only accepted candidates should improve the bitset implementation.
- Exact commands:
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - First run: best `19.63 ms`, report `6a3960b2836494ceadaff63c`.
  - Rerun: best `22.71 ms`, report `6a3960bc836494ceadaff643`.
  - Third run: best `20.34 ms`, report `6a3960ca836494ceadaff64b`.
- Decision:
  - The result is noisy and does not clearly beat the bitset baseline (`19.97 ms`). Revert the owned-candidate change and keep the validated bitset representation only.

## Final validation after greedy-DAG optimization pass

- Status: complete
- Final code keeps:
  - staged child collection and one-shot candidate entry allocation;
  - dense reachable cost-key ids;
  - `Vec<(key_id, cost)> + FixedBitSet` cost-set representation;
  - old `Arc<DagCostSet<_>>` return from `compute_cost_hyperedge`, because owned candidates were not validated.
- Exact commands:
  - `cargo fmt --check`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test files extract_vec_bench`
  - `cargo test --test files proof_support_snapshot`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking extract-vec-bench`
  - `git diff --check`
- Observed result:
  - Focused tests passed.
  - Final greedy-DAG CodSpeed run: best `20.79 ms`, relative stddev `2.03%`, report `6a396143c93a1577f817bfed`.
  - Final tree canary: best `11.07 ms`, relative stddev `8.14%`, report `6a39614dc93a1577f817bff0`.
  - `git diff --check` passed.

## Experiment 15: clone largest child cost set

- Status: complete
- Question: can we port the behavior-preserving set-union tricks from <https://github.com/egraphs-good/extraction-gym/blob/903ba0f818b50608fe20ae9e0f03c35cb27bc50a/src/extract/faster_greedy_dag.rs> to the bitset-backed greedy-DAG extractor?
- Hypothesis:
  - Starting each candidate from the largest child `DagCostSet` should avoid reinserting that largest set entry-by-entry.
  - This should preserve the current greedy candidate cost; any output-quality change would be a bug.
- Exact commands:
  - `cargo fmt --check`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - Clone-largest-only run: best `18.41 ms`, relative stddev `1.58%`, report `6a397d684edabacae4cbe290`.
  - Previous final-code greedy-DAG canary was best `20.79 ms`, so this is about 11% faster by best time.
- Decision:
  - Keep the change. It is behavior-preserving and directly reduces the remaining per-candidate merge work.

## Experiment 16: skip child cost set when its root is already shared

- Status: rejected for now
- Question: can we adapt `global_greedy_dag`'s early shared-subterm stop to the current bitset-backed cost sets?
- Hypothesis: if a candidate already contains a child root key, the child's selected DAG is already paid and the child merge can be skipped.
- Observed result:
  - A first implementation incorrectly used `entries[0]` as the child root key. That is not a valid invariant after cloning/merging because the root key is appended when a cost set is built.
  - Even with an explicit root-key field, this optimization can change quality when two paths reach the same eclass with different selected sub-DAGs. The current implementation unions entries from both child selections; skipping would pick the first reachable selection instead.
- Decision:
  - Do not keep this as a behavior-preserving optimization. Revisit only with explicit quality/oracle checks and a representation that records the root key of each `DagCostSet`.

## Experiment 17: worklist parent propagation with full-scan safety pass

- Status: complete
- Question: can we adapt `faster_greedy_dag`'s parent worklist without losing correctness for egglog rows and containers?
- Hypothesis:
  - Build `eq child sort -> (function, child column)` indexes.
  - Seed with one full scan, then when a `(sort, value)` improves, only probe rows whose direct eq child column matches that value.
  - After each worklist drain, run a full safety scan to preserve fixed-point completeness for dependencies hidden inside container values or any missed indexing case.
- Quality/correctness expectation:
  - This should preserve the same fixed-point semantics as the previous repeated full scan because the safety scan repeats until no update occurs.
  - The direct parent worklist only changes how quickly obvious dependent rows are revisited.
- Exact commands:
  - `cargo fmt`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `codspeed samply record --save-only --presymbolicate -o /private/tmp/egg-smol-greedy-dag-worklist-profile.json.gz -- ../egglog_repro/.egglog_repro_cache/builds/799279f743b0c7e4/target/release/bench /Users/saul/p/egg-smol/tests/greedy-dag-vec-extract.egg 0 1 200`
- Observed result:
  - First run: best `14.16 ms`, relative stddev `0.97%`, report `6a397e0b4edabacae4cbe29a`.
  - Confirmation run: best `14.18 ms`, report `6a397e174edabacae4cbe29d`.
  - Clone-largest baseline was best `18.41 ms`, so this is about 23% faster by best time.
  - Profile check: `egglog_bridge::EGraph::for_each` inclusive dropped from `43.91%` in the bitset profile to `15.86%`; `for_each_matching_col` rose from `9.63%` to `18.66%`, as expected for parent probes.
- Decision:
  - Keep the worklist implementation with the safety scan. It preserves fixed-point coverage while cutting repeated full-table scans on the benchmark workload.

## Experiment 18: stack-backed child vectors

- Status: complete
- Question: after the worklist change, does avoiding per-candidate heap allocation for low-arity child vectors help?
- Hypothesis:
  - Most extraction candidate rows have small arity, so using `SmallVec<[...; 4]>` for `child_cost_sets` and `child_costs` should reduce allocation overhead without changing behavior.
- Exact commands:
  - `cargo fmt --check`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - First run: best `13.88 ms`, relative stddev `2.32%`, report `6a397e98f49bf12cd3aa60a9`.
  - Confirmation run: best `13.62 ms`, relative stddev `4.00%`, report `6a397ea4f49bf12cd3aa60aa`.
  - Worklist baseline was best `14.16-14.18 ms`, so this is a small 2-4% best-time improvement.
- Decision:
  - Keep the change. `smallvec` was already a workspace dependency; this adds it to the `egglog` crate and removes a small hot-loop allocation source.

## Experiment 19: materialize only winning candidate cost sets

- Status: rejected
- Question: after the worklist and `SmallVec` changes, does avoiding full `DagCostSet` materialization for losing candidates help enough to justify scratch state in the extractor?
- Hypothesis:
  - `relax_hyperedge` often computes candidate cost sets that do not improve the target. Reusing an epoch-marked scratch vector to compute the candidate total first should avoid allocating `FixedBitSet`/entry storage for losing candidates.
- Exact commands:
  - `cargo fmt`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - First run: best `13.66 ms`, relative stddev `2.08%`, report `6a398d7ade74dd3d7e15160d`.
  - Confirmation run: best `13.57 ms`, relative stddev `1.98%`, report `6a398d87de74dd3d7e151610`.
  - The accepted Experiment 18 baseline was best `13.62 ms`, so this is effectively tied and below the threshold for keeping extra mutable scratch state plus a second hyperedge-cost path.
- Decision:
  - Reject and remove the scratch implementation. The extra complexity did not produce a clear walltime win.

## Experiment 20: exact producer dependencies through containers

- Status: complete
- Question: can we remove the full safety scan from the greedy-DAG fixed point by using exact reachable producer dependencies, including eq values nested inside containers?
- Hypothesis:
  - Root discovery already visits each reachable producer row. If it stores those rows and indexes each producer by every eq dependency in its child values, then an improved `(sort, value)` can enqueue exactly the producers that might change.
  - This should preserve the same candidate set as the safety-scan implementation while avoiding repeated backend table scans and matching-column probes in the fixed-point loop.
- Quality/correctness expectation:
  - This is intended as a scheduling/data-layout optimization only. The selected node and DAG cost should remain governed by the same `compute_cost_hyperedge` path.
  - Dependencies are deduplicated per producer so repeated use of the same eq value in one row does not enqueue redundant relaxations.
- Exact commands:
  - `cargo fmt --check`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - Intermediate implementation: best `9.09 ms`, relative stddev `1.80%`, report `6a398ec7de74dd3d7e15162b`.
  - Intermediate confirmation: best `9.15 ms`, relative stddev `2.46%`, report `6a398ed071bb38ed292405e9`.
  - Cleaned final implementation: best `9.44 ms`, relative stddev `1.46%`, report `6a398f1f71bb38ed292405ef`.
  - Final confirmation: best `9.29 ms`, relative stddev `1.81%`, report `6a398f2971bb38ed292405f0`.
  - Experiment 18 baseline was best `13.62 ms`, so the final implementation is about 31-32% faster by best time.
- Decision:
  - Keep the exact producer-dependency implementation and delete the obsolete backend-scan safety loop. The extra producer table is justified by the measured walltime drop and has an in-code comment documenting that rationale.

## Experiment 21: dense producer worklist keys

- Status: rejected
- Question: after exact producer propagation, does using dense `DagCostKey` ids in the pending queue and dependency map improve the worklist loop?
- Hypothesis:
  - The producer worklist no longer needs owned `(String, Value)` keys once `key_ids` exist. Replacing queue keys with dense `usize` ids should reduce hashing/cloning in propagation.
- Exact commands:
  - `cargo fmt`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - First run: best `9.40 ms`, relative stddev `3.26%`, report `6a398fb271bb38ed292405f9`.
  - Confirmation run: best `9.38 ms`, relative stddev `2.28%`, report `6a398fbe71bb38ed292405fa`.
  - This is tied with or slightly slower than the simpler exact-producer implementation (`9.29-9.44 ms` final-code runs).
- Decision:
  - Reject and revert. Dense queue keys did not produce a clear walltime win, and the conversion made the prepare/relax path more complex.

## Research idea disposition

- Exact producer-dependency worklist through containers: tried in Experiment 20 and kept. This was the largest remaining win.
- Materialize cost sets only for winning candidates: tried in Experiment 19 and rejected. The measured change was below noise while adding mutable scratch state and a second candidate-cost path.
- Lower-bound rejection: not implemented in this generic extractor. It is only safe under nonnegative monotone cost assumptions, but the current `DagCostModel<C>` remains generic over signed/custom costs.
- Deterministic complete relaxation: kept as a design constraint for Experiment 20. The producer table changes scheduling and storage, while still evaluating the same reachable producer rows with the same `compute_cost_hyperedge` logic.
- Tiny exact oracle / ILP quality checks: deferred. Useful for a future quality-focused extractor suite, but not necessary to validate these behavior-preserving scheduling/data-structure changes.
- Quality-changing boosted/pruned extractors: deferred to separate extractor modes. They should not be hidden behind `:extractor greedy-dag` because quality tradeoffs need explicit user control.

## Citation map for implemented ideas

- Greedy DAG paid-set scoring and counting shared subterms once.
  - Direct implementation source: <https://github.com/egraphs-good/extraction-gym/blob/903ba0f818b50608fe20ae9e0f03c35cb27bc50a/src/extract/faster_greedy_dag.rs>, especially `CostSet` and `calculate_cost_set`; the simpler baseline is <https://github.com/egraphs-good/extraction-gym/blob/903ba0f818b50608fe20ae9e0f03c35cb27bc50a/src/extract/greedy_dag.rs>.
  - Background only: e-boost frames DAG extraction as a heuristic-vs-exact tradeoff and keeps exact optimal solving separate from fast heuristics: <https://arxiv.org/abs/2508.13020>.
  - Current PR adaptation: egglog keys paid dependencies by `(sort, value)`, not by extraction-gym `ClassId`, because egglog values are sort-local and primitive/container values can also contribute cost.

- Clone-largest child cost-set union.
  - Direct implementation source: <https://github.com/egraphs-good/extraction-gym/blob/903ba0f818b50608fe20ae9e0f03c35cb27bc50a/src/extract/faster_greedy_dag.rs#L57-L62> clones the largest child set before inserting smaller sets.
  - Local validation: Experiment 15 confirmed this still improved the bitset-backed egglog implementation.

- Worklist propagation from improved children to producers.
  - Direct implementation source: <https://github.com/egraphs-good/extraction-gym/blob/903ba0f818b50608fe20ae9e0f03c35cb27bc50a/src/extract/faster_greedy_dag.rs#L91-L135> builds parent lists and re-enqueues parents after a class improves.
  - Local validation: Experiment 17 tested the first egglog worklist version; Experiment 20 replaced the safety-scan version with exact producer dependencies.

- Root-aware reachable producer slice.
  - Motivation: extraction-gym issue #49 asks about DAG extractors ignoring the root argument and extracting all e-classes instead: <https://github.com/egraphs-good/extraction-gym/issues/49>.
  - Current PR adaptation: root discovery starts from requested `(sort, value)` roots and stores only reachable producer rows.

- Exact dependency index including eq values nested in containers.
  - Source: egglog-specific design, not from extraction-gym or the papers.
  - Local validation: Experiment 20 showed exact producer dependencies removed the full safety scan and improved the benchmark.

- Dense `(sort, value)` cost-key ids and `FixedBitSet` membership.
  - Source: local profiling and data-structure experiment, not an external paper.
  - Local validation: Experiment 13 replaced hash-table-backed cost sets with dense ids plus `FixedBitSet`; Experiment 23 rejected linear scans after a large regression.

- Staged child collection and one-shot candidate allocation.
  - Source: local profiling and allocation experiment.
  - Local validation: Experiment 12 showed precomputing final candidate entry capacity reduced resize/rehash cost.

- Avoiding subtraction and preserving arbitrary `Cost` implementations.
  - Source: egglog API design constraint. The extractor maintains cached totals and unions newly seen entries, rather than deriving marginal deltas by subtracting shared child totals.
  - Local rationale: Experiment 13 records the complexity target: DAG scoring still visits selected entries, but the representation makes that union cheaper without assuming numeric subtraction.

- Keeping boosted, pruned, and quality-changing heuristics out of `:extractor greedy-dag`.
  - Background: e-boost combines heuristic extraction, adaptive pruning, and initialized exact solving as separate techniques: <https://arxiv.org/abs/2508.13020>.
  - Cautionary evidence: extraction-gym issue #19 reports `faster-greedy-dag` having a worse cumulative DAG cost than `greedy-dag`, and issue #28 reports correctness problems in `global-greedy-dag`: <https://github.com/egraphs-good/extraction-gym/issues/19>, <https://github.com/egraphs-good/extraction-gym/issues/28>.
  - Current decision: quality-changing boosted/pruned ideas should be separate extractor modes, not hidden inside `:extractor greedy-dag`.

- Debug fixed-point assertion over discovered producers.
  - Source: local research-review follow-up after Experiment 20.
  - Scope: it catches missed reachable producer updates after the worklist drains; it does not prove global greedy quality or equivalence to an exact solver.

- Exact extraction papers and ILP/treewidth approaches.
  - Background only for this PR: `E-Graphs as Circuits, and Optimal Extraction via Treewidth` describes an exact parameterized approach, but no implemented code path here uses that algorithm: <https://arxiv.org/abs/2408.17042>.
  - Future use: these sources are better fits for oracle tests or separate exact extractor work than for comments on the current greedy implementation.

## Final validation after research-idea pass

- Status: complete
- Exact commands:
  - `cargo fmt --check`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test files extract_vec_bench`
  - `cargo test --test files proof_support_snapshot`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking extract-vec-bench`
  - `git diff --check`
- Observed result:
  - Focused tests passed.
  - Final serial greedy-DAG canary: best `9.37 ms`, relative stddev `1.94%`, report `6a39904bde74dd3d7e15164f`.
  - Final serial tree canary: best `9.98 ms`, relative stddev `1.35%`, report `6a39905571bb38ed2924061e`.
  - `git diff --check` passed.
  - The earlier parallel CodSpeed pair was discarded for decision-making because concurrent walltime runs can interfere with each other.

## Research review follow-up

- Status: complete
- Reviewer: Peirce research agent (`019ef08f-8aed-7360-976d-9687d16c61a8`)
- Findings:
  - No obvious missed dependency bug in the exact producer table: direct eq children and eq values nested in containers are collected with canonical keys, and candidates still go through `compute_cost_hyperedge`.
  - Main remaining risk is semantic equivalence to the old full-scan/safety-scan schedule; a future generated differential/oracle test would be useful.
  - Producer storage is a memory/performance scaling tradeoff on very large reachable egraphs, not a correctness issue.
- Follow-up change:
  - Added a debug-only fixed-point assertion over the discovered producer table. This catches missed-dependency bugs where the worklist drains but a reachable producer can still improve. It has no release/CodSpeed cost.
- Exact commands:
  - `cargo fmt --check`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test files extract_vec_bench`
  - `cargo test --test files proof_support_snapshot`
  - `git diff --check`
- Observed result:
  - All commands passed.

## Diff-reduction pass

- Status: in progress
- Goal: reduce `src/extract.rs` review surface and remove performance-only abstractions unless CodSpeed still justifies them.
- Cleanup changes kept so far:
  - Inlined single-use greedy-DAG helpers (`empty_cost_set`, `merge_cost_set`) and the debug fixed-point assertion.
  - Removed the unused `function_to_dag` root/row prepass and restored streaming row extraction.
  - Removed `GreedyDagProducer::target`, deriving the target from the producer row output column.
  - Removed `GreedyDagExtractor::funcs` and `ReachableExtractionBuilder::{funcs_set, funcs}` by ranking greedy variants from the already-discovered producer rows.
  - Built greedy-DAG `sort_ids` from reachable cost keys only.
  - Made `merge_child_cost_sets` concrete over `Arc<DagCostSet<C>>` children.
  - Added reviewer-facing comments for extracted root ordering, reachable producer indexing, and the tree extractor topological-rank invariant.
- Exact commands after the structural cleanup:
  - `cargo fmt --check`
  - `cargo test --test integration_test greedy_dag`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - Focused tests passed.
  - Cleaned-code greedy-DAG baseline: best `9.31 ms`, relative stddev `3.25%`, report `6a39c569b3ef5b3641e39d91`.

## Experiment 22: replace SmallVec child buffers with Vec

- Status: complete
- Question: does the `SmallVec` dependency still pay for itself after exact producer propagation and cleanup?
- Hypothesis:
  - `SmallVec` was previously a small 2-4% win, but after the larger worklist change it may be in the noise. Plain `Vec` would remove a performance-only dependency from the `egglog` crate.
- Exact commands:
  - `cargo fmt --check`
  - `cargo test --test integration_test greedy_dag`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test files extract_vec_bench`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - First run: best `9.42 ms`, relative stddev `2.93%`, report `6a39c5bab3ef5b3641e39d9b`.
  - Confirmation run: best `9.23 ms`, relative stddev `1.74%`, report `6a39c5c5b072bb35f7798247`.
  - Cleaned-code `SmallVec` baseline was best `9.31 ms`, so plain `Vec` is performance-neutral on this workload.
- Decision:
  - Keep the plain `Vec` version and remove the `egglog` crate's direct `smallvec` dependency.

## Experiment 23: replace FixedBitSet membership with linear entry scans

- Status: rejected
- Question: can `DagCostSet` drop its `FixedBitSet` membership cache and use linear scans over compact `entries` instead?
- Hypothesis:
  - If selected cost sets stay small after exact producer propagation, linear membership could remove the `fixedbitset` dependency and simplify `DagCostSet`.
- Exact commands:
  - `cargo fmt`
  - `cargo test --test integration_test greedy_dag`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - Vec-only membership run: best `15.72 ms`, relative stddev `1.10%`, report `6a39c62cb3ef5b3641e39daa`.
  - The accepted bitset-backed version was best `9.23-9.42 ms`, so removing the bitset regressed the benchmark by roughly 65-70%.
- Decision:
  - Reject and restore `FixedBitSet`. The field and dependency are still justified by measured performance, and the code comment on `DagCostSet` documents why it exists.

## Final validation after diff-reduction pass

- Status: complete
- Exact commands:
  - `cargo fmt --check`
  - `cargo test --test integration_test greedy_dag`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test files extract_vec_bench`
  - `cargo test --test files proof_support_snapshot`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking extract-vec-bench`
- Observed result:
  - Focused tests passed.
  - Final greedy-DAG canary: best `9.15 ms`, relative stddev `1.78%`, report `6a39c6a5b3ef5b3641e39dbf`.
  - Final tree canary: best `9.73 ms`, relative stddev `1.38%`, report `6a39c6afb3ef5b3641e39dc2`.

## Experiment 24: interner/secondary-map abstraction canary

- Status: complete
- Question: did moving the bitset-backed DAG cost-set storage behind the
  `Interner`/`AggregatedSparseSecondaryMap` helper abstraction regress the copied
  greedy-DAG workload?
- Baseline:
  - Previous accepted greedy-DAG canary after diff reduction: best `9.15 ms`,
    relative stddev `1.78%`.
  - Previous accepted tree canary after diff reduction: best `9.73 ms`,
    relative stddev `1.38%`.
- Exact commands:
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking extract-vec-bench`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - Greedy-DAG run 1: best `9.37 ms`, relative stddev `3.32%`, report
    `6a3a0571844322f190cb0bbc`.
  - Tree canary: best `10.11 ms`, relative stddev `1.18%`, report
    `6a3a057c844322f190cb0bbd`.
  - Greedy-DAG confirmation: best `9.33 ms`, relative stddev `2.58%`, report
    `6a3a0586844322f190cb0bbe`.
- Decision:
  - Keep the abstraction. The current greedy-DAG results are within local
    walltime noise of the accepted range, while the helper names and docs make
    the data structure easier to review.

## Experiment 25: intern fixed-point dependency keys

- Status: complete
- Question: should the greedy-DAG fixed-point state keep using
  string-keyed `(sort, value)` maps after `DagCostKey` ids already exist?
- Change:
  - Convert the final `producers_by_dependency`, `costs`, `parent_edge`, and
    pending worklist state to `InternId<DagCostKey>`.
  - Keep the one-time reachable-builder state string-keyed because discovery
    still talks to `ArcSort` and `funcs_by_sort` by sort name.
  - Store `producer_id` in `parent_edge` instead of cloning the selected
    function name and row values.
- Exact commands:
  - `cargo fmt --check`
  - `cargo test --test integration_test greedy_dag`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test files extract_vec_bench`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking extract-vec-bench`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - Greedy-DAG run 1: best `9.40 ms`, relative stddev `2.09%`, report
    `6a3a06c30a018d8e5a43daec`.
  - Tree canary: best `10.13 ms`, relative stddev `3.23%`, report
    `6a3a06ce0a018d8e5a43daf4`.
  - Greedy-DAG confirmation: best `9.19 ms`, relative stddev `2.08%`, report
    `6a3a06db0a018d8e5a43dafd`.
- Decision:
  - Keep the refactor. It is performance-neutral to slightly positive on this
    workload, removes repeated sort-name keys from the fixed-point state, and
    makes the final data structures line up with `DagCostSet`.

## Experiment 26: move clone-largest union into `AggregatedSparseSecondaryMap`

- Status: complete
- Question: should the greedy-DAG-specific `merge_child_cost_sets` helper be
  part of the reusable secondary-map data structure instead?
- Change:
  - Added `AggregatedSparseSecondaryMap::union_by_cloning_largest`, which
    takes the interner explicitly for empty unions and uses the clone-largest
    strategy from extraction-gym's `faster-greedy-dag`.
  - Removed the extractor-local `merge_child_cost_sets` wrapper.
  - Kept the operation generic over borrowed map owners, so the extractor can
    pass `Arc<DagCostSet<_>>` without making `secondary_map.rs` depend on `Arc`.
- Exact commands:
  - `cargo check`
  - `cargo test secondary_map`
  - `cargo fmt --check`
  - `cargo test --test integration_test greedy_dag`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test files extract_vec_bench`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `git diff --check`
- Observed result:
  - Focused tests passed.
  - Greedy-DAG canary: best `9.20 ms`, relative stddev `2.82%`, report
    `6a3a08b58d121c1a93b46656`.
- Decision:
  - Keep the refactor. It preserves the accepted walltime range and puts the
    efficient union behavior with the data structure that owns the membership
    representation.

## Experiment 27: separate monoid aggregation from extraction costs

- Status: complete
- Question: should the secondary-map helper depend directly on the extraction
  `Cost` trait, or should it own a smaller algebraic aggregation trait?
- Change:
  - Moved `CommutativeMonoid` into `secondary_map.rs`.
  - Made `Cost` a blanket domain marker over `CommutativeMonoid`.
  - Removed the secondary-map payload trait so `AggregatedSparseSecondaryMap`
    is generic over any commutative-monoid payload, not just costs.
- Exact commands:
  - `cargo check`
  - `cargo test secondary_map`
  - `cargo test --test integration_test greedy_dag`
  - `cargo test --test integration_test test_tree_extract_accepts_tree_only_cost_model`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test files extract_vec_bench`
  - `cargo test --test files proof_support_snapshot`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `git diff --check`
- Observed result:
  - Focused tests passed.
  - Greedy-DAG canary: best `9.26 ms`, relative stddev `3.05%`, report
    `6a3a0d918dcfd2917a5e0d26`.
- Decision:
  - Keep the refactor. It removes the import cycle between extraction cost
    traits and the secondary-map implementation without changing the observed
    greedy-DAG walltime range.

## Experiment 28: intern reachable builder keys during discovery

- Status: complete
- Question: should `ReachableExtractionBuilder` use interned `(sort, value)`
  ids while discovering reachable rows instead of collecting string-keyed
  `(String, Value)` sets and converting them afterward?
- Change:
  - Added `sort_ids` and `key_ids` to `ReachableExtractionBuilder`.
  - Removed the `cost_keys` set and the post-discovery conversion pass.
  - Changed builder `seen` and producer reverse dependencies to use
    `InternId<DagCostKey>` directly.
- Exact commands:
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `cargo fmt --check`
  - `cargo check`
  - `cargo test --test integration_test greedy_dag`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test files proof_support_snapshot`
  - `cargo test secondary_map`
  - `git diff --check`
- Observed result:
  - Pre-change canary: best `9.40 ms`, relative stddev `1.97%`, report
    `6a3a10fc8dcfd2917a5e0d78`.
  - Builder-interned canary: best `9.14 ms`, relative stddev `2.44%`,
    report `6a3a11688dcfd2917a5e0d82`.
- Decision:
  - Keep the refactor. It removes an extra conversion pass, improves type
    clarity for builder state, and stays within the accepted walltime range.

## Experiment 29: use dense secondary maps for fixed-point state

- Status: complete
- Question: should long-lived fixed-point maps use the interned id-space
  directly instead of hashing `InternId<DagCostKey>`?
- Change:
  - Added a plain `SecondaryMap<K, V>` keyed by `InternId<K>`.
  - Converted `producers_by_dependency`, `costs`, and `parent_edge` in
    `GreedyDagExtractor` to `SecondaryMap`.
- Exact commands:
  - `cargo fmt --check`
  - `cargo check`
  - `cargo test --test integration_test greedy_dag`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - Dense secondary-map canary: best `9.11 ms`, relative stddev `1.60%`,
    report `6a3a11d68d121c1a93b466c5`.
- Decision:
  - Keep the refactor. It is slightly positive on this workload and clarifies
    that these maps are associated with the fixed `DagCostKey` id universe.
    The tradeoff is dense `Option` slots for all reachable cost keys.

## Experiment 30: compose sparse storage under aggregated cost sets

- Status: complete
- Question: should cached aggregation stay embedded directly in
  `AggregatedSparseSecondaryMap`, or should aggregation wrap a reusable sparse
  secondary map?
- Change:
  - Added `SparseSecondaryMap<K, V>` for sparse entries plus bitset
    membership.
  - Rebuilt `AggregatedSparseSecondaryMap<K, V>` as
    `SparseSecondaryMap<K, V>` plus cached `total`.
- Exact commands:
  - `cargo fmt --check`
  - `cargo check`
  - `cargo test secondary_map`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - Composed aggregate canary: best `9.08 ms`, relative stddev `2.30%`,
    report `6a3a126f81e46b19c3bc46fd`.
- Decision:
  - Keep the composition. It makes the abstraction boundary clearer without a
    measurable penalty: sparse membership/iteration is separate from cached
    monoid aggregation.

## Experiment 31: use a secondary set for worklist membership

- Status: complete
- Question: should fixed-point worklist membership use the same interned
  universe as the secondary maps instead of a hash set?
- Change:
  - Added `SecondarySet<K>` backed by `FixedBitSet`.
  - Used `SecondarySet<DagCostKey>` for greedy-DAG `pending_set`.
  - Reused `SecondarySet` inside `SparseSecondaryMap` for membership.
- Exact commands:
  - `cargo fmt --check`
  - `cargo check`
  - `cargo test secondary_map`
  - `cargo test --test integration_test greedy_dag`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test files proof_support_snapshot`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
  - `git diff --check`
- Observed result:
  - Secondary-set canary: best `8.98 ms`, relative stddev `2.03%`, report
    `6a3a12df81e46b19c3bc470c`.
- Decision:
  - Keep the refactor. It is still in the accepted walltime range, slightly
    improves the canary, and makes maps/sets consistently tied to the interned
    id-space.

## Experiment 32: simplify clone-largest union bookkeeping

- Status: complete
- Question: can `AggregatedSparseSecondaryMap::union_by_cloning_largest`
  reduce call-site and helper bookkeeping without hurting the hot merge path?
- Change:
  - Removed the explicit capacity argument from `union_by_cloning_largest`.
  - Moved the `sum(child_set.len()) + 1` reservation into the helper.
  - Combined capacity calculation and largest-child selection into one pass.
  - Inlined the one-use `insert_missing_from` helper.
- Exact commands:
  - `cargo fmt --check`
  - `cargo check`
  - `cargo test secondary_map`
  - `cargo codspeed build -m walltime --bench ci_benchmarking`
  - `env CODSPEED_PROFILER_ENABLED=false codspeed run -m walltime -- cargo codspeed run -m walltime --bench ci_benchmarking greedy-dag-vec-extract`
- Observed result:
  - Greedy-DAG canary: best `8.56 ms`, relative stddev `2.59%`, report
    `6a3aa2c6d805349beed26de7`.
- Decision:
  - Keep the cleanup. It removes redundant caller-side capacity bookkeeping and
    one helper while staying at or below the accepted walltime range.

## Experiment 33: poach Herbie Taylor 7 external canary

- Status: complete
- Question: does the default tree extractor regress on a larger external
  extraction workload, and how does the current greedy-DAG extractor compare?
- Input:
  - Source: <https://github.com/ajpal/poach/blob/1de0e5bf540130ea1e1520bf006907f33590a286/infra/nightly-resources/test-files/herbie-math-taylor/taylor7.egg>.
  - Downloaded raw file to
    `/private/tmp/egg-smol-taylor7-bench.cQNVez/taylor7.egg`.
  - The pinned file does not run unchanged on either clean upstream or the PR
    binary because it uses `(set (constN) ...)` on zero-argument constructors.
    The compatibility input rewrites those 10,548 setup actions to
    `(union (constN) ...)` for both binaries.
  - The greedy-DAG input additionally rewrites all 1,363 final extract commands
    to add `:extractor greedy-dag`.
- Binaries:
  - Clean upstream/main worktree at `e4a65359`, copied to
    `/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-upstream-main`.
  - Current dirty PR worktree at `e4a65359`, copied to
    `/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current`.
- Exact command:
  - `hyperfine --warmup 1 --runs 3 --export-json /private/tmp/egg-smol-taylor7-bench.cQNVez/hyperfine-taylor7-current-vs-upstream.json --command-name 'upstream tree' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-upstream-main /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor7-compat.egg > /dev/null 2> /dev/null' --command-name 'current tree' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor7-compat.egg > /dev/null 2> /dev/null' --command-name 'current greedy-dag' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor7-compat-greedy-dag.egg > /dev/null 2> /dev/null'`
- Observed result:
  - Upstream tree: `26.516 s ± 0.072 s`.
  - Current tree: `27.293 s ± 0.109 s`.
  - Current greedy-DAG: `1.644 s ± 0.012 s`.
- Decision:
  - On this compatibility version of Taylor 7, current tree is about 2.9%
    slower than clean upstream tree. This is a larger external canary than
    `extract-vec-bench`, so the PR description should not claim a universal
    zero-regression result for tree extraction.
  - Current greedy-DAG is about 16.6x faster than current tree on this workload,
    likely because the file asks for up to one million variants per root and
    greedy-DAG variants rank root alternatives while using one best extraction
    for child e-classes.

## Experiment 34: checked-in Taylor 51 canary

- Status: complete
- Question: does the checked-in `tests/taylor51.egg` benchmark show the same
  tree before/after and tree-vs-greedy-DAG pattern as the larger Taylor 7
  input?
- Input:
  - Original checked-in file copied to
    `/private/tmp/egg-smol-taylor7-bench.cQNVez/taylor51.egg`.
  - Greedy-DAG variant mechanically rewrites all 324 final extract commands to
    add `:extractor greedy-dag`.
  - Unlike Taylor 7, this file needed no `(set (constN) ...)` compatibility
    rewrite.
- Binaries:
  - Reused the preserved `egglog-upstream-main` and `egglog-current` release
    binaries from Experiment 33.
- Exact command:
  - `hyperfine --warmup 3 --runs 10 --export-json /private/tmp/egg-smol-taylor7-bench.cQNVez/hyperfine-taylor51-current-vs-upstream.json --command-name 'upstream tree' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-upstream-main /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor51.egg > /dev/null 2> /dev/null' --command-name 'current tree' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor51.egg > /dev/null 2> /dev/null' --command-name 'current greedy-dag' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor51-greedy-dag.egg > /dev/null 2> /dev/null'`
- Observed result:
  - Upstream tree: `740.6 ms ± 3.2 ms`.
  - Current tree: `759.5 ms ± 1.5 ms`.
  - Current greedy-DAG: `150.7 ms ± 0.9 ms`.
- Decision:
  - Current tree is about 2.55% slower than clean upstream tree on this
    checked-in benchmark.
  - Current greedy-DAG is about 5.04x faster than current tree on this workload.

## Experiment 35: root-local sort-discovery prepass for tree extraction

- Status: rejected
- Question: can the tree extractor recover cheap sort-level filtering without
  restoring static container `inner_sorts` by doing a root-local prepass that
  discovers only output sort names, then running the existing full-table
  Bellman-Ford over those active sorts?
- Change tested:
  - Added `discover_reachable_output_sorts`, which starts from requested
    `(sort, value)` roots, follows actual container `inner_values`, probes
    producer rows by matching output value, and records eq-sort names.
  - `EGraph::extract_best` and `extract_variants` passed roots into
    `Extractor::prepare`; proof extraction and `print-function` stayed on the
    broad scan path.
  - After discovery, tree extraction still used full-table Bellman-Ford for the
    active output sorts rather than a value-local row graph.
- Validation before timing:
  - `cargo check`
  - `cargo test --test integration_test greedy_dag`
  - `cargo test --test files extract_vec_bench`
  - `cargo test --test files proof_support_snapshot`
  - `cargo test --test files greedy_dag_vec_extract`
- Timing probes:
  - `tests/taylor51.egg`, `hyperfine --warmup 3 --runs 10`: upstream tree
    `736.0 ms ± 3.4 ms`, current tree before prepass `780.1 ms ± 33.3 ms`,
    current tree with sort prepass `868.2 ms ± 3.4 ms`.
  - Taylor 7 compatibility input, `hyperfine --warmup 0 --runs 1`: current
    tree before prepass `26.885 s`, current tree with sort prepass `28.334 s`.
  - `tests/extract-vec-bench.egg`, `hyperfine --warmup 10 --runs 30`: upstream
    tree `232.4 ms ± 3.3 ms`, current tree before prepass
    `232.1 ms ± 4.2 ms`, current tree with sort prepass `234.1 ms ± 14.9 ms`.
- Decision:
  - Revert the experiment. The prepass cost is paid per extract command and
    did not save enough full-table work; it clearly regressed the Taylor
    canaries and was neutral/noisy on `extract-vec-bench`.
  - If we want to recover upstream tree filtering, the next plausible route is
    a cheaper sort-level mechanism, likely a narrow static API for normal
    containers with a conservative fallback for unknown containers, or reverting
    the sort API changes more directly.

## Experiment 36: direct sort-API revert for tree extraction

- Status: accepted
- Question: does reverting the sort API changes and seeding the tree extractor
  from requested root sorts recover the upstream tree-extraction performance
  without changing the new public multi-root APIs?
- Change tested:
  - Restored `Sort::inner_sorts` / `ContainerSort::inner_sorts`.
  - Restored the `UnstableFn` partial-argument sort cache used by extraction.
  - Removed the `container_sort_as_any` downcast API.
  - Kept public extraction APIs as `extract_best(roots, cost_model)` and
    `extract_variants(roots, nvariants, cost_model)`, deriving root sorts from
    those requested roots internally.
  - Restored tree extractor setup to a sort-level BFS from those root sorts.
- Validation before timing:
  - `cargo fmt`
  - `cargo check`
  - `cargo test --test integration_test greedy_dag`
  - `cargo test --test files extract_vec_bench`
  - `cargo test --test files greedy_dag_vec_extract`
  - `cargo test --test extraction_proof_mode`
  - `cargo test --test files proof_support_snapshot`
  - `cargo fmt --check`
  - `git diff --check`
- Binaries:
  - Reused preserved upstream/current binaries from Experiments 33-35.
  - Built current sort-reverted release binary and copied it to
    `/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current-sort-api-reverted`.
- Exact commands:
  - `cargo build --release --bin egglog`
  - `cp target/release/egglog /private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current-sort-api-reverted`
  - `hyperfine --warmup 2 --runs 10 --export-json /private/tmp/egg-smol-taylor7-bench.cQNVez/hyperfine-taylor51-sort-api-reverted.json '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-upstream-main /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor51.egg' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor51.egg' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current-sort-api-reverted /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor51.egg' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current-sort-api-reverted /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor51-greedy-dag.egg'`
  - `hyperfine --runs 1 --export-json /private/tmp/egg-smol-taylor7-bench.cQNVez/hyperfine-taylor7-sort-api-reverted-smoke.json '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-upstream-main /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor7-compat.egg' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor7-compat.egg' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current-sort-api-reverted /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor7-compat.egg' '/private/tmp/egg-smol-taylor7-bench.cQNVez/bin/egglog-current-sort-api-reverted /private/tmp/egg-smol-taylor7-bench.cQNVez/taylor7-compat-greedy-dag.egg'`
- Observed result:
  - Taylor 51 upstream tree: `751.3 ms ± 10.3 ms`.
  - Taylor 51 old current tree: `797.7 ms ± 50.9 ms`.
  - Taylor 51 sort-reverted current tree: `760.8 ms ± 5.3 ms`.
  - Taylor 51 sort-reverted greedy-DAG: `152.3 ms ± 5.0 ms`.
  - Taylor 7 smoke upstream tree: `26.610 s`.
  - Taylor 7 smoke old current tree: `27.346 s`.
  - Taylor 7 smoke sort-reverted current tree: `26.580 s`.
  - Taylor 7 smoke sort-reverted greedy-DAG: `1.626 s`.
- Decision:
  - Keep the direct sort-API revert. It recovers the tree extractor from the
    broad-scan regression on both Taylor canaries while preserving the new
    public multi-root extraction APIs.
  - Greedy-DAG remains materially faster on these workloads, but this
    experiment is specifically about eliminating the incidental tree slowdown.
