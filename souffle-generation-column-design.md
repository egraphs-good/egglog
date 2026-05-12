# Generation column for full `(run N)` snapshot parity

Companion to `souffle-strata-design.md` and `souffle-backend-plan.md`. This
document records the design for getting **exact** `(run N)` semantic parity
between the souffle backend and default egglog — what's needed to make
`tests/files.rs` snapshots match across treatments the way proof mode does.

## Problem statement

The strata setup (snap/buffer/canon split) achieves `(check ...)` parity:
souffle's UF gives the same eclass equality as default egglog's. But it does
**not** achieve `(print-size F)` parity.

Empirical data, `math-microbenchmark.egg`:

| Backend | `(run 1)` Add | `(run 11)` Add |
|---|---|---|
| Default egglog | 20 | 641,743 |
| Souffle (strata, current) | ~3 | ~13 (canonical) / ~384 (buffer) |

The gap is structural, not a constant factor. Two semantic mismatches stack:

1. **Subsumption**: the rebuild rule's subsumption clause deletes the old
   row when canonicalized output replaces it. Egglog's function table keeps
   the (c0, c1) key around — its output may have been canonicalized in place,
   but the row didn't get deleted. Net effect: souffle's canonical view has
   far fewer rows than egglog's function table even before counting derivation
   layers.

2. **No within-iter cascading**: souffle's user rules read `_snap`, which is
   frozen at the start of each outer iteration. Egglog's semi-naive lets
   rule A produce a tuple and rule B in the same `(run)` iteration match
   it. In our setup B has to wait for snap to refresh next outer iter.
   The growth rate per iter is therefore much smaller — souffle's `(run 11)`
   reaches only ~1 layer of derivation depth, while egglog reaches the full
   exponential cascade.

Both have to be addressed for full snapshot parity.

## Design: generation column ("wave")

Idea: encode iteration explicitly as a column on every relation. Bodies
match tuples with `wave < CURRENT_WAVE`; heads write `wave = CURRENT_WAVE`.
Outer-saturate advances `CURRENT_WAVE` by 1 per iteration. N outer iters ⇒
N user rounds ⇒ matches `(run N)`.

### Relation shape

```
.decl Eg_AddView(c0: number, c1: number, c2: Math, out: number, wave: number)
.decl Eg_AddView_buffer(c0: number, c1: number, c2: Math, out: number, wave: number)
// no _snap relations any more
```

The drain rule preserves the wave from buffer to canonical. Initial facts
get `wave = 0`. User rule writes get `wave = CURRENT_WAVE`.

### CURRENT_WAVE exposure

The fork already has `outer_loop_counter` as a RAM variable. We expose it
to user code via a one-row built-in relation:

```
.decl _IterCounter(n: number)
// runtime-populated: at the start of each outer iter, contains the single
// row (current_counter_value)
```

In the fork's outer-saturate emission, after the counter increment, emit
RAM that clears `_IterCounter` and inserts the current value.

User rules reference it as `_IterCounter(K)` and use K to filter:

```
Eg_UF_Math_buffer(t1, t2, 0, K) :-
  _IterCounter(K),
  Eg_AddView(a, b, t1, _, w1), w1 < K,
  ...rest of body referencing waves < K...
```

### Rebuild rule handling

The rebuild rule (subsumption: replace c2 with leader) needs special care.
Two reasonable designs:

- **Preserve wave**: rebuild keeps the original tuple's wave. The
  canonicalized tuple has the same wave as the original. This means
  subsequent rules can still see it (wave threshold met). Print-size
  stays equal to the number of distinct (c0, c1) introductions.

- **Don't subsume at all** (match default egglog literally): the rebuild
  rule writes the canonicalized version *without* deleting the original.
  This is closer to egglog's behavior where the function table row stays
  with an updated output. Print-size grows toward egglog's exact count.

I think we want **don't subsume** — that's the only way print-size matches
default egglog's flat e-node count. Subsumption made the strata setup
tractable; with bounded iterations + generation column we don't need it.

### Within-iter cascading

Once user rules read directly from the canonical view (no `_snap`), and
write with `wave = CURRENT_WAVE`, semi-naive within the user-rule SCC
*does* cascade — within outer iter K, rule A writes wave=K, but the body
filter `wave < K` excludes those. So no within-iter cascading; one logical
pass per outer iter. This is what we want for `(run N)` parity.

Wait — but egglog cascades within an iter, no?

Re-reading egglog's source on `run` semantics: each `(run)` iteration is
"semi-naive over the delta from the previous iteration." Rule A's outputs
become next iter's delta. So egglog also does *one logical pass per
`(run)` iter*. The 641K growth is from compounding across 11 passes, not
from within-iter cascading. Good — that matches the generation-column
design exactly.

## Encoder rule-emission site map (intel from Explore agents, 2026-05)

Twenty distinct view-touching emission sites in `src/proofs/proof_encoding.rs`,
across 7 rule categories. Most are **SYSTEM** rules (write/read the canonical
view); the small handful of **USER** sites is where the wave-filter must
attach.

| Lines | Category | Direction | Path |
|---|---|---|---|
| 577 | view decl | — | `term_and_view` |
| 333, 339 | delete/subsume body reads | SYSTEM read | `delete_and_subsume` |
| 334, 340 | delete/subsume head writes | SYSTEM write | `delete_and_subsume` |
| 374-375, 421-422, 437-438 | merge body reads | SYSTEM read | `handle_merge_fn` |
| 411, 440 | merge head writes (set/delete) | SYSTEM write | `handle_merge_fn` |
| 462-463 | congruence body | SYSTEM read | `handle_congruence` |
| 606-607, 613-614 | drain head | SYSTEM write | `term_and_view` (strata only) |
| 719 (via 1186) | rebuild body | SYSTEM read | `rebuild_rule` |
| 769 (via update_view) | rebuild head set | SYSTEM write | `rebuild_rule` |
| 780 | rebuild head delete | SYSTEM write | `rebuild_rule` |
| **824** | **custom-fn user-rule read** | **USER read** | `instrument_fact_expr` |
| **939** | **constructor user-rule read** | **USER read** | `instrument_fact_expr` |
| **1165** | **user-rule head write** | **USER write** | `add_term_and_view` |

User reads use `view_name_for_user_read` (resolves to `_snap` under strata,
canonical otherwise). User writes use `view_name_for_user_write` (resolves
to `_buffer` under strata).

## Affected snapshots

Just one: `src/proofs/snapshots/egglog__proofs__proof_tests__tests__doc_example_add_function1.snap`.
16 `__AddView` references; all need an extra trailing arg.

Test that produces it: `doc_example_add_function1()` in
`src/proofs/proof_tests.rs:29`.

## Translator-side user/system detection

The encoder doesn't put an explicit AST flag on rules, but **`rule.ruleset`**
is reliable. System rules always land in one of these 6 rulesets (defined
in `EncodingNames` in `proof_encoding_helpers.rs`):

- `path_compress_ruleset_name`
- `single_parent_ruleset_name`
- `uf_function_index_ruleset_name`
- `rebuilding_ruleset_name`
- `rebuilding_cleanup_ruleset_name`
- `delete_subsume_ruleset_name`

User rules either have an empty ruleset or one the user declared. They
**never** use the 6 above. The translator's `translate_rule` should check
`rule.ruleset` against this set: hit ⇒ system (pass wave through), miss ⇒
user (emit `IterCounter(K)` body atom and `wave < K` filter, head wave = K).

## Implementation plan

### 1. Fork: expose iteration counter

- Add `_IterCounter` synthetic relation, declared automatically.
- In `UnitTranslator::translateProgram`, inside the outer-saturate loop
  body, emit RAM that clears `_IterCounter` and inserts a single row with
  the current `outer_loop_counter` value.
- Reserve the name `_IterCounter` (or document that user programs can't use
  it).

Estimated effort: half a day. Pattern is the same as the snap refresh.

### 2. Encoder: add wave column

- `proof_encoding.rs`: every view/buffer function declaration gets a
  trailing `wave: i64` column.
- Every emitted rule body atom gets a fresh wave-var; every head atom
  passes through a wave value.
- Drain rules: pass wave through unchanged.
- Rebuild rules: pass wave through, drop the subsumption (no `delete`
  action).
- UF rules: pass wave through.
- Initial facts: `wave = 0` (or maybe `wave = iter at which the fact was
  declared` if we want top-level interleaved with rules — but for v0,
  treat all initial facts as wave 0).

Estimated effort: 1-2 days. Touches a lot of `view_table`,
`add_term_and_view`, and rule emission paths. Each rule needs the wave-var
plumbed through both the resolved AST and the eventual emitted ResolvedRule.

### 3. Translator: emit wave filter on user rules

- User-rule body translation: insert `_IterCounter(K)` as a body atom and
  `body_atom_wave < K` as a constraint for each body atom.
- User-rule head translation: write `wave = K` as the head's wave column.
- Drain/rebuild/UF rules: pass through (no filter — they're "system"
  rules that operate on all waves).
- (check) facts: rewrite to read from canonical (no `< K` filter — they
  see the final state).

Estimated effort: 1 day. The hook points already exist (`translate_rule`
already does body/action translation); this is wiring through one more
column.

### 4. Adapter / tests

- `runner.rs`: parse the new column-shape printsize output — `<rel>\t<n>`
  format unchanged, just the row count.
- `tests/souffle_files.rs`: add a `print_size_parity_against_default` test
  that compares `(print-size)` outputs between souffle backend and default
  egglog, file by file.
- `tests/files.rs`: add a `souffle` treatment alongside `proof`,
  comparing against `shared_snapshot`.

Estimated effort: half a day, once the prior pieces work.

## Risks / open questions

- **Top-level `(let v X)` and globals**: globals are inlined as records.
  Their wave is what? Probably 0 (declared before iter 1). Need to thread
  through.
- **`(extract)`**: not in scope for this design, but the canonical
  representative needed for extraction would need to be picked
  deterministically; "highest wave" or "lowest wave" both have meaning.
- **Performance**: every relation gets one more column. Souffle's record
  representations are wide; one more `number` field is cheap but
  measurable. Probably fine.
- **`(saturate)` schedules**: egglog has `(saturate (...))` for fixpoint
  iteration. We'd map this to `outer-saturate = LARGE_NUMBER` with the
  generation column still in place — saturation naturally happens when no
  rule writes new tuples.
- **Multiple rulesets**: math-microbenchmark uses one. The encoder's
  multiple-rulesets handling (`__rebuilding`, `__parent`, etc.) might
  interact oddly. Each ruleset's rules need waves consistent within the
  ruleset; cross-ruleset interactions need thought.

## Why this is the right path

- **Precise `(run N)` semantics.** Every test file's snapshot can match
  exactly, no caveats.
- **Datalog-native.** The generation column is the standard way to encode
  bounded iteration of non-monotonic operations in Datalog; it doesn't
  require fork hacks beyond exposing the counter.
- **Decouples from subsumption.** Once iterations are explicit, we don't
  need the canonical view to be small for termination — souffle's
  bounded-outer-iter handles that. Subsumption can be removed, restoring
  parity with egglog's flat function table.

The strata setup remains a valuable intermediate. It demonstrates that
the cycle-breaking via snapshots works for eclass equality (`(check)`
parity). The generation column is the next layer that adds precise
iteration counting on top.
