# Seminaive Encoding: Hand-Written `Add` Example

A worked-by-hand example of what the seminaive encoding pass should
produce, using the running `Add`/commutativity program from
[`src/proofs/proof_encoding.md`](src/proofs/proof_encoding.md).

The point of writing this by hand: force the open design questions
in [`seminaive-encoding-experiment.md`](seminaive-encoding-experiment.md)
§8 to a decision before mechanizing the rewrite. Once this target
runs correctly under the existing backend with `seminaive=false`,
the encoder's job is to produce this shape mechanically.

**Scope.** Non-proof mode only. Proof mode is a straightforward
extension — every function whose output type is currently `Proof`
just keeps tracking proofs through the parallel `_ts` table the same
way Unit-output functions do.

---

## 1. Source (after term encoding, before seminaive encoding)

Reproduced verbatim from `proof_encoding.md` §1. Term encoding has
already lowered `union`, deferred deletes, and turned rebuilding
into rules. This is the input our pass receives.

```text
(ruleset parent)
(ruleset single_parent)
(ruleset uf_function_index)
(ruleset rebuilding)
(ruleset rebuilding_cleanup)
(ruleset delete_subsume_ruleset)

(run-schedule
    (saturate
       rebuilding_cleanup
       (saturate single_parent)
       (saturate parent)
       (saturate uf_function_index)
       rebuilding)
    delete_subsume_ruleset)

(sort Math)
(function UF_Math (Math Math) Unit :merge old :internal-hidden)
(function UF_Mathf (Math) Math :merge new)

(rule ((UF_Math a b)
       (UF_Math b c)
       (!= b c))
      ((delete (UF_Math a b))
       (set (UF_Math a c) ()))
       :ruleset parent :name "uf_update")

(rule ((UF_Math a b)
       (UF_Math a c)
       (!= b c)
       (= (ordering-max b c) b))
      ((delete (UF_Math a b))
       (set (UF_Math b c) ()))
       :ruleset single_parent :name "singleparentuf_update")

(rule ((UF_Math a b))
      ((set (UF_Mathf a) b))
       :ruleset uf_function_index :name "uf_function_index_update")

(constructor Add (i64 i64) Math)
(function AddView (i64 i64 Math) Unit :merge old :internal-term-constructor Add)
(constructor to_delete_Add (i64 i64) view)
(constructor to_subsume_Add (i64 i64) view)

(rule ((AddView c0 c1 new)
       (AddView c0 c1 old)
       (!= old new)
       (= (ordering-max old new) new))
      ((set (UF_Math (ordering-max new old) (ordering-min new old)) ()))
       :ruleset rebuilding :name "congruence_rule")

(rule ((= v9 (AddView c0 c1 c2))
       (= c2_leader (UF_Mathf c2)))
      ((set (AddView c0 c1 c2_leader) ())
       (delete (AddView c0 c1 c2)))
        :ruleset rebuilding :name "rebuild_rule")

(function v2 () Math :no-merge)
(set (v2) (Add 1 2))
(set (AddView 1 2 (v2)) ())
(set (UF_Math (v2) (v2)) ())

(rule ((= v3 (AddView a b v4)))
      ((let v5 (Add a b))
       (set (AddView a b v5) ())
       (set (UF_Math v5 v5) ())
       (let v6 (Add b a))
       (set (AddView b a v6) ())
       (set (UF_Math v6 v6) ())
       (set (UF_Math (ordering-max v5 v6) (ordering-min v5 v6)) ()))
       :name "commutativity")

(rule ((to_delete_Add c0 c1)
       (AddView c0 c1 out))
      ((delete (AddView c0 c1 out))
       (delete (to_delete_Add c0 c1)))
        :ruleset delete_subsume_ruleset :name "delete_rule")
(rule ((to_subsume_Add c0 c1)
       (AddView c0 c1 out))
      ((subsume (AddView c0 c1 out)))
        :ruleset delete_subsume_ruleset :name "delete_rule_subsume")

(to_delete_Add 1 2)
```

---

## 2. Decisions baked into this example

These are the open-question answers chosen for the hand-written
target. They're decisions we should *commit to* in this experiment
or revisit if the design proves awkward.

**D1. Parallel `_ts` tables, not bundled outputs.** Every queried
function `f` gets a parallel `f_ts` function with the same key
columns and an `i64` output for the row's epoch. The base function
is unchanged. Reasons: works uniformly across non-proof and proof
mode (no `Pair Proof i64` gymnastics); base function semantics
unchanged; the encoding stays mechanical. Cost: doubles the number
of body atoms in queries.

**D2. `_ts` only for queried functions.** Functions whose rows are
never read by any rule body (e.g. the term table `Add`, which is
write-only in the term encoding) don't get a parallel `_ts`.

**D3. `_ts` merge mode mirrors the base function.** `:merge old` →
`:merge old` (first epoch wins, never changes); `:merge new` →
`:merge new` (latest epoch wins). This way ts updates iff the row's
"identity for downstream rules" updates.

**D4. Focused-atom-only seminaive.** The focused atom carries
`(>= ts_i (last_run_at_<rule_name>))`. Non-focused atoms are
unrestricted. This matches the existing backend's seminaive
semantics (per `egglog-bridge/src/rule.rs:869–919` from earlier
exploration). We are *not* adding `(< ts_j (next_ts))` defenses on
non-focused atoms — they're optional and add planner load.

**D5. Globals updated by a Rust-side schedule hook, not by egglog
rules.** `next_ts` and `last_run_at_<rule>` are read-only from the
encoded program's perspective. The schedule executor in
`src/lib.rs` bumps them between iterations. (See §5 for
pseudocode.) This is the smallest non-pure-encoding piece; trying
to express it inside egglog runs into the saturation-loop problem
where a "bump-counter" rule would never terminate.

**D6. Primitive atoms don't get variants.** Only function-table
atoms generate seminaive variants. Primitive atoms like `(!= b c)`
or `(= (ordering-max b c) b)` are passed through unchanged in every
variant.

**D7. Variants over `:no-merge` constructor calls in actions.**
Constructor calls like `(let v5 (Add a b))` insert into the term
table without a parallel ts (per D2). No change needed.

---

## 3. Encoded program

### 3.1 Schedule and bookkeeping globals

```text
(ruleset parent)
(ruleset single_parent)
(ruleset uf_function_index)
(ruleset rebuilding)
(ruleset rebuilding_cleanup)
(ruleset delete_subsume_ruleset)

;; Schedule unchanged. The Rust executor bumps next_ts and the
;; per-rule last_run_at globals between iterations.
(run-schedule
    (saturate
       rebuilding_cleanup
       (saturate single_parent)
       (saturate parent)
       (saturate uf_function_index)
       rebuilding)
    delete_subsume_ruleset)

;; --- Seminaive bookkeeping globals ---
;; Read-only from the encoded program's perspective. The schedule
;; executor sets these via direct backend calls; they look like
;; :merge new functions to egglog so updates take effect.
(function next_ts () i64 :merge new)
(set (next_ts) 0)

(function last_run_at_uf_update () i64 :merge new)
(set (last_run_at_uf_update) 0)

(function last_run_at_singleparentuf_update () i64 :merge new)
(set (last_run_at_singleparentuf_update) 0)

(function last_run_at_uf_function_index_update () i64 :merge new)
(set (last_run_at_uf_function_index_update) 0)

(function last_run_at_congruence_rule () i64 :merge new)
(set (last_run_at_congruence_rule) 0)

(function last_run_at_rebuild_rule () i64 :merge new)
(set (last_run_at_rebuild_rule) 0)

(function last_run_at_commutativity () i64 :merge new)
(set (last_run_at_commutativity) 0)

(function last_run_at_delete_rule () i64 :merge new)
(set (last_run_at_delete_rule) 0)

(function last_run_at_delete_rule_subsume () i64 :merge new)
(set (last_run_at_delete_rule_subsume) 0)
```

### 3.2 Sort and function declarations with parallel `_ts` tables

```text
(sort Math)

;; UF tables
(function UF_Math (Math Math) Unit :merge old :internal-hidden)
(function UF_Math_ts (Math Math) i64 :merge old)   ;; D3: mirrors :merge old

(function UF_Mathf (Math) Math :merge new)
(function UF_Mathf_ts (Math) i64 :merge new)       ;; D3: mirrors :merge new

;; Term and view tables
(constructor Add (i64 i64) Math)                   ;; D2: no Add_ts
(function AddView (i64 i64 Math) Unit :merge old :internal-term-constructor Add)
(function AddView_ts (i64 i64 Math) i64 :merge old)

;; Deferred-action helpers
(constructor to_delete_Add (i64 i64) view)
(function to_delete_Add_ts (i64 i64) i64 :merge old)
;; Note: to_delete_Add is a constructor (returns a fresh `view` ID).
;; Its parallel _ts is keyed only on the i64 inputs — we don't need
;; to track the freshly allocated view ID; the rule that consumes
;; it does so by argument pattern, not by the view value.
(constructor to_subsume_Add (i64 i64) view)
(function to_subsume_Add_ts (i64 i64) i64 :merge old)
```

### 3.3 Rule rewrites

For each rule, we annotate every function-table atom with a paired
`_ts` atom, then expand into N variants (one per function atom),
where variant `i` carries the focused-atom `>= last_run_at`
predicate. Every action that inserts to a tracked function also
inserts to its `_ts` parallel; every deletion does likewise.

#### `uf_update` (2 function atoms → 2 variants)

```text
(rule ((UF_Math a b)
       (UF_Math_ts a b ts1)
       (>= ts1 (last_run_at_uf_update))   ;; FOCUS: first UF_Math
       (UF_Math b c)
       (UF_Math_ts b c ts2)               ;; bound but unconstrained
       (!= b c))
      ((delete (UF_Math a b))
       (delete (UF_Math_ts a b))
       (set (UF_Math a c) ())
       (set (UF_Math_ts a c) (next_ts)))
       :ruleset parent :name "uf_update@1")

(rule ((UF_Math a b)
       (UF_Math_ts a b ts1)               ;; bound but unconstrained
       (UF_Math b c)
       (UF_Math_ts b c ts2)
       (>= ts2 (last_run_at_uf_update))   ;; FOCUS: second UF_Math
       (!= b c))
      ((delete (UF_Math a b))
       (delete (UF_Math_ts a b))
       (set (UF_Math a c) ())
       (set (UF_Math_ts a c) (next_ts)))
       :ruleset parent :name "uf_update@2")
```

#### `singleparentuf_update` (2 function atoms → 2 variants)

```text
(rule ((UF_Math a b)
       (UF_Math_ts a b ts1)
       (>= ts1 (last_run_at_singleparentuf_update))   ;; FOCUS
       (UF_Math a c)
       (UF_Math_ts a c ts2)
       (!= b c)
       (= (ordering-max b c) b))
      ((delete (UF_Math a b))
       (delete (UF_Math_ts a b))
       (set (UF_Math b c) ())
       (set (UF_Math_ts b c) (next_ts)))
       :ruleset single_parent :name "singleparentuf_update@1")

(rule ((UF_Math a b)
       (UF_Math_ts a b ts1)
       (UF_Math a c)
       (UF_Math_ts a c ts2)
       (>= ts2 (last_run_at_singleparentuf_update))   ;; FOCUS
       (!= b c)
       (= (ordering-max b c) b))
      ((delete (UF_Math a b))
       (delete (UF_Math_ts a b))
       (set (UF_Math b c) ())
       (set (UF_Math_ts b c) (next_ts)))
       :ruleset single_parent :name "singleparentuf_update@2")
```

#### `uf_function_index_update` (1 function atom → 1 variant)

```text
(rule ((UF_Math a b)
       (UF_Math_ts a b ts1)
       (>= ts1 (last_run_at_uf_function_index_update)))   ;; FOCUS
      ((set (UF_Mathf a) b)
       (set (UF_Mathf_ts a) (next_ts)))
       :ruleset uf_function_index :name "uf_function_index_update@1")
```

#### `congruence_rule` (2 function atoms → 2 variants)

```text
(rule ((AddView c0 c1 new)
       (AddView_ts c0 c1 new ts1)
       (>= ts1 (last_run_at_congruence_rule))   ;; FOCUS
       (AddView c0 c1 old)
       (AddView_ts c0 c1 old ts2)
       (!= old new)
       (= (ordering-max old new) new))
      ((set (UF_Math (ordering-max new old) (ordering-min new old)) ())
       (set (UF_Math_ts (ordering-max new old) (ordering-min new old)) (next_ts)))
       :ruleset rebuilding :name "congruence_rule@1")

(rule ((AddView c0 c1 new)
       (AddView_ts c0 c1 new ts1)
       (AddView c0 c1 old)
       (AddView_ts c0 c1 old ts2)
       (>= ts2 (last_run_at_congruence_rule))   ;; FOCUS
       (!= old new)
       (= (ordering-max old new) new))
      ((set (UF_Math (ordering-max new old) (ordering-min new old)) ())
       (set (UF_Math_ts (ordering-max new old) (ordering-min new old)) (next_ts)))
       :ruleset rebuilding :name "congruence_rule@2")
```

#### `rebuild_rule` (2 function atoms → 2 variants)

```text
(rule ((= v9 (AddView c0 c1 c2))
       (AddView_ts c0 c1 c2 ts1)
       (>= ts1 (last_run_at_rebuild_rule))   ;; FOCUS: AddView
       (= c2_leader (UF_Mathf c2))
       (UF_Mathf_ts c2 ts2))
      ((set (AddView c0 c1 c2_leader) ())
       (set (AddView_ts c0 c1 c2_leader) (next_ts))
       (delete (AddView c0 c1 c2))
       (delete (AddView_ts c0 c1 c2)))
        :ruleset rebuilding :name "rebuild_rule@1")

(rule ((= v9 (AddView c0 c1 c2))
       (AddView_ts c0 c1 c2 ts1)
       (= c2_leader (UF_Mathf c2))
       (UF_Mathf_ts c2 ts2)
       (>= ts2 (last_run_at_rebuild_rule)))   ;; FOCUS: UF_Mathf
      ((set (AddView c0 c1 c2_leader) ())
       (set (AddView_ts c0 c1 c2_leader) (next_ts))
       (delete (AddView c0 c1 c2))
       (delete (AddView_ts c0 c1 c2)))
        :ruleset rebuilding :name "rebuild_rule@2")
```

#### `commutativity` (1 function atom → 1 variant)

```text
(rule ((= v3 (AddView a b v4))
       (AddView_ts a b v4 ts1)
       (>= ts1 (last_run_at_commutativity)))   ;; FOCUS
      ((let v5 (Add a b))
       (set (AddView a b v5) ())
       (set (AddView_ts a b v5) (next_ts))
       (set (UF_Math v5 v5) ())
       (set (UF_Math_ts v5 v5) (next_ts))
       (let v6 (Add b a))
       (set (AddView b a v6) ())
       (set (AddView_ts b a v6) (next_ts))
       (set (UF_Math v6 v6) ())
       (set (UF_Math_ts v6 v6) (next_ts))
       (set (UF_Math (ordering-max v5 v6) (ordering-min v5 v6)) ())
       (set (UF_Math_ts (ordering-max v5 v6) (ordering-min v5 v6)) (next_ts)))
       :name "commutativity@1")
```

#### `delete_rule` and `delete_rule_subsume` (2 function atoms each → 2 variants each)

```text
(rule ((to_delete_Add c0 c1)
       (to_delete_Add_ts c0 c1 ts1)
       (>= ts1 (last_run_at_delete_rule))   ;; FOCUS
       (AddView c0 c1 out)
       (AddView_ts c0 c1 out ts2))
      ((delete (AddView c0 c1 out))
       (delete (AddView_ts c0 c1 out))
       (delete (to_delete_Add c0 c1))
       (delete (to_delete_Add_ts c0 c1)))
        :ruleset delete_subsume_ruleset :name "delete_rule@1")

(rule ((to_delete_Add c0 c1)
       (to_delete_Add_ts c0 c1 ts1)
       (AddView c0 c1 out)
       (AddView_ts c0 c1 out ts2)
       (>= ts2 (last_run_at_delete_rule)))   ;; FOCUS
      ((delete (AddView c0 c1 out))
       (delete (AddView_ts c0 c1 out))
       (delete (to_delete_Add c0 c1))
       (delete (to_delete_Add_ts c0 c1)))
        :ruleset delete_subsume_ruleset :name "delete_rule@2")

(rule ((to_subsume_Add c0 c1)
       (to_subsume_Add_ts c0 c1 ts1)
       (>= ts1 (last_run_at_delete_rule_subsume))   ;; FOCUS
       (AddView c0 c1 out)
       (AddView_ts c0 c1 out ts2))
      ((subsume (AddView c0 c1 out)))
        :ruleset delete_subsume_ruleset :name "delete_rule_subsume@1")

(rule ((to_subsume_Add c0 c1)
       (to_subsume_Add_ts c0 c1 ts1)
       (AddView c0 c1 out)
       (AddView_ts c0 c1 out ts2)
       (>= ts2 (last_run_at_delete_rule_subsume)))   ;; FOCUS
      ((subsume (AddView c0 c1 out)))
        :ruleset delete_subsume_ruleset :name "delete_rule_subsume@2")
```

### 3.4 Initial inserts

The `(Add 1 2)` initial insert and the `to_delete_Add 1 2` request
need ts stamps too:

```text
(function v2 () Math :no-merge)
(set (v2) (Add 1 2))
(set (AddView 1 2 (v2)) ())
(set (AddView_ts 1 2 (v2)) (next_ts))
(set (UF_Math (v2) (v2)) ())
(set (UF_Math_ts (v2) (v2)) (next_ts))

(to_delete_Add 1 2)
(set (to_delete_Add_ts 1 2) (next_ts))
```

(Subtle point: the schedule executor must ensure `next_ts` has been
bumped to `1` before these run, so they don't collide with the
initial-state `0` that all `last_run_at_*` are set to. Otherwise
the first iteration's seminaive scans miss them.)

---

## 4. What rule firings should look like

To sanity-check, walk through the first iteration of the
`commutativity` rule.

- Initial state: `next_ts = 1` (after Rust bumps it once for the
  initial inserts), `last_run_at_commutativity = 0`.
- All initial-insert rows have `ts = 1`.
- Bump: schedule executor sets `next_ts = 2`.
- Run `commutativity@1`: focused atom `(AddView_ts a b v4 ts1)` with
  `(>= ts1 0)`. All AddView rows match (ts1 = 1 ≥ 0).
  - Match found: `a=1, b=2, v4=(v2)`.
  - Actions fire: `(let v5 (Add 1 2))` → reuses `(v2)`; `(let v6 (Add 2 1))`
    → fresh Math ID; the various `set` actions stamp `next_ts = 2`.
- Update: schedule executor sets `last_run_at_commutativity = 2`.

On the next iteration, `last_run_at_commutativity = 2`, so the focus
atom requires `ts >= 2`. The original AddView row with `ts = 1` is
filtered out — so the rule only re-fires if a *new* AddView row
appeared (e.g. `(AddView 2 1 v6)` from this iteration's actions).

This is exactly the behavior built-in seminaive provides.

---

## 5. Schedule-executor pseudocode

The Rust-side hook lives in `src/lib.rs`'s schedule executor (the
loop that drives `run-schedule`). Pseudocode:

```rust
fn run_one_iteration(ruleset: RulesetId) -> bool {
    // Bump next_ts before the iteration. Any rows inserted by rules
    // in this iteration carry this new ts.
    let new_ts = read_global("next_ts") + 1;
    set_global("next_ts", new_ts);

    // Run all rules in the ruleset (each rule is now a set of N
    // seminaive variants, all with `seminaive=false` set on the
    // backend).
    let report = backend.run_rules(ruleset_rules);

    // After the iteration, mark each rule that ran as having run at
    // `new_ts`. Next iteration, these rules' focused atoms will
    // filter `ts >= new_ts`.
    for rule in ruleset_rules {
        let src_name = rule.source_name();  // strips the @i suffix
        set_global(format!("last_run_at_{src_name}"), new_ts);
    }

    report.updated
}
```

Two subtleties:

- *Per source rule, not per variant.* All N variants of source rule
  R share `last_run_at_R`. After the iteration we update the single
  global, not N globals.
- *Initial-state setup.* Before the schedule starts, all
  `last_run_at_*` are 0 and `next_ts` is 0. The executor bumps
  `next_ts` once before processing initial `set`/`union` commands,
  so those rows have ts ≥ 1. This way the first iteration's `ts >=
  0` predicates match them.

---

## 6. What this teaches us before mechanizing

Things this hand-written target makes concrete and testable:

1. **Doubled atoms, doubled inserts.** Every join and every
   insertion grows by a factor of ~2. If this slows the planner
   noticeably even at this small scale, that's an early warning for
   §5.5 of the experiment plan. We can run this hand-written program
   *now*, today, against the existing backend and see.
2. **Naming convention.** `f_ts` parallel tables, `last_run_at_<src>`
   per source rule, `<src>@i` for variants. The encoder emits
   exactly these names; debugging is easier with stable conventions.
3. **Action mirroring is mechanical.** Every `(set (f args...) val)`
   gets a paired `(set (f_ts args...) (next_ts))`. Every `(delete
   (f args...))` gets a paired `(delete (f_ts args...))`. No
   conditional logic.
4. **Constructor calls don't need ts.** `(let v5 (Add a b))` is
   pure (doesn't insert into a tracked table from the encoder's
   perspective — the `Add` term table is not queried). Confirmed
   by D2.

---

## 7. Open questions still to validate

These are the remaining unknowns even with this target written
down:

1. **Will the free-join planner pick reasonable orders?** This
   target has, for example, `congruence_rule@1` joining four atoms
   instead of two. The planner must figure out that `AddView_ts c0
   c1 new ts1` with `(>= ts1 X)` is highly selective. Try it; if
   it picks badly, we file the planner-hint mitigation per
   experiment-plan §5.5.
2. **Does `:merge new` on `last_run_at_*` interact correctly with
   saturation detection?** If a `last_run_at` global is updated
   between iterations, that's a fact-base change. We need the
   schedule executor to update these globals via a path that
   doesn't count as "rule activity" for saturation purposes —
   otherwise `(saturate ...)` never terminates.
3. **What happens with `:merge new` on `UF_Mathf_ts`?** When
   `UF_Mathf` updates, the `ts` updates too. But what if a *rule*
   has already read the old `UF_Mathf_ts` value within the same
   iteration? The scan and the update can race. The standard answer
   is "rule firings see a snapshot per step"; need to confirm
   that's preserved.
4. **Initial-insert timing.** Bumping `next_ts` to 1 before initial
   inserts is a small but fiddly piece of executor logic. Worth
   getting right early.

If any of these look genuinely ambiguous after a first pass on the
encoder, we revisit and possibly amend D1–D7.

---

## 8. Result: this target runs.

The hand-written target was extracted into two files at the repo
root and exercised against the existing backend:

- [`seminaive-encoding-add-baseline.egg`](seminaive-encoding-add-baseline.egg)
  — the original 7-line program (sort, constructor, rule, run,
  check). Runs cleanly with and without `--naive`.
- [`seminaive-encoding-add.egg`](seminaive-encoding-add.egg) — the
  full seminaive-encoded form (parallel `_ts` tables, 14 rule
  variants, bookkeeping globals). Runs cleanly under `--naive` and
  the final `(check ...)` passes. Verified the check is genuine by
  flipping its arguments — that produces "Check failed".

### What this validates

- **D1 (parallel `_ts` tables)** composes with term-encoding view
  tables and UF tables without conflicts.
- **D3 (mirroring merge modes)** doesn't break anything: `:merge
  old` and `:merge new` `_ts` tables coexist with their bases.
- **D5 (Rust-side global updates)** isn't yet exercised (we hold
  `next_ts=1` and all `last_run_at=0` for the whole run), but the
  *read* path works — binding `(= now (next_ts))` in the body and
  using `now` in actions is the right idiom.
- **D6 (primitive atoms don't get variants)** worked: `(!= b c)`,
  `(= (ordering-max b c) b)`, `(guard ...)` are passed through
  unchanged in every variant.

One small surprise found while writing it: egglog rejects function
*lookups* of non-constructor functions in **action** position
(`LookupInRuleDisallowed` at `src/typechecking.rs:867`). So
`(set (UF_Math_ts a c) (next_ts))` in an action is rejected, but
the equivalent `(= now (next_ts))` in the body followed by
`(set (UF_Math_ts a c) now)` in the action is fine. The encoder
just needs to bind every `(next_ts)` read in the body and
substitute the binding in actions. Mechanical.

### What this does *not* validate

- Performance. This is one rule iteration on a trivial program. No
  conclusion about doubled-atom planner overhead until we run the
  encoder on a real benchmark.

---

## 9. Result: schedule-executor hook works.

The hook was implemented in `src/lib.rs`'s `step_rules` (the
function that runs one ruleset iteration). It does two things:

1. **Before** invoking the backend's `run_rules`, look up the
   nullary i64 function `next_ts`. If it exists, increment it by
   one. (No-op for programs that don't use the encoding.)
2. **After** the backend returns, for every rule that just ran,
   compute the source name (`rule_name.split('@').next()`) and
   set `last_run_at_<src>` to the bumped value if that function
   exists.

Helpers `bump_next_ts_global` and `update_last_run_at_globals`
encapsulate the work; the `step_rules` body picks up rule names
alongside ids and calls them around `backend.run_rules`. About 60
lines of new code, all behind name-based feature detection so the
hook is invisible to programs that don't define `next_ts`.

### Verification

[`seminaive-encoding-add.egg`](seminaive-encoding-add.egg) was
amended to start `next_ts` at 0 and to print the final values of
all bookkeeping globals via `(extract ...)`. Run under `--naive`:

```
$ egglog --naive seminaive-encoding-add.egg
25                              # next_ts (bumped 25 times across the run)
13                              # last_run_at_commutativity
22                              # last_run_at_uf_update
21                              # last_run_at_singleparentuf_update
23                              # last_run_at_uf_function_index_update
24                              # last_run_at_congruence_rule
24                              # last_run_at_rebuild_rule
25                              # last_run_at_delete_rule
25                              # last_run_at_delete_rule_subsume
```

Each value matches when its source rule's ruleset was last
invoked: `commutativity` ran during the user's `(run 1)` (which
came before the maintenance schedule's many iterations);
`uf_update`/`singleparentuf_update`/`uf_function_index_update` ran
during the maintenance saturate; `congruence_rule`/`rebuild_rule`
during the outer rebuilding ruleset; the delete rules during the
final `delete_subsume_ruleset` run. The check at the end of the
program still passes.

### What this validates

- The hook fires once per `step_rules` call (= once per ruleset
  iteration), not per rule.
- Per-source-rule attribution works: variants `commutativity@1`,
  `commutativity@2`, ... all roll up to `last_run_at_commutativity`.
- The seminaive predicates now actually filter — rules see only
  rows added since their source last ran. Correctness is preserved
  (the check still passes).
- The hook is name-based and inactive for programs without
  `next_ts`, so it doesn't disturb the existing test suite.

### What still remains

- The encoder itself. With D1–D7 confirmed and the hook in place,
  the remaining work is mechanical: walk the term-encoded program
  and emit the shape this hand-written file demonstrates.
- Benchmarking against the codspeed suite per
  [`seminaive-encoding-experiment.md`](seminaive-encoding-experiment.md)
  §6 to confirm the doubled-atom shape doesn't tank the planner's
  selectivity estimation.
