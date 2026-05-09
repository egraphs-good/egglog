# souffle_compat strata: buffer/canon/snap split

Status: design doc, implementation pending.

This is the encoder-side counterpart to the Souffle fork's `.snapshot` and
named-strata support. The goal is to make the souffle_compat-encoded form
**actually preserve `(run N)` semantics** when run through Souffle.

## The problem

Today's `--souffle-compat` produces an encoded form where user rules and
rebuild rules read and write the same view + UF tables. Souffle's SCC
analysis (which is data-dependency-driven) lumps them all into one SCC.

Consequences when running through Souffle:
1. **`(run N)` is not faithful.** Bounding the SCC at N iterations bounds
   user rules AND rebuild rules together. Egglog's semantics is "fire user
   rules N times, with rebuild saturating between each firing" — that
   requires user rules in a separate stratum from rebuild.
2. **Less rebuild work per outer iteration.** User rules see non-canonical
   intermediate state, may fire on redundant matches, may miss canonical
   matches that rebuild hasn't computed yet.
3. **Programs with intentional `(run N)` cutoff produce different results.**

## The fix: split each rebuilt relation into buffer + canon + snap

For each constructor `C` and each rebuilt function `F`:

```
__C_buffer    — written by user rules (where they used to write __CView)
__C_canon     — written by rebuild rules (drains the buffer + canonicalizes)
__C_canon_snap — read by user rules (snapshot of __C_canon refreshed at
                 outer-loop boundaries; declared via .snapshot directive)
```

User rules:
- READ `__C_canon_snap` (instead of `__CView`)
- WRITE `__C_buffer` (instead of `__CView`)

Rebuild rules:
- READ both `__C_canon` (existing canonical state) and `__C_buffer`
  (pending writes from user)
- WRITE `__C_canon` (drains buffer + canonicalizes existing rows)

Souffle's SCC analysis then puts user rules and rebuild rules in separate
SCCs (they don't share writes; the canon→snap edge is runtime-only via
`.snapshot`, not a rule). With `--pragma "outer-saturate"`, the outer loop
iterates the two SCCs alternately.

`(run N)` becomes `.limititerations <user_rule_relation>(n=N)` which
bounds the user-rule SCC's inner loop.

## Why this matches egglog's `(seq (run user) (saturate rebuild))`

| Egglog step | Souffle equivalent |
|---|---|
| `(run user)` fires user rules once | User SCC iterates, bounded by `.limititerations`. With N=1 it fires once. |
| `(saturate rebuild)` runs rebuild to fixpoint | Rebuild SCC iterates without bound, naturally to fixpoint via Souffle's semi-naive. |
| Outer schedule loops | `.pragma "outer-saturate"` wraps the whole thing. |
| Snapshots refresh between outer iterations | `.snapshot R_snap = R` directive. |

## Which relations need splitting?

In the current souffle_compat encoded form, these are written by both user
and rebuild rules:

1. **`__CView`** for each constructor `C` (view tables) — clear case for
   the split.
2. **`__UF_<Sort>`** — user rules write union edges; rebuild's path
   compression and single_parent rules read+write.
3. **`__<Sort>Proof`** (proof mode) — user rules set proofs; rebuild reads
   them.
4. **`__to_delete_<C>` / `__to_subsume_<C>`** — these are USER-written
   markers; rebuild "drains" them by deleting the marked rows. With our
   live-view drain pattern in the translator, the helper relations stay
   simple — no split needed.

## Implementation phases

1. **Flag plumbing.** Add `EncodingState::souffle_compat_strata` (extends
   `souffle_compat`). New CLI flag `--souffle-compat-strata`. Default off;
   tests stay green; downstream stages (translator) consume the flag.
2. **View table split.** In `term_and_view`, emit both `__CView_buffer`
   and `__CView_canon`. Update `delete_and_subsume` to point at the
   correct one. Update `handle_congruence` and `rebuild_rule` to read
   buffer + canon and write canon. Add `.snapshot` directive emission.
3. **User rule rewrite.** When `souffle_compat_strata` is on, walk user
   rules' bodies and actions: rewrite reads of `__CView` to
   `__CView_canon_snap`; rewrite writes to `__CView_buffer`.
4. **UF + proof split.** Same pattern for `__UF_<Sort>` and
   `__<Sort>Proof`.
5. **Translator integration.** When the translator sees a `(run N)`,
   emit `.limititerations` on the corresponding user-rule SCC. Currently
   we just set `outer-saturate=100` for any RunSchedule; with strata we
   can do better.
6. **End-to-end test.** Source program with `(run 1)` round-trips through
   the pipeline and matches native egglog's output.

## Estimated effort

~200-400 lines of egglog-side changes plus translator updates. Two-three
days of focused work.

The Souffle fork already supports everything we need on the engine side
(`.snapshot`, `.limititerations`, `.pragma "outer-saturate"`).

## Decision point

Without this work:
- `--souffle-compat` is correct for programs that don't depend on bounded
  iteration (lumped fixpoint reaches the same answer, just slower).
- Programs with semantic `(run N)` cutoff get different answers in
  Souffle vs native egglog.
- Snapshots in `tests/files.rs` will not match for any program where
  `(run N)` matters.

With this work:
- Full `(run N)` semantic fidelity.
- Schedule expressions in egglog (`(seq ... (saturate ...))`) map to
  Souffle's outer-saturate properly.
- `tests/files.rs` snapshots have a chance of matching after we also add
  output normalization (separate concern).
