import Egglog.Compiler

/-!
# Eliminating a bounds check with a three-way join

The raw egglog version in `lean/tests/bounds-check.egg` joins the constructor
row `InBounds(i, n)` with the relations `at-most(i, k)` and `below(k, n)`.
Its range analysis inserts relation rows for maxima and successors. Here the
range facts are equalities to `yes` with a meaning in the natural-number
model. Proving the range-analysis rules establishes those equalities for every
register assignment. The two equality premises and the bounds-check pattern
then share a substitution, including the existentially matched `k`.

These commands run both examples from the repository root.

```sh
(cd lean && lake build lean-egglog Egglog.Compiler && lake env lean Egglog/Examples/BoundsCheck.lean)
lean/.lake/build/bin/lean-egglog --raw lean/tests/bounds-check.egg
```
-/

open Egglog Egglog.Compiler

namespace BoundsCheck

private def truth (p : Prop) [Decidable p] : Nat := if p then 1 else 0

private theorem truth_eq_one {p : Prop} [Decidable p] : truth p = 1 ↔ p := by
  by_cases h : p <;> simp [truth, h]

egg_signature Ops : Nat where
  | maximum x y => max x y
  | next x => x + 1
  | copy x => x
  | yes => 1
  | atMost i k => truth (i ≤ k)
  | below k n => truth (k < n)
  | inBounds i n => truth (i < n)
  otherwise 0

open Ops

/-- The middle value `k` is bound by the two range facts, independently of
    the bounds-check node. All three matches must agree on their shared keys. -/
def eliminateCheck : GuardedRule model :=
  egg_rule (i, k, n) : inBounds i n ==> yes
    when (lo : atMost i k = yes) (hi : below k n = yes) by
    exact truth_eq_one.mpr (Nat.lt_of_le_of_lt (truth_eq_one.mp lo) (truth_eq_one.mp hi))

private def maxLeft : VerifiedRule model :=
  egg_rule (x, y) : atMost x (maximum x y) ==> yes by
    exact truth_eq_one.mpr (Nat.le_max_left x y)

private def maxRight : VerifiedRule model :=
  egg_rule (x, y) : atMost y (maximum x y) ==> yes by
    exact truth_eq_one.mpr (Nat.le_max_right x y)

private def belowNext : VerifiedRule model :=
  egg_rule (x) : below x (next x) ==> yes by exact truth_eq_one.mpr (Nat.lt_succ_self x)

private def copyValue : VerifiedRule model :=
  egg_rule (x) : copy x ==> x by rfl

/-- Seed a bounds check and separate range-analysis roots in the same graph.
    The result is a proof for every assignment to the symbolic registers. -/
def proveInBounds? (i n : Pat) (facts : List Pat) (propagateCopies : Bool) :
    Option (PLift (∀ regs, i.eval model regs < n.eval model regs)) := do
  let (g, check) := (Certified.Session.empty model).add (inBounds i n)
  let (g, target) := g.add yes
  let g := facts.foldl (fun g fact => (g.add fact).1) g
  let g := g.run (egg_schedule { run #[maxLeft, maxRight, belowNext], guarded #[eliminateCheck] })
  let g := if propagateCopies then g.run (egg_schedule { run #[copyValue], guarded #[eliminateCheck] }) else g
  let ⟨same⟩ ← g.equivalent? check target
  return ⟨fun regs => truth_eq_one.mp (same regs)⟩

private def i := Pat.var 0
private def k := maximum i (.var 1)
private def n := next (copy k)
private def facts := [atMost i k, below (copy k) n]

#guard (proveInBounds? i n facts true).isSome
#guard (proveInBounds? i n facts false).isNone
#guard (proveInBounds? i n [atMost i k] true).isNone
#guard (proveInBounds? i n [below (copy k) n] true).isNone

-- Both premises are individually true. The second uses a different middle
-- register, so it cannot justify a check involving i and that register's length.
private def other := maximum (.var 2) (.var 3)
#guard (proveInBounds? i (next other) [atMost i k, below other (next other)] true).isNone

end BoundsCheck
