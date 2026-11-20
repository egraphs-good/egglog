import Egglog.DSL

/-!
# Eliminating a bounds check with a relational compiler pass

An index, a length and a run-time bounds check belong to the source language.
The bounds analysis records an upper bound in a relation and a strict lower
bound in a partial function. Writes to the same key merge using maximum.
A rule joining the check with both tables proves the check true. The only
input to `optimize` is a source AST.

Run from the repository root.

```sh
(cd lean && lake build Egglog.Examples.BoundsCheck)
```
-/

open Egglog Egglog.Compiler

namespace BoundsCheck

private def truth (p : Prop) [Decidable p] : Nat := if p then 1 else 0

egg_language Expr using Ops : Nat with_vars reg where
  | maximum x y => max x y
  | next x => x + 1
  | copy x => x
  | yes => 1
  | inBounds i n => truth (i < n)
  otherwise 0

egg_relations R : Nat where
  | atMost i k => i ≤ k

-- The value v of lower(n) carries the erased proposition v < n.
egg_functions F : Nat over Ops.model where
  | lower n ↦ v => v < n
    merge old new => Ops.maximum old new
    preserving ho hn by exact Nat.max_lt.mpr ⟨ho, hn⟩

def ir := Expr.language (F.model (R.model Ops.model))

open Ops

def maxLeft : RelRule ir.model :=
  egg_rule (x, y, k) : when (h : k = maximum x y) yield R.atMost x k by
    change k = max x y at h
    change x ≤ k
    exact h.symm ▸ Nat.le_max_left x y

def maxRight : RelRule ir.model :=
  egg_rule (x, y, k) : when (h : k = maximum x y) yield R.atMost y k by
    change k = max x y at h
    change y ≤ k
    exact h.symm ▸ Nat.le_max_right x y

/-- A length has two strict lower bounds. Both writes share the same
    constructor match; their merge may construct a new value term. -/
def lowerNextMax : RelRule ir.model :=
  egg_rule (n, x, y) : when (h : n = next (maximum x y)) do {
    set F.lower n := x by
      change n = max x y + 1 at h
      change x < n
      exact h.symm ▸ Nat.lt_succ_of_le (Nat.le_max_left x y),
    set F.lower n := y by
      change n = max x y + 1 at h
      change y < n
      exact h.symm ▸ Nat.lt_succ_of_le (Nat.le_max_right x y)
  }

def checkFromLower : RelRule ir.model :=
  egg_rule (i, k, n) : when (upper : R.atMost i k) (lower : F.lower n ↦ k)
    rewrite inBounds i n ==> yes by
    change truth (i < n) = 1
    exact if_pos (Nat.lt_of_le_of_lt (show i ≤ k from upper) (show k < n from lower))

def copyValue : VerifiedRule ir.model :=
  egg_rule (x) : copy x ==> x by rfl

def phases : Certified.Schedule ir.model := egg_schedule {
  relational #[maxLeft, maxRight, lowerNextMax], run #[copyValue],
  relational #[lowerNextMax], relational #[checkFromLower]
}

def optimize (input : Expr) : Expr := (ir.optimize phases input Ops.cost).output

theorem optimize_correct (input : Expr) (regs : Nat → Nat) :
    (optimize input).eval regs = input.eval regs :=
  (ir.optimize phases input Ops.cost).sound regs

private def i := Expr.reg 0
private def k := Expr.maximum i (.reg 1)
private def check := Expr.inBounds i (.next (.copy k))

-- Copy propagation makes the argument of next match the maximum pattern.
-- Only the merged value of lower connects this length with the atMost row.
#guard (ir.optimize (egg_schedule {
  relational #[maxLeft, maxRight, lowerNextMax], relational #[checkFromLower]
}) check Ops.cost).output == check
#guard optimize check == .yes
#guard (ir.optimize (egg_schedule { run #[copyValue], relational #[maxLeft, maxRight],
  relational #[checkFromLower] }) check Ops.cost).output != .yes

-- A weak bound alone does not justify a strict check. The case i = k can occur.
#guard optimize (.inBounds i k) == .inBounds i k

-- Two valid facts with distinct witnesses must not fold this inner check.
-- The right child ensures that the table also contains the fact (i, k).
private def other := Expr.maximum (.reg 2) (.reg 3)
private def disjoint := Expr.maximum (.inBounds i (.next other)) k
#guard optimize disjoint == disjoint

end BoundsCheck
