import Egglog.Examples.Rlibm.Types

/-!
# Rounding cells for bfloat16 logarithms

The specification is independent of the polynomial, its coefficients, and the
range reduction. To compare log_b(N/D) to a rational midpoint p/q, compare
N^q * b^max(-p,0) and D^q * b^max(p,0). This clears the denominators in
(N/D)^q versus b^p, using the strict monotonicity of the logarithm for b > 1.
Both sides are natural numbers. Midpoint ties belong to the even encoding.
-/

namespace Rlibm

def LogBase.radix : LogBase → Nat
  | .two => 2
  | .ten => 10

structure Dyadic where
  mantissa : Int
  exponent : Int
  deriving Repr, DecidableEq

def Dyadic.numerator (v : Dyadic) : Int := v.mantissa * (2 : Int) ^ v.exponent.toNat
def Dyadic.denominator (v : Dyadic) : Nat := 2 ^ (-v.exponent).toNat

/-- Decode a finite bfloat16 word, including subnormals, as mantissa * 2^exponent. -/
def decode (bits : Nat) : Dyadic :=
  let magnitude := bits % 0x8000
  let field := magnitude / 128
  let m : Int := (if field == 0 then 0 else 128) + (magnitude % 128 : Nat)
  ⟨if bits < 0x8000 then m else -m, (max 1 field : Nat) - (134 : Int)⟩

/-- Lower and upper neighbors in numerical order; callers use nonzero finite words
    whose two neighbors are finite. The sign bit reverses the encoding order. -/
def below (y : Nat) : Nat := if y < 0x8000 then y - 1 else y + 1
def above (y : Nat) : Nat := if y < 0x8000 then y + 1 else y - 1

def midpoint (x y : Dyadic) : Dyadic :=
  let e := min x.exponent y.exponent
  ⟨x.mantissa * 2 ^ (x.exponent - e).toNat + y.mantissa * 2 ^ (y.exponent - e).toNat, e-1⟩

def lowerMid (y : Nat) : Dyadic := midpoint (decode (below y)) (decode y)
def upperMid (y : Nat) : Dyadic := midpoint (decode y) (decode (above y))

/-- For positive x, the ratio of the two integers is x^q / b^p,
    where the proposed rounding boundary is p/q. -/
def powerSides (b : LogBase) (x cut : Dyadic) : Nat × Nat :=
  let n := x.numerator.natAbs
  let d := x.denominator
  let p := cut.numerator
  let q := cut.denominator
  (n ^ q * b.radix ^ (-p).toNat, d ^ q * b.radix ^ p.toNat)

/-- Compare the exact integers, allowing equality precisely when the low
    significand bit of the proposed result is zero. -/
def belowOrTie (even : Bool) (a b : Nat) : Bool :=
  if even then decide (a ≤ b) else decide (a < b)

/-- The upstream bfloat16 library's deterministic special-value convention.
    log(1) is +0 in round-to-nearest mode. -/
def specialLog? (x : Nat) : Option Nat :=
  if x == 0 || x == 0x8000 then some 0xff80
  else if x == 0x7f80 then some 0x7f80
  else if x > 0x7f80 then some 0xffff
  else if x == 0x3f80 then some 0
  else none

/-- The polynomial's output must lie in the mathematical log rounding cell.
    Values whose adjacent encodings are infinite cannot be correct for the
    finite bfloat16 logarithm. -/
def CorrectLog (base : LogBase) (x y : Nat) : Prop :=
  x < 65536 ∧ match specialLog? x with
  | some result => y = result
  | none =>
    y < 65536 ∧ 0 < y % 0x8000 ∧ y % 0x8000 < 0x7f7f ∧
    let lo := powerSides base (decode x) (lowerMid y)
    let hi := powerSides base (decode x) (upperMid y)
    if y % 2 = 0 then lo.2 ≤ lo.1 ∧ hi.1 ≤ hi.2 else lo.2 < lo.1 ∧ hi.1 < hi.2

/-- The preliminary size checks bound the work even on an invalid candidate.
    They only reject. Every acceptance must still pass both exact inequalities. -/
def checkLog (base : LogBase) (x y : Nat) : Bool :=
  if x ≥ 65536 then false else
  match specialLog? x with
  | some result => y == result
  | none =>
    let mag := y % 0x8000
    if y ≥ 65536 || mag < 0x3a80 || mag > 0x4380 then false else
    let lo := powerSides base (decode x) (lowerMid y)
    let hi := powerSides base (decode x) (upperMid y)
    let even := y % 2 == 0
    belowOrTie even lo.2 lo.1 && belowOrTie even hi.1 hi.2

theorem checkLog_sound {b x y} (ok : checkLog b x y = true) : CorrectLog b x y := by
  unfold checkLog at ok
  split at ok
  · contradiction
  rename_i limit
  refine ⟨by omega, ?_⟩
  cases h : specialLog? x with
  | some result =>
    simp [h] at ok ⊢
    exact ok
  | none =>
    simp only [h] at ok ⊢
    split at ok
    · contradiction
    rename_i small
    have bounds : y < 65536 ∧ 0x3a80 ≤ y % 0x8000 ∧ y % 0x8000 ≤ 0x4380 := by
      simpa [Bool.or_eq_true, not_or, and_assoc] using small
    refine ⟨bounds.1, by omega, by omega, ?_⟩
    by_cases parity : y % 2 = 0 <;> simpa [belowOrTie, parity] using ok

end Rlibm
