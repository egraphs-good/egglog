import Egglog.Examples.Rlibm.KernelPages
import Egglog.Examples.Rlibm.Polynomial
import Egglog.Examples.Rlibm.Spec

namespace Rlibm.Cells

def quadratic (b : LogBase) (m : Nat) : Quadratic :=
  (quadratics b)[m / 8 % 16]?.getD default

def narrowValue (b : LogBase) (raw : Nat) : Int :=
  let (e, m) := rangeReduce raw
  let c := quadratic b m
  let t : Int := m % 8
  e * exponentStep b + (c.c0 + t*(c.c1 + t*c.c2))

def narrow (b : LogBase) (raw : Nat) : Nat :=
  if raw == 0 || raw == 0x8000 then 0xff80
  else if raw == 0x7f80 then 0x7f80
  else if raw > 0x7f80 then 0xffff
  else roundQ40 (narrowValue b raw)

/-- Test a candidate Q40 value against the rational endpoints of a rounding
    cell, crossing denominators exactly. This predicate is linear in `z`.
    Zero is the sole Q40 value that rounds to bfloat16 zero. -/
def inCell (bits : Nat) (z : Int) : Bool :=
  if bits == 0 then z == 0
  else
    let lo := lowerMid bits
    let hi := upperMid bits
    let left := z * lo.denominator - lo.numerator * 2^40
    let right := hi.numerator * 2^40 - z * hi.denominator
    if bits % 2 == 0 then left ≥ 0 && right ≥ 0 else left > 0 && right > 0

/-- For fixed e,t these are precisely linear forms in the unknown LP
    coefficients. Horner evaluation is justified by integer distributivity. -/
def linearForm (e k t c0 c1 c2 c3 : Int) : Int :=
  e*k + (c0 + (t*c1 + (t*t*c2 + t*t*t*c3)))

theorem linear_eq (e k t c0 c1 c2 c3 : Int) :
    accum e k t c0 c1 c2 c3 = linearForm e k t c0 c1 c2 c3 := by
  simp only [accum, linearForm, Int.mul_add, Int.mul_assoc]

/-- For each finite input certify the actual rounded outputs and the two
    LP rows independently. The cubic is certified against log in Certificate. -/
def testWord (b : LogBase) (x : Nat) : Bool :=
  let y := core b x
  narrow b x == y && (if 0 < x && x < 0x7f80 then
    inCell y (reduced accum b x) && inCell y (narrowValue b x) else true)

set_option Elab.async false
set_option maxRecDepth 8000
set_option maxHeartbeats 0

certify_bfloat16 (testWord .two) as two
certify_bfloat16 (testWord .ten) as ten

private theorem certified (b : LogBase) (x : Fin 65536) : testWord b x = true := by
  cases b
  · exact two x
  · exact ten x

theorem narrow_eq (b : LogBase) (x : Fin 65536) : narrow b x = core b x :=
  eq_of_beq (Bool.and_eq_true_iff.mp (certified b x)).1

theorem feasible (b : LogBase) (x : Fin 65536) (pos : 0 < x.val) (finite : x.val < 0x7f80) :
    inCell (core b x) (reduced accum b x) = true ∧ inCell (core b x) (narrowValue b x) = true := by
  have h := (Bool.and_eq_true_iff.mp (certified b x)).2
  simpa [pos, finite] using h

end Rlibm.Cells
