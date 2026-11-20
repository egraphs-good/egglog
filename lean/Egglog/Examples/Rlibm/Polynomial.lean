import Egglog.Examples.Rlibm.Coefficients

namespace Rlibm

abbrev Accumulator := Int → Int → Int → Int → Int → Int → Int → Int

/-- The LP's polynomial and exponent compensation share a Q40 scale. -/
def accum (e k t c0 c1 c2 c3 : Int) : Int :=
  e*k + (c0 + t*(c1 + t*(c2 + t*c3)))

/-- For positive finite inputs, x = 2^e * (1 + m/128).
    Normalize subnormals before selecting a polynomial segment. -/
def rangeReduce (raw : Nat) : Int × Nat :=
  let field := raw / 128 % 256
  let m := raw % 128
  if field = 0 then
    let shift := 7 - m.log2
    (-(126 + (shift : Int)), (m <<< shift) - 128)
  else ((field : Int) - 127, m)

def bucket (m : Nat) : Fin 8 := ⟨m / 16 % 8, Nat.mod_lt _ (by decide)⟩

def cubic (base : LogBase) (i : Fin 8) : Cubic := (coefficients base)[i.val]?.getD default

/-- Round a signed Q40 integer to bfloat16 with ties to even. Every nonzero
    Q40 value has magnitude at least 2^-40, within the normal exponent range. -/
def roundQ40 (z : Int) : Nat := Id.run do
  let n := z.natAbs
  if n == 0 then return 0
  let s := n.log2
  let cut := s - 7
  let unit := 2^cut
  let high := n / unit
  let rem := n % unit
  let significand := if s < 7 then n <<< (7-s) else
    high + (if 2*rem > unit || (2*rem == unit && high % 2 == 1) then 1 else 0)
  -- A significand of 256 carries into the adjacent exponent field.
  let bits := min 0x7f80 ((s + 86) * 128 + significand)
  return (if z < 0 then 0x8000 else 0) + bits

def reduced (fn : Accumulator) (b : LogBase) (raw : Nat) : Int :=
  let (e,m) := rangeReduce raw
  let c := cubic b (bucket m)
  fn e (exponentStep b) (m%16) c.c0 c.c1 c.c2 c.c3

def evalWith (fn : Accumulator) (b : LogBase) (raw : Nat) : Nat :=
  if raw == 0 || raw == 0x8000 then 0xff80
  else if raw == 0x7f80 then 0x7f80
  else if raw > 0x7f80 then 0xffff
  else roundQ40 (reduced fn b raw)

def core := evalWith accum

private theorem mulBound {a b : Int} {u v : Nat} (ha : a.natAbs ≤ u) (hb : b.natAbs ≤ v) :
    (a*b).natAbs ≤ u*v := by rw [Int.natAbs_mul]; exact Nat.mul_le_mul ha hb

private theorem addBound {a b : Int} {u v : Nat} (ha : a.natAbs ≤ u) (hb : b.natAbs ≤ v) :
    (a+b).natAbs ≤ u+v := Nat.le_trans (Int.natAbs_add_le a b) (Nat.add_le_add ha hb)

set_option maxRecDepth 4096 in
theorem exponentBound (raw : Nat) : (rangeReduce raw).1.natAbs ≤ 133 := by
  simp only [rangeReduce]
  split
  · have subnormal : ∀ i : Fin 8, (-(126 + (i.val : Int))).natAbs ≤ 133 := by decide
    exact subnormal ⟨_, by omega⟩
  · have normal : ∀ i : Fin 256, ((i.val : Int) - 127).natAbs ≤ 133 := by decide
    exact normal ⟨_, Nat.mod_lt _ (by decide)⟩

theorem accum_fits {t c0 c1 c2 c3 : Int} (b : LogBase) (raw : Nat)
    (ht : t.natAbs ≤ 15) (h0 : c0.natAbs ≤ 2^40) (h1 : c1.natAbs ≤ 2^40)
    (h2 : c2.natAbs ≤ 2^40) (h3 : c3.natAbs ≤ 2^40) :
    (accum (rangeReduce raw).1 (exponentStep b) t c0 c1 c2 c3).natAbs < 2^63 := by
  have hk : (exponentStep b).natAbs ≤ 2^40 := by cases b <;> decide
  exact Nat.lt_of_le_of_lt (addBound (mulBound (exponentBound raw) hk)
    (addBound h0 (mulBound ht (addBound h1 (mulBound ht (addBound h2 (mulBound ht h3))))))) (by decide)

/-- All extracted schedules may use unsigned word arithmetic. The final Q40
    accumulator fits a signed 64-bit word even before the bfloat16 rounding. -/
theorem reduced_fits (b : LogBase) (raw : Nat) : (reduced accum b raw).natAbs < 2^63 := by
  have hc : ∀ i : Fin 8, (cubic b i).c0.natAbs ≤ 2^40 ∧ (cubic b i).c1.natAbs ≤ 2^40 ∧
      (cubic b i).c2.natAbs ≤ 2^40 ∧ (cubic b i).c3.natAbs ≤ 2^40 := by cases b <;> decide
  obtain ⟨h0,h1,h2,h3⟩ := hc (bucket (rangeReduce raw).2)
  have ht : (((rangeReduce raw).2 % 16 : Nat) : Int).natAbs ≤ 15 := by
    simp only [Int.natAbs_natCast]; omega
  exact accum_fits b raw ht h0 h1 h2 h3

end Rlibm
