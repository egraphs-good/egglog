import Egglog.Examples.Rlibm.Polynomial
import Egglog.Examples.Rlibm.Spec
import Egglog.Examples.Rlibm.KernelPages

/-!
# Strict binary64 model of the published bfloat16 log2 and log10

The rounding of each separate operation is computed with integers. A finite
binary64 value is stored as sign, significand, and binary scale. Overflow or
division by zero returns `none`; the domain theorem checks that neither occurs.
The original bfloat16-to-float cast, `frexpf`, multiplication by two and
promotion to binary64 are exact for the normalized 8-bit significand.
-/
namespace Rlibm.Upstream

private structure F64 where
  negative : Bool
  sig : Nat
  scale : Int
  deriving Inhabited, DecidableEq

private def twice52 : Nat := 2^52

/-- The IEEE round-to-nearest-even choice between adjacent integers.
    At a midpoint the low bit of the lower candidate decides. -/
private def nearest (num den : Nat) : Nat :=
  let q := num / den
  let r := 2 * (num % den)
  if r > den || (r == den && q % 2 == 1) then q+1 else q

/-- Nearest rounding is at most one half unit from the exact rational.
    At either boundary the selected integer is even. -/
private theorem nearest_cell (n d : Nat) (hd : 0 < d) :
    let q := nearest n d
    2*(q*d) ≤ 2*n+d ∧ 2*n ≤ 2*(q*d)+d ∧
    ((2*(q*d) = 2*n+d ∨ 2*n = 2*(q*d)+d) → q % 2 = 0) := by
  have divmod := Nat.mod_add_div n d
  rw [Nat.mul_comm d (n/d)] at divmod
  have rem := Nat.mod_lt n hd
  simp only [nearest]
  split
  next h =>
    simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
    rw [Nat.add_mul]
    simp only [Nat.one_mul]
    omega
  next h =>
    simp only [Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
    omega

/-- Floor of log2(n/d) for positive n,d, using the integer significand. -/
private def binade (n d : Nat) : Int :=
  let guess : Int := (n.log2 : Int) - d.log2
  let below := if guess < 0 then (n <<< (-guess).toNat) < d else n < (d <<< guess.toNat)
  if below then guess-1 else guess

/-- Round the exact real (-1)^sign × n/d × 2^scale. For subnormals the
    last place is fixed at 2^-1074. A carry can advance the binade. -/
private def propose (negative : Bool) (n d : Nat) (scale : Int) : Option F64 := do
  guard (d != 0)
  if n == 0 then return ⟨negative, 0, 0⟩
  let last := max (-1074) (binade n d + scale - 52)
  let shift := scale - last
  let q := if shift >= 0 then nearest (n <<< shift.toNat) d else nearest n (d <<< (-shift).toNat)
  if q == 0 then return ⟨negative, 0, 0⟩
  let carry := q == 2 * twice52
  let e := if carry then last+1 else last
  guard (decide (e ≤ 971))
  return ⟨negative, if carry then twice52 else q, e⟩

/-- Compare n/d × 2^scale with a boundary h × 2^unit using positive
    integer scaling. The returned pair has the same order as its arguments. -/
private def scaledPair (n d : Nat) (scale : Int) (h : Nat) (unit : Int) : Nat × Nat :=
  if scale ≥ unit then (n <<< (scale-unit).toNat, d*h) else (n, (d*h) <<< (unit-scale).toNat)

/-- Exact midpoint criterion for a finite IEEE binary64 candidate. Adjacent
    significands are spaced by one unit. At a power of two the lower neighbor
    is half a unit away, except at the normal/subnormal boundary. In units
    of 2^(place-2), this gives midpoint radii of one or two. -/
private def ieeeCell (sign : Bool) (n d : Nat) (scale : Int) (x : F64) : Prop :=
  let place := if x.sig == 0 then -1074 else x.scale
  let lowerRadius := if x.sig == twice52 && place > -1074 then 1 else 2
  let low := scaledPair n d scale (4*x.sig - lowerRadius) (place-2)
  let high := scaledPair n d scale (4*x.sig + 2) (place-2)
  let even := x.sig % 2 = 0
  d > 0 ∧ x.negative = sign ∧
    (if x.sig == 0 then x.scale = 0 else
      -1074 ≤ x.scale ∧ x.scale ≤ 971 ∧ x.sig < 2*twice52 ∧ (x.scale = -1074 ∨ twice52 ≤ x.sig)) ∧
    (low.2 < low.1 ∨ (low.2 = low.1 ∧ even)) ∧ (high.1 < high.2 ∨ (high.1 = high.2 ∧ even))

private instance {sign n d scale x} : Decidable (ieeeCell sign n d scale x) := by
  unfold ieeeCell; infer_instance

/-- No exponent or rounding algorithm is trusted to select the IEEE result.
    Check the exact input against both midpoints before using a candidate. -/
private def round64 (sign : Bool) (n d : Nat) (scale : Int) : Option F64 := do
  let x ← propose sign n d scale
  if ieeeCell sign n d scale x then some x else none

private theorem round64_sound {sign n d scale x} (h : round64 sign n d scale = some x) :
    ieeeCell sign n d scale x := by
  unfold round64 at h
  cases hx : propose sign n d scale with
  | none => simp [hx] at h
  | some result =>
    by_cases ok : ieeeCell sign n d scale result
    · simp [hx, ok] at h
      exact h ▸ ok
    · simp [hx, ok] at h

private def F64.signed (x : F64) : Int := if x.negative then -(x.sig : Int) else x.sig

private def F64.plus (x y : F64) : Option F64 :=
  let e := min x.scale y.scale
  let z := (x.signed <<< (x.scale - e).toNat) + (y.signed <<< (y.scale - e).toNat)
  round64 (if z == 0 then x.negative && y.negative else decide (z < 0)) z.natAbs 1 e

private def F64.times (x y : F64) : Option F64 :=
  round64 (x.negative != y.negative) (x.sig * y.sig) 1 (x.scale + y.scale)

private def F64.divide (x y : F64) : Option F64 :=
  round64 (x.negative != y.negative) x.sig y.sig (x.scale - y.scale)

/-- All literals below are finite binary64 words. -/
private def fromBits (u : Nat) : F64 :=
  let field := (u >>> 52) % 2048
  let sig := u % twice52 + (if field == 0 then 0 else twice52)
  ⟨u >= 2^63, sig, if sig == 0 then 0 else (max 1 field : Nat) - (1075 : Int)⟩

private def F64.bits (x : F64) : Nat :=
  let sign := if x.negative then 2^63 else 0
  if x.sig < twice52 then sign + x.sig
  else sign + ((x.scale + 1075).toNat <<< 52) + (x.sig - twice52)

/-- RoundDoubleToBFloat16 in upstream include/bfloat16.hpp. For normals the
    retained, guard, and sticky bits are 45, 44, and the low 44, respectively. -/
private def cast (u : Nat) : Nat := Id.run do
  if 0x47eff00000000000 ≤ u && u ≤ 0x7ff0000000000000 then return 0x7f80
  if 0xc7eff00000000000 ≤ u && u ≤ 0xfff0000000000000 then return 0xff80
  let field := (u >>> 52) % 2048
  if field == 2047 then return u >>> 48
  let e : Int := (field : Int) - 1023
  let sign := (u >>> 48) &&& 0x8000
  if e ≤ -135 then return sign
  let frac := u % twice52
  if e == -134 then return sign + (if frac != 0 then 1 else 0)
  if e == -133 then return sign + nearest (twice52 + frac) twice52
  if e ≤ -127 then return sign + nearest (twice52 + frac) (2^((-e)-81).toNat)
  return sign + (e + 126).toNat * 128 + nearest (twice52 + frac) (2^45)

/-- Published decimal literals written as their nearest binary64 bit patterns,
    in execution order (highest degree first). -/
private def constants : LogBase → Nat × Nat × Nat
  | .two => (0x3fe76233ae28706e, 0x3fee53e1a6a150b6, 0x400715f779f69699)
  | .ten => (0x3fe5799d41a9dfe2, 0x3fee9b86270e4e2c, 0x4007156bbfe58ba6)

/-- The normalized input has value (128 + m) × 2^-7. Each call to
    plus/times/divide below rounds separately; there is no fused multiply-add. -/
private def fraction (base : LogBase) (m : Nat) : Option F64 := do
  let dx : F64 := ⟨false, 128+m, -7⟩
  let numerator ← dx.plus ⟨true, 1, 0⟩
  let denominator ← dx.plus ⟨false, 1, 0⟩
  let t ← numerator.divide denominator
  let t2 ← t.times t
  let (a,b,c) := constants base
  let v ← (fromBits a).times t2
  let v ← v.plus (fromBits b)
  let v ← v.times t2
  let v ← v.plus (fromBits c)
  v.times t

/-- The verifier evaluates each of the 128 mantissa polynomials once per base.
    The elaborator proposes their binary64 words; the bounded theorem below
    checks each entry against the unreduced strict Horner implementation. -/
elab "fraction_words% " radix:num : term => do
  let base : LogBase ← match radix.getNat with
    | 2 => pure .two
    | 10 => pure .ten
    | _ => throwErrorAt radix "expected logarithm base 2 or 10"
  return Lean.toExpr ((Array.range 128).map fun i => ((fraction base i).map F64.bits).getD 0)

private def fracWords : LogBase → Array Nat
  | .two => fraction_words% 2
  | .ten => fraction_words% 10

private def cached (base : LogBase) (m : Nat) : Option F64 :=
  if m < 128 then (fracWords base)[m]?.map fromBits else fraction base m

private def entryOK (base : LogBase) (m : Nat) : Bool := decide (cached base m = fraction base m)
set_option maxRecDepth 8000 in
set_option maxHeartbeats 0 in
private theorem valuesTwo : KernelPages.checkSpan (entryOK .two) 0 7 = true := by decide +kernel
set_option maxRecDepth 8000 in
set_option maxHeartbeats 0 in
private theorem valuesTen : KernelPages.checkSpan (entryOK .ten) 0 7 = true := by decide +kernel

private theorem cached_eq (base : LogBase) (m : Nat) : cached base m = fraction base m := by
  by_cases hm : m < 128
  · have valid := KernelPages.span_sound (test := entryOK base) (start := 0) 7
        (by cases base with | two => exact valuesTwo | ten => exact valuesTen) (Nat.zero_le m) hm
    exact of_decide_eq_true valid
  · simp [cached, hm]

private def finish (base : LogBase) (e : Int) (f : F64) : Option Nat := do
  let y ← f.plus ⟨e < 0, e.natAbs, 0⟩
  let z ← match base with
    | .two => some y
    | .ten => y.divide (fromBits 0x400a934f0979a371)
  return cast z.bits

/-- The exponent and 7-bit fraction returned by rangeReduce give exactly the
    decoded bfloat16 input, so (128+m)/256 and e+1 satisfy the frexpf contract.
    The integer equality expresses the two dyadics in a common power-of-two unit. -/
def FrexpContract (raw : Nat) : Prop :=
  let d := decode raw
  let m := (rangeReduce raw).2
  let s := (rangeReduce raw).1 - 7
  let unit := min d.exponent s
  m < 128 ∧ d.mantissa * 2^(d.exponent - unit).toNat = ((128 + m : Nat) : Int) * 2^(s - unit).toNat

instance (raw : Nat) : Decidable (FrexpContract raw) := by unfold FrexpContract; infer_instance

/-- Model of the strict C++ log routine on the raw 16-bit input encoding.
    On finite positive words, rangeReduce returns the exact exponent and
    the seven fraction bits of twice the mantissa returned by frexpf. -/
private def runWith (poly : LogBase → Nat → Option F64) (base : LogBase) (raw : Nat) : Option Nat := do
  if raw == 0 || raw == 0x8000 then return 0xff80
  if raw == 0x7f80 then return 0x7f80
  if raw > 0x7f80 then return 0xffff
  let (e,m) := rangeReduce raw
  finish base e (← poly base m)

def implementation := runWith fraction

/-- An execution with a small table certified against every mantissa value.
    It serves only to amortize the kernel replay over repeated exponents. -/
def replay := runWith cached

theorem replay_eq : replay = implementation := by
  have h : cached = fraction := funext fun b => funext (cached_eq b)
  simp only [replay, implementation, h]

#guard implementation .two 0x0001 == some 0xc305
#guard implementation .ten 0x3f80 == some 0
#guard implementation .ten 0x4120 == some 0x3f80
#guard implementation .two 0xffff == some 0xffff

end Rlibm.Upstream
