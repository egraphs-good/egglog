import Egglog.Examples.Rlibm.Certificate
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Tactic

namespace Rlibm

noncomputable def Dyadic.real (x : Dyadic) : ℝ := x.numerator / x.denominator

private theorem Dyadic.real_eq (x : Dyadic) :
    x.real = x.mantissa * (2 : ℝ) ^ x.exponent := by
  cases h : x.exponent with
  | ofNat n => simp [Dyadic.real, Dyadic.numerator, Dyadic.denominator, h]
  | negSucc n =>
    simp [Dyadic.real, Dyadic.numerator, Dyadic.denominator, h, zpow_negSucc, div_eq_mul_inv]

theorem midpoint_real (x y : Dyadic) : (midpoint x y).real = (x.real + y.real) / 2 := by
  let e := min x.exponent y.exponent
  have hx : (↑(x.exponent - e).toNat : ℤ) = x.exponent - e := by
    exact Int.toNat_of_nonneg (sub_nonneg.mpr (min_le_left ..))
  have hy : (↑(y.exponent - e).toNat : ℤ) = y.exponent - e := by
    exact Int.toNat_of_nonneg (sub_nonneg.mpr (min_le_right ..))
  rw [Dyadic.real_eq, Dyadic.real_eq, Dyadic.real_eq]
  change (↑(x.mantissa * 2 ^ (x.exponent - e).toNat + y.mantissa * 2 ^ (y.exponent - e).toNat) : ℝ) *
    2 ^ (e-1) = _
  push_cast
  rw [← zpow_natCast (2 : ℝ) (x.exponent - e).toNat, hx,
      ← zpow_natCast (2 : ℝ) (y.exponent - e).toNat, hy]
  rw [zpow_sub₀ (by norm_num : (2:ℝ) ≠ 0), zpow_sub₀ (by norm_num : (2:ℝ) ≠ 0),
      zpow_sub₀ (by norm_num : (2:ℝ) ≠ 0)]
  have he : (2:ℝ) ^ e ≠ 0 := zpow_ne_zero e (by norm_num)
  norm_num
  field_simp

/-- The logarithm of the exact ratio checked by the kernel. -/
private theorem logSides (b : LogBase) (x c : Dyadic) (hx : 0 < x.numerator) :
    Real.log (powerSides b x c).1 - Real.log (powerSides b x c).2 =
      c.denominator * Real.log x.real - c.numerator * Real.log b.radix := by
  have hr : (0 : ℝ) < b.radix := by cases b <;> norm_num [LogBase.radix]
  have hd : (0 : ℝ) < x.denominator := by unfold Dyadic.denominator; positivity
  have hn : (0 : ℝ) < x.numerator := by exact_mod_cast hx
  have absn : (↑x.numerator.natAbs : ℝ) = x.numerator := by
    have eq : ((x.numerator.natAbs : ℕ) : ℤ) = x.numerator := by omega
    calc
      (↑x.numerator.natAbs : ℝ) = (((x.numerator.natAbs : ℕ) : ℤ) : ℝ) := by simp
      _ = _ := congrArg (fun z : ℤ => (z : ℝ)) eq
  have split : (↑c.numerator.toNat : ℝ) - ↑(-c.numerator).toNat = c.numerator := by
    exact_mod_cast (show (c.numerator.toNat : ℤ) - (-c.numerator).toNat = c.numerator by omega)
  simp only [powerSides, Nat.cast_mul, Nat.cast_pow, absn, Dyadic.real, Real.log_div hn.ne' hd.ne']
  rw [Real.log_mul (by positivity) (by positivity), Real.log_mul (by positivity) (by positivity)]
  simp only [Real.log_pow]
  linear_combination -(Real.log b.radix) * split

/-- Both strict and weak comparisons, including the reverse orientation, follow
    from the sign of the difference of the two exact integer logarithms. -/
theorem compare_realLog (b : LogBase) (x c : Dyadic) (hx : 0 < x.numerator) :
    let sides := powerSides b x c
    (sides.1 ≤ sides.2 ↔ Real.logb b.radix x.real ≤ c.real) ∧
    (sides.2 ≤ sides.1 ↔ c.real ≤ Real.logb b.radix x.real) := by
  have hb : 1 < (b.radix : ℝ) := by cases b <;> norm_num [LogBase.radix]
  have hlog := Real.log_pos hb
  have hc : (0 : ℝ) < c.denominator := by unfold Dyadic.denominator; positivity
  have apos : 0 < x.numerator.natAbs := Int.natAbs_pos.mpr hx.ne'
  have bpos : 0 < b.radix := by cases b <;> decide
  have denpos : 0 < x.denominator := by unfold Dyadic.denominator; positivity
  have hl : 0 < ((powerSides b x c).1 : ℝ) := by unfold powerSides; positivity
  have hr : 0 < ((powerSides b x c).2 : ℝ) := by unfold powerSides; positivity
  have eqn := logSides b x c hx
  constructor
  · rw [← Nat.cast_le (α := ℝ), ← Real.log_le_log_iff hl hr, Real.logb,
      show c.real = (c.numerator : ℝ) / c.denominator from rfl, div_le_div_iff₀ hlog hc]
    constructor <;> intro h <;> nlinarith
  · rw [← Nat.cast_le (α := ℝ), ← Real.log_le_log_iff hr hl, Real.logb,
      show c.real = (c.numerator : ℝ) / c.denominator from rfl, div_le_div_iff₀ hc hlog]
    constructor <;> intro h <;> nlinarith

private theorem input_positive {x : ℕ} (h : specialLog? x = none) : 0 < (decode x).numerator := by
  have range : 0 < x ∧ x < 0x7f80 := by
    unfold specialLog? at h
    split at h <;> try contradiction
    split at h <;> try contradiction
    split at h <;> try contradiction
    rename_i first second third
    simp at first second third
    omega
  have sign : x < 0x8000 := by omega
  simp only [Dyadic.numerator, decode, Nat.mod_eq_of_lt sign, if_pos sign]
  apply Int.mul_pos _ (by positivity)
  by_cases field : x / 128 = 0
  · have rem : x % 128 = x := Nat.mod_eq_of_lt (by omega)
    simp [field, rem, range.1]
  · simp [field]
    omega

noncomputable def RealCell (b : LogBase) (x y : ℕ) : Prop :=
    let target := Real.logb b.radix (decode x).real
    let lower := ((decode (below y)).real + (decode y).real) / 2
    let upper := ((decode y).real + (decode (above y)).real) / 2
    if y % 2 = 0 then lower ≤ target ∧ target ≤ upper else lower < target ∧ target < upper

/-- For finite nonspecial inputs, every accepted result contains the true
    Mathlib logarithm between the arithmetic midpoints to its two adjacent
    bfloat16 values. Even output significands include both endpoints. -/
theorem correctLog_real {b x y} (correct : CorrectLog b x y)
    (finite : specialLog? x = none) :
    RealCell b x y := by
  unfold RealCell
  obtain ⟨_, cell⟩ := correct
  rw [finite] at cell
  obtain ⟨_, _, _, cell⟩ := cell
  let lower := compare_realLog b (decode x) (lowerMid y) (input_positive finite)
  let upper := compare_realLog b (decode x) (upperMid y) (input_positive finite)
  simp only
  rw [← midpoint_real, ← midpoint_real]
  change if y % 2 = 0 then (lowerMid y).real ≤ _ ∧ _ ≤ (upperMid y).real
    else (lowerMid y).real < _ ∧ _ < (upperMid y).real
  split <;> rename_i parity
  · rw [if_pos parity] at cell
    exact ⟨lower.2.mp cell.1, upper.1.mp cell.2⟩
  · rw [if_neg parity] at cell
    exact ⟨lt_of_not_ge fun h => (not_le_of_gt cell.1) (lower.1.mpr h),
      lt_of_not_ge fun h => (not_le_of_gt cell.2) (upper.2.mpr h)⟩

/-- Replays the core certificate in this same kernel and interprets the result
    directly in terms of the real logarithm for both library routines. -/
theorem core_correct_real (b : LogBase) (x : Fin 65536) :
    match specialLog? x.val with
    | some word => core b x.val = word
    | none => RealCell b x.val (core b x.val) := by
  have cert := core_correct b x
  cases h : specialLog? x.val with
  | some word => simpa [h] using cert.2
  | none => exact correctLog_real cert h

/-- The exact zero result bypasses the adjacent-nonzero-word midpoint test. -/
theorem log_one_real (b : LogBase) :
    core b 0x3f80 = 0 ∧ Real.logb b.radix (decode 0x3f80).real = 0 := by
  constructor
  · have h := core_correct_real b ⟨0x3f80, by decide⟩
    exact h
  · have one : (decode 0x3f80).real = 1 := by
      unfold Dyadic.real
      rw [show (decode 0x3f80).numerator = 128 by decide,
        show (decode 0x3f80).denominator = 128 by decide]
      norm_num
    rw [one, Real.logb_one]

#print axioms core_correct_real

end Rlibm
