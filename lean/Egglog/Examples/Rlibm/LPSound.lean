import Egglog.Examples.Rlibm.CellsCertificate

namespace Rlibm.Cells

private def bump (n u : Nat) : Nat :=
  if 2*(n%u) > u || (2*(n%u) == u && n/u%2 == 1) then 1 else 0

private def rnSig (n u : Nat) := n/u + bump n u

private theorem sig_mono {a b u : Nat} (h : a ≤ b) : rnSig a u ≤ rnSig b u := by
  have q := Nat.div_le_div_right h (c := u)
  by_cases same : a/u = b/u
  · have rem : a%u ≤ b%u := by
      have aeq := Nat.mod_add_div a u
      have beq := Nat.mod_add_div b u
      rw [same] at aeq
      omega
    simp only [rnSig, bump, same, Bool.or_eq_true, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
    split <;> split <;> omega
  · have qa : bump a u ≤ 1 := by unfold bump; split <;> omega
    unfold rnSig
    omega

private theorem sig_range {n : Nat} (hn : 128 ≤ n) :
    128 ≤ rnSig n (2^(n.log2-7)) ∧ rnSig n (2^(n.log2-7)) ≤ 256 := by
  have nz : n ≠ 0 := by omega
  have hs : 7 ≤ n.log2 := (Nat.le_log2 nz).2 hn
  let u := 2^(n.log2-7)
  have hu : 0 < u := Nat.pow_pos (by decide)
  have p : 128*u = 2^n.log2 := by
    change 2^7 * 2^(n.log2-7) = 2^n.log2
    rw [← Nat.pow_add, Nat.add_sub_of_le hs]
  have low : 128 ≤ n/u := by
    apply (Nat.le_div_iff_mul_le hu).2
    rw [p]
    exact ((Nat.log2_eq_iff nz).1 rfl).1
  have upper : n/u < 256 := by
    apply (Nat.div_lt_iff_lt_mul hu).2
    have := Nat.lt_log2_self (n := n)
    rw [Nat.pow_succ, ← p] at this
    omega
  have r : bump n u ≤ 1 := by unfold bump; split <;> omega
  change 128 ≤ n/u + bump n u ∧ n/u + bump n u ≤ 256
  omega

private def magBits (n : Nat) := min 0x7f80 ((n.log2+86)*128 + rnSig n (2^(n.log2-7)))

private theorem magBits_mono {a b : Nat} (ha : 128 ≤ a) (h : a ≤ b) : magBits a ≤ magBits b := by
  suffices (a.log2+86)*128 + rnSig a (2^(a.log2-7)) ≤
      (b.log2+86)*128 + rnSig b (2^(b.log2-7)) by unfold magBits; omega
  have hb := Nat.le_trans ha h
  by_cases hs : a.log2 = b.log2
  · rw [hs]
    apply Nat.add_le_add_left
    exact sig_mono h
  · have order : a.log2 ≤ b.log2 := (Nat.le_log2 (by omega)).2
        (Nat.le_trans (((Nat.log2_eq_iff (by omega : a ≠ 0)).1 rfl).1) h)
    have left := (sig_range ha).2
    have right := (sig_range hb).1
    omega

private theorem round_magnitude {z : Int} (h : 128 ≤ z.natAbs) :
    roundQ40 z = (if z < 0 then 0x8000 else 0) + magBits z.natAbs := by
  have nz : z.natAbs ≠ 0 := by omega
  have lg := (Nat.le_log2 nz (k := 7)).2 h
  simp [roundQ40, magBits, rnSig, bump, nz, Nat.not_lt.mpr lg]


/-- If the endpoint magnitudes are at least 128 and have the same sign,
    equality of the rounded endpoints forces equality of all interior results. -/
private theorem sandwich {l z r : Int} {bits : Nat} (lo : l ≤ z) (hi : z ≤ r)
    (signs : (128 : Int) ≤ l ∨ r ≤ -128)
    (hl : roundQ40 l = bits) (hr : roundQ40 r = bits) : roundQ40 z = bits := by
  rcases signs with pos | neg
  · have al : 128 ≤ l.natAbs := by omega
    have az : l.natAbs ≤ z.natAbs := by omega
    have ar : z.natAbs ≤ r.natAbs := by omega
    have u := magBits_mono al az
    have v := magBits_mono (Nat.le_trans al az) ar
    rw [round_magnitude al] at hl
    rw [round_magnitude (z := r) (by omega)] at hr
    rw [round_magnitude (z := z) (by omega)]
    simp only [if_neg (by omega : ¬l < 0), if_neg (by omega : ¬z < 0),
      if_neg (by omega : ¬r < 0), Nat.zero_add] at hl hr ⊢
    omega
  · have ar : 128 ≤ r.natAbs := by omega
    have az : r.natAbs ≤ z.natAbs := by omega
    have al : z.natAbs ≤ l.natAbs := by omega
    have u := magBits_mono ar az
    have v := magBits_mono (Nat.le_trans ar az) al
    rw [round_magnitude ar] at hr
    rw [round_magnitude (z := l) (by omega)] at hl
    rw [round_magnitude (z := z) (by omega)]
    simp only [if_pos (by omega : l < 0), if_pos (by omega : z < 0),
      if_pos (by omega : r < 0)] at hl hr ⊢
    omega

private def scaled (cut : Dyadic) := cut.numerator * 2^40
private def grid (cut : Dyadic) := scaled cut / (cut.denominator : Int)
private def integralGrid (c : Dyadic) : Prop := grid c * c.denominator = scaled c

private def pad (word : Nat) : Int := if word % 2 = 0 then 0 else 1
private def first (word : Nat) := grid (lowerMid word) + pad word
private def last (word : Nat) := grid (upperMid word) - pad word

private def Endpoints (word : Nat) : Prop :=
  integralGrid (lowerMid word) ∧ integralGrid (upperMid word) ∧
  ((128 : Int) ≤ first word ∨ last word ≤ -128) ∧
  roundQ40 (first word) = word ∧ roundQ40 (last word) = word

private instance (word : Nat) : Decidable (Endpoints word) := by
  unfold Endpoints integralGrid
  infer_instance

/-- The LP endpoints are exact dyadics on the Q40 grid. Strict midpoint
    inequalities remove one integer on each end of an odd output cell. -/
private theorem cell_grid_bounds {word : Nat} (nz : word ≠ 0)
    (hl : integralGrid (lowerMid word)) (hr : integralGrid (upperMid word))
    {z : Int} (cell : inCell word z = true) : first word ≤ z ∧ z ≤ last word := by
  have dl : 0 < ((lowerMid word).denominator : Int) := by
    exact Int.natCast_pos.mpr (Nat.pow_pos (by decide))
  have dr : 0 < ((upperMid word).denominator : Int) := by
    exact Int.natCast_pos.mpr (Nat.pow_pos (by decide))
  unfold integralGrid scaled at hl hr
  simp only [inCell, beq_iff_eq, nz, ↓reduceIte] at cell
  rw [← hl, ← hr] at cell
  by_cases even : word % 2 = 0
  · simp only [even, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at cell
    simp only [first, last, pad, even, ↓reduceIte, Int.add_zero, Int.sub_zero]
    constructor
    · apply (Int.mul_le_mul_right dl).mp
      omega
    · apply (Int.mul_le_mul_right dr).mp
      omega
  · simp only [even, ↓reduceIte, Bool.and_eq_true, decide_eq_true_eq] at cell
    have l : grid (lowerMid word) < z := (Int.mul_lt_mul_right dl).mp (by omega)
    have r : z < grid (upperMid word) := (Int.mul_lt_mul_right dr).mp (by omega)
    simp only [first, last, pad, even, ↓reduceIte]
    omega

private def supported (word : Nat) : Bool :=
  0x3a80 ≤ word % 0x8000 && word % 0x8000 ≤ 0x4380

private def checkEndpoints (word : Nat) := !supported word || decide (Endpoints word)

set_option maxHeartbeats 0
set_option maxRecDepth 8000
set_option Elab.async false

-- Only the endpoints need to be tested once per relevant result word.
certify_bfloat16 checkEndpoints as endpoints

/-- Any Q40 value satisfying the two LP inequalities rounds to the proposed
    result in the normal magnitude range used by the bfloat16 logarithms.
    This theorem quantifies over arbitrary unbounded signed integers z. -/
theorem roundQ40_of_inCell (word : Fin 65536) (z : Int)
    (range : word.val = 0 ∨ (0x3a80 ≤ word.val % 0x8000 ∧ word.val % 0x8000 ≤ 0x4380))
    (cell : inCell word.val z = true) : roundQ40 z = word.val := by
  rcases range with zero | range
  · have hz : z = 0 := by simpa [inCell, zero] using cell
    simp [zero, hz, roundQ40]
  have ok : Endpoints word.val := by
    simpa [checkEndpoints, supported, range.1, range.2] using endpoints word
  obtain ⟨hl, hr, signs, left, right⟩ := ok
  obtain ⟨a,b⟩ := cell_grid_bounds (by omega) hl hr cell
  exact sandwich a b signs left right

/-- A certificate for the target word can be reused by any set of polynomial
    coefficients whose unrounded Q40 accumulator satisfies the same LP row. -/
theorem correct_of_feasible (base : LogBase) (input word : Fin 65536) (z : Int)
    (reference : CorrectLog base input.val word.val)
    (range : word.val = 0 ∨ (0x3a80 ≤ word.val % 0x8000 ∧ word.val % 0x8000 ≤ 0x4380))
    (feasible : inCell word.val z = true) : CorrectLog base input.val (roundQ40 z) := by
  rw [roundQ40_of_inCell word z range feasible]
  exact reference

end Rlibm.Cells
