import Egglog.Examples.ParaBitSemantics
import Init.Data.BitVec.Lemmas

open Egglog

namespace ParaBit

/-- Bit `i` in the infinite two's-complement expansion. -/
def digit : Int → Nat → Bool
  | .ofNat n, i => n.testBit i
  | .negSucc n, i => !n.testBit i

private theorem mixed_digits (a b : Nat) : ¬∀ i, a.testBit i = !b.testBit i := by
  intro h
  let n := a + b
  have bound : n < 2^n := Nat.lt_two_pow_self
  have ha : a < 2^n := by dsimp [n] at *; omega
  have hb : b < 2^n := by dsimp [n] at *; omega
  have := h n
  simp [Nat.testBit_lt_two_pow ha, Nat.testBit_lt_two_pow hb] at this

theorem eq_of_digits {x y : Int} (h : ∀ i, digit x i = digit y i) : x = y := by
  cases x <;> cases y
  case ofNat.ofNat a b => exact congrArg Int.ofNat (Nat.eq_of_testBit_eq h)
  case negSucc.negSucc a b =>
    exact congrArg Int.negSucc (Nat.eq_of_testBit_eq (fun i => by simpa [digit] using h i))
  case ofNat.negSucc a b => exact False.elim (mixed_digits a b h)
  case negSucc.ofNat a b => exact False.elim (mixed_digits b a (fun i => (h i).symm))

theorem digit_not (x : Int) (i : Nat) : digit (bitNot x) i = !digit x i := by
  cases x with
  | ofNat a => have e : bitNot (.ofNat a) = .negSucc a := by simp [bitNot] <;> omega
               rw [e]; rfl
  | negSucc a => have e : bitNot (.negSucc a) = .ofNat a := by simp [bitNot] <;> omega
                 rw [e]; simp [digit]

theorem digit_and (x y : Int) (i : Nat) : digit (bitAnd x y) i = (digit x i && digit y i) := by
  cases x <;> cases y <;> simp [bitAnd, digit] <;> grind

theorem digit_or (x y : Int) (i : Nat) : digit (bitOr x y) i = (digit x i || digit y i) := by
  simp [bitOr, digit_and, digit_not]

theorem digit_xor (x y : Int) (i : Nat) : digit (bitXor x y) i = (digit x i ^^ digit y i) := by
  cases x <;> cases y <;> simp [bitXor, digit] <;> grind

theorem digit_trunc (w x : Int) (i : Nat) :
    digit (trunc w x) i = (decide (i < w.toNat) && digit x i) := by
  have hn : 0 ≤ trunc w x := Int.emod_nonneg _ (Int.ne_of_gt (Int.pow_pos (by decide)))
  have h : trunc w x = Int.ofNat (BitVec.ofInt w.toNat x).toNat := by
    simp [BitVec.toNat_ofInt, trunc, modulus] at *
    omega
  rw [h]
  change (BitVec.ofInt w.toNat x).getLsbD i = _
  cases x <;> simp [digit, BitVec.getLsbD_ofNat] <;> grind

variable {S : Type} {v : S → Nat → Int}
open Op

theorem digit_zero (i : Nat) : digit 0 i = false := Nat.zero_testBit i
theorem digit_ones (i : Nat) : digit (-1) i = true := by
  change (!Nat.testBit 0 i) = true
  simp

private theorem minusOnePat : lit (-1) = .app 34 [] := rfl

local macro "bit_identity" : tactic => `(tactic|
  (funext e
   apply eq_of_digits
   intro i
   simp +zetaDelta [Pat.eval, Pat.evalArgs, point, scalar, Op.model, band, bor, bxor, bnot,
     bw, zero, minusOnePat, List.map]
   simp only [digit_and, digit_or, digit_xor, digit_not, digit_trunc, digit_zero, digit_ones]
   grind))

private def andComm : VerifiedRule (point v) :=
  egg_rule (a,b) : band a b ==> band b a by bit_identity

private def andAssoc : VerifiedRule (point v) :=
  egg_rule (a,b,c) : band (band a b) c ==> band a (band b c) by bit_identity

private def andSelf : VerifiedRule (point v) :=
  egg_rule (a) : band a a ==> a by bit_identity

private def andZero : VerifiedRule (point v) :=
  egg_rule (a) : band a zero ==> zero by bit_identity

private def andOnes : VerifiedRule (point v) :=
  egg_rule (a) : band a (lit (-1)) ==> a by bit_identity

private def andRemoveLeft : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (band (bw w a) b) ==> bw w (band a b) by bit_identity

private def andRemoveRight : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (band a (bw w b)) ==> bw w (band a b) by bit_identity

private def andRemoveOuter : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (band (bw w a) (bw w b)) ==> band (bw w a) (bw w b) by bit_identity

private def orComm : VerifiedRule (point v) :=
  egg_rule (a,b) : bor a b ==> bor b a by bit_identity

private def orAssoc : VerifiedRule (point v) :=
  egg_rule (a,b,c) : bor (bor a b) c ==> bor a (bor b c) by bit_identity

private def orSelf : VerifiedRule (point v) :=
  egg_rule (a) : bor a a ==> a by bit_identity

private def orZero : VerifiedRule (point v) :=
  egg_rule (a) : bor a zero ==> a by bit_identity

private def orOnes : VerifiedRule (point v) :=
  egg_rule (a) : bor a (lit (-1)) ==> lit (-1) by bit_identity

private def orRemoveLeft : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (bor (bw w a) b) ==> bw w (bor a b) by bit_identity

private def orRemoveRight : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (bor a (bw w b)) ==> bw w (bor a b) by bit_identity

private def orRemoveOuter : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (bor (bw w a) (bw w b)) ==> bor (bw w a) (bw w b) by bit_identity

private def xorComm : VerifiedRule (point v) :=
  egg_rule (a,b) : bxor a b ==> bxor b a by bit_identity

private def xorAssoc : VerifiedRule (point v) :=
  egg_rule (a,b,c) : bxor (bxor a b) c ==> bxor a (bxor b c) by bit_identity

private def xorSelf : VerifiedRule (point v) :=
  egg_rule (a) : bxor a a ==> zero by bit_identity

private def xorZero : VerifiedRule (point v) :=
  egg_rule (a) : bxor a zero ==> a by bit_identity

private def xorOnes : VerifiedRule (point v) :=
  egg_rule (a) : bxor a (lit (-1)) ==> bnot a by bit_identity

private def xorRemoveLeft : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (bxor (bw w a) b) ==> bw w (bxor a b) by bit_identity

private def xorRemoveRight : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (bxor a (bw w b)) ==> bw w (bxor a b) by bit_identity

private def xorRemoveOuter : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (bxor (bw w a) (bw w b)) ==> bxor (bw w a) (bw w b) by bit_identity

private def notRemove : VerifiedRule (point v) :=
  egg_rule (w,a) : bw w (bnot (bw w a)) ==> bw w (bnot a) by bit_identity

private def notNot : VerifiedRule (point v) :=
  egg_rule (a) : bnot (bnot a) ==> a by bit_identity

private def notZero : VerifiedRule (point v) :=
  egg_rule () : bnot zero ==> lit (-1) by bit_identity

private def notOnes : VerifiedRule (point v) :=
  egg_rule () : bnot (lit (-1)) ==> zero by bit_identity

private def andComplement : VerifiedRule (point v) :=
  egg_rule (a) : band a (bnot a) ==> zero by bit_identity

private def orComplement : VerifiedRule (point v) :=
  egg_rule (a) : bor a (bnot a) ==> lit (-1) by bit_identity

private def xorComplement : VerifiedRule (point v) :=
  egg_rule (a) : bxor a (bnot a) ==> lit (-1) by bit_identity

private def absorbAnd : VerifiedRule (point v) :=
  egg_rule (a,b) : band a (bor a b) ==> a by bit_identity

private def absorbOr : VerifiedRule (point v) :=
  egg_rule (a,b) : bor a (band a b) ==> a by bit_identity

private def andDistribute : VerifiedRule (point v) :=
  egg_rule (a,b,c) : band a (bor b c) ==> bor (band a b) (band a c) by bit_identity

private def factorAnd : VerifiedRule (point v) :=
  egg_rule (a,b,c) : bor (band a b) (band a c) ==> band a (bor b c) by bit_identity

private def orDistribute : VerifiedRule (point v) :=
  egg_rule (a,b,c) : bor a (band b c) ==> band (bor a b) (bor a c) by bit_identity

private def factorOr : VerifiedRule (point v) :=
  egg_rule (a,b,c) : band (bor a b) (bor a c) ==> bor a (band b c) by bit_identity

private def deMorganAnd : VerifiedRule (point v) :=
  egg_rule (a,b) : bnot (band a b) ==> bor (bnot a) (bnot b) by bit_identity

private def deMorganOr : VerifiedRule (point v) :=
  egg_rule (a,b) : bnot (bor a b) ==> band (bnot a) (bnot b) by bit_identity

private def xorCancel : VerifiedRule (point v) :=
  egg_rule (a,b) : bxor a (bxor a b) ==> b by bit_identity

def bitRules : Array (VerifiedRule (point v)) := #[
  andComm, andAssoc, andSelf, andZero, andOnes, andRemoveLeft,
  andRemoveRight, andRemoveOuter, orComm, orAssoc, orSelf, orZero,
  orOnes, orRemoveLeft, orRemoveRight, orRemoveOuter, xorComm, xorAssoc,
  xorSelf, xorZero, xorOnes, xorRemoveLeft, xorRemoveRight, xorRemoveOuter,
  notRemove, notNot, notZero, notOnes, andComplement, orComplement,
  xorComplement, absorbAnd, absorbOr, andDistribute, factorAnd, orDistribute,
  factorOr, deMorganAnd, deMorganOr, xorCancel]

private theorem condition {P : Prop} [Decidable P] : truth P = 1 ↔ P := by
  by_cases h : P <;> simp [truth, h]

private theorem fixedTrunc {x w : Int} : trunc w x = x ↔ Fits x w := by
  have pos : 0 < modulus w := Int.pow_pos (by decide)
  constructor
  · intro h
    exact ⟨h ▸ Int.emod_nonneg x (Int.ne_of_gt pos), h ▸ Int.emod_lt_of_pos x pos⟩
  · intro ⟨lo,hi⟩
    exact Int.emod_eq_of_lt lo hi

private theorem digitFits {x w : Int} : Fits x w ↔ ∀ i, w.toNat ≤ i → digit x i = false := by
  rw [← fixedTrunc]
  constructor
  · intro h i hi
    rw [← h, digit_trunc]
    simp [Nat.not_lt.mpr hi]
  · intro h
    apply eq_of_digits
    intro i
    rw [digit_trunc]
    by_cases hi : i < w.toNat
    · simp [hi]
    · simp [hi, h i (Nat.le_of_not_lt hi)]

private def fitAnd : GuardedRule (point v) :=
  egg_rule (x,y,w) : fits (band x y) w ==> one when (hx : fits x w = one) by
    funext e
    have h : Fits (x e) (w e) := condition.mp (congrFun hx e)
    change truth (Fits (bitAnd (x e) (y e)) (w e)) = 1
    apply condition.mpr
    apply digitFits.mpr
    intro i hi
    simp only [digit_and, digitFits.mp h i hi, Bool.false_and]

private def fitOr : GuardedRule (point v) :=
  egg_rule (x,y,w) : fits (bor x y) w ==> one when (hx : fits x w = one) (hy : fits y w = one) by
    funext e
    have h : Fits (x e) (w e) := condition.mp (congrFun hx e)
    have k : Fits (y e) (w e) := condition.mp (congrFun hy e)
    change truth (Fits (bitOr (x e) (y e)) (w e)) = 1
    apply condition.mpr
    apply digitFits.mpr
    intro i hi
    simp [digit_or, digitFits.mp h i hi, digitFits.mp k i hi]

private def fitXor : GuardedRule (point v) :=
  egg_rule (x,y,w) : fits (bxor x y) w ==> one when (hx : fits x w = one) (hy : fits y w = one) by
    funext e
    have h : Fits (x e) (w e) := condition.mp (congrFun hx e)
    have k : Fits (y e) (w e) := condition.mp (congrFun hy e)
    change truth (Fits (bitXor (x e) (y e)) (w e)) = 1
    apply condition.mpr
    apply digitFits.mpr
    intro i hi
    simp [digit_xor, digitFits.mp h i hi, digitFits.mp k i hi]

private def stripWideAnd : GuardedRule (point v) :=
  egg_rule (p,q,x,y) : bw p (band (bw q x) y) ==> bw p (band x y) when (h : Op.le p q = one) by
    funext e
    have leq : p e ≤ q e := condition.mp (congrFun h e)
    change trunc (p e) (bitAnd (trunc (q e) (x e)) (y e)) = trunc (p e) (bitAnd (x e) (y e))
    apply eq_of_digits
    intro i
    simp only [digit_and, digit_trunc]
    by_cases hi : i < (p e).toNat
    · have hq : i < (q e).toNat := by omega
      simp [hi, hq]
    · simp [hi]

private def stripWideOr : GuardedRule (point v) :=
  egg_rule (p,q,x,y) : bw p (bor (bw q x) y) ==> bw p (bor x y) when (h : Op.le p q = one) by
    funext e
    have leq : p e ≤ q e := condition.mp (congrFun h e)
    change trunc (p e) (bitOr (trunc (q e) (x e)) (y e)) = trunc (p e) (bitOr (x e) (y e))
    apply eq_of_digits
    intro i
    simp only [digit_or, digit_trunc]
    by_cases hi : i < (p e).toNat
    · have hq : i < (q e).toNat := by omega
      simp [hi, hq]
    · simp [hi]

private def stripWideXor : GuardedRule (point v) :=
  egg_rule (p,q,x,y) : bw p (bxor (bw q x) y) ==> bw p (bxor x y) when (h : Op.le p q = one) by
    funext e
    have leq : p e ≤ q e := condition.mp (congrFun h e)
    change trunc (p e) (bitXor (trunc (q e) (x e)) (y e)) = trunc (p e) (bitXor (x e) (y e))
    apply eq_of_digits
    intro i
    simp only [digit_xor, digit_trunc]
    by_cases hi : i < (p e).toNat
    · have hq : i < (q e).toNat := by omega
      simp [hi, hq]
    · simp [hi]

def bitGuarded : Array (GuardedRule (point v)) := #[fitAnd, fitOr, fitXor, stripWideAnd, stripWideOr, stripWideXor]

end ParaBit
