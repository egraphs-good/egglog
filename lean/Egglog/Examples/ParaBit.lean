import Egglog.Examples.ParaBitControl
import Egglog.Examples.ParaBitMixed
import Egglog.Examples.ParaBitDecision

open Egglog
namespace ParaBit

mutual
  private theorem eval_closed (p : Pat) (h : p.variables = []) (m : Model α) (e f : Nat → α) :
      p.eval m e = p.eval m f := by
    cases p with
    | var => simp [Pat.variables] at h
    | app op args => exact congrArg (m.interp op) (evalArgs_closed args h m e f)
  private theorem evalArgs_closed (ps : List Pat) (h : Pat.argsVariables ps = []) (m : Model α) (e f : Nat → α) :
      Pat.evalArgs ps m e = Pat.evalArgs ps m f := by
    cases ps with
    | nil => rfl
    | cons p ps =>
      simp only [Pat.argsVariables, List.append_eq_nil_iff] at h
      change _ :: _ = _ :: _
      rw [eval_closed p h.1 m e f, evalArgs_closed ps h.2 m e f]
end

private theorem truth_eq {P : Prop} [Decidable P] : truth P = 1 ↔ P := by
  by_cases h : P <;> simp [truth, h]

variable {S : Type} {v : S → Nat → Int}
open Op


private theorem modulus_pos (w : Int) : 0 < modulus w := Int.pow_pos (by decide)
private theorem fits_trunc (w x : Int) : Fits (trunc w x) w :=
  ⟨Int.emod_nonneg _ (Int.ne_of_gt (modulus_pos w)), Int.emod_lt_of_pos _ (modulus_pos w)⟩

private theorem mod_mono {p q : Int} (h : p ≤ q) : modulus p ≤ modulus q :=
  Int.ofNat_le.mpr (Nat.pow_le_pow_right (by decide) (Int.toNat_le_toNat h))
private theorem small_trunc {p q : Int} (h : p ≤ q) (x : Int) : trunc p (trunc q x) = trunc p x :=
  Int.emod_emod_of_dvd _ (Int.ofNat_dvd.mpr (Nat.pow_dvd_pow 2 (Int.toNat_le_toNat h)))
private theorem drop_trunc {p q : Int} (h : p ≤ q) (x : Int) : trunc q (trunc p x) = trunc p x :=
  Int.emod_eq_of_lt (fits_trunc p x).1 (Int.lt_of_lt_of_le (fits_trunc p x).2 (mod_mono h))

private theorem sum_fit {a b p q w : Int} (ha : Fits a p) (hb : Fits b q)
    (hp : p < w) (hq : q < w) (hw : 0 < w) : Fits (a + b) w := by
  have h (i : Int) (hi : i < w) : modulus i * 2 ≤ modulus w := by
    change 2 ^ i.toNat * 2 ≤ 2 ^ w.toNat
    rw [← Int.pow_succ]
    exact Int.ofNat_le.mpr (Nat.pow_le_pow_right (by decide) (by omega : i.toNat + 1 ≤ w.toNat))
  obtain ⟨ha₀, ha₁⟩ := ha
  obtain ⟨hb₀, hb₁⟩ := hb
  have := h p hp
  have := h q hq
  constructor <;> omega

private def addComm : VerifiedRule (point v) :=
  egg_rule (a,b) : add a b ==> add b a by
    funext e
    exact Int.add_comm _ _

private def addAssoc : VerifiedRule (point v) :=
  egg_rule (a,b,c) : add (add a b) c ==> add a (add b c) by
    funext e
    exact Int.add_assoc _ _ _

private def mulComm : VerifiedRule (point v) :=
  egg_rule (a,b) : mul a b ==> mul b a by
    funext e
    exact Int.mul_comm _ _

private def mulAssoc : VerifiedRule (point v) :=
  egg_rule (a,b,c) : mul (mul a b) c ==> mul a (mul b c) by
    funext e
    exact Int.mul_assoc _ _ _

private def subSelf : VerifiedRule (point v) :=
  egg_rule (a) : sub a a ==> zero by
    funext e
    exact Int.sub_self _

private def addZero : VerifiedRule (point v) :=
  egg_rule (a) : add a zero ==> a by
    funext e
    exact Int.add_zero _

private def mulZero : VerifiedRule (point v) :=
  egg_rule (a) : mul a zero ==> zero by
    funext e
    exact Int.mul_zero _

private def mulOne : VerifiedRule (point v) :=
  egg_rule (a) : mul a one ==> a by
    funext e
    exact Int.mul_one _

private def divOne : VerifiedRule (point v) :=
  egg_rule (a) : Op.div a one ==> a by
    funext e
    exact Int.fdiv_one _

private def subZero : VerifiedRule (point v) :=
  egg_rule (a) : sub a zero ==> a by
    funext e
    exact Int.sub_zero _

private def doubleNeg : VerifiedRule (point v) :=
  egg_rule (a) : neg (neg a) ==> a by
    funext e
    exact Int.neg_neg _

private def subToAdd : VerifiedRule (point v) :=
  egg_rule (a,b) : sub a b ==> add a (neg b) by
    funext e
    exact Int.sub_eq_add_neg

private def mulTwo : VerifiedRule (point v) :=
  egg_rule (a) : mul a two ==> add a a by
    funext e
    change a e * 2 = a e + a e
    omega

private def dist : VerifiedRule (point v) :=
  egg_rule (a,b,c) : mul a (add b c) ==> add (mul a b) (mul a c) by
    funext e
    exact Int.mul_add _ _ _

private def shiftLeft : VerifiedRule (point v) :=
  egg_rule (a,b) : shl a b ==> mul a (Op.pow two b) by
    funext e
    rfl

private def shiftRight : VerifiedRule (point v) :=
  egg_rule (a,b) : shr a b ==> Op.div a (Op.pow two b) by
    funext e
    exact (Int.fdiv_eq_ediv_of_nonneg _ (Int.le_of_lt (modulus_pos _))).symm

private def powZero : VerifiedRule (point v) :=
  egg_rule (a) : Op.pow a zero ==> one by
    funext e
    exact Int.pow_zero _

private def powOne : VerifiedRule (point v) :=
  egg_rule (a) : Op.pow a one ==> a by
    funext e
    exact Int.pow_one _

private def sameSel : VerifiedRule (point v) :=
  egg_rule (a,c) : sel c a a ==> a by
    funext e
    change (if _ then a e else a e) = a e
    split <;> rfl

private def selZero : VerifiedRule (point v) :=
  egg_rule (a,b) : sel zero a b ==> b by
    funext e
    rfl

private def selOne : VerifiedRule (point v) :=
  egg_rule (a,b) : sel one a b ==> a by
    funext e
    rfl

private def bwZero : VerifiedRule (point v) :=
  egg_rule (w) : bw w zero ==> zero by
    funext e
    exact Int.zero_emod _

private def repeatBw : VerifiedRule (point v) :=
  egg_rule (w,a) : bw w (bw w a) ==> bw w a by
    funext e
    exact Int.emod_emod _ _

private def fitsBw : VerifiedRule (point v) :=
  egg_rule (w,a) : fits (bw w a) w ==> one by
    funext e
    exact truth_eq.mpr (fits_trunc _ _)

private def fitsZero : VerifiedRule (point v) :=
  egg_rule (w) : fits zero w ==> one by
    funext e
    exact truth_eq.mpr ⟨Int.le_refl 0, modulus_pos _⟩

private def leRefl : VerifiedRule (point v) :=
  egg_rule (w) : Op.le w w ==> one by
    funext e
    exact truth_eq.mpr (Int.le_refl _)

private def doubleOne : VerifiedRule (point v) :=
  egg_rule () : add one one ==> two by
    funext e
    rfl

private def ltToLe : GuardedRule (point v) :=
  egg_rule (a,b) : Op.le a b ==> one when (ltab : Op.lt a b = one) by
    funext e
    have ltabv : a e < b e := truth_eq.mp (congrFun ltab e)
    exact truth_eq.mpr (Int.le_of_lt ltabv)

private def transLe : GuardedRule (point v) :=
  egg_rule (a,b,c) : Op.le a c ==> one when (leab : Op.le a b = one) (lebc : Op.le b c = one) by
    funext e
    have leabv : a e ≤ b e := truth_eq.mp (congrFun leab e)
    have lebcv : b e ≤ c e := truth_eq.mp (congrFun lebc e)
    exact truth_eq.mpr (Int.le_trans leabv lebcv)

private def ltLe : GuardedRule (point v) :=
  egg_rule (a,b,c) : Op.lt a c ==> one when (ltab : Op.lt a b = one) (lebc : Op.le b c = one) by
    funext e
    have ltabv : a e < b e := truth_eq.mp (congrFun ltab e)
    have lebcv : b e ≤ c e := truth_eq.mp (congrFun lebc e)
    exact truth_eq.mpr (Int.lt_of_lt_of_le ltabv lebcv)

private def leLt : GuardedRule (point v) :=
  egg_rule (a,b,c) : Op.lt a c ==> one when (leab : Op.le a b = one) (ltbc : Op.lt b c = one) by
    funext e
    have leabv : a e ≤ b e := truth_eq.mp (congrFun leab e)
    have ltbcv : b e < c e := truth_eq.mp (congrFun ltbc e)
    exact truth_eq.mpr (Int.lt_of_le_of_lt leabv ltbcv)

private def fitElim : GuardedRule (point v) :=
  egg_rule (x,w) : bw w x ==> x when (fitxw : fits x w = one) by
    funext e
    have fitxwv : Fits (x e) (w e) := truth_eq.mp (congrFun fitxw e)
    exact Int.emod_eq_of_lt fitxwv.1 fitxwv.2

/-- Either branch fits at the same width, hence so does the selected word. -/
private def fitsSel : GuardedRule (point v) :=
  egg_rule (c,a,b,w) : fits (sel c a b) w ==> one
      when (ha : fits a w = one) (hb : fits b w = one) by
    funext e
    change truth (Fits (if c e % 2 = 1 then a e else b e) (w e)) = 1
    split
    · exact congrFun ha e
    · exact congrFun hb e

private def wideBw : GuardedRule (point v) :=
  egg_rule (x,p,w) : bw w (bw p x) ==> bw p x when (lepw : Op.le p w = one) by
    funext e
    have lepwv : p e ≤ w e := truth_eq.mp (congrFun lepw e)
    exact drop_trunc lepwv _

private def narrowBw : GuardedRule (point v) :=
  egg_rule (x,p,w) : bw w (bw p x) ==> bw w x when (lewp : Op.le w p = one) by
    funext e
    have lewpv : w e ≤ p e := truth_eq.mp (congrFun lewp e)
    exact small_trunc lewpv _

private def fitsAdd : GuardedRule (point v) :=
  egg_rule (a,b,p,q,w) : fits (add a b) w ==> one when (fitap : fits a p = one) (fitbq : fits b q = one) (ltpw : Op.lt p w = one) (ltqw : Op.lt q w = one) (posw : Op.lt zero w = one) by
    funext e
    have fitapv : Fits (a e) (p e) := truth_eq.mp (congrFun fitap e)
    have fitbqv : Fits (b e) (q e) := truth_eq.mp (congrFun fitbq e)
    have ltpwv : p e < w e := truth_eq.mp (congrFun ltpw e)
    have ltqwv : q e < w e := truth_eq.mp (congrFun ltqw e)
    have poswv : 0 < w e := truth_eq.mp (congrFun posw e)
    exact truth_eq.mpr (sum_fit fitapv fitbqv ltpwv ltqwv poswv)

private def fitOne : GuardedRule (point v) :=
  egg_rule (w) : fits one w ==> one when (posw : Op.lt zero w = one) by
    funext e
    have poswv : 0 < w e := truth_eq.mp (congrFun posw e)
    have hh := mod_mono (p := 1) (q := w e) (by omega)
    change 2 ≤ modulus (w e) at hh
    change truth (Fits 1 (w e)) = 1
    exact truth_eq.mpr ⟨by decide, by omega⟩

private def removeAddL : GuardedRule (point v) :=
  egg_rule (a,b,p,w) : bw w (add (bw p a) b) ==> bw w (add a b) when (lewp : Op.le w p = one) by
    funext e
    have lewpv : w e ≤ p e := truth_eq.mp (congrFun lewp e)
    change (trunc (p e) (a e) + b e) % modulus (w e) = (a e + b e) % modulus (w e)
    rw [Int.add_emod, show trunc (p e) (a e) % modulus (w e) = trunc (w e) (a e) from small_trunc lewpv _]
    exact (Int.add_emod _ _ _).symm

private def removeAddR : GuardedRule (point v) :=
  egg_rule (a,b,p,w) : bw w (add a (bw p b)) ==> bw w (add a b) when (lewp : Op.le w p = one) by
    funext e
    have lewpv : w e ≤ p e := truth_eq.mp (congrFun lewp e)
    change (a e + trunc (p e) (b e)) % modulus (w e) = (a e + b e) % modulus (w e)
    rw [Int.add_emod, show trunc (p e) (b e) % modulus (w e) = trunc (w e) (b e) from small_trunc lewpv _]
    exact (Int.add_emod _ _ _).symm

private def removeSubL : GuardedRule (point v) :=
  egg_rule (a,b,p,w) : bw w (sub (bw p a) b) ==> bw w (sub a b) when (lewp : Op.le w p = one) by
    funext e
    have lewpv : w e ≤ p e := truth_eq.mp (congrFun lewp e)
    change (trunc (p e) (a e) - b e) % modulus (w e) = (a e - b e) % modulus (w e)
    rw [Int.sub_emod, show trunc (p e) (a e) % modulus (w e) = trunc (w e) (a e) from small_trunc lewpv _]
    exact (Int.sub_emod _ _ _).symm

private def removeSubR : GuardedRule (point v) :=
  egg_rule (a,b,p,w) : bw w (sub a (bw p b)) ==> bw w (sub a b) when (lewp : Op.le w p = one) by
    funext e
    have lewpv : w e ≤ p e := truth_eq.mp (congrFun lewp e)
    change (a e - trunc (p e) (b e)) % modulus (w e) = (a e - b e) % modulus (w e)
    rw [Int.sub_emod, show trunc (p e) (b e) % modulus (w e) = trunc (w e) (b e) from small_trunc lewpv _]
    exact (Int.sub_emod _ _ _).symm

private def removeMulL : GuardedRule (point v) :=
  egg_rule (a,b,p,w) : bw w (mul (bw p a) b) ==> bw w (mul a b) when (lewp : Op.le w p = one) by
    funext e
    have lewpv : w e ≤ p e := truth_eq.mp (congrFun lewp e)
    change (trunc (p e) (a e) * b e) % modulus (w e) = (a e * b e) % modulus (w e)
    rw [Int.mul_emod, show trunc (p e) (a e) % modulus (w e) = trunc (w e) (a e) from small_trunc lewpv _]
    exact (Int.mul_emod _ _ _).symm

private def removeMulR : GuardedRule (point v) :=
  egg_rule (a,b,p,w) : bw w (mul a (bw p b)) ==> bw w (mul a b) when (lewp : Op.le w p = one) by
    funext e
    have lewpv : w e ≤ p e := truth_eq.mp (congrFun lewp e)
    change (a e * trunc (p e) (b e)) % modulus (w e) = (a e * b e) % modulus (w e)
    rw [Int.mul_emod, show trunc (p e) (b e) % modulus (w e) = trunc (w e) (b e) from small_trunc lewpv _]
    exact (Int.mul_emod _ _ _).symm


private theorem product_fit {a b p q w : Int} (ha : Fits a p) (hb : Fits b q)
    (hp : 0 ≤ p) (hq : 0 ≤ q) (h : p + q ≤ w) : Fits (a * b) w := by
  constructor
  · exact Int.mul_nonneg ha.1 hb.1
  · have h1 := Int.mul_le_mul_of_nonneg_right (Int.le_of_lt ha.2) hb.1
    have h2 := Int.mul_lt_mul_of_pos_left hb.2 (modulus_pos p)
    have h3 := mod_mono h
    simp only [modulus, Int.toNat_add hp hq, Int.pow_add] at h3
    exact Int.lt_of_lt_of_le (Int.lt_of_le_of_lt h1 h2) h3

private def fitsMul : GuardedRule (point v) :=
  egg_rule (a,b,p,q,w) : fits (mul a b) w ==> one when
    (ha : fits a p = one) (hb : fits b q = one) (hp : Op.lt zero p = one)
    (hq : Op.lt zero q = one) (hw : Op.le (add p q) w = one) by
    funext e
    exact truth_eq.mpr (product_fit (truth_eq.mp (congrFun ha e)) (truth_eq.mp (congrFun hb e))
      (Int.le_of_lt (truth_eq.mp (congrFun hp e))) (Int.le_of_lt (truth_eq.mp (congrFun hq e)))
      (truth_eq.mp (congrFun hw e)))

private def bwTwo : VerifiedRule (point v) :=
  egg_rule () : bw two two ==> two by rfl

private def bwSingleOne : VerifiedRule (point v) :=
  egg_rule () : bw one one ==> one by rfl

private def addOneToTwo : VerifiedRule (point v) :=
  egg_rule (a) : add a a ==> mul two a by
    funext e
    change a e + a e = 2 * a e
    omega

private def subAddCancel : VerifiedRule (point v) :=
  egg_rule (a,b) : sub (add a b) b ==> a by
    funext e
    exact Int.add_sub_cancel _ _

private def addMuxPair : VerifiedRule (point v) :=
  egg_rule (c,a,b,x,y) : add (sel c a b) (sel c x y) ==> sel c (add a x) (add b y) by
    funext e
    change (if c e % 2 = 1 then a e else b e) + (if c e % 2 = 1 then x e else y e) =
      (if c e % 2 = 1 then a e + x e else b e + y e)
    split <;> simp_all

private theorem mod_sum {a b : Int} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    modulus (a + b) = modulus a * modulus b := by
  simp [modulus, Int.toNat_add ha hb, Int.pow_add]

private def joinShl : GuardedRule (point v) :=
  egg_rule (a,x,y,p,q) : shl (shl a x) y ==> shl a (add x y) when
      (hx : fits x p = one) (hy : fits y q = one) by
    funext e
    have hx' : Fits (x e) (p e) := truth_eq.mp (congrFun hx e)
    have hy' : Fits (y e) (q e) := truth_eq.mp (congrFun hy e)
    change a e * modulus (x e) * modulus (y e) = a e * modulus (x e + y e)
    rw [mod_sum hx'.1 hy'.1, Int.mul_assoc]

private def joinShr : GuardedRule (point v) :=
  egg_rule (a,x,y,p,q) : shr (shr a x) y ==> shr a (add x y) when
      (hx : fits x p = one) (hy : fits y q = one) by
    funext e
    have hx' : Fits (x e) (p e) := truth_eq.mp (congrFun hx e)
    have hy' : Fits (y e) (q e) := truth_eq.mp (congrFun hy e)
    change a e / modulus (x e) / modulus (y e) = a e / modulus (x e + y e)
    rw [mod_sum hx'.1 hy'.1, Int.ediv_ediv_of_nonneg (Int.le_of_lt (modulus_pos _))]

private def shrFits : GuardedRule (point v) :=
  egg_rule (x,y,p,w) : fits (shr x y) w ==> one when
      (hp : fits x p = one) (hw : Op.le p w = one) by
    funext e
    have hx : Fits (x e) (p e) := truth_eq.mp (congrFun hp e)
    have hw' : p e ≤ w e := truth_eq.mp (congrFun hw e)
    exact truth_eq.mpr ⟨Int.ediv_nonneg hx.1 (Int.le_of_lt (modulus_pos _)),
      Int.lt_of_le_of_lt (Int.ediv_le_self _ hx.1) (Int.lt_of_lt_of_le hx.2 (mod_mono hw'))⟩

private def shlFits : GuardedRule (point v) :=
  egg_rule (a,b,p,q,s) : fits (shl a b) s ==> one when
      (ha : fits a p = one) (hb : fits b q = one) (hp : Op.lt zero p = one)
      (hw : Op.le (add p (sub (Op.pow two q) one)) s = one) by
    funext e
    have ha' : Fits (a e) (p e) := truth_eq.mp (congrFun ha e)
    have hb' : Fits (b e) (q e) := truth_eq.mp (congrFun hb e)
    have pp : 0 < p e := truth_eq.mp (congrFun hp e)
    have ww : p e + (modulus (q e) - 1) ≤ s e := truth_eq.mp (congrFun hw e)
    have hh : p e + b e ≤ s e := by have := hb'.2; omega
    have bound := mod_mono hh
    rw [mod_sum (Int.le_of_lt pp) hb'.1] at bound
    exact truth_eq.mpr ⟨Int.mul_nonneg ha'.1 (Int.le_of_lt (modulus_pos _)),
      Int.lt_of_lt_of_le (Int.mul_lt_mul_of_pos_right ha'.2 (modulus_pos _)) bound⟩

private def shiftedSum : VerifiedRule (point v) :=
  egg_rule (a,b,c) : shr (add (shl a c) b) c ==> add a (shr b c) by
    funext e
    exact Int.mul_add_ediv_right _ _ (Int.ne_of_gt (modulus_pos _))

private def shlTrunc : GuardedRule (point v) :=
  egg_rule (a,b,p,w) : bw w (shl (bw p a) b) ==> bw w (shl a b) when
    (h : Op.le w p = one) by
    funext e
    have hw : w e ≤ p e := truth_eq.mp (congrFun h e)
    change (trunc (p e) (a e) * modulus (b e)) % modulus (w e) = (a e * modulus (b e)) % modulus (w e)
    rw [Int.mul_emod, show trunc (p e) (a e) % modulus (w e) = trunc (w e) (a e) from small_trunc hw _]
    exact (Int.mul_emod _ _ _).symm

private def zeroSub : VerifiedRule (point v) :=
  egg_rule (a) : sub zero a ==> neg a by
    funext e
    change 0 - a e = -a e
    omega

private def negSum : VerifiedRule (point v) :=
  egg_rule (a,b) : neg (add a b) ==> add (neg a) (neg b) by
    funext e
    change -(a e + b e) = -a e + -b e
    omega

private def negSub : VerifiedRule (point v) :=
  egg_rule (a,b) : neg (sub a b) ==> sub b a by
    funext e
    change -(a e - b e) = b e - a e
    omega

private def subSub : VerifiedRule (point v) :=
  egg_rule (a,b,c) : sub a (sub b c) ==> add (sub a b) c by
    funext e
    change a e - (b e - c e) = (a e - b e) + c e
    omega

private def addSubCancel : VerifiedRule (point v) :=
  egg_rule (a,b) : add a (sub b a) ==> b by
    funext e
    change a e + (b e - a e) = b e
    omega

private def subSum : VerifiedRule (point v) :=
  egg_rule (a,b,c) : sub a (add b c) ==> sub (sub a c) b by
    funext e
    change a e - (b e + c e) = a e - c e - b e
    omega

private def sumSubSelf : VerifiedRule (point v) :=
  egg_rule (a,b) : sub a (add a b) ==> neg b by
    funext e
    change a e - (a e + b e) = -b e
    omega

private def diffSubSelf : VerifiedRule (point v) :=
  egg_rule (a,b) : sub (sub a b) a ==> neg b by
    funext e
    change a e - b e - a e = -b e
    omega

private def halfSucc : VerifiedRule (point v) :=
  egg_rule (a) : Op.div (add a one) two ==> add (Op.div a two) (bw one a) by
    funext e
    change (a e + 1).fdiv 2 = (a e).fdiv 2 + a e % 2
    simp only [Int.fdiv_eq_ediv_of_nonneg _ (by decide : (0:Int) ≤ 2)]
    omega

private def boolSelect : VerifiedRule (point v) :=
  egg_rule (c) : sel c one zero ==> bw one c by
    funext e
    change (if c e % 2 = 1 then (1:Int) else 0) = c e % 2
    split <;> omega

private def mulNegs : VerifiedRule (point v) :=
  egg_rule (a,b) : mul (neg a) (neg b) ==> mul a b by
    funext e
    exact Int.neg_mul_neg _ _

/-- Division by a positive radix factor commutes with discarding complete
    multiples of the product of the input and output radices. -/
private theorem quotient_low (a m n : Int) (hn : n ≠ 0) :
    ((a % (m * n)) / n) % m = (a / n) % m := by
  let k := a / (m * n)
  have h : a % (m * n) + (k * m) * n = a := by
    rw [Int.mul_assoc]
    exact Int.emod_add_ediv_mul _ _
  calc
    ((a % (m * n)) / n) % m = ((a % (m * n)) / n + k * m) % m := (Int.add_mul_emod_self_right ..).symm
    _ = ((a % (m * n) + (k * m) * n) / n) % m := by rw [Int.add_mul_ediv_right _ _ hn]
    _ = _ := by rw [h]

private theorem trunc_shift_right {w p k : Int} (hw : 0 ≤ w) (hk : 0 ≤ k) (hp : w + k ≤ p) (x : Int) :
    trunc w (trunc p x / modulus k) = trunc w (x / modulus k) := by
  have low (a : Int) : trunc w (trunc (w+k) a / modulus k) = trunc w (a / modulus k) := by
    unfold trunc
    rw [mod_sum hw hk]
    exact quotient_low _ _ _ (Int.ne_of_gt (modulus_pos k))
  rw [← low (trunc p x), small_trunc hp, low]

private def shrinkQuotient : GuardedRule (point v) :=
  egg_rule (a,w,p,k) : bw w (shr (bw p a) k) ==> bw w (shr a k) when
    (hw : Op.lt zero w = one) (hk : Op.lt zero k = one) (hp : Op.le k (sub p w) = one) by
    funext e
    have hw' : 0 < w e := truth_eq.mp (congrFun hw e)
    have hk' : 0 < k e := truth_eq.mp (congrFun hk e)
    have hp' : k e ≤ p e - w e := truth_eq.mp (congrFun hp e)
    change trunc (w e) (trunc (p e) (a e) / modulus (k e)) = trunc (w e) (a e / modulus (k e))
    exact trunc_shift_right (Int.le_of_lt hw') (Int.le_of_lt hk') (by omega) _

private def leSucc : GuardedRule (point v) :=
  egg_rule (a,b) : Op.le a (add b one) ==> one when (h : Op.le a b = one) by
    funext e
    have hv : a e ≤ b e := truth_eq.mp (congrFun h e)
    change truth (a e ≤ b e + 1) = 1
    exact truth_eq.mpr (by omega)

private def pushMulSub : VerifiedRule (point v) :=
  egg_rule (a,b,c) : mul a (sub b c) ==> sub (mul a b) (mul a c) by
    funext e
    exact Int.mul_sub _ _ _

private def pushMulNeg : VerifiedRule (point v) :=
  egg_rule (a,b) : mul a (neg b) ==> neg (mul a b) by
    funext e
    exact Int.mul_neg _ _

private def signedMux : VerifiedRule (point v) :=
  egg_rule (w,c,a,b) : signed w (sel c a b) ==> sel c (signed w a) (signed w b) by
    funext e
    change signedVal (w e) (if c e % 2 = 1 then a e else b e) =
      (if c e % 2 = 1 then signedVal (w e) (a e) else signedVal (w e) (b e))
    split <;> rfl

private def lowAllOnes : VerifiedRule (point v) :=
  egg_rule () : bw one (lit (-1)) ==> one by rfl

private def bitIncrement : VerifiedRule (point v) :=
  egg_rule (x,c) : add x (bw one c) ==> sel c (add x one) x by
    funext e
    change x e + c e % 2 = if c e % 2 = 1 then x e + 1 else x e
    split <;> omega

private def factorCastMux : VerifiedRule (point v) :=
  egg_rule (w,c,a,b) : sel c (bw w a) (bw w b) ==> bw w (sel c a b) by
    funext e
    change (if c e % 2 = 1 then trunc (w e) (a e) else trunc (w e) (b e)) =
      trunc (w e) (if c e % 2 = 1 then a e else b e)
    split <;> rfl

private def minusOneMul : VerifiedRule (point v) :=
  egg_rule (x) : mul (lit (-1)) x ==> neg x by
    funext e
    exact Int.neg_one_mul _

private def narrowAnd : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (band a b) ==> band (bw w a) (bw w b) by
    funext e
    change trunc (w e) (bitAnd (a e) (b e)) = bitAnd (trunc (w e) (a e)) (trunc (w e) (b e))
    apply eq_of_digits
    intro i
    simp only [digit_trunc, digit_and]
    grind

private def narrowOr : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (bor a b) ==> bor (bw w a) (bw w b) by
    funext e
    change trunc (w e) (bitOr (a e) (b e)) = bitOr (trunc (w e) (a e)) (trunc (w e) (b e))
    apply eq_of_digits
    intro i
    simp only [digit_trunc, digit_or]
    grind

private def narrowXor : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (bxor a b) ==> bxor (bw w a) (bw w b) by
    funext e
    change trunc (w e) (bitXor (a e) (b e)) = bitXor (trunc (w e) (a e)) (trunc (w e) (b e))
    apply eq_of_digits
    intro i
    simp only [digit_trunc, digit_xor]
    grind

private def narrowSignedAdd : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (add (signed w a) b) ==> bw w (add a b) by
    funext e
    change (signedVal (w e) (a e) + b e) % modulus (w e) = (a e + b e) % modulus (w e)
    rw [Int.add_emod, show signedVal (w e) (a e) % modulus (w e) = trunc (w e) (a e) from trunc_signed _ _]
    exact (Int.add_emod _ _ _).symm

private def narrowSignedSub : VerifiedRule (point v) :=
  egg_rule (w,a,b) : bw w (sub (signed w a) b) ==> bw w (sub a b) by
    funext e
    change (signedVal (w e) (a e) - b e) % modulus (w e) = (a e - b e) % modulus (w e)
    rw [Int.sub_emod, show signedVal (w e) (a e) % modulus (w e) = trunc (w e) (a e) from trunc_signed _ _]
    exact (Int.sub_emod _ _ _).symm

private def widenSigned : GuardedRule (point v) :=
  egg_rule (p,q,x) : signed q (signed p x) ==> signed p x when
    (h : Op.lt p q = one) (hq : Op.lt zero q = one) by
    funext e
    have hp' : p e < q e := truth_eq.mp (congrFun h e)
    have hq' : 0 < q e := truth_eq.mp (congrFun hq e)
    have h1 := fits_trunc (p e) (2 * trunc (p e - 1) (x e))
    have h2 := fits_trunc (p e) (x e)
    have := twice_modulus hp' hq'
    change signedVal (q e) (signedVal (p e) (x e)) = signedVal (p e) (x e)
    apply signed_small <;> unfold Fits at h1 h2 <;> unfold signedVal <;> omega

private def flipLowBits : VerifiedRule (point v) :=
  egg_rule (w,x) : bxor (bw w (lit (-1))) (bw w x) ==> bw w (bnot x) by
    funext e
    change bitXor (trunc (w e) (-1)) (trunc (w e) (x e)) = trunc (w e) (bitNot (x e))
    apply eq_of_digits
    intro i
    simp only [digit_xor, digit_trunc, digit_ones, digit_not]
    grind

private def plainRules : Array (VerifiedRule (point v)) :=
  #[flipLowBits, bitIncrement, factorCastMux, minusOneMul, narrowAnd, narrowOr,
    narrowXor, narrowSignedAdd, narrowSignedSub, pushMulSub, pushMulNeg, signedMux,
    lowAllOnes, zeroSub, negSum, negSub, subSub, addSubCancel,
    subSum, sumSubSelf, diffSubSelf, halfSucc, boolSelect, mulNegs,
    subAddCancel, shiftedSum, addMuxPair, bwTwo, bwSingleOne, addOneToTwo,
    addComm, addAssoc, mulComm, mulAssoc, subSelf, addZero,
    mulZero, mulOne, divOne, subZero, doubleNeg, subToAdd,
    mulTwo, dist, shiftLeft, shiftRight, powZero, powOne,
    sameSel, selZero, selOne, bwZero, repeatBw, fitsBw,
    fitsZero, leRefl, doubleOne]

private def guards : Array (GuardedRule (point v)) :=
  #[widenSigned, shrinkQuotient, leSucc, joinShl, joinShr, shrFits,
    shlFits, shlTrunc, fitsMul, ltToLe, transLe, ltLe,
    leLt, fitElim, wideBw, narrowBw, fitsAdd, fitOne,
    removeAddL, removeAddR, removeSubL, removeSubR, removeMulL, removeMulR]

/-- Demand for the relational analysis follows truncations in the query.
    Seeding a predicate creates a node. Only a rule with a checked proof can
    establish that the predicate is true. -/
private structure Demands where
  hasBitwise : Bool := false
  pureBitwise : Bool := true
  hasControl : Bool := false
  widths : List Pat := []
  facts : List Pat := []

private def addNew (xs : List Pat) (x : Pat) := if xs.contains x then xs else x :: xs

mutual
  private def demand (d : Demands) (p : Pat) : Demands :=
    match p with
    | .var _ => d
    | .app _ args =>
      let isGate := match p with
        | Op.band _ _ | Op.bor _ _ | Op.bxor _ _ | Op.bnot _ => true
        | _ => false
      let isControl := match p with
        | Op.signed _ _ | Op.sel _ _ _ => true
        | _ => false
      let d := {d with hasBitwise := d.hasBitwise || isGate,
                       hasControl := d.hasControl || isControl,
                       pureBitwise := d.pureBitwise && (args.isEmpty || isGate || (match p with | Op.bw _ _ => true | _ => false))}
      let d := match p with
        | Op.bw w x => {d with widths := addNew d.widths w, facts := addNew (addNew d.facts (Op.fits x w)) (Op.fits p w) }
        | Op.signed w _ => {d with widths := addNew d.widths w}
        | _ => d
      demandArgs d args
  private def demandArgs (d : Demands) : List Pat → Demands
    | [] => d
    | x :: xs => demandArgs (demand d x) xs
end

private abbrev Admitted (q : Query) := {ρ : Nat → Int // q.Holds ρ}
private abbrev queryModel (q : Query) := point (Subtype.val : Admitted q → Nat → Int)

private def importPremises (q : Query) (ps : Array {p // p ∈ q.assumptions})
    (s : Certified.Session (queryModel q)) :
    Option (Certified.Session (queryModel q)) :=
  ps.foldlM (init := s) fun s ⟨h, hh⟩ => do
    if hg : h.variables = [] then
      let (s, a) := s.add h
      let (s, b) := s.add Op.one
      s.merge? a b (fun assignment => by
        funext e
        rw [eval_point, eval_closed h hg _ _ (fun _ => 0)]
        exact e.property h hh)
    else none


abbrev Query.Evidence (q : Query) := PLift (∀ ρ, q.Holds ρ → q.Equivalent ρ)

/-- Replay hints are indices into checked premises, seed facts and proved rules. -/
inductive ReplayHint where
  | lanes
  | diagrams
  | saturation (keep : List Nat)
  | ground (keep : List Nat) (steps : List Certified.GroundStep)
  deriving Repr

structure SaturationInputs (q : Query) where
  premises : Array {p // p ∈ q.assumptions}
  seeds : Array Pat
  plain : Array (VerifiedRule (queryModel q))
  conditional : Array (GuardedRule (queryModel q))

private def saturationInputs (q : Query) (d : Demands) : SaturationInputs q := Id.run do
  let mut seeds := d.facts.toArray
  for w in d.widths do
    for u in d.widths do seeds := (seeds.push (Op.le w u)).push (Op.lt w u)
  let vars := (Subtype.val : Admitted q → Nat → Int)
  let rules := if d.hasBitwise then plainRules ++ bitRules ++ mixedRules else plainRules
  let rules := if d.hasControl then rules ++ controlRules vars else rules
  let guards := if d.hasBitwise then guards ++ bitGuarded else guards
  let guards := if d.hasControl then guards ++ controlGuards vars else guards
  return ⟨q.assumptions.attach.toArray, seeds, rules, guards⟩

def Query.saturationInputs (q : Query) : SaturationInputs q :=
  ParaBit.saturationInputs q (demand (demand {} q.lhs) q.rhs)

def SaturationInputs.count (ins : SaturationInputs q) :=
  ins.premises.size + ins.seeds.size + ins.plain.size + ins.conditional.size

private def selectIndices (xs : Array α) (keep : List Nat) (base : Nat) : Array α :=
  keep.foldl (fun out i =>
    if i < base then out else
      match xs[i - base]? with | some x => out.push x | none => out) #[]

def SaturationInputs.slice (ins : SaturationInputs q) (keep : List Nat) : SaturationInputs q :=
  let a := ins.premises.size
  let b := a + ins.seeds.size
  let c := b + ins.plain.size
  ⟨selectIndices ins.premises keep 0, selectIndices ins.seeds keep a,
   selectIndices ins.plain keep b, selectIndices ins.conditional keep c⟩

def SaturationInputs.initial? {q : Query} (ins : SaturationInputs q) := do
  let s ← importPremises q ins.premises (Certified.Session.empty (queryModel q))
  let (s, l) := s.add q.lhs
  let (s, r) := s.add q.rhs
  let s := ins.seeds.foldl (fun (g : Certified.Session (queryModel q)) p => (g.add p).1) s
  return (s, l, r)

private def evidence {q : Query} (s : Certified.Session (queryModel q))
    (l : Certified.Handle (queryModel q) q.lhs) (r : Certified.Handle (queryModel q) q.rhs) :
    Option q.Evidence := do
  let h ← s.equivalent? l r
  return ⟨fun ρ hr => by
    let e : Admitted q := ⟨ρ,hr⟩
    have he := congrFun (h.down (fun _ _ => 0)) e
    simpa only [eval_point, e, Query.Equivalent, denote] using he⟩

private def unconditional (r : VerifiedRule m) : GuardedRule m :=
  ⟨r.toRewrite, [], fun env _ => r.sound env⟩

private def SaturationInputs.ground? {q : Query} (ins : SaturationInputs q)
    (steps : List Certified.GroundStep) : Option q.Evidence := do
  let (s, l, r) ← ins.initial?
  let rules := ins.plain.map unconditional ++ ins.conditional
  evidence (s.replaySteps rules steps) l r

private def SaturationInputs.record? {q : Query} (ins : SaturationInputs q)
    (rounds : Nat) : Option (List Certified.GroundStep) := do
  let (initial, l, r) ← ins.initial?
  let rules := ins.plain.map unconditional
  let mut s := initial
  let mut steps := #[]
  for _ in [:rounds] do
    if (s.equivalent? l r).isSome then break
    let (s', plain) := s.recordRound rules
    let (s', guardedSteps) := s'.recordRound ins.conditional rules.size
    s := s'
    steps := steps ++ plain ++ guardedSteps
  guard (s.equivalent? l r).isSome
  guard (ins.ground? steps.toList).isSome
  return steps.toList

def SaturationInputs.prove? {q : Query} (ins : SaturationInputs q) (rounds : Nat) :
    Option q.Evidence := do
  let (s, l, r) ← ins.initial?
  let rules := ins.plain
  let analysis := ins.conditional
  let stages := [egg_schedule { run rules }, egg_schedule { guarded analysis }]
  let s := s.runUntilEqual l r stages rounds
  evidence s l r

private def proveWithDemands (q : Query) (demands : Demands) (rounds : Nat) :
    Option q.Evidence := do
  let lanes := if demands.hasBitwise then Decision.compareLanes q else .compared
  if let .proved h := lanes then return ⟨h⟩
  let needsDiagram := match lanes with | .needsDiagram => true | _ => false
  let decisionFirst := needsDiagram && demands.pureBitwise
  if decisionFirst then
    if let some h := Decision.equivalent? q.lhs q.rhs (8 * rounds) then
      return ⟨fun ρ _ => h.down ρ⟩
  if let some h := (ParaBit.saturationInputs q demands).prove? rounds then return h
  guard (needsDiagram && !decisionFirst)
  let h ← Decision.equivalent? q.lhs q.rhs (8 * rounds)
  return ⟨fun ρ _ => h.down ρ⟩

/-- A successful search provides a single universally quantified conditional
    equivalence for unbounded integer assignments. The abstract environment
    and all invariant witnesses are erased from compiled search. -/
def Query.prove? (q : Query) (rounds : Nat := 6) :
    Option (PLift (∀ ρ, q.Holds ρ → q.Equivalent ρ)) :=
  proveWithDemands q (demand (demand {} q.lhs) q.rhs) rounds

/-- Try range propagation and paired selection before general algebra on word
    control queries. Each of the two phases has its own round bound. The second
    phase continues from the first phase's graph and uses the ordinary rules.
    Other queries use the ordinary solver, including its bitwise procedures. -/
def Query.proveStaged? (q : Query) (rounds : Nat := 6) : Option q.Evidence := do
  let d := demand (demand {} q.lhs) q.rhs
  if !d.hasControl || d.hasBitwise then return ← proveWithDemands q d rounds
  let ins := ParaBit.saturationInputs q d
  let (s, l, r) ← ins.initial?
  let simplify := #[fitsBw, fitsZero, leRefl, repeatBw, bwZero, sameSel, selZero, selOne]
  let range := #[fitsSel, fitElim]
  let pair := #[addMuxPair]
  let truncate := #[bwMux]
  let stages := [egg_schedule { run simplify }, egg_schedule { guarded range },
                 egg_schedule { run pair }, egg_schedule { run truncate }]
  let s := s.runUntilEqual l r stages rounds
  if let some h := evidence s l r then return h
  let rules := ins.plain
  let analysis := ins.conditional
  let s := s.runUntilEqual l r [egg_schedule { run rules }, egg_schedule { guarded analysis }] rounds
  evidence s l r

def Query.replay? (q : Query) (rounds : Nat) (hint : ReplayHint) : Option q.Evidence := do
  match hint with
  | .lanes => Decision.replayLanes? q
  | .diagrams =>
    let h ← Decision.equivalent? q.lhs q.rhs (8 * rounds)
    return ⟨fun ρ _ => h.down ρ⟩
  | .saturation keep => (q.saturationInputs.slice keep).prove? rounds
  | .ground keep steps => (q.saturationInputs.slice keep).ground? steps

/-- Native export can try deleting blocks of hints. Every successful deletion
    is tested using the same sound runner as the kernel. Only the surviving
    indices are embedded in the emitted theorem. -/
private def minimize (items : List α) (valid : List α → Bool) : List α := Id.run do
  let mut keep := items
  let mut block := max 1 ((items.length + 1) / 2)
  for _ in [:items.length.log2 + 2] do
    let mut pos := 0
    for _ in [:keep.length] do
      if pos ≥ keep.length then break
      let candidate := keep.take pos ++ keep.drop (pos + block)
      if valid candidate then keep := candidate else pos := pos + block
    if block == 1 then break
    block := (block + 1) / 2
  return keep

def Query.replayHint? (q : Query) (rounds : Nat) : Option ReplayHint := do
  let d := demand (demand {} q.lhs) q.rhs
  if d.hasBitwise && (q.replay? rounds .lanes).isSome then return .lanes
  let inputs := ParaBit.saturationInputs q d
  if (inputs.prove? rounds).isSome then
    let keep := minimize (List.range inputs.count) (fun keep => ((inputs.slice keep).prove? rounds).isSome)
    let selected := inputs.slice keep
    if let some transcript := selected.record? rounds then
      let steps := minimize transcript fun steps => (selected.ground? steps).isSome
      return .ground keep steps
    return .saturation keep
  if d.hasBitwise && (q.replay? rounds .diagrams).isSome then return .diagrams
  none

end ParaBit
