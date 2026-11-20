import Egglog.Examples.ParaBitSemantics

open Egglog
namespace ParaBit
open Op

private def mux (c a b : Int) := if c % 2 = 1 then a else b

private theorem mux_map (f : Int → Int) (c a b : Int) :
    f (mux c a b) = mux c (f a) (f b) := by
  by_cases h : c % 2 = 1 <;> simp [mux, h]

private theorem mux_zip (f : Int → Int → Int) (c a b x y : Int) :
    f (mux c a b) (mux c x y) = mux c (f a x) (f b y) := by
  by_cases h : c % 2 = 1 <;> simp [mux, h]

variable {S : Type} {v : S → Nat → Int}

def bwMux : VerifiedRule (point v) :=
  egg_rule (p,c,a,b) : bw p (sel c a b) ==> sel c (bw p a) (bw p b) by
    funext e
    exact mux_map (trunc (p e)) _ _ _

private def addMux : VerifiedRule (point v) :=
  egg_rule (c,a,b,x) : add x (sel c a b) ==> sel c (add x a) (add x b) by
    funext e
    exact mux_map (x e + ·) _ _ _

private def mulMux : VerifiedRule (point v) :=
  egg_rule (c,a,b,x) : mul x (sel c a b) ==> sel c (mul x a) (mul x b) by
    funext e
    exact mux_map (x e * ·) _ _ _

private def xorMux : VerifiedRule (point v) :=
  egg_rule (c,a,b,x) : bxor x (sel c a b) ==> sel c (bxor x a) (bxor x b) by
    funext e
    exact mux_map (bitXor (x e)) _ _ _

private def andMux : VerifiedRule (point v) :=
  egg_rule (c,a,b,x) : band x (sel c a b) ==> sel c (band x a) (band x b) by
    funext e
    exact mux_map (bitAnd (x e)) _ _ _

private def orMux : VerifiedRule (point v) :=
  egg_rule (c,a,b,x) : bor x (sel c a b) ==> sel c (bor x a) (bor x b) by
    funext e
    exact mux_map (bitOr (x e)) _ _ _

private def subZip : VerifiedRule (point v) :=
  egg_rule (c,a,b,x,y) : sub (sel c a b) (sel c x y) ==> sel c (sub a x) (sub b y) by
    funext e
    exact mux_zip (· - ·) _ _ _ _ _

private def truncBitMux : VerifiedRule (point v) :=
  egg_rule (a,b,c) : sel (bw one c) a b ==> sel c a b by
    funext e
    change mux ((c e) % 2) (a e) (b e) = mux (c e) (a e) (b e)
    simp [mux]

private theorem twice_low (w x : Int) :
    trunc w (2 * trunc (w-1) x) = trunc w (2*x) := by
  by_cases hw : w ≤ 0
  · simp [trunc, modulus, Int.toNat_of_nonpos hw]
  · have h : w.toNat = (w-1).toNat + 1 := by omega
    have hm : modulus w = 2 * modulus (w-1) := by
      unfold modulus
      rw [h, Int.pow_succ, Int.mul_comm]
    simp only [trunc, hm, Int.mul_emod_mul_of_pos _ _ (by decide : (0:Int) < 2), Int.emod_emod]

theorem trunc_signed (w x : Int) : trunc w (signedVal w x) = trunc w x := by
  rw [signedVal, twice_low]
  change ((2*x) % modulus w - x % modulus w) % modulus w = x % modulus w
  rw [Int.emod_sub_emod, Int.sub_emod_emod, show 2*x-x = x by omega]

private def dropSigned : VerifiedRule (point v) :=
  egg_rule (p,x) : bw p (signed p x) ==> bw p x by
    funext e
    exact trunc_signed _ _

theorem signed_trunc (w x : Int) : signedVal w (trunc w x) = signedVal w x := by
  unfold signedVal
  rw [twice_low, twice_low]
  unfold trunc
  rw [Int.mul_emod 2 (x % modulus w), Int.emod_emod, ← Int.mul_emod]

private def signedInput : VerifiedRule (point v) :=
  egg_rule (p,x) : signed p (bw p x) ==> signed p x by
    funext e
    exact signed_trunc _ _

theorem signed_small {x w : Int} (lo : -(modulus w) ≤ 2*x) (hi : 2*x < modulus w) :
    signedVal w x = x := by
  have pos : 0 < modulus w := Int.pow_pos (by decide)
  rw [signedVal, twice_low]
  unfold trunc
  by_cases h : 0 ≤ x
  · rw [Int.emod_eq_of_lt (by omega) hi, Int.emod_eq_of_lt h (by omega)]
    omega
  · have wrap (a : Int) (ha : -modulus w ≤ a) (hn : a < 0) :
        a % modulus w = modulus w + a := by
      rw [← Int.add_emod_left (modulus w) a, Int.emod_eq_of_lt (by omega) (by omega)]
    rw [wrap (2*x) lo (by omega), wrap x (by omega) (by omega)]
    omega

theorem twice_modulus {p w : Int} (hp : p < w) (hw : 0 < w) :
    2 * modulus p ≤ modulus w := by
  change 2 * 2 ^ p.toNat ≤ 2 ^ w.toNat
  rw [Int.mul_comm, ← Int.pow_succ]
  exact Int.ofNat_le.mpr (Nat.pow_le_pow_right (by decide) (by omega : p.toNat + 1 ≤ w.toNat))

private theorem holds {p : Prop} [Decidable p] (h : truth p = 1) : p := by
  by_cases hp : p <;> simp_all [truth]

private def signedFits : GuardedRule (point v) :=
  egg_rule (q,x,p) : signed q x ==> x when (hf : fits x p = one)
      (hlt : Op.lt p q = one) (hp : Op.lt zero q = one) by
    funext e
    obtain ⟨ha,hb⟩ : Fits (x e) (p e) := holds (congrFun hf e)
    have lt : p e < q e := holds (congrFun hlt e)
    have pos : 0 < q e := holds (congrFun hp e)
    have := twice_modulus lt pos
    change signedVal (q e) (x e) = x e
    apply signed_small <;> omega

private def signedDiff : GuardedRule (point v) :=
  egg_rule (w,x,y,p,q) : signed w (sub x y) ==> sub x y
      when (hx : fits x p = one) (hy : fits y q = one)
      (hp : Op.lt p w = one) (hq : Op.lt q w = one) (hw : Op.lt zero w = one) by
    funext e
    obtain ⟨xa,xb⟩ : Fits (x e) (p e) := holds (congrFun hx e)
    obtain ⟨ya,yb⟩ : Fits (y e) (q e) := holds (congrFun hy e)
    have pw : p e < w e := holds (congrFun hp e)
    have qw : q e < w e := holds (congrFun hq e)
    have pos : 0 < w e := holds (congrFun hw e)
    have := twice_modulus pw pos
    have := twice_modulus qw pos
    change signedVal (w e) (x e - y e) = x e - y e
    apply signed_small <;> omega

def controlGuards (v : S → Nat → Int) : Array (GuardedRule (point v)) := #[signedFits, signedDiff]

def controlRules (v : S → Nat → Int) : Array (VerifiedRule (point v)) :=
  #[bwMux, addMux, mulMux, xorMux, andMux, orMux, subZip, truncBitMux, dropSigned, signedInput]

end ParaBit
