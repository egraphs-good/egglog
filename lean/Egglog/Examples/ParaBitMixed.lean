import Egglog.Examples.ParaBitBitwise
import Egglog.Examples.ParaBitControl
import Init.Data.BitVec.Bitblast

open Egglog
namespace ParaBit
open Op

private theorem wordBit (w : Nat) (x : Int) (i : Nat) :
    (BitVec.ofInt w x).getLsbD i = (decide (i < w) && digit x i) := by
  cases x <;> simp [digit, BitVec.getLsbD_ofNat] <;> grind

private theorem words_ext {x y : Int} (h : ∀ w, BitVec.ofInt w x = BitVec.ofInt w y) : x = y := by
  apply eq_of_digits
  intro i
  have hb := congrArg (fun (z : BitVec (i+1)) => z.getLsbD i) (h (i+1))
  simpa only [wordBit, Nat.lt_succ_self, decide_true, Bool.true_and] using hb

/-- Disjoint bit positions have carry-free addition at every word size. -/
private theorem disjoint_add {x y : Int} (h : bitAnd x y = 0) : x + y = bitOr x y := by
  apply words_ext
  intro w
  have disjoint : BitVec.ofInt w x &&& BitVec.ofInt w y = 0#w := by
    apply BitVec.eq_of_getLsbD_eq
    intro i hi
    have hc := congrArg (fun z => digit z i) h
    simp only [digit_and, digit_zero] at hc
    simp only [BitVec.getLsbD_and, BitVec.getLsbD_zero, wordBit, hi, decide_true, Bool.true_and, hc]
  rw [BitVec.ofInt_add, BitVec.add_eq_or_of_and_eq_zero _ _ disjoint]
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_or, wordBit, hi, decide_true, Bool.true_and, digit_or]

private theorem andXorSum (x y : Int) : bitAnd x y + bitXor x y = bitOr x y := by
  rw [disjoint_add]
  · apply eq_of_digits
    intro i
    simp only [digit_or, digit_and, digit_xor]
    grind
  · apply eq_of_digits
    intro i
    simp only [digit_and, digit_xor, digit_zero]
    grind

/-- The mutually disjoint minterms of the Boolean ring. -/
private theorem sum_bits (x y : Int) : x + y = bitXor x y + 2 * bitAnd x y := by
  let a := bitAnd x (bitNot y)
  let b := bitAnd (bitNot x) y
  let c := bitAnd x y
  have ac : a + c = x := by
    rw [disjoint_add] <;> apply eq_of_digits <;> intro i <;> simp only [a,c,digit_and,digit_or,digit_not,digit_zero] <;> grind
  have bc : b + c = y := by
    rw [disjoint_add] <;> apply eq_of_digits <;> intro i <;> simp only [b,c,digit_and,digit_or,digit_not,digit_zero] <;> grind
  have ab : a + b = bitXor x y := by
    rw [disjoint_add] <;> apply eq_of_digits <;> intro i <;> simp only [a,b,digit_and,digit_or,digit_xor,digit_not,digit_zero] <;> grind
  change x + y = bitXor x y + 2 * c
  omega

variable {S : Type} {v : S → Nat → Int}

private def joinBits : VerifiedRule (point v) :=
  egg_rule (x,y) : add (band x y) (bxor x y) ==> bor x y by funext e; exact andXorSum _ _
private def diffBits : VerifiedRule (point v) :=
  egg_rule (x,y) : sub (bor x y) (bxor x y) ==> band x y by
    funext e; change bitOr (x e) (y e) - bitXor (x e) (y e) = bitAnd (x e) (y e)
    have := andXorSum (x e) (y e); omega
private def xorDifference : VerifiedRule (point v) :=
  egg_rule (x,y) : sub (bor x y) (band x y) ==> bxor x y by
    funext e; change bitOr (x e) (y e) - bitAnd (x e) (y e) = bitXor (x e) (y e)
    have := andXorSum (x e) (y e); omega
private def carrySum : VerifiedRule (point v) :=
  egg_rule (x,y) : add (bxor x y) (mul two (band x y)) ==> add x y by
    funext e; exact (sum_bits _ _).symm
private def notSubtract : VerifiedRule (point v) :=
  egg_rule (x) : bnot x ==> sub (lit (-1)) x by
    funext e; change -x e - 1 = -1 - x e; omega
private def negNot : VerifiedRule (point v) :=
  egg_rule (x) : add (bnot x) one ==> neg x by
    funext e; change -x e - 1 + 1 = -x e; omega
private def negUnitMul : VerifiedRule (point v) :=
  egg_rule (x) : mul (lit (-1)) x ==> neg x by
    funext e; exact Int.neg_one_mul _
private def complementXor : VerifiedRule (point v) :=
  egg_rule (x,y) : bxor (band x y) y ==> bnot (bor x (bnot y)) by
    funext e
    change bitXor (bitAnd (x e) (y e)) (y e) = bitNot (bitOr (x e) (bitNot (y e)))
    apply eq_of_digits; intro i; simp only [digit_not, digit_xor, digit_and, digit_or]; grind

/-- Affine transfer rules expose cancellation across complement, subtraction and
    addition without expanding arbitrary additions into Boolean gates. -/
private def notDiff : VerifiedRule (point v) :=
  egg_rule (x,y) : bnot (sub x y) ==> add y (bnot x) by
    funext e; change -(x e - y e) - 1 = y e + (-x e - 1); omega
private def notSum : VerifiedRule (point v) :=
  egg_rule (x,y) : bnot (add x y) ==> sub (bnot x) y by
    funext e; change -(x e + y e) - 1 = (-x e - 1) - y e; omega
private def diffNot : VerifiedRule (point v) :=
  egg_rule (x,y) : sub x (bnot y) ==> add y (add x one) by
    funext e; change x e - (-y e - 1) = y e + (x e + 1); omega
private def sumNot : VerifiedRule (point v) :=
  egg_rule (x,y) : add (bnot x) y ==> sub (sub y one) x by
    funext e; change -x e - 1 + y e = y e - 1 - x e; omega

private theorem oneSign (a : Int) : signedVal 1 a = (if a % 2 = 1 then -1 else 0) := by
  change ((2 * (a % 1)) % 2) - a % 2 = _
  split <;> omega

private def signedBooleanMask : VerifiedRule (point v) :=
  egg_rule (x,y) : band (signed one x) y ==> sel x y zero by
    funext e
    change bitAnd (signedVal 1 (x e)) (y e) = (if x e % 2 = 1 then y e else 0)
    rw [oneSign]; split <;> apply eq_of_digits <;> intro i <;> simp [digit_and, digit_ones, digit_zero]

/-- BWLang signed conversion agrees with Lean’s signed bitvector interpretation. -/
theorem signed_as_word (w x : Int) : signedVal w x = (BitVec.ofInt w.toNat x).toInt := by
  let t := BitVec.ofInt w.toNat x
  have same : trunc w t.toInt = trunc w x := by
    apply eq_of_digits
    intro i
    have h : BitVec.ofInt w.toNat t.toInt = BitVec.ofInt w.toNat x := BitVec.ofInt_toInt
    simpa only [wordBit, digit_trunc] using congrArg (fun (z : BitVec w.toNat) => z.getLsbD i) h
  have small : signedVal w t.toInt = t.toInt :=
    signed_small (BitVec.le_two_mul_toInt (x := t)) (BitVec.two_mul_toInt_lt (x := t))
  rw [← signed_trunc w x, ← same, signed_trunc, small]

private theorem wordAnd (x y : Int) (w : Nat) :
    BitVec.ofInt w (bitAnd x y) = BitVec.ofInt w x &&& BitVec.ofInt w y := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [wordBit, digit_and, BitVec.getLsbD_and, hi, decide_true, Bool.true_and]

private def andSigns : VerifiedRule (point v) :=
  egg_rule (w,x,y) : band (signed w x) (signed w y) ==> signed w (band x y) by
    funext e
    change bitAnd (signedVal (w e) (x e)) (signedVal (w e) (y e)) = signedVal (w e) (bitAnd (x e) (y e))
    apply words_ext; intro u
    simp only [signed_as_word, wordAnd]
    exact BitVec.signExtend_and.symm

def mixedRules : Array (VerifiedRule (point v)) := #[joinBits, diffBits, xorDifference, carrySum,
  notSubtract, negNot, negUnitMul, complementXor, notDiff, notSum, diffNot, sumNot, signedBooleanMask, andSigns]
end ParaBit
