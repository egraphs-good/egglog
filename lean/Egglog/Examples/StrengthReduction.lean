import Egglog.Compiler

open Egglog Egglog.Compiler

namespace StrengthReduction

/-- Natural-number IR with a variable-count left shift. -/
inductive Expr where
  | arg (n : Nat)
  | one
  | plus (a b : Expr)
  | times (a b : Expr)
  | pow2 (count : Expr)
  | shl (value count : Expr)
  deriving DecidableEq, Repr

def Expr.eval (inputs : Nat → Nat) : Expr → Nat
  | .arg n => inputs n
  | .one => 1
  | .plus a b => a.eval inputs + b.eval inputs
  | .times a b => a.eval inputs * b.eval inputs
  | .pow2 n => 2 ^ n.eval inputs
  | .shl a n => a.eval inputs <<< n.eval inputs

egg_signature Ops : Nat where
  | one => 1
  | plus a b costing 2 => a + b
  | times a b costing 4 => a * b
  | pow2 n => 2 ^ n
  | shl a n => a <<< n
  otherwise 0

namespace Expr

egg_codec encode, decode : Expr using Ops where
  | arg i <=> .var i

end Expr

def ir : Language Expr Nat where
  model := Ops.model
  denote src env := src.eval env
  encode := Expr.encode
  decode := Expr.decode
  encode_sound src env := by
    induction src <;> simp_all [Expr.encode, Expr.eval, Ops.model, Pat.eval, Pat.evalArgs]

open Ops

def doubling : VerifiedRule model :=
  egg_rule (x) : plus x x ==> times x (pow2 one) by
    change x + x = x * 2
    exact (Nat.mul_two x).symm

def twoOnes : VerifiedRule model :=
  egg_rule () : plus one one ==> pow2 one by rfl

/-- The exponent `n` is introduced by matching the guard against the class of
`y`. Other rules can discover that class equality in an earlier phase. -/
def mulPow2 : GuardedRule model :=
  egg_rule (x, y, n) : times x y ==> shl x n when (h : y = pow2 n) by
    change y = 2 ^ n at h
    change x * y = x <<< n
    rw [h, Nat.shiftLeft_eq]

def mulOnes : GuardedRule model :=
  egg_rule (x, y) : times x y ==> one when (hx : x = one) (hy : y = one) by
    change x = 1 at hx
    change y = 1 at hy
    change x * y = 1
    rw [hx, hy]

def phases : Certified.Schedule ir.model := egg_schedule {
  run #[doubling, twoOnes],
  guarded #[mulPow2, mulOnes]
}

def optimize (input : Expr) := ir.optimize phases input Ops.cost

end StrengthReduction
