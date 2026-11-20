import Egglog.Compiler

open Egglog Egglog.Compiler

namespace BooleanPass

/-- An existing source IR. Register reads refer to the compiler's caller-supplied inputs. -/
inductive Expr where
  | reg (i : Nat)
  | lit (value : Bool)
  | neg (x : Expr)
  | conj (a b : Expr)
  | disj (a b : Expr)
  | mux (guard a b : Expr)
  deriving DecidableEq, Repr

def Expr.eval (regs : Nat → Bool) : Expr → Bool
  | .reg i => regs i
  | .lit value => value
  | .neg x => !(x.eval regs)
  | .conj a b => a.eval regs && b.eval regs
  | .disj a b => a.eval regs || b.eval regs
  | .mux c a b => if c.eval regs then a.eval regs else b.eval regs

egg_signature Ops : Bool where
  | yes => true
  | no => false
  | neg x => !x
  | conj a b => a && b
  | disj a b => a || b
  | mux c a b => if c then a else b
  otherwise false

namespace Expr

egg_codec encode, decode : Expr using Ops where
  | reg i <=> .var i
  | lit true <=> Ops.yes
  | lit false <=> Ops.no

end Expr

theorem Expr.encode_sound (src : Expr) (regs : Nat → Bool) :
    src.encode.eval Ops.model regs = src.eval regs := by
  induction src with
  | lit value => cases value <;> rfl
  | reg i => rfl
  | _ => simp_all [encode, eval, Ops.model, Pat.eval, Pat.evalArgs]

def ir : Language Expr Bool where
  model := Ops.model
  encode := Expr.encode
  decode := Expr.decode
  denote src env := src.eval env
  encode_sound := Expr.encode_sound

open Ops

def andTrue : VerifiedRule model :=
  egg_rule (x) : conj x yes ==> x by exact Bool.and_true x

def falseOr : VerifiedRule model :=
  egg_rule (x) : disj no x ==> x by exact Bool.false_or x

def doubleNeg : VerifiedRule model :=
  egg_rule (a) : neg (neg a) ==> a by exact Bool.not_not a

def sameBranch : VerifiedRule model :=
  egg_rule (c, x) : mux c x x ==> x by exact ite_self x

def algebra := #[andTrue, falseOr, doubleNeg]
def control := #[sameBranch]

def phases : Certified.Schedule ir.model := egg_schedule {
  run control,
  saturate 4 { run algebra, run control }
}

def optimize (input : Expr) : Expr := (ir.optimize phases input).output

/-- Every program optimized by the pass has the same meaning in every register file. -/
theorem optimize_correct (input : Expr) (regs : Nat → Bool) :
    (optimize input).eval regs = input.eval regs := by
  unfold optimize
  exact (ir.optimize phases input).sound regs

end BooleanPass
