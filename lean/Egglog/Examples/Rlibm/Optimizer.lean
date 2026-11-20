import Egglog.DSL
import Egglog.Examples.Rlibm.Polynomial

/-!
# Scheduling a range-reduced cubic

The seven input registers are the binary exponent, its compensation factor,
the reduced mantissa, and four polynomial coefficients. All operands are exact
fixed-point integers. The only rounding to bfloat16 occurs after reconstruction.
-/

open Egglog Egglog.Compiler

namespace Rlibm.Schedule

/-- The power-of-two component of the split compensation factor. For Q40
    this is 1/4; the residual is exact for both logarithm bases. -/
def quarter : Int := 2^38

egg_language Expr using Ops : Int with_vars arg where
  | step t a b => a + t * b
  | compensate e k p => e * k + p
  | block t a b => a + (t * t) * b
  | splitOffset e k t a b => (e * quarter + t * b) + ((k - quarter) * e + a)
  otherwise 0

def source : Expr :=
  let t := Expr.arg 2
  .compensate (.arg 0) (.arg 1) (.step t (.arg 3) (.step t (.arg 4) (.step t (.arg 5) (.arg 6))))

open Ops

/-- Form the compensated constant while the remaining Horner chain evaluates. -/
def early : VerifiedRule Ops.model :=
  egg_rule (e, k, t, a, b) : compensate e k (step t a b) ==> step t (step e a k) b by
    change e * k + (a + t * b) = (a + e * k) + t * b
    omega

/-- Evaluate the low- and high-degree coefficient pairs as independent linear forms. -/
def estrin : VerifiedRule Ops.model :=
  egg_rule (t, a, b, c, d) : step t a (step t b (step t c d)) ==>
    block t (step t a b) (step t c d) by
    change a + t * (b + t * (c + t*d)) = a + t*b + (t*t)*(c+t*d)
    rw [Int.mul_add, Int.mul_assoc, Int.add_assoc]

/-- Split the range compensation into a shift and a residual multiply. -/
def splitFactor : VerifiedRule Ops.model :=
  egg_rule (e, k, t, a, b) : compensate e k (step t a b) ==> splitOffset e k t a b by
    change e*k + (a + t*b) = (e*quarter + t*b) + ((k - quarter)*e + a)
    rw [Int.sub_mul, Int.mul_comm k e, Int.mul_comm quarter e]
    omega

def phases : Certified.Schedule Ops.model := egg_schedule { saturate { run #[early, estrin, splitFactor] } }

inductive Profile where
  | horner | estrin | split
  deriving Repr, DecidableEq

private def tag : Pat → Nat
  | .app op _ => op
  | .var _ => 0

/-- Extraction preferences expose each schedule for measurement. These weights
    distinguish instruction shapes; they are not fitted CPU cycle estimates. -/
def cost (profile : Profile) (op : Nat) : Nat :=
  let x := Pat.var 0
  if op == tag (Ops.step x x x) then 4
  else if op == tag (Ops.compensate x x x) then 9
  else if op == tag (Ops.block x x x) then if profile == .estrin then 1 else 10
  else if op == tag (Ops.splitOffset x x x x x) then if profile == .split then 1 else 20
  else 1

private def ir := Expr.language Ops.model

def selected (p : Profile) : Expr := (ir.optimize phases source (cost p)).output

theorem selected_eq (p : Profile) (env : Nat → Int) :
    (selected p).eval env = source.eval env :=
  (ir.optimize phases source (cost p)).sound env

/-- Concrete extraction regressions check that all three cost profiles choose
    different programs using the same rules and input. -/
private def hornerShape : Expr :=
  let t := Expr.arg 2
  .step t (.step (.arg 0) (.arg 3) (.arg 1)) (.step t (.arg 4) (.step t (.arg 5) (.arg 6)))

private def estrinShape : Expr :=
  let t := Expr.arg 2
  .block t (.step t (.step (.arg 0) (.arg 3) (.arg 1)) (.arg 4)) (.step t (.arg 5) (.arg 6))

private def splitShape : Expr :=
  let t := Expr.arg 2
  .splitOffset (.arg 0) (.arg 1) t (.arg 3) (.step t (.arg 4) (.step t (.arg 5) (.arg 6)))

#guard selected .horner == hornerShape
#guard selected .estrin == estrinShape
#guard selected .split == splitShape

/-- Arithmetic used by the emitted C kernel. Unsigned C11 addition and
    multiplication have exactly the wraparound semantics of `BitVec 64`. -/
inductive WordExpr where
  | arg (i : Nat)
  | imm (z : Int)
  | add (a b : WordExpr)
  | mul (a b : WordExpr)
  deriving Repr

def WordExpr.word (env : Nat → BitVec 64) : WordExpr → BitVec 64
  | .arg i => env i
  | .imm z => BitVec.ofInt 64 z
  | .add a b => a.word env + b.word env
  | .mul a b => a.word env * b.word env

def WordExpr.integer (env : Nat → Int) : WordExpr → Int
  | .arg i => env i
  | .imm z => z
  | .add a b => a.integer env + b.integer env
  | .mul a b => a.integer env * b.integer env

theorem WordExpr.word_eq (expr : WordExpr) (env : Nat → Int) :
    expr.word (fun i => BitVec.ofInt 64 (env i)) = BitVec.ofInt 64 (expr.integer env) := by
  induction expr <;> simp only [word, integer, BitVec.ofInt_add, BitVec.ofInt_mul, *]

def Expr.lower : Expr → WordExpr
  | .arg i => .arg i
  | .step t a b => .add a.lower (.mul t.lower b.lower)
  | .compensate e k p => .add (.mul e.lower k.lower) p.lower
  | .block t a b => .add a.lower (.mul (.mul t.lower t.lower) b.lower)
  | .splitOffset e k t a b =>
    .add (.add (.mul e.lower (.imm quarter)) (.mul t.lower b.lower))
      (.add (.mul (.add k.lower (.imm (-quarter))) e.lower) a.lower)

theorem Expr.lower_sound (expr : Expr) (env : Nat → Int) : expr.lower.integer env = expr.eval env := by
  induction expr <;> simp only [Expr.lower, WordExpr.integer, Expr.eval, *] <;> rfl

/-- Machine wraparound cannot invalidate an intermediate reassociation. The
    result is the low 64 bits of the original exact accumulator on all operands. -/
theorem selected_word_eq (p : Profile) (env : Nat → Int) :
    (selected p).lower.word (fun i => BitVec.ofInt 64 (env i)) =
      BitVec.ofInt 64 (source.eval env) := by
  rw [WordExpr.word_eq, Expr.lower_sound, selected_eq]

def inputs (e k t c0 c1 c2 c3 : Int) : Nat → Int
  | 0 => e | 1 => k | 2 => t | 3 => c0 | 4 => c1 | 5 => c2 | 6 => c3 | _ => 0

@[noinline] private def evaluator (tree : Expr) : Accumulator :=
  fun e k t c0 c1 c2 c3 => tree.eval (inputs e k t c0 c1 c2 c3)

structure Prepared where
  program : Expr
  eval : Accumulator

/-- Run equality saturation during preparation and capture the extracted AST
    in the returned native Lean closure. Reuse that closure for each input. -/
@[noinline] def prepare (p : Profile) : Prepared :=
  let program := selected p
  ⟨program, evaluator program⟩

theorem accumulator_eq (p : Profile) (e k t c0 c1 c2 c3 : Int) :
    (prepare p).eval e k t c0 c1 c2 c3 = e*k + (c0 + t*(c1 + t*(c2 + t*c3))) :=
  selected_eq p (inputs e k t c0 c1 c2 c3)

theorem prepare_eq (p : Profile) : (prepare p).eval = accum := by
  funext e k t c0 c1 c2 c3
  exact accumulator_eq p e k t c0 c1 c2 c3

theorem optimized_eq_core (p : Profile) (base : LogBase) (raw : Nat) :
    evalWith (prepare p).eval base raw = core base raw := by
  rw [prepare_eq]
  rfl

/-- Execute the arithmetic IR with machine semantics, interpreting the final
    64 bits as signed for the common rounding routine. -/
def machine (p : Profile) : Accumulator := fun e k t c0 c1 c2 c3 =>
  let env := inputs e k t c0 c1 c2 c3
  ((selected p).lower.word (fun i => BitVec.ofInt 64 (env i))).toInt

theorem reduced_machine_eq (p : Profile) (base : LogBase) (raw : Nat) :
    reduced (machine p) base raw = reduced accum base raw := by
  simp only [reduced, machine, selected_word_eq]
  have fits := reduced_fits base raw
  change (BitVec.ofInt 64 (reduced accum base raw)).toInt = reduced accum base raw
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> omega

theorem machine_eq_core (p : Profile) (base : LogBase) (raw : Nat) :
    evalWith (machine p) base raw = core base raw := by
  simp only [core, evalWith, reduced_machine_eq]

end Rlibm.Schedule
