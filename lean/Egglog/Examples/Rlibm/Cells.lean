import Egglog.DSL
import Egglog.Examples.Rlibm.CellsCertificate

namespace Rlibm.Cells

/-- The unsigned quadratic in the emitted C uses modulo-2^64 addition and
    multiplication. Its signed result equals the exact fixed-point value. -/
def narrowWord (b : LogBase) (raw : Nat) : BitVec 64 :=
  let (e,m) := rangeReduce raw
  let c := quadratic b m
  let word := BitVec.ofInt 64
  let t := word (m % 8 : Nat)
  word e * word (exponentStep b) + (word c.c0 + t*(word c.c1 + t*word c.c2))

theorem narrowWord_exact (b : LogBase) (raw : Nat) :
    (narrowWord b raw).toInt = narrowValue b raw := by
  have hc : ∀ i : Fin 16,
      let c := ((quadratics b)[i.val]?).getD default
      c.c0.natAbs ≤ 2^40 ∧ c.c1.natAbs ≤ 2^40 ∧ c.c2.natAbs ≤ 2^40 := by cases b <;> decide
  obtain ⟨h0,h1,h2⟩ := hc ⟨(rangeReduce raw).2 / 8 % 16, Nat.mod_lt _ (by decide)⟩
  have ht : (((rangeReduce raw).2 % 8 : Nat) : Int).natAbs ≤ 15 := by
    simp only [Int.natAbs_natCast]; omega
  have emod (n : Nat) : ((n % 8 : Nat) : Int) = (n : Int) % 8 := rfl
  have fits : (narrowValue b raw).natAbs < 2^63 := by
    simpa only [narrowValue, quadratic, accum, emod, Int.mul_zero, Int.add_zero] using
      accum_fits b raw ht h0 h1 h2 (show (0 : Int).natAbs ≤ 2^40 by decide)
  have h : narrowWord b raw = BitVec.ofInt 64 (narrowValue b raw) := by
    simp only [narrowWord, narrowValue, emod, BitVec.ofInt_add, BitVec.ofInt_mul]
  rw [h]
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> omega

open Egglog Egglog.Compiler

egg_language Expr using Ops : Nat with_vars input where
  | wide2 x => core .two (x % 65536)
  | narrow2 x => narrow .two (x % 65536)
  | wide10 x => core .ten (x % 65536)
  | narrow10 x => narrow .ten (x % 65536)
  otherwise 0

/- A candidate is eligible for extraction only when it gives the baseline's
    rounded word for every valuation of the input. -/
egg_functions Candidates : Nat over Ops.model where
  | log2 x ↦ y => y = core .two (x % 65536)
    merge old _new => old
    preserving valid _incoming by exact valid
  | log10 x ↦ y => y = core .ten (x % 65536)
    merge old _new => old
    preserving valid _incoming by exact valid

private def ir := Expr.language (Candidates.model Ops.model)
open Ops

def offer2 : RelRule ir.model :=
  egg_rule (x, y) : when (_source : y = wide2 x) set Candidates.log2 x := narrow2 x by
    exact narrow_eq .two ⟨x % 65536, Nat.mod_lt _ (by decide)⟩

def offer10 : RelRule ir.model :=
  egg_rule (x, y) : when (_source : y = wide10 x) set Candidates.log10 x := narrow10 x by
    exact narrow_eq .ten ⟨x % 65536, Nat.mod_lt _ (by decide)⟩

def resegment2 : RelRule ir.model :=
  egg_rule (x, y) : when (valid : Candidates.log2 x ↦ y) rewrite wide2 x ==> y by
    exact valid.symm

def resegment10 : RelRule ir.model :=
  egg_rule (x, y) : when (valid : Candidates.log10 x ↦ y) rewrite wide10 x ==> y by
    exact valid.symm

private def source (b : LogBase) : Expr :=
  match b with | .two => .wide2 (.input 0) | .ten => .wide10 (.input 0)

private def phases : Certified.Schedule ir.model := egg_schedule {
  relational #[offer2, offer10], relational #[resegment2, resegment10]
}

/-- Estimate the per-input multiplication count, preferring the quadratic. -/
private def cost (op : Nat) : Nat :=
  match Ops.wide2 (.var 0), Ops.wide10 (.var 0) with
  | .app a _, .app b _ => if op == a || op == b then 4 else 3
  | _, _ => 3

def selected (b : LogBase) := (ir.optimize phases (source b) cost).output

#guard selected .two == .narrow2 (.input 0)
#guard selected .ten == .narrow10 (.input 0)

@[noinline] private def evaluate (tree : Expr) : Nat → Nat := fun raw => tree.eval (fun _ => raw)
structure Prepared where
  program : Expr
  eval : Nat → Nat

/-- Keep extraction in the preparation call when compiling the Lean evaluator. -/
@[noinline] def prepare (b : LogBase) : Prepared :=
  let tree := selected b
  ⟨tree, evaluate tree⟩

theorem prepare_eq_core (b : LogBase) (raw : Fin 65536) : (prepare b).eval raw = core b raw := by
  have h := (ir.optimize phases (source b) cost).sound (fun _ => raw.val)
  change (selected b).eval (fun _ => raw.val) = (source b).eval (fun _ => raw.val) at h
  change (selected b).eval _ = _
  rw [h]
  cases b
  · change core .two (raw.val % 65536) = core .two raw.val
    rw [Nat.mod_eq_of_lt raw.isLt]
  · change core .ten (raw.val % 65536) = core .ten raw.val
    rw [Nat.mod_eq_of_lt raw.isLt]

-- These are different approximants before the single final rounding.
#guard narrowValue .two 0x3f82 != reduced accum .two 0x3f82
#guard narrowValue .ten 0x3f82 != reduced accum .ten 0x3f82

/-- Rounding down to Q16 before the final bfloat16 rounding of log2 changes the answer by one bfloat16
    encoding on an input just below one. -/
theorem premature_rounding_changes_log2 :
    let v := reduced accum .two 0x3f77
    roundQ40 v = 0xbd53 ∧ roundQ40 (v / 2^24 * 2^24) = 0xbd54 ∧
      inCell 0xbd53 (v / 2^24 * 2^24) = false := by decide +kernel

end Rlibm.Cells
