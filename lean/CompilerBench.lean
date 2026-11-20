import Egglog.Examples.Boolean
import Egglog.Examples.BoundsCheck
import Egglog.Examples.StrengthReduction
import Egglog.Parser

open Egglog

private def register (i : String) : Except String Nat :=
  match i.toNat? with
  | some n => .ok n
  | none => .error "expected a register number"

private partial def readBool : SExpr → Except String BooleanPass.Expr
  | .list #[.atom "Reg", .atom i] => return .reg (← register i)
  | .list #[.atom "Yes"] => pure (.lit true)
  | .list #[.atom "No"] => pure (.lit false)
  | .list #[.atom "Neg", x] => return .neg (← readBool x)
  | .list #[.atom "Conj", a, b] => return .conj (← readBool a) (← readBool b)
  | .list #[.atom "Disj", a, b] => return .disj (← readBool a) (← readBool b)
  | .list #[.atom "Mux", c, a, b] => return .mux (← readBool c) (← readBool a) (← readBool b)
  | _ => throw "unknown Boolean expression"

private partial def readNat : SExpr → Except String StrengthReduction.Expr
  | .list #[.atom "Reg", .atom i] => return .arg (← register i)
  | .list #[.atom "One"] => pure .one
  | .list #[.atom "Plus", a, b] => return .plus (← readNat a) (← readNat b)
  | .list #[.atom "Times", a, b] => return .times (← readNat a) (← readNat b)
  | .list #[.atom "Pow2", n] => return .pow2 (← readNat n)
  | .list #[.atom "Shl", a, n] => return .shl (← readNat a) (← readNat n)
  | _ => throw "unknown natural-number expression"

private def readFile (read : SExpr → Except String α) (path : String) : IO α := do
  let #[sexp] ← IO.ofExcept (parseSExprs (← IO.FS.readFile path))
    | throw (IO.userError "expected one program")
  IO.ofExcept (read sexp)

private def timePass [BEq α] (read : SExpr → Except String α) (pass : α → α)
    (input expected : String) : IO Unit := do
  let ref ← IO.mkRef (← readFile read input)
  let want ← readFile read expected
  let before ← IO.monoNanosNow
  ref.modify pass
  let after ← IO.monoNanosNow
  unless (← ref.get) == want do throw (IO.userError "unexpected optimized IR")
  IO.println s!"compiler_ns={after - before}"

namespace BoundsSchedule

open BoundsCheck

private def sample (i : Nat) : Pat × Pat × Pat :=
  let x := Pat.var (2*i)
  let y := Pat.var (2*i+1)
  let k := Ops.maximum x y
  let n := Ops.next (Ops.copy k)
  (Ops.inBounds x n, n, k)

/-- Time only the shared certified schedule on a preseeded graph, then check
    the equality for every source check and the result of every function merge. -/
def measure (size : Nat) : IO Unit := do
  let mut g := Certified.Session.empty ir.model
  let (s, yes) := g.add Ops.yes
  g := s
  for i in [:size] do
    let (expr, _, _) := sample i
    g := (g.add expr).1
  let state ← IO.mkRef g
  let start ← IO.monoNanosNow
  state.modify (·.run phases)
  let stop ← IO.monoNanosNow
  let mut result ← state.get
  for i in [:size] do
    let (check, n, k) := sample i
    let (s, root) := result.add check
    result := s
    unless (result.equivalent? root yes).isSome do throw (IO.userError s!"check {i} survived")
    unless (result.getCell? (F.lower n).name [n]).map Subtype.val == some k do
      throw (IO.userError s!"wrong merged bound for check {i}")
  IO.println s!"schedule_ns={stop-start}"

end BoundsSchedule

def main (args : List String) : IO Unit :=
  match args with
  | ["bounds-schedule", size] => do
    let some n := size.toNat? | throw (IO.userError "expected a natural root count")
    BoundsSchedule.measure n
  | ["boolean", input, expected] => timePass readBool BooleanPass.optimize input expected
  | ["strength", input, expected] =>
    timePass readNat (fun e => (StrengthReduction.optimize e).output) input expected
  | _ => throw (IO.userError "expected boolean|strength INPUT EXPECTED or bounds-schedule ROOTS")
