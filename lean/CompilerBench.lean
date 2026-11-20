import Egglog.Examples.Boolean
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

def main (args : List String) : IO Unit :=
  match args with
  | ["boolean", input, expected] => timePass readBool BooleanPass.optimize input expected
  | ["strength", input, expected] =>
    timePass readNat (fun e => (StrengthReduction.optimize e).output) input expected
  | _ => throw (IO.userError "expected boolean|strength INPUT EXPECTED")
