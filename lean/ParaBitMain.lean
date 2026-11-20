import Egglog.Examples.BWLang
import Egglog.Relations

open Egglog BWLang

/-- On pure bitwise expressions, propagate packed truth tables immediately.
    On mixed expressions, simplify first so arithmetic identities and width facts
    expose Boolean structure to the same table pass. All paths retain bounded
    algebra and decision-diagram fallbacks; each stage stops when the goal joins. -/
def BWLang.relationalSchedule (hasBitwise : Bool) (bound : Nat) (simplifyBitsFirst := false) :
    List (String × Nat) :=
  let simplify := ("", bound)
  let algebra := (List.replicate 3 [("algebra", 1), ("", min bound 5)]).flatten
  let truthTable := ("truth-table", 40)
  let diagrams := ("bdd", 160)
  if hasBitwise then
    let first := if simplifyBitsFirst then [simplify, truthTable] else [truthTable, simplify]
    first ++ [diagrams, simplify] ++ algebra
  else [simplify] ++ algebra ++ [truthTable, diagrams, simplify]

private partial def bitwiseTerm : Term → Bool
  | .app op xs => ["and", "or", "xor", "not"].contains op || xs.any bitwiseTerm
  | _ => false

private partial def hasWordArithmetic : Term → Bool
  | .app "bw" [_, x] => hasWordArithmetic x
  | .app op xs => !["and", "or", "xor", "not"].contains op || xs.any hasWordArithmetic
  | _ => false

private def scheduleFor (p : Problem) (bound : Nat) :=
  let bits := (p.lhs :: p.rhs :: p.conditions).any bitwiseTerm
  BWLang.relationalSchedule bits bound (bits && (hasWordArithmetic p.lhs || hasWordArithmetic p.rhs))

private def explore (db : Relational.Database) (stages : List (String × Nat)) : Except String String := do
  let eq := call "=" [.atom "$lhs", .atom "$rhs"]
  let mut db := db
  let mut exhausted := false
  for (set, rounds) in stages do
    let initial := db.epoch
    let opts := [.atom (toString rounds), .atom ":until", eq]
    let cmd := call "run" (if set.isEmpty then opts else (.atom set :: opts))
    db ← Relational.exec db cmd
    exhausted := exhausted || db.epoch - initial == rounds
    match Relational.exec db (call "check" [eq]) with
    | .ok _ => return "solved"
    | .error "check failed" => pure ()
    | .error msg => throw msg
  return if exhausted then "round-limit" else "unproved"

private partial def render : SExpr → String
  | .atom s =>
    if s.startsWith "\"" && s.endsWith "\"" then (s.drop 1 |>.dropEnd 1 |>.toString).quote else s
  | .list xs => "(" ++ String.intercalate " " (xs.toList.map render) ++ ")"

/-- Emit a standalone egglog input for the crate's Rust CLI or another interpreter. -/
private def emitEgg (source : String) (p : Problem) (bound : Nat) : IO Unit := do
  IO.println source
  let cmds ← IO.ofExcept p.commands
  for cmd in cmds do IO.println (render cmd)
  let eq := call "=" [.atom "$lhs", .atom "$rhs"]
  for (set, n) in scheduleFor p bound do
    let opts := [.atom (toString n), .atom ":until", eq]
    IO.println (render (call "run" (if set.isEmpty then opts else .atom set :: opts)))
  IO.println (render (call "check" [eq]))

/-- Run a declarative theory on one unchanged CAV artifact input. -/
def main (args : List String) : IO Unit := do
  let (emit, theory, file, bound) ← match args with
    | [theory, file, n] => pure (false, theory, file, n)
    | ["--emit-egg", theory, file, n] => pure (true, theory, file, n)
    | _ => throw (IO.userError "expected [--emit-egg] THEORY FILE.bwlang ROUNDS")
  let some n := bound.toNat? | throw (IO.userError "expected a natural round limit")
  let problem ← IO.ofExcept (BWLang.parse (← IO.FS.readFile file))
  let source ← IO.FS.readFile theory
  if emit then return ← emitEgg source problem n
  let setup ← IO.ofExcept (Relational.runSource source)
  let cmds ← IO.ofExcept problem.commands
  let ref ← IO.mkRef (Except.ok "" : Except String String)
  let start ← IO.monoNanosNow
  ref.set ((cmds.foldlM Relational.exec setup).bind (explore · (scheduleFor problem n)))
  let stop ← IO.monoNanosNow
  IO.println (← IO.ofExcept (← ref.get))
  IO.println s!"solver_ns={stop-start}"
