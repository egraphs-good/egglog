import Egglog.Examples.ParaBitInput

private partial def leanTerm : BWLang.Term → String
  | .literal i => s!".literal ({i})"
  | .symbol name => ".symbol " ++ name.quote
  | .app op args => ".app " ++ op.quote ++ " [" ++ String.intercalate ", " (args.map leanTerm) ++ "]"

private def leanTerms (ts : List BWLang.Term) : String :=
  "[" ++ String.intercalate ", " (ts.map leanTerm) ++ "]"

/-- Prefer a small successful bound for replay. The initial check also detects
    an unproved input without testing every lower bound. The returned bound
    was explicitly checked even when a scheduler takes different branches. -/
private def replayBound (solves : Nat → Bool) (limit : Nat) : Option Nat := do
  guard (solves limit)
  let mut lo := 0
  let mut hi := limit
  for _ in [:limit.log2 + 1] do
    if lo >= hi then break
    let mid := (lo + hi) / 2
    if solves mid then hi := mid else lo := mid + 1
  return hi

private def emitTheorem (p : BWLang.Problem) (file : System.FilePath) (rounds : Nat) (staged : Bool) : IO Unit := do
  let q ← IO.ofExcept p.prepare
  -- Reducing the staged bound can cut its cheap first phase short and
  -- enter general algebra on a much larger set of terms.
  let bound := if staged then
    if (q.proveStaged? rounds).isSome then some rounds else none
    else replayBound (fun n => (q.prove? n).isSome) rounds
  let some replay := bound |
    throw (IO.userError "the certified solver did not establish this equivalence within the round bound")
  let stem := file.fileStem.getD "input"
  let id := if stem.toList.all (fun c => c.isAlphanum || c == '_') then "Case_" ++ stem
    else "Escaped_" ++ String.intercalate "_" (stem.toList.map fun c => toString c.toNat)
  IO.println "import Egglog.Examples.ParaBitInput"
  IO.println s!"\nnamespace BWLang.Certificates.{id}\n"
  IO.println "def problem : BWLang.Problem := {"
  IO.println ("  lhs := " ++ leanTerm p.lhs)
  IO.println ("  rhs := " ++ leanTerm p.rhs)
  IO.println ("  conditions := " ++ leanTerms p.conditions)
  IO.println ("  positiveWidths := " ++ leanTerms p.positiveWidths ++ "\n}\n")
  IO.println "set_option maxRecDepth 100000 in"
  IO.println "set_option maxHeartbeats 100000000 in"
  IO.println "theorem equivalence : ∀ (ρ : String → Int), problem.Holds ρ →"
  IO.println "    problem.lhs.eval ρ = problem.rhs.eval ρ :="
  if staged then
    IO.println s!"  ((problem.proveStaged? {replay}).get (by decide +kernel)).down"
  else if let some hint := q.query.replayHint? replay then
    IO.println s!"  ((problem.replay? {replay} ({reprStr hint})).get (by decide +kernel)).down"
  else
    IO.println s!"  ((problem.prove? {replay}).get (by decide +kernel)).down"
  IO.println s!"\nend BWLang.Certificates.{id}"

def main (args : List String) : IO Unit := do
  let (emit, staged, file, bound) ← match args with
    | [file, n] => pure (false, false, file, n)
    | ["--staged", file, n] => pure (false, true, file, n)
    | ["--emit-lean", file, n] => pure (true, false, file, n)
    | ["--staged", "--emit-lean", file, n] | ["--emit-lean", "--staged", file, n] => pure (true, true, file, n)
    | _ => throw (IO.userError "expected [--emit-lean] [--staged] FILE.bwlang ROUNDS")
  let some n := bound.toNat? | throw (IO.userError "expected a natural round limit")
  let problem ← IO.ofExcept (BWLang.parse (← IO.FS.readFile file))
  if emit then return ← emitTheorem problem file n staged
  let q ← IO.ofExcept problem.prepare
  let ref ← IO.mkRef false
  let start ← IO.monoNanosNow
  ref.set (if staged then q.proveStaged? n else q.prove? n).isSome
  let stop ← IO.monoNanosNow
  IO.println (if ← ref.get then "solved" else "unproved")
  IO.println s!"solver_ns={stop-start}"
