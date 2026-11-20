import Egglog.Relations
import Egglog.Naturals

open Egglog

private def checked (src : String) :=
  runCheckedSource src Arithmetic.signature Arithmetic.rules

private def readProgram (path : String) : IO (Array SExpr) := do
  IO.ofExcept (parseSExprs (← IO.FS.readFile path))

private def printOutput (s : ProgramState) : IO Unit := do
  for line in s.output do IO.println line

-- Keep setup, checking, and output outside the interval used for saturation.
private def timed (raw : Bool) (setup schedule checks : String) (signal := false) : IO Unit := do
  let src ← IO.FS.readFile setup
  let db ← IO.ofExcept (if raw then Relational.runSource src else
    (checked src).map fun st => { program := st : Relational.Database })
  let #[cmd@(.list items)] ← readProgram schedule
    | throw (IO.userError "expected one benchmark schedule")
  match items[0]? with
  | some (.atom "run") | some (.atom "run-schedule") => pure ()
  | _ => throw (IO.userError "expected run or run-schedule")
  -- The effectful read and write anchor pure native computation between the clocks.
  let result ← IO.mkRef (Except.ok db : Except String Relational.Database)
  let stdout ← IO.getStdout
  if signal then
    stdout.putStrLn "ready"
    stdout.flush
    let go ← (← IO.getStdin).getLine
    unless go == "go\n" do throw (IO.userError "expected benchmark start signal")
  let before ← IO.monoNanosNow
  result.modify (·.bind (fun snapshot => Relational.exec snapshot cmd))
  let after ← IO.monoNanosNow
  if signal then
    stdout.putStrLn "done"
    stdout.flush
  let mut db ← IO.ofExcept (← result.get)
  -- Validation may materialize a witness relation over the final graph.
  -- It runs after the end signal, just like the Rust comparison driver.
  for cmd in ← readProgram checks do
    db ← IO.ofExcept (Relational.exec db cmd)
  printOutput db.program
  IO.println s!"schedule_ns={after - before}"

def main (args : List String) : IO Unit := do
  match args with
  | [file] | ["--checked", file] => printOutput (← IO.ofExcept (checked (← IO.FS.readFile file)))
  | ["--naturals", file] =>
    printOutput (← IO.ofExcept (runCheckedSource (← IO.FS.readFile file) Naturals.signature Naturals.rules))
  | ["--raw", file] =>
    let db ← IO.ofExcept (Relational.runSource (← IO.FS.readFile file))
    printOutput db.program
  | ["--time-run", setup, schedule, checks] => timed false setup schedule checks
  | ["--time-raw", setup, schedule, checks] => timed true setup schedule checks
  | ["--signal-raw", setup, schedule, checks] => timed true setup schedule checks true
  | _ => throw (IO.userError "usage: lean-egglog [--checked|--naturals|--raw] FILE")
