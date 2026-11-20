import EggCCMain

open EggCC EggCC.Driver

namespace EggCCTests

-- The then region's slot zero is input y and its else region's slot one is x.
private def x := project 0 .arg
private def y := project 1 .arg
private def inputs := Expr.pair y (.pair x .nil)
private def region := Expr.branch (.lt x y) inputs (.pair (project 0 .arg) .nil)
  (.pair (project 1 .arg) .nil)
private def src := Expr.head region
private def target := Expr.select (.lt x y) y x

theorem swapped_environment_is_preserved (args : Value) :
    (optimize src).output.eval args = src.eval args := optimize_correct ..

#guard (optimize src).output == target
#guard (src.eval (.pair (.data (.word 1)) (.pair (.data (.word 9)) .nil))) == .data (.word 9)
#guard (src.eval (.pair (.data (.word 9)) (.pair (.data (.word 1)) .nil))) == .data (.word 9)
#guard (Expr.add (.num (2^64 - 1)) (.num 1)).eval .nil == .data (.word 0)
#guard (Expr.lt (.num (2^63)) (.num 0)).eval .nil == .data (.flag true)

private def printRegion : Program :=
  ⟨#[.token], .pair (.select .yes (.output (.num 2) x) x) .nil, 0⟩
#guard printRegion.validate.isOk == false

private def badProjection : Program := ⟨#[.token], .pair (project 2 .arg) .nil, 0⟩
#guard badProjection.validate.isOk == false
#guard (parse "{\"inputs\":[\"state\"],\"nodes\":[{\"op\":\"mul\",\"args\":[1,1]}],\"root\":0}").isOk == false

private def checkNested : IO Unit := do
  let src ← IO.ofExcept (parse (← IO.FS.readFile "bench/eggcc/if_dead_code_nested.json"))
  let out := {src with body := (optimize src.body).output}
  IO.ofExcept out.validate
  let bril ← IO.ofExcept (toBril out)
  unless (bril.splitOn " = select ").length == 7 && (bril.splitOn "  print ").length == 4 do
    throw (IO.userError "expected six selects and three ordered prints")
  for (n, expected) in [("-1", #["1", "true", "1"]), ("0", #["1", "true", "2"]),
      ("2", #["0", "false", "2"]), ("3", #["0", "false", "4"]), ("4", #["0", "false", "3"])] do
    let arg ← IO.ofExcept (readInputs src.inputs [n])
    let before ← IO.ofExcept (traceOf (src.body.eval arg))
    let after ← IO.ofExcept (traceOf (out.body.eval arg))
    unless before == expected && after == expected do throw (IO.userError s!"wrong trace at {n}")

#eval checkNested

end EggCCTests
