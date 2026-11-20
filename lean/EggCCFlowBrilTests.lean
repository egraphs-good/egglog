import Egglog.Examples.EggCCFlowBril

namespace EggCCFlowBrilTests
open EggCC EggCC.Driver

private def tuple (xs : List Expr) : Expr := xs.foldr Expr.pair .nil
private def slot (i : Nat) := project i .arg

-- Even a state-only program that emits no data instructions has a valid
-- Bril basic block ending in a return.
#guard (flowToBril #[.token] (.pure .arg)).toOption == some "@main() {\n  ret;\n}\n"

-- The continuation flag aliases an input register whose next value is false.
-- The next pair swaps two input registers; both old values must be saved.
def swapAndStop : Flow :=
  let initial := tuple [.yes, .num 11, .num 22, slot 0]
  let next := tuple [.no, slot 2, slot 1, .output (slot 1) (slot 3)]
  let after := tuple [.output (slot 1) (slot 3)]
  .seq (.pure initial) (.seq (.loop (.pure (.pair (slot 0) next))) (.pure after))

#guard (flowToBril #[.token] swapAndStop).isOk
#guard swapAndStop.run 16 (.pair (.state []) .nil) == some
  (.pair (.state [.word 11, .word 22, .word 11]) .nil)

-- The incoming token can occur more than once in a saved tuple. It can only
-- be consumed again if its value is still the current print stream.
#guard (flowToBril #[.token] (.letVal (.pure (.output (.num 8) (slot 0)))
  (.pure (tuple [.output (.num 9) (.head (.tail .arg))])))).isOk == false

-- Speculating one arm would emit a print even when the other arm is chosen.
#guard (flowToBril #[.bool, .token] (.pure (tuple
  [.select (slot 0) (.output (.num 2) (slot 1)) (slot 1)]))).isOk == false

-- An effectful iteration must pass the new token to its next iteration.
#guard (flowToBril #[.token] (.loop (.letVal (.pure (.output (.num 1) (slot 0)))
  (.pure (.pair .no (.tail .arg)))))).isOk == false

-- The two tuple fields refer to the same state-producing expression. Lower it
-- once, then read one of the aliases on the single output path.
#guard Id.run do
  let shared := Expr.output (.num 12) (slot 0)
  let f := Flow.seq (.pure (.pair shared shared)) (.pure (tuple [slot 0]))
  let .ok text := flowToBril #[.token] f | return false
  return (text.splitOn "  print ").length == 2

-- A print in one branch is followed by a print on either exit path.
def branchTrace : Flow :=
  .seq (.branch (slot 0) (.pure (tuple [.output (.num 8) (slot 1)])) (.pure (tuple [slot 1])))
    (.pure (tuple [.output (.num 2) (slot 0)]))

#guard (flowToBril #[.bool,.token] branchTrace).isOk
#guard branchTrace.run 0 (.pair (.data (.flag true)) (.pair (.state []) .nil)) ==
  some (.pair (.state [.word 2,.word 8]) .nil)
#guard branchTrace.run 0 (.pair (.data (.flag false)) (.pair (.state []) .nil)) ==
  some (.pair (.state [.word 2]) .nil)

-- Reusing the state before a print on just one branch fails at the join.
#guard Id.run do
  let f := Flow.letVal (.pure (.output (.num 8) (slot 1)))
    (.branch (project 0 (.tail .arg))
       (.pure (tuple [slot 0])) (.pure (tuple [project 1 (.tail .arg)])))
  return !(flowToBril #[.bool,.token] f).isOk

-- A lazy conditional and a lazy select have equal trace-valued denotations.
-- The cheapest candidate places a print in an arm of Bril's eager select.
private def printingBranch : Flow := .pure (.branch (slot 0) .arg
  (tuple [.output (.num 2) (slot 1)]) (tuple [slot 1]))

example (k : Nat) (i : Value) :
    Flow.observe k printingBranch.optimize i = Flow.observe k printingBranch i :=
  congrFun (Flow.optimize_observe printingBranch k) i

#guard match flowToBril #[.bool,.token] printingBranch.optimize with
  | .error e => e == "cannot speculate print effects in a select"
  | _ => false

#guard Id.run do
  let .ok selected := optimizeToBril #[.bool,.token] printingBranch | return false
  return selected.program == printingBranch && selected.text.contains "br arg0" &&
    (selected.text.splitOn "  print ").length == 2 && !selected.text.contains " = select "

-- Dropping the second field preserves the functional result but hides a print
-- with no consumer. Reject an ill-formed source even if extraction erases it.
private def discardedPrint : Flow := .pure
  (tuple [.head (.pair (slot 1) (.output (.num 9) (slot 1)))])

#guard (flowToBril #[.bool,.token] discardedPrint.optimize).isOk
#guard match optimizeToBril #[.bool,.token] discardedPrint with
  | .error e => e == "source cannot be lowered to Bril: main discarded a print state"
  | _ => false

end EggCCFlowBrilTests
