import Egglog.Examples.EggCCFlow

open EggCC

namespace EggCCFlowTests

private def word (n : Nat) := Value.data (.word (BitVec.ofNat 64 n))
private def printed (n : Nat) := Atom.word (BitVec.ofNat 64 n)

/-- The final iteration is observable and must execute even with zero backedges. -/
private def signedExit : Flow := .loop (.pure (.pair
    (.lt (.num 3) (.num (2^64-1))) (.output (.num 4) .arg)))
#guard Flow.optimize signedExit == .pure (.output (.num 4) .arg)
#guard Flow.run 0 signedExit (.state [.flag true]) == some (.state [printed 4, .flag true])

private def spin : Flow := .loop (.pure (.pair .yes (.output (.num 7) .arg)))
#guard Flow.observe 2 (Flow.optimize spin) (.state [.flag true]) ==
  .paused (.state [printed 7, printed 7, printed 7, .flag true])
#guard Flow.observe 0 (.loop (.pure (.num 42))) .invalid == .stuck

-- A negative signed word is less than zero: the backedge must survive.
private def signedContinue : Flow := .loop (.pure (.pair
    (.lt (.num (2^64-1)) (.num 0)) .arg))
#guard Flow.observe 2 (Flow.optimize signedContinue) (word 9) == .paused (word 9)

-- A letVal binding explicitly retains the outer environment for the nested loop.
private def scopedBody : Flow := .letVal (.pure (.num 3))
    (.seq (.loop (.pure (.pair .no (.tail .arg)))) (.pure (.output (.num 2) .arg)))
#guard Flow.run 0 (Flow.optimize scopedBody) (.state [printed 8]) ==
  some (.state [printed 2, printed 8])

-- An effect can be a shared SSA definition before the post-test. Removing
-- the backedge must still execute that definition exactly once.
private def printedBody : Flow := .letVal (.pure (.output (.num 3) .arg))
    (.pure (.pair .no (.head .arg)))
#guard (Flow.optimize (.loop printedBody)) == .letVal (.pure (.output (.num 3) .arg))
    (.pure (.head .arg))
#guard Flow.run 0 (Flow.optimize (.loop printedBody)) (.state []) == some (.state [printed 3])

-- Eliminating an unreachable outer test after an infinite inner loop preserves
-- the first paused inner state as well as the absence of any final result.
private def nestedSpin : Flow := .loop (.seq spin (.pure (.pair .no .arg)))
#guard Flow.observe 2 (Flow.optimize nestedSpin) (.state []) ==
  .paused (.state [printed 7, printed 7, printed 7])

-- Both arms can establish a false post-test through different control shapes.
-- Matching branch, then loop, needs two e-graph rounds and congruence.
private def foldedArms : Flow := .loop (.branch (.head .arg)
    (.letVal (.pure (.num 5)) (.pure (.pair .no (.head .arg))))
    (.pure (.pair (.le (.num 9) (.num 3)) (.num 11))))
#guard Flow.optimize foldedArms == .branch (.head .arg)
    (.letVal (.pure (.num 5)) (.pure (.head .arg))) (.pure (.num 11))
#guard Flow.run 0 (Flow.optimize foldedArms) (.pair (.data (.flag false)) .nil) == some (word 11)

-- The rule is available to ordinary user-defined schedules and proves an
-- open equivalence for arbitrary scoped computations represented by atoms.
private def throughBinder : Flow.Control Nat :=
  .loop (.letVal (.atom 0) (.withFalseExit (.atom 1)))
#guard (Flow.Control.language.proveEq? (Flow.ControlOps.schedule 5) throughBinder
    (.letVal (.atom 0) (.atom 1))).isSome
#guard !(Flow.Control.language.proveEq? (Flow.ControlOps.schedule 5)
    (.loop (.atom 0)) (.atom 0)).isSome

-- The induction uses the upstream flag/next-tuple convention: no finite
-- execution of a body with an always-true continuation can be the last.
private theorem spin_never_returns (input out : Value) : ¬ Flow.Evaluates spin input out := by
  intro h
  change Flow.Iterates (Flow.Evaluates (.pure (.pair .yes (.output (.num 7) .arg)))) input out at h
  induction h with
  | stop step =>
    change Value.pair (.data (.flag true)) _ = Value.pair (.data (.flag false)) _ at step
    cases step
  | more _ _ ih => exact ih

example (i o : Value) : ¬ Flow.Evaluates (Flow.optimize spin) i o :=
  fun h => spin_never_returns i o ((Flow.optimize_correct ..).mp h)

end EggCCFlowTests
