import Egglog.Session
import Egglog.Examples.Boolean
import Egglog.Examples.StrengthReduction

open Egglog Egglog.Compiler

namespace BooleanPass

def exampleIR : Expr := .mux (.neg (.neg (.reg 7))) (.conj (.reg 3) (.lit true))
  (.disj (.lit false) (.reg 3))

#guard optimize exampleIR == .reg 3
#guard (ir.proveEq? phases exampleIR (.reg 3)).isSome
#guard (ir.proveEq? phases (.reg 7) (.reg 3)).isNone
#guard (ir.optimize? (egg_schedule { repeat 0 { run algebra } }) exampleIR).map
  (·.output) == some exampleIR

private def runUntil : Certified.Schedule ir.model :=
  egg_schedule { until (x) (Ops.conj x Ops.yes = x) { run algebra } }

-- The existential stop condition is checked against existing class equalities.
#guard (ir.optimize runUntil (.neg (.neg (.reg 8)))).output == .reg 8
#guard (ir.optimize (egg_schedule { until () (Ops.yes = Ops.yes) { run algebra } })
  (.conj (.reg 8) (.lit true))).output == .conj (.reg 8) (.lit true)
#guard (ir.optimize (egg_schedule { saturate { run algebra, run control } }) exampleIR).output == .reg 3

private def conjunctiveStop : Certified.Schedule ir.model :=
  egg_schedule { until (x) (Ops.conj x Ops.yes = Ops.conj x Ops.yes) (Ops.neg x = Ops.neg x)
    { run algebra } }

private def withNeg (i : Nat) : Expr := .disj (.neg (.reg i)) (.conj (.reg 1) (.lit true))
#guard (ir.optimize conjunctiveStop (withNeg 1)).output == withNeg 1
#guard (ir.optimize conjunctiveStop (withNeg 2)).output == .disj (.neg (.reg 2)) (.reg 1)

-- The generated codec checks arity, including a known operator used with missing operands.
#guard (Expr.decode (.app 5 [])).isNone
#guard (Expr.decode (.app 99 [])).isNone

example : True := by
  fail_if_success
    let _ : VerifiedRule Ops.model := egg_rule (a, b) : Ops.conj a b ==> a by rfl
  trivial

-- The output variable `b` below has no binding even though the equality is semantically valid.
example : True := by
  fail_if_success
    let _ : GuardedRule Ops.model :=
      egg_rule (a, b) : a ==> Ops.mux Ops.yes a b when (h : a = a) by rfl
  trivial

-- A mistaken decoder cannot invalidate the pass theorem.
def badIR : Language Expr Bool := { ir with
  decode := Decoder.checked Expr.encode (fun _ => some (.lit false)) }

#guard (badIR.optimize? phases (.reg 1)).isNone
#guard (badIR.optimize phases (.reg 1)).output == .reg 1

/-- A closed query can supply a reusable theorem about arbitrary inputs. -/
theorem foldedReg : ir.Equivalent (.conj (.reg 7) (.lit true)) (.reg 7) := by
  egg_prove ir using (egg_schedule { run #[andTrue] })

/-- A reusable e-class analysis for exactly known Boolean constants. -/
private def knownBool : Certified.ValueAnalysis Ops.model Bool where
  meaning := id
  transfer op args := some (Ops.model.interp op args)
  sound op args value h := by simpa using h

-- An open condition prevents evaluation of the mux until its equal branches merge.
private def openMux : Expr := .neg (.mux (.reg 1) (.lit true) (.lit true))
#guard (ir.proveEq? (egg_schedule { analyze knownBool }) openMux (.lit false)).isNone
#guard (ir.proveEq? (egg_schedule { run control, analyze knownBool }) openMux (.lit false)).isSome
#guard (ir.proveEq? (egg_schedule { analyze knownBool }) (.reg 1) (.reg 2)).isNone

end BooleanPass

namespace StrengthReduction

-- The intermediate product exists only inside the e-graph. Its exponent is discovered by the guard.
#guard (optimize (.plus (.arg 5) (.arg 5))).output == .shl (.arg 5) .one
#guard (optimize (.times (.arg 5) (.pow2 (.arg 8)))).output == .shl (.arg 5) (.arg 8)
#guard (optimize (.times (.arg 5) (.plus .one .one))).output == .shl (.arg 5) .one
#guard (optimize (.times .one .one)).output == .one
#guard (optimize (.times (.arg 5) .one)).output == .times (.arg 5) .one
#guard (optimize (.times (.arg 5) (.arg 8))).output == .times (.arg 5) (.arg 8)

private def reverseAnchor : GuardedRule Ops.model :=
  egg_rule (x, y, n) : Ops.times x y ==> Ops.shl x n when (h : Ops.pow2 n = y) by
    change 2 ^ n = y at h
    change x * y = x <<< n
    rw [← h, Nat.shiftLeft_eq]

#guard (ir.optimize (egg_schedule { guarded #[reverseAnchor] })
  (.times (.arg 4) (.pow2 (.arg 6))) Ops.cost).output == .shl (.arg 4) (.arg 6)

/-- The shift exponent is introduced by the joined premise and kernel checked. -/
theorem shiftPower : ir.Equivalent (.times (.arg 0) (.pow2 .one)) (.shl (.arg 0) .one) := by
  egg_prove ir using (egg_schedule { guarded #[mulPow2] })

set_option maxRecDepth 40000 in
set_option maxHeartbeats 6000000 in
theorem shiftDiscoveredPower : ir.Equivalent (.times (.arg 0) (.plus .one .one)) (.shl (.arg 0) .one) := by
  egg_prove ir using phases

private def mulPowers : GuardedRule Ops.model :=
  egg_rule (x, y, a, b) : Ops.times x y ==> Ops.pow2 (Ops.plus a b)
    when (hx : x = Ops.pow2 a) (hy : y = Ops.pow2 b) by
    change x = 2 ^ a at hx
    change y = 2 ^ b at hy
    change x * y = 2 ^ (a + b)
    rw [hx, hy, Nat.pow_add]

-- Both exponents are introduced by separate guard patterns in the same query.
#guard (ir.proveEq? (egg_schedule { guarded #[mulPowers] })
  (.times (.pow2 (.arg 1)) (.pow2 (.arg 2))) (.pow2 (.plus (.arg 1) (.arg 2)))).isSome

set_option maxRecDepth 40000 in
set_option maxHeartbeats 6000000 in
theorem joinedPowers : ir.Equivalent (.times (.pow2 .one) (.pow2 .one)) (.pow2 (.plus .one .one)) := by
  egg_prove ir using (egg_schedule { guarded #[mulPowers] })

end StrengthReduction

namespace CertifiedRegression

-- A transcript remains usable after a fresh graph is allocated. Removing
-- the rule which establishes its guard must invalidate the final equality.
open StrengthReduction.Ops in
private def groundReplay : Bool := Id.run do
  let s := Certified.Session.empty StrengthReduction.ir.model
  let (s, l) := s.add (times (.var 2) (plus one one))
  let (s, r) := s.add (shl (.var 2) one)
  let first := (Certified.NamedRule.plain "two ones" StrengthReduction.twoOnes).rule
  let second := StrengthReduction.mulPow2
  let (s₁, a) := s.recordRound #[first]
  let (_, b) := s₁.recordRound #[second] 1
  let rules := #[first, second]
  return (s.replaySteps rules (a ++ b).toList |>.equivalent? l r).isSome &&
         (s.replaySteps rules b.toList |>.equivalent? l r).isNone

#guard groundReplay

/-- A nested match becomes available under an existing outer constructor when
    `falseOr` merges its child with an existing negation. -/
private def nestedPass : Certified.Schedule BooleanPass.ir.model :=
  egg_schedule { run #[BooleanPass.doubleNeg], run #[BooleanPass.falseOr], run #[BooleanPass.doubleNeg] }

#guard (BooleanPass.ir.optimize nestedPass (.neg (.disj (.lit false) (.neg (.reg 9))))).output == .reg 9

/-- All target roots exist before the second search. Its guard becomes
    satisfiable when an intervening phase merges the multiplier with a power. -/
private def delayedGuard : Certified.Schedule StrengthReduction.ir.model :=
  egg_schedule { guarded #[StrengthReduction.mulPow2], run #[StrengthReduction.twoOnes],
                 guarded #[StrengthReduction.mulPow2] }

#guard (StrengthReduction.ir.optimize delayedGuard (.times (.arg 2) (.plus .one .one))
  StrengthReduction.Ops.cost).output == .shl (.arg 2) .one

private def expand : VerifiedRule Arithmetic.model :=
  ⟨⟨.var 0, .app 3 [.var 0, .app 0 []]⟩, fun e => (Nat.add_zero (e 0)).symm⟩

-- Every class can gain a cyclic zero-cost alternative. Extraction must retain
-- an actual finite term and obey the user-supplied additive cost.
#guard (Certified.optimize Arithmetic.model (.app 4 [.var 9, .app 1 []])
  (.saturate 8 (.run #[expand, Arithmetic.mulOne])) (fun _ => 0)).val == .var 9

-- A failed merge and a foreign root between valid extractions must preserve
-- both the graph and the shared recovery cache.
#guard Id.run do
  let s := Certified.Session.empty Arithmetic.model
  let (_, foreign) := s.add (.var 7)
  let arg : Pat := .var 6
  let repeated : Pat := .app 3 [arg, arg]
  let (s, good) := s.add arg
  let (s, parent) := s.add repeated
  let s := s.mergeIfOwned foreign foreign (fun _ => rfl)
  let (batch, first) := s.extractor.batch.extract? parent
  let (batch, missing) := batch.extract? foreign
  let (_, last) := batch.extract? good
  return first.map (·.val) == some repeated && missing.isNone && last.map (·.val) == some arg

end CertifiedRegression


namespace BooleanPass

private def shared (reg : Nat) : Bool := Id.run do
  let (g, left) := ir.session.add (.conj (.reg reg) (.lit true))
  let (g, right) := g.add (.reg reg)
  let (g, parLeft) := g.add (.neg (.conj (.reg reg) (.lit true)))
  let (g, parRight) := g.add (.neg (.reg reg))
  if (g.equivalent? parLeft parRight).isSome then return false
  let some merged := g.merge? left right (fun env => Bool.and_true (env reg)) | return false
  if (merged.equivalent? parLeft parRight).isNone then return false
  let results := (g.run phases).extractor.batch
  let (results, child) := results.extract? left
  let (_, parent) := results.extract? parLeft
  return (child.map (·.output) == some (.reg reg)) &&
         (parent.map (·.output) == some (.neg (.reg reg)))

#guard shared 13

-- Branches can intern different syntax in the same numeric slot.
#guard Id.run do
  let (g, a) := ir.session.add (.reg 1)
  let (_, b) := ir.session.add (.reg 2)
  return (g.equivalent? a b).isNone && (g.extractor.extract? b).isNone

end BooleanPass

namespace SchedulerTest

open BooleanPass

-- Postpone the entire first batch. The unchanged graph must receive its
-- delayed matches on the next iteration. Repeated choices fire only once.
private def postpone : Certified.MatchScheduler where
  State := Bool
  initial := false
  choose ready _ _ batch := if ready then (true, Array.range batch.size ++ #[0]) else (true, #[])
private def rules := #[Certified.NamedRule.plain "and-true" andTrue]
private def input := Expr.conj (.reg 4) (.lit true)
#guard (ir.optimize (egg_schedule { with_scheduler rules with postpone for 1 }) input).output == input
#guard (ir.optimize (egg_schedule { with_scheduler rules with postpone for 2 }) input).output == .reg 4

end SchedulerTest
