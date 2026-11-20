import Egglog.QueryProof
import Egglog.Naturals
import Egglog.Parser
import Lean.Elab.Command

open Egglog

namespace ProofTests

private abbrev zero : Pat := .app 0 []
private abbrev one : Pat := .app 1 []
private abbrev add (x y : Pat) : Pat := .app 3 [x, y]

-- A syntactically altered rule cannot enter through the checked registry.
#guard (Rewrite.mk (add (.var 0) zero) one).certify? Arithmetic.rules |>.isNone
#guard Arithmetic.addZero.toRewrite.certify? Arithmetic.rules |>.isSome

-- add(one, zero) and one share a class. The addition row refers back to the
-- result class, as happens after a simplifying rewrite.
private def rows : Snapshot := [⟨0, [], 0⟩, ⟨1, [], 1⟩, ⟨3, [1, 0], 1⟩]
private def value := rows.inferValues Arithmetic.model 0

example (env : Nat → Nat) :
    (add one zero).eval Arithmetic.model env = one.eval Arithmetic.model env :=
  Snapshot.checkEq_sound (rows := rows) (value := value) (by decide) env

-- A merge of zero and one invalidates the interpretation, even if an e-graph
-- claims a positive equality result for this pair.
private def corrupt : Snapshot := [⟨0, [], 0⟩, ⟨1, [], 0⟩]
example : corrupt.checkEq Arithmetic.model (corrupt.inferValues Arithmetic.model 0) zero one = false :=
  by decide

private def realGraph (corruptGraph : Bool) : EGraph := Id.run do
  let (a, g) := EGraph.empty.add ⟨0, #[]⟩
  let (b, g) := g.add ⟨1, #[]⟩
  let (ab, g) := g.add ⟨3, #[b, a]⟩
  let g := g.merge ab b
  return (if corruptGraph then g.merge a b else g).rebuild

private def checkedGraph (g : EGraph) : Bool :=
  let s := g.snapshot
  s.checkEq Arithmetic.model (s.inferValues Arithmetic.model 0) (add one zero) one

#guard checkedGraph (realGraph false)
#guard !checkedGraph (realGraph true)

-- Only closed literal rows cross from the compiled engine into a replay proof.
open Lean Elab Command in
private def reflectRows (name : Name) (g : EGraph) : CommandElabM Unit := do
  let rows ← g.snapshot.mapM fun row =>
    `(ClassRow.mk $(quote row.op) $(quote row.args) $(quote row.result))
  let rows := rows.toArray
  elabCommand (← `(private def $(mkIdent name) : Snapshot := [$rows,*]))

open Lean Elab Command in
run_cmd
  let source ← IO.FS.readFile "../tests/web-demo/naturals.egg"
  let state ← match runCheckedSource source Naturals.signature Naturals.rules with
    | .ok state => pure state
    | .error msg => throwError "{msg}"
  reflectRows `naturalsRun state.graph

private abbrev z := Pat.app 0 []
private abbrev s (p : Pat) := Pat.app 1 [p]
private abbrev two := s (s z)
private abbrev three := s two
private abbrev plus (p q : Pat) := Pat.app 2 [p, q]

-- The final check in the upstream file compares two times three to two plus
-- (two plus two). Kernel reduction verifies the witness for this run.
set_option maxRecDepth 8192 in
example (env : Nat → Nat) :
    (Pat.app 3 [two, three]).eval Naturals.model env =
      (plus two (plus two two)).eval Naturals.model env :=
  Snapshot.checkEq_sound (rows := naturalsRun)
    (value := naturalsRun.inferValues Naturals.model 0) (by decide) env

namespace Symbolic

-- The first five operators retain their checked arithmetic meanings. Three
-- fresh, nullary symbols stand for arbitrary inputs throughout the run.
private abbrev c (id : Nat) := Pat.app id []
private abbrev mul (a b : Pat) := Pat.app 4 [a, b]
private abbrev lhs := add (mul (c 5) (add (c 6) (c 7))) zero
private abbrev rhs := add (mul (c 5) (c 6)) (mul (c 5) (c 7))

private def interpretation (x y z : Nat) : Model Nat where
  interp op args := match op, args with
    | 5, [] => x | 6, [] => y | 7, [] => z
    | _, _ => Arithmetic.model.interp op args

private def rules (x y z : Nat) : Array (VerifiedRule (interpretation x y z)) :=
  #[⟨Arithmetic.mulAdd.toRewrite, fun env => Arithmetic.mulAdd.sound env⟩,
    ⟨Arithmetic.addZero.toRewrite, fun env => Arithmetic.addZero.sound env⟩]

open Lean Elab Command in
private partial def quoteTree : Tree → CommandElabM (TSyntax `term)
  | .app op kids => do
    let kids ← kids.mapM quoteTree
    `(Pat.app $(quote op) [$kids,*])

open Lean Elab Command in
run_cmd
  let (_, g) := instantiateBound lhs #[] EGraph.empty
  -- The search only reads the rule syntax. All valuations give the same graph.
  let g := Compiled.run g ((rules 0 0 0).map (·.toRewrite)) 3
  reflectRows `graph g
  -- Each extracted term proposes a symbolic representative for its class.
  -- The proof below checks that proposal against every row in the snapshot.
  let terms ← (Array.range g.nodes.size).mapM fun id => do
    let some witness := g.extract id | throwError "class has no ground representative"
    quoteTree witness.term
  elabCommand (← `(private def $(mkIdent `terms) : Array Pat := #[$terms,*]))

private def valueAt (x y z : Nat) (id : Id) :=
  terms[id]!.eval (interpretation x y z) (fun _ => 0)

-- The distributive row follows from its general rule. All other constructor
-- equations reduce by reflexivity, including the outer addition of zero.
private theorem consistent (x y z : Nat) :
    graph.consistent (interpretation x y z) (valueAt x y z) := by
  simp only [Snapshot.consistent, graph, List.all_cons, List.all_nil, Bool.and_eq_true]
  repeat' constructor
  all_goals apply decide_eq_true
  all_goals first | rfl | exact Nat.mul_add x y z

set_option maxRecDepth 8192 in
theorem distributeInSum (x y z : Nat) : x * (y + z) + 0 = x * y + x * z :=
  Snapshot.query_sound (m := interpretation x y z) (p := lhs) (q := rhs)
    (fun _ => 0) (consistent x y z) (id := (graph.resolve? lhs).getD 0) (by decide) (by decide)

end Symbolic
end ProofTests
