import Egglog.Examples.ParaBitInput

open Egglog
namespace ParaBit

private structure Opened (globals : Nat → Int) (source : Pat) where
  pat : Pat
  correct : ∀ env, pat.eval (scalar globals) env = denote env source

private structure OpenedArgs (globals : Nat → Int) (source : List Pat) where
  pats : List Pat
  correct : ∀ env, Pat.evalArgs pats (scalar globals) env = Pat.evalArgs source (scalar env) (fun _ => 0)

private theorem fixedOperator (op : Nat) (xs : List Int) (globals env : Nat → Int)
    (h : ¬(32 ≤ op ∧ (op - 32) % 3 = 0)) :
    (scalar globals).interp op xs = (scalar env).interp op xs := by
  simp only [scalar]
  split
  · rfl
  · have hn : (op - 32) % 3 ≠ 0 := by omega
    split <;> first | rfl | contradiction

mutual
  private def openTerm (globals : Nat → Int) (p : Pat) : Option (Opened globals p) :=
    match p with
    | .var _ => none
    | .app op args => do
      if h : 32 ≤ op ∧ (op - 32) % 3 = 0 then
        guard args.isEmpty
        return ⟨.var ((op - 32) / 3), fun env => by
          simp [denote, Pat.eval, scalar, show ¬op < 32 by omega, h.2]⟩
      else
        let xs ← openArgs globals args
        return ⟨.app op xs.pats, fun env =>
          (congrArg ((scalar globals).interp op) (xs.correct env)).trans (fixedOperator op _ _ _ h)⟩
  private def openArgs (globals : Nat → Int) (ps : List Pat) : Option (OpenedArgs globals ps) :=
    match ps with
    | [] => return ⟨[], fun _ => rfl⟩
    | p :: ps => do
      let x ← openTerm globals p
      let xs ← openArgs globals ps
      return ⟨x.pat :: xs.pats, fun env => by
        simp only [Pat.evalArgs, x.correct, xs.correct, denote]⟩
end

private theorem evalArgs_map (ps : List Pat) (m : Model α) (env : Nat → α) :
    Pat.evalArgs ps m env = ps.map (fun p => p.eval m env) := by
  induction ps with
  | nil => rfl
  | cons x xs ih => simp only [Pat.evalArgs, List.map_cons, ih]

/-- Install a universally proved BWLang equivalence as an open, directional
    compiler rule. Every interned source symbol becomes a match variable, and
    every side condition becomes an e-class equality to the integer one.
    The target language uses the BWLang scalar operations. Its globals may
    have any interpretation, independent of the solver's symbol valuation. -/
def Query.toGuardedRule? (q : Query) (proved : ∀ ρ, q.Holds ρ → q.Equivalent ρ)
    (globals : Nat → Int := fun _ => 0) : Option (GuardedRule (scalar globals)) := do
  let lhs ← openTerm globals q.lhs
  let rhs ← openTerm globals q.rhs
  let conds ← openArgs globals q.assumptions
  let available := lhs.pat.variables ++ conds.pats.flatMap Pat.variables
  guard (rhs.pat.variables.all available.contains)
  let premises := conds.pats.map (fun h => (h, Op.one))
  return ⟨⟨lhs.pat, rhs.pat⟩, premises, fun env facts => by
    rw [lhs.correct, rhs.correct]
    apply proved env
    intro term member
    have hm : denote env term ∈ q.assumptions.map (denote env) := List.mem_map_of_mem (f := denote env) member
    change denote env term ∈ q.assumptions.map (fun t => t.eval (scalar env) (fun _ => 0)) at hm
    rw [← evalArgs_map, ← conds.correct, evalArgs_map] at hm
    obtain ⟨p, hp, val⟩ := List.mem_map.mp hm
    exact val.symm.trans (facts (p, Op.one) (List.mem_map_of_mem (f := fun h => (h, Op.one)) hp))⟩

/-- Solve once, then retain the returned rule for a compiler schedule. -/
def Query.certifyRule? (q : Query) (rounds : Nat := 6) (globals : Nat → Int := fun _ => 0)
    (staged : Bool := false) :
    Option (GuardedRule (scalar globals)) := do
  let proof ← if staged then q.proveStaged? rounds else q.prove? rounds
  q.toGuardedRule? proof.down globals

end ParaBit

/-- Certify a parsed ParaBit input as a rule for an independent compiler pass. -/
def BWLang.Problem.certifyRule? (p : BWLang.Problem) (rounds : Nat := 6)
    (globals : Nat → Int := fun _ => 0) (staged : Bool := false) :
    Option (GuardedRule (ParaBit.scalar globals)) := do
  let prepared ← p.prepare.toOption
  prepared.query.certifyRule? rounds globals staged
