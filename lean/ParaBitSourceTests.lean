import Egglog.Examples.ParaBitInput

open BWLang
namespace ParaBitSourceTests
open Term

private def w (name : String) (x : Term) := app "bw" [symbol name, x]
private def ltW (p q : String) := app "<" [symbol p, symbol q]
private def qWidth := symbol "q"
private def rWidth := symbol "r"
private def tWidth := symbol "t"

/-- ROVER commutativity_add, with the same source variables and width premises as the artifact. -/
def rover : Problem :=
  let a := w "p" (symbol "a")
  let b := w "q" (symbol "b")
  ⟨w "r" (app "+" [a, b]), w "r" (app "+" [b, a]), [], [rWidth, symbol "p", qWidth]⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem rover_source_equivalence : ∀ ρ, rover.Holds ρ → rover.lhs.eval ρ = rover.rhs.eval ρ :=
  ((rover.prove? 1).get (by decide +kernel)).down

private def negated := app "-" [w "r" (literal 0), w "r" (w "q" (symbol "newvar0"))]
private def signedQ := app "signed" [qWidth, w "q" (symbol "newvar0")]

/-- Both Hydra inputs explicitly assume strictly increasing widths. -/
def hydra310 : Problem :=
  ⟨w "r" negated, w "r" signedQ, [ltW "q" "r"], [rWidth, qWidth]⟩
def hydra2703 : Problem :=
  ⟨w "t" (app "signed" [rWidth, w "r" negated]), w "t" signedQ,
    [ltW "q" "r", ltW "q" "t", ltW "r" "t"], [tWidth, rWidth, qWidth]⟩

def countermodel : String → Int
  | "q" => 2 | "r" => 3 | "t" => 4 | _ => 1

example : hydra310.Holds countermodel ∧ hydra310.lhs.eval countermodel = 7 ∧
    hydra310.rhs.eval countermodel = 1 := by
  constructor
  · intro t ht
    simp [Problem.premises, hydra310] at ht
    rcases ht with rfl | rfl | rfl <;> decide
  · decide

example : hydra2703.Holds countermodel ∧ hydra2703.lhs.eval countermodel = 15 ∧
    hydra2703.rhs.eval countermodel = 1 := by
  constructor
  · intro t ht
    simp [Problem.premises, hydra2703] at ht
    rcases ht with rfl | rfl | rfl | rfl | rfl | rfl <;> decide
  · decide

private def branchSelect (right : Term) : Term :=
  app "sel" [symbol "c", w "w" (symbol "a"), right]

private def castSelect (right : Term) : Problem :=
  let choice := branchSelect right
  ⟨w "w" choice, choice, [], []⟩

private def boundedSelect := castSelect (w "w" (symbol "b"))

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem bounded_select_equivalence :
    ∀ ρ, boundedSelect.Holds ρ → boundedSelect.lhs.eval ρ = boundedSelect.rhs.eval ρ :=
  ((boundedSelect.proveStaged? 2).get (by decide +kernel)).down

private def unboundedSelect := castSelect (symbol "b")
private def selectWitness : String → Int
  | "w" => 1 | "b" => 2 | _ => 0

example : unboundedSelect.Holds selectWitness ∧
    unboundedSelect.lhs.eval selectWitness ≠ unboundedSelect.rhs.eval selectWitness := by
  constructor
  · intro t ht
    simp [Problem.premises, unboundedSelect, castSelect] at ht
  · decide

#guard !(unboundedSelect.proveStaged? 2).isSome

end ParaBitSourceTests
