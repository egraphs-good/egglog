import Egglog.Examples.ParaBit

open Egglog ParaBit ParaBit.Op

namespace ParaBitTests

/-- ROVER's commutativity_add. The inputs and widths remain independent. -/
def commutativityAdd : Query :=
  let x := bw (symbol 0) (symbol 3)
  let y := bw (symbol 1) (symbol 4)
  ⟨bw (symbol 2) (add x y), bw (symbol 2) (add y x), []⟩

def inside : Pat := bw (symbol 0) (symbol 2)
def nested : Pat := bw (symbol 1) inside

def widen : Query := ⟨nested, inside, [Op.le (symbol 0) (symbol 1)]⟩
def invalidWiden : Query := ⟨nested, inside, []⟩

/-- The conclusion can be false when its width precondition is dropped. -/
example : ¬invalidWiden.Equivalent (fun i => if i = 1 then 1 else 3) := by
  unfold Query.Equivalent
  decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
/-- Kernel computation produces an independently usable conditional rewrite. -/
theorem widening_equivalence : ∀ ρ, widen.Holds ρ → widen.Equivalent ρ :=
  ((widen.prove? 1).get (by decide +kernel)).down

/-- ROVER add_assoc_4 requires range propagation on both intermediate sums. -/
def assocFullPrecision : Query :=
  let a := bw (symbol 0) (symbol 6)
  let b := bw (symbol 1) (symbol 7)
  let c := bw (symbol 2) (symbol 8)
  let left := bw (symbol 5) (add (bw (symbol 3) (add a b)) c)
  let right := bw (symbol 5) (add a (bw (symbol 4) (add b c)))
  ⟨left, right, [Op.lt (symbol 0) (symbol 3), Op.lt (symbol 1) (symbol 3),
                 Op.lt (symbol 1) (symbol 4), Op.lt (symbol 2) (symbol 4),
                 Op.lt zero (symbol 3), Op.lt zero (symbol 4)]⟩

#guard (assocFullPrecision.prove? 4).isSome
#guard commutativityAdd.prove? 1 |>.isSome
#guard widen.prove? 1 |>.isSome
#guard !(invalidWiden.prove? 2).isSome

end ParaBitTests

namespace ParaBitTests.Diagrams

/-- The published Alive input AndOrXor_2647_values_0, including every width cast. -/
private def alive : Query :=
  let w := symbol 0
  let x := bw w (symbol 1)
  let y := bw w (symbol 2)
  ⟨bw w (bxor (bw w (band x y)) (bw w (bxor x y))), bw w (bor x y), []⟩

#guard (Decision.equivalent? alive.lhs alive.rhs 26).isSome
#guard !(Decision.equivalent? (band (symbol 1) (symbol 2)) (bor (symbol 1) (symbol 2)) 12).isSome

end ParaBitTests.Diagrams

namespace ParaBitTests.Mixed

/-- A carry-free sum, quantified over arbitrary integer operands. -/
private def carryFree : Query :=
  let x := symbol 0; let y := symbol 1
  ⟨add (band x y) (bxor x y), bor x y, []⟩

private def extendAnd : Query :=
  let w := symbol 0; let x := symbol 1; let y := symbol 2
  ⟨band (signed w x) (signed w y), signed w (band x y), []⟩

#guard (carryFree.prove? 1).isSome
#guard (extendAnd.prove? 1).isSome

end ParaBitTests.Mixed

namespace ParaBitTests.OrderedMasks

/-- A strict or weak width bound makes the inner wide mask redundant. -/
private def masked (conditions : List Pat) : Query :=
  let p := symbol 0
  let w := symbol 1
  let x := symbol 2
  let y := symbol 3
  ⟨bw p (band (bw w x) (bw w y)), bw p (band x y), conditions⟩

private def solvedByLanes (conditions : List Pat) : Bool :=
  match Decision.compareLanes (masked conditions) with
  | .proved _ => true
  | _ => false

#guard solvedByLanes [Op.lt (symbol 0) (symbol 1)]
#guard solvedByLanes [Op.le (symbol 0) (symbol 1)]
#guard !(solvedByLanes [])
#guard !(solvedByLanes [Op.le (symbol 1) (symbol 0)])

end ParaBitTests.OrderedMasks
