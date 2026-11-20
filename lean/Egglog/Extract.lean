import Egglog.Core
import Std.Data.TreeSet

namespace Egglog

/-- Shared subtrees use the same immutable Lean value. Tree cost counts each occurrence. -/
inductive Tree where
  | app (op : Nat) (args : Array Tree)
  deriving Inhabited, Repr

structure Extracted where
  cost : Nat
  term : Tree
  deriving Inhabited, Repr

private structure Pending where
  missing : Nat
  cost : Nat
  height : Nat := 1
  deriving Inhabited

attribute [local instance] lexOrd

/-- Extract a minimum additive-cost tree. Ties favor lower tree height and then
    lower enode ID. A class is finalized only after its chosen children, even
    when a cycle consists entirely of zero-cost constructors. -/
def EGraph.extract (g : EGraph) (id : Id) (cost : Nat → Nat := fun _ => 1) :
    Option Extracted := Id.run do
  if id ≥ g.nodes.size then return none
  let g := g.indexParents
  let target := g.find id
  let mut chosen : Array (Option Tree) := Array.replicate g.nodes.size none
  let mut pending := Array.emptyWithCapacity g.nodes.size
  -- Each ready enode has a complete finite derivation, ordered by its score.
  let mut ready : Std.TreeSet (Nat × Nat × Id) := {}
  for i in [:g.nodes.size] do
    let n := g.nodes[i]!
    let p : Pending := { missing := n.args.size, cost := cost n.op }
    pending := pending.push p
    if p.missing == 0 && g.extractable i then ready := ready.insert (p.cost, p.height, i)
  for _ in [:g.nodes.size] do
    let some key := ready.min? | break
    ready := ready.erase key
    let (price, height, i) := key
    let root := g.find i
    if chosen[root]!.isSome then continue
    let n := g.nodes[i]!
    let term := Tree.app n.op (n.args.map (fun k => chosen[g.find k]!.get!))
    if root == target then return some ⟨price, term⟩
    chosen := chosen.set! root (some term)
    for parent in g.parentNodes root do
      let p := pending[parent]!
      let updated : Pending := {
        missing := p.missing - 1, cost := p.cost + price, height := max p.height (height + 1) }
      pending := pending.set! parent updated
      if updated.missing == 0 && g.extractable parent then ready := ready.insert (updated.cost, updated.height, parent)
  return none

end Egglog
