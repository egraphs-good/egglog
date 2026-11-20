import Egglog.Extract

open Egglog

-- Exercise a dirty graph in which a fresh canonical copy predates repair
-- of the original enode, then force another collision involving its parent.
private def checkCongruence : IO Unit := do
  let (a, g) := EGraph.empty.add ⟨0, #[]⟩
  let (b, g) := g.add ⟨1, #[]⟩
  let (fb, g) := g.add ⟨2, #[b]⟩
  let (hfb, g) := g.add ⟨3, #[fb, fb]⟩
  let g := g.merge a b
  let (fa, g) := g.add ⟨2, #[a]⟩
  let (hfa, g) := g.add ⟨3, #[fa, fa]⟩
  let g := g.rebuild
  unless g.find fa == g.find fb && g.find hfa == g.find hfb do
    throw (IO.userError "congruence did not reach the grandparent")
  unless (g.opNodes 2).size == 1 && (g.matchingNodes hfb 3).size == 1 do
    throw (IO.userError "congruent duplicates still appear in matching indices")
  let epoch := g.mergeCount
  let g := g.addToClass fa ⟨4, #[b]⟩
  unless g.lookup? ⟨4, #[a]⟩ == some (g.find fb) && g.mergeCount == epoch do
    throw (IO.userError "direct insertion into a compacted class failed")
  let g := (g.addToClass a ⟨4, #[a]⟩).rebuild
  unless g.find a == g.find fb && g.mergeCount > epoch && (g.matchingNodes b 4).size == 1 do
    throw (IO.userError "existing right-hand side was not merged")

private def checkExtraction : IO Unit := do
  let (a, g) := EGraph.empty.add ⟨0, #[]⟩
  let (b, g) := g.add ⟨1, #[]⟩
  let (h, g) := g.add ⟨3, #[a, a]⟩
  let g := ((g.addToClass a ⟨2, #[a]⟩).merge a b).rebuild
  let some out := g.extract h (fun op => #[5, 2, 0, 7][op]!)
    | throw (IO.userError "finite tree missed in zero-cost cycle")
  unless out.cost == 11 do throw (IO.userError "tree cost must charge both children")
  let .app 3 #[.app 1 #[], .app 1 #[]] := out.term
    | throw (IO.userError "wrong weighted extraction")
  let some best := g.extract b | throw (IO.userError "missing minimum-cost constant")
  let .app 0 #[] := best.term | throw (IO.userError "tie did not select the earlier enode")

private def checkCollisions : IO Unit := do
  let mut g := EGraph.empty
  let mut leaves := #[]
  -- Nullary symbols with matching high/low halves share the same folded hash.
  -- Their distinct keys force full equality checks through growth and repair.
  for i in [:96] do
    -- Promote both the dense constructor table and sparse symbol table mid-trace.
    if i == 32 then g := g.reserveNodes 32768
    let (leaf, next) := g.add ⟨i * (2^32 + 1), #[]⟩
    let (_, next) := next.add ⟨3, #[leaf]⟩
    leaves := leaves.push leaf
    g := next
  for i in [:96] do
    unless g.lookup? ⟨i * (2^32 + 1), #[]⟩ == some leaves[i]! do
      throw (IO.userError "lost colliding constant during resize")
    if i % 2 == 1 then g := g.merge leaves[i-1]! leaves[i]!
  g := g.rebuild
  for i in [:96] do
    unless g.lookup? ⟨3, #[leaves[i]!]⟩ == g.lookup? ⟨3, #[leaves[2*(i/2)]!]⟩ do
      throw (IO.userError "lost a probe chain past a tombstone")
  unless (g.opNodes 3).size == 48 do throw (IO.userError "incorrect repaired row count")

private def checkBinary : IO Unit := do
  let (a, g) := EGraph.empty.add ⟨0, #[]⟩
  let (b, g) := g.add ⟨1, #[]⟩
  let (f, g) := g.add2 2 a b
  let g := g.reserveNodes 32768
  let (same, g) := g.add ⟨2, #[a, b]⟩
  unless f == same do throw (IO.userError "binary and general paths disagree")
  let (h, g) := g.add ⟨3, #[a, b]⟩
  let g := g.add2ToClass f 3 a b
  let g := (g.add2ToClass a 4 a b).rebuild
  unless g.find f == g.find h && g.lookup? ⟨4, #[a, b]⟩ == some (g.find a) do
    throw (IO.userError "binary fused hit or miss failed")
  -- These parents are added after the first union materializes the reverse index.
  let (fa, g) := g.add2 5 a a
  let (fb, g) := g.add2 5 b b
  let g := (g.merge a b).rebuild
  unless g.find fa == g.find fb do throw (IO.userError "pending parent edges missed")

private def checkRowStorage : IO Unit := do
  let (a, g) := EGraph.empty.add ⟨0, #[]⟩
  let (b, g) := g.add ⟨1, #[]⟩
  let (packed, g) := g.add2 63 a b
  let (wide, g) := g.add2 64 a b
  let huge := 2^67 + 63
  let (large, g) := g.add2 huge a b
  let (ternary, g) := g.add ⟨63, #[a, b, a]⟩
  let g := g.reserveNodes 32768
  for (id, op) in [(packed, 63), (wide, 64), (large, huge)] do
    unless g.nodes.op id == op && g.nodes.arity id == 2 &&
        g.nodes.child id 0 == a && g.nodes.child id 1 == b &&
        g.lookup? ⟨op, #[a, b]⟩ == some id do
      throw (IO.userError "packed and overflow row contents differ")
  unless g.nodes.arity ternary == 3 && g.nodes.child ternary 2 == a do
    throw (IO.userError "lost variadic arguments")
  -- Materialize a parent index after a merge, while old keys still exist.
  let g := (g.merge a b).indexParents
  let (fresh, g) := g.add2 63 a a
  let g := g.rebuild
  unless g.find fresh == g.find packed && (g.matchingNodes packed 63).size == 1 do
    throw (IO.userError "rebuild lost a pending key across index invalidation")
  let (ff, g) := g.add2 7 packed wide
  let (gg, g) := g.add2 7 wide packed
  let g := (g.indexParents.merge wide fresh).rebuild
  unless g.find ff == g.find gg && g.find packed == g.find fresh &&
      g.oldest fresh == packed do
    throw (IO.userError "indexed congruence lost a retired handle")

def main : IO Unit := do
  checkCongruence
  checkExtraction
  checkCollisions
  checkBinary
  checkRowStorage
