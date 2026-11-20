import Egglog.Compiled
import Egglog.Extract

open Egglog

namespace CompiledTests

private abbrev app := Pat.app
private abbrev x := Pat.var 0
private abbrev y := Pat.var 1
private abbrev a := app 0 []
private abbrev b := app 1 []
private def seed (terms : List Pat) :=
  terms.foldl (fun g term => (instantiateBound term #[] g).2) EGraph.empty

private partial def resolveTree (g : EGraph) (t : Tree) : Option Id := do
  match t with
  | .app op xs => g.lookup? ⟨op, ← xs.mapM (resolveTree g)⟩

-- Extract one ground witness for every class and use it to compare *all* rows
-- modulo graph-specific representatives, including rows added by either matcher.
private def includes (g other : EGraph) : Bool := Id.run do
  let mut images : Std.HashMap Id Id := {}
  let mut used : Std.HashSet Id := {}
  for id in [:g.nodes.size] do
    if g.find id != id then continue
    let some witness := g.extract id | return false
    let some image := resolveTree other witness.term | return false
    if used.contains image then return false
    images := images.insert id image
    used := used.insert image
  for i in [:g.nodes.size] do
    let node := g.nodes[i]!
    let key := node.args.map fun arg => images[g.find arg]!
    if other.lookup? ⟨node.op, key⟩ != images.get? (g.find i) then return false
  return true

private def agree (g : EGraph) (rs : Array Rewrite) : Bool :=
  (List.range 6).all fun rounds =>
    let fast := Compiled.run g rs rounds
    let ref := runRounds g rs rounds
    includes fast ref && includes ref fast

-- Old parent; new child constructor attached directly to an existing class.
private def childGraph := seed [app 10 [app 11 [a]]]
private def childRules : Array Rewrite :=
  #[⟨app 11 [x], app 12 [x]⟩, ⟨app 10 [app 12 [x]], app 13 [x]⟩]
#guard agree childGraph childRules
#guard (resolveTree (Compiled.run childGraph childRules 1) (.app 13 #[.app 0 #[]])).isNone
#guard (resolveTree (Compiled.run childGraph childRules 2) (.app 13 #[.app 0 #[]])).isSome

-- Dirty parents, old nested matches and repeated-variable guards become enabled
-- when already existing distinct roots merge. Subsidiary congruences retire rows.
private def dirtyGraph := seed [app 20 [app 21 [a], app 21 [b]], app 22 [app 10 [a]],
                                 app 23 [app 10 [b]], app 24 [b]]
private def dirtyRules : Array Rewrite :=
  #[⟨a, b⟩, ⟨app 20 [x, x], app 25 [x]⟩, ⟨app 22 [app 10 [b]], app 26 []⟩,
    ⟨app 24 [a], app 27 []⟩, ⟨app 23 [app 10 [x]], app 28 [x, x, x]⟩]
#guard agree dirtyGraph dirtyRules
#guard (resolveTree (Compiled.run dirtyGraph dirtyRules 2) (.app 25 #[.app 21 #[.app 0 #[]]])).isSome

-- Mixed arities at the same operator ID, sparse variable IDs, a variable-rooted
-- fallback, multiple nested subpatterns, deep children and an unbound RHS.
private def fallbackGraph :=
  seed [app 30 [app 31 [a], app 32 [b]], app 30 [a], app 33 [app 34 [app 35 [b]]]]
private def fallbackRules : Array Rewrite :=
  #[⟨app 30 [app 31 [x], app 32 [y]], app 37 [y, x]⟩,
    ⟨app 33 [app 34 [app 35 [x]]], app 38 [x]⟩,
    ⟨app 30 [Pat.var 7], app 39 [Pat.var 7]⟩, ⟨app 30 [x], y⟩,
    ⟨Pat.var 3, Pat.var 3⟩]
#guard agree fallbackGraph fallbackRules

-- The shared column can be in either child position. A later merge can
-- enable an old triple by making its previously unequal join keys equal.
private abbrev z := Pat.var 9
private def forkGraph := seed [
  app 40 [app 41 [a, b], app 42 [a, a]], app 40 [app 41 [b, b], app 42 [b, a]],
  app 40 [app 41 [a], app 42 [a, a]], app 40 [app 41 [a, a]], app 48 [a]]

private def forkRules : Array Rewrite := #[
  ⟨app 40 [app 41 [x, y], app 42 [z, x]], app 43 [x, app 44 [y, z]]⟩,
  ⟨app 40 [app 41 [x, x], app 42 [x, z]], app 45 [z, x]⟩,
  ⟨app 40 [app 41 [x, y], app 42 [y, x]], app 46 [y, x, y]⟩,
  ⟨app 40 [app 41 [x, y], app 42 [y, x]], Pat.var 99⟩,
  ⟨app 48 [x], app 48 [b]⟩, ⟨a, b⟩]

#guard (ForkJoin.compile forkRules[0]!).isSome
#guard (ForkJoin.compile forkRules[3]!).isNone
#guard agree forkGraph forkRules

end CompiledTests
