import Egglog.Core
import Egglog.Verified

namespace Egglog

abbrev Bindings := Array (Option Id)

def bindVar (env : Bindings) (varId : Nat) (value : Id) : Option Bindings :=
  match env[varId]?.join with
  | some prev => if prev == value then some env else none
  | none =>
    if varId == env.size then some (env.push (some value))
    else if varId < env.size then some (env.set! varId (some value))
    else some ((env ++ Array.replicate (varId - env.size) none).push (some value))

-- Linear variable sequences follow one path and allocate a result only on success.
-- Nested applications branch over the child's e-class members for that operator.
private partial def matchFrom (g : EGraph) (ps : List Pat) (args : Array Id) (i : Nat)
    (env : Bindings) (hits : Array Bindings) : Array Bindings := Id.run do
  match ps with
  | [] => return hits.push env
  | p :: rest =>
    let id := g.find args[i]!
    match p with
    | .var v =>
      let some next := bindVar env v id | return hits
      return matchFrom g rest args (i + 1) next hits
    | .app op [] =>
      let some c := (if g.maskVersion == 0 then g.lookup? ⟨op, #[]⟩ else g.lookupActive? ⟨op, #[]⟩) | return hits
      if c == id then return matchFrom g rest args (i + 1) env hits
      return hits
    | .app op kids =>
      let mut hits := hits
      for member in g.matchingNodes id op do
        let n := g.nodes[member]!
        if n.args.size == kids.length then
          for e in matchFrom g kids n.args 0 env #[] do
            hits := matchFrom g rest args (i + 1) e hits
      return hits

def matchArgs (g : EGraph) (ps : List Pat) (args : Array Id)
    (env : Bindings) : Array Bindings :=
  if ps.length == args.size then matchFrom g ps args 0 env #[] else #[]

structure Hit where
  root : Id
  env : Bindings

private def search (g : EGraph) (lhs : Pat) : Array Hit := Id.run do
  let mut hits := #[]
  match lhs with
  | .app op args =>
    for id in g.opNodes op do
      let n := g.nodes[id]!
      for env in matchArgs g args n.args #[] do
        hits := hits.push ⟨id, env⟩
  | .var v =>
    for id in [:g.nodes.size] do
      if g.find id == id then
        if let some e := bindVar #[] v id then hits := hits.push ⟨id, e⟩
  return hits

mutual
  /-- Instantiate a pattern whose variables have already been bound by the body. -/
  partial def instantiateBound (rhs : Pat) (env : Bindings) (g : EGraph) : Id × EGraph :=
    match rhs with
    | .var v => (env[v]!.get!, g)
    | .app op kids =>
      let (ids, g) := instantiateArgs kids env g
      g.add ⟨op, ids⟩

  private partial def instantiateArgs (kids : List Pat) (env : Bindings) (g : EGraph) :
      Array Id × EGraph := Id.run do
    let mut g := g
    let mut ids := Array.emptyWithCapacity kids.length
    for kid in kids do
      let (id, next) := instantiateBound kid env g
      ids := ids.push id
      g := next
    return (ids, g)
end

private partial def allBound (p : Pat) (isBound : Nat → Bool) : Bool :=
  match p with
  | .var i => isBound i
  | .app _ kids => kids.all fun kid => allBound kid isBound

def instantiate (rhs : Pat) (env : Bindings) (g : EGraph) : Option (Id × EGraph) :=
  if allBound rhs (fun i => env[i]?.join.isSome) then some (instantiateBound rhs env g)
  else none

private partial def collectVars (p : Pat) (vs : Std.HashSet Nat) : Std.HashSet Nat :=
  match p with
  | .var i => vs.insert i
  | .app _ kids => kids.foldl (fun vs p => collectVars p vs) vs

/-- Matches are collected in one snapshot, with no outstanding congruence repairs. -/
structure MatchBatch where
  private entries : Array (Pat × Array Hit)

def searchRound (g : EGraph) (rules : Array Rewrite) : MatchBatch :=
  ⟨rules.filterMap fun r =>
    let vs := collectVars r.lhs {}
    if allBound r.rhs vs.contains then some (r.rhs, search g r.lhs) else none⟩

def MatchBatch.append (a b : MatchBatch) : MatchBatch := ⟨a.entries ++ b.entries⟩

/-- Filter matches and specialize heads using only their snapshot bindings. -/
def MatchBatch.filterHeadsM (b : MatchBatch) (f : Pat → Bindings → Except String (Option Pat)) :
    Except String MatchBatch := do
  let mut entries := #[]
  for (rhs, hits) in b.entries do
    for hit in hits do
      if let some p ← f rhs hit.env then entries := entries.push (p, #[hit])
  return ⟨entries⟩

/-- Apply a batch to its source graph, possibly extended with other same-round heads.
The caller performs one rebuild after all heads have been applied. -/
def applyRound (g : EGraph) (batch : MatchBatch) : EGraph := Id.run do
  let mut g := g
  for (rhs, hits) in batch.entries do
    for hit in hits do
      match rhs with
      | .var v => g := g.merge hit.root hit.env[v]!.get!
      | .app op kids =>
        let (ids, next) := instantiateArgs kids hit.env g
        g := next.addToClass hit.root ⟨op, ids⟩
  return g

/-- Each iteration selects matches in the rebuilt graph, applies all heads,
and restores congruence before the next search. A fixed point ends the run early. -/
def runRounds (initial : EGraph) (rules : Array Rewrite) (limit : Nat) : EGraph := Id.run do
  let mut g := initial.rebuild
  for _ in [:limit] do
    let before := g.version
    let batch := searchRound g rules
    g := (applyRound g batch).rebuild
    if g.version == before then break
  return g

/-- Executes the very same syntax trees used by the rules' semantic proofs. -/
def runVerified (g : EGraph) (rules : Array (VerifiedRule m)) (limit : Nat) : EGraph :=
  runRounds g (rules.map (·.toRewrite)) limit

end Egglog
