import Egglog.Rules

namespace Egglog.ForkJoin

private inductive Head where
  | binaryRight (outer first inner a b : Nat)
  | term (p : Pat)
  deriving Inhabited

structure Plan where
  private outer : Nat
  private left : Nat
  private right : Nat
  private leftKey : Nat
  private rightKey : Nat
  private equal : Array (Nat × Nat)
  private head : Head
  deriving Inhabited

mutual
  private def project (vars : List Nat) : Pat → Option Pat
    | .var i => return .var (← vars.idxOf? i)
    | .app f args => return .app f (← projectArgs vars args)
  private def projectArgs (vars : List Nat) : List Pat → Option (List Pat)
    | [] => some []
    | p :: ps => return (← project vars p) :: (← projectArgs vars ps)
end

/-- Compile a binary root joining two binary children on a shared variable.
Every occurrence of a repeated variable contributes an equality constraint. -/
def compile (r : Rewrite) : Option Plan := do
  let .app f [.app l [.var a, .var b], .app k [.var c, .var d]] := r.lhs | none
  let vars := [a, b, c, d]
  let (leftKey, rightKey) ← [(0, 2), (1, 2), (0, 3), (1, 3)].find? fun (i, j) => vars[i]! == vars[j]!
  let mut eq := #[]
  for i in [:4] do
    let first := vars.idxOf vars[i]!
    if first != i && (first != leftKey || i != rightKey) then eq := eq.push (first, i)
  let rhs ← project vars r.rhs
  let head : Head := match rhs with
    | .app op [.var x, .app inner [.var y, .var z]] => .binaryRight op x inner y z
    | _ => .term rhs
  return ⟨f, l, k, leftKey, rightKey - 2, eq, head⟩

private def buckets (g : @& EGraph) (op : Nat) : Array (Array Id) := Id.run do
  let mut groups := Array.replicate g.nodes.size #[]
  for i in g.opNodes op do
    if g.nodes.arity i == 2 then groups := groups.modify (g.find i) (·.push i)
  return groups

@[inline] private def field (rows : @& NodeRows) (l r k : Nat) : Id :=
  if k < 2 then rows.child l k else rows.child r (k - 2)

private def matchesExtra (rows : @& NodeRows) (l r : Id) (eq : @& Array (Nat × Nat)) : Bool :=
  eq.all fun (a, b) => field rows l r a == field rows l r b

structure Hits where
  roots : Array Id := #[]
  left : Array Id := #[]
  right : Array Id := #[]
  deriving Inhabited

/-- A triple fired previously only if both nested rows already belonged to the
outer operands and every variable equality held at that round's boundary. -/
@[inline] private def oldMatch (old : @& NodeRows) (reps : @& Array Id) (p : @& Plan) (root l r : Nat) : Bool :=
  root < old.size && l < old.size && r < old.size &&
    old.child root 0 == reps[l]! && old.child root 1 == reps[r]! &&
    old.child l p.leftKey == old.child r p.rightKey &&
    (p.equal.isEmpty || matchesExtra old l r p.equal)

def search (g : @& EGraph) (old : @& NodeRows) (reps : @& Array Id) (p : @& Plan) : Hits := Id.run do
  let roots := g.opNodes p.outer
  if roots.isEmpty || (g.opNodes p.left).isEmpty || (g.opNodes p.right).isEmpty then return {}
  let left := buckets g p.left
  let right := if p.left == p.right then left else buckets g p.right
  let mut hits : Hits := {}
  for root in roots do
    if g.nodes.arity root != 2 then continue
    let ls := left[g.nodes.child root 0]!
    let rs := right[g.nodes.child root 1]!
    if ls.isEmpty || rs.isEmpty then continue
    for l in ls do
      let key := g.nodes.child l p.leftKey
      for r in rs do
        if key != g.nodes.child r p.rightKey then continue
        if !p.equal.isEmpty && !matchesExtra g.nodes l r p.equal then continue
        if oldMatch old reps p root l r then continue
        hits := ⟨hits.roots.push root, hits.left.push l, hits.right.push r⟩
  return hits

def fire (g : EGraph) (p : @& Plan) (hits : @& Hits) : EGraph := Id.run do
  let mut g := g
  match p.head with
  | .binaryRight op x inner a b =>
    for i in [:hits.roots.size] do
      let root := hits.roots[i]!
      let l := hits.left[i]!
      let r := hits.right[i]!
      let first := field g.nodes l r x
      let (t, next) := g.add2 inner (field g.nodes l r a) (field g.nodes l r b)
      g := next.add2ToClass root op first t
  | .term rhs =>
    for i in [:hits.roots.size] do
      let root := hits.roots[i]!
      let l := hits.left[i]!
      let r := hits.right[i]!
      let env := #[some (g.nodes.child l 0), some (g.nodes.child l 1),
                   some (g.nodes.child r 0), some (g.nodes.child r 1)]
      match rhs with
      | .var v => g := g.merge root env[v]!.get!
      | .app op xs =>
        let mut args := Array.emptyWithCapacity xs.length
        for x in xs do
          let (a, next) := instantiateBound x env g
          args := args.push a
          g := next
        g := g.addToClass root ⟨op, args⟩
  return g

end Egglog.ForkJoin
