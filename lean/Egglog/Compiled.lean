import Egglog.ForkJoin

namespace Egglog

/- A compiled pattern projects variables directly out of one or two matching
   constructor rows. Deeper joins use the recursive matcher. Both paths consume
   the same Rewrite value that carries the semantic proof at registration. -/
namespace Compiled

private inductive Slot where
  | outer (i : Nat) | inner (i : Nat) | constRef (i : Nat) | temp (i : Nat)
  deriving Inhabited, BEq

private inductive Inputs where
  | zero | one (a : Slot) | two (a b : Slot) | many (args : Array Slot)
  deriving Inhabited, BEq

private def inputs (args : Array Slot) : Inputs :=
  match args with
  | #[] => .zero
  | #[a] => .one a
  | #[a, b] => .two a b
  | rest => .many rest

private structure Call where
  op : Nat
  fields : Inputs
  deriving Inhabited, BEq

private inductive Head where
  | unify (slot : Slot)
  | insert (call : Call)
  deriving Inhabited

private structure Locals where
  vars : Std.HashMap Nat Slot := {}
  equal : Array (Slot × Slot) := #[]
  literals : Array (Slot × Nat) := #[]
  constants : Array Pat := #[]
  calls : Array Call := #[]
  deriving Inhabited

private structure Plan where
  op : Nat
  arity : Nat
  nested : Option (Nat × Nat × Nat)  -- position, op, arity
  locals : Locals
  head : Head
  inverseCovered : Bool
  deriving Inhabited

private abbrev Build := StateT Locals Option

private def leaf (p : Pat) (slot : Slot) : Build Unit := do
  match p with
  | .var v =>
    let l ← get
    if let some prev := l.vars.get? v then modify fun s => { s with equal := s.equal.push (prev, slot) }
    else modify fun s => { s with vars := s.vars.insert v slot }
  | .app op [] => modify fun s => { s with literals := s.literals.push (slot, op) }
  | _ => failure

private partial def isGround : Pat → Bool
  | .var _ => false
  | .app _ kids => kids.all isGround

private partial def project (p : Pat) : Build Slot := do
  match p with
  | .var v => return ← (← get).vars.get? v
  | .app op kids =>
    if isGround p then
      let s ← get
      if let some i := s.constants.idxOf? p then return .constRef i
      set { s with constants := s.constants.push p }
      return .constRef s.constants.size
    let fields := inputs (← kids.mapM project).toArray
    let s ← get
    let call : Call := ⟨op, fields⟩
    if let some i := s.calls.idxOf? call then return .temp i
    set { s with calls := s.calls.push call }
    return .temp s.calls.size

private def compile (r : Rewrite) : Option Plan := do
  let .app op fields := r.lhs | none
  let action : Build _ := do
    let mut nested := none
    for (p, col) in fields.zipIdx do
      match p with
      | .app child (x :: xs) =>
        if nested.isSome then failure
        nested := some (col, child, xs.length + 1)
        for (q, i) in (x :: xs).zipIdx do leaf q (.inner i)
      | _ => leaf p (.outer col)
    let head ← match r.rhs with
      | .app op kids =>
        if isGround r.rhs then pure (.unify (← project r.rhs))
        else pure (.insert ⟨op, inputs (← kids.mapM project).toArray⟩)
      | .var _ => pure (.unify (← project r.rhs))
    return (nested, head)
  let ((nested, head), locals) ← action.run {}
  -- If a projection permutes arguments and is its own inverse, every row
  -- it creates already has its return edge in the same class.
  let inverseCovered := Id.run do
    let .app dest ys := r.rhs | return false
    if op != dest || ys.length != fields.length then return false
    let some src := fields.mapM (fun | Pat.var v => some v | _ => none) | return false
    let some dst := ys.mapM (fun | Pat.var v => some v | _ => none) | return false
    if !src.Nodup || !dst.Nodup then return false
    let perm := dst.map (fun v => src.idxOf v)
    return perm.zipIdx.all fun (j, i) => perm[j]? == some i
  return ⟨op, fields.length, nested, locals, head, inverseCovered⟩

@[inline] private def readSlot (r n consts temps : @& Array Id) : Slot → Id
  | .outer i => r[i]!
  | .inner i => n[i]!
  | .constRef i => consts[i]!
  | .temp i => temps[i]!

@[inline] private def mkNode (call : Call) (r n consts temps : @& Array Id) : ENode :=
  let atSlot := readSlot r n consts temps
  ⟨call.op, match call.fields with
    | .zero => #[]
    | .one a => #[atSlot a]
    | .two a b => #[atSlot a, atSlot b]
    | .many args => args.map atSlot⟩

-- Most nested algebraic heads construct one or two intermediate applications.
-- Keep those results in scalar locals across each hit.
@[inline] private def scalarSlot (r n consts : @& Array Id) (t0 t1 : Id) : Slot → Id
  | .outer i => r[i]!
  | .inner i => n[i]!
  | .constRef i => consts[i]!
  | .temp i => if i == 0 then t0 else t1

@[inline] private def scalarArgs (args : Inputs) (read : Slot → Id) : Array Id :=
  match args with
  | .zero => #[]
  | .one a => #[read a]
  | .two a b => #[read a, read b]
  | .many xs => xs.map read

@[inline] private def addCall (g : EGraph) (call : Call) (r n consts : @& Array Id)
    (t0 t1 : Id) : Id × EGraph :=
  let read := scalarSlot r n consts t0 t1
  match call.fields with
  | .two a b => g.add2 call.op (read a) (read b)
  | fields => g.add ⟨call.op, scalarArgs fields read⟩

@[inline] private def finishCall (g : EGraph) (root : Id) (call : Call) (r n consts : @& Array Id)
    (t0 t1 : Id) : EGraph :=
  let read := scalarSlot r n consts t0 t1
  match call.fields with
  | .two a b => g.add2ToClass root call.op (read a) (read b)
  | fields => g.addToClass root ⟨call.op, scalarArgs fields read⟩

private structure Hits where
  roots : Array Id := #[]
  children : Array Id := #[]
  deriving Inhabited

private structure Previous where
  nodes : NodeRows := {}
  reps : Array Id := #[]

/-- Two old rows already joined if the inner result inhabited this outer
    operand's class in the previous snapshot. A union can only add matches.
    This test also works when essential child IDs changed during rebuild. -/
private def nestedRows (g : @& EGraph) (prev : @& Previous) (p : Plan)
    (roots : Array Id) (col op arity : Nat) : Hits := Id.run do
  let mut hits : Hits := {}
  let withOp := g.opNodes op
  for id in roots do
    if g.nodes.arity id != p.arity then continue
    let target := g.nodes.child id col
    let old := if id < prev.nodes.size then some (prev.nodes.child id col) else none
    let members := g.members target
    if members.size <= withOp.size then
      for child in members do
        if g.nodes.op child != op || g.nodes.arity child != arity then continue
        if let some k := old then
          if child < prev.reps.size && prev.reps[child]! == k then continue
        hits := ⟨hits.roots.push id, hits.children.push child⟩
    else
      for child in withOp do
        if g.find child != target || g.nodes.arity child != arity then continue
        if let some k := old then
          if child < prev.reps.size && prev.reps[child]! == k then continue
        hits := ⟨hits.roots.push id, hits.children.push child⟩
  return hits

private def searchPlan (g : @& EGraph) (prev : @& Previous) (p : Plan) (cutoff : Nat) (rekeyed : Bool)
    (covered : Nat × Nat) : Hits := Id.run do
  let mut literals := #[]
  for (slot, op) in p.locals.literals do
    let some id := g.lookup? ⟨op, #[]⟩ | return {}
    literals := literals.push (slot, id)
  let accept := fun (r n : Array Id) =>
    p.locals.equal.all (fun (a,b) => readSlot r n #[] #[] a == readSlot r n #[] #[] b) &&
    literals.all (fun (a,id) => readSlot r n #[] #[] a == id)
  -- Existing row IDs and their children stay stable across iterations without
  -- representative changes. A new nullary row can attach to an existing class.
  let lower := if !literals.isEmpty || (rekeyed && (p.nested.isSome || !p.locals.equal.isEmpty))
               then 0 else cutoff
  let roots := g.opNodes p.op
  if roots.isEmpty then return {}
  if let some (col, op, arity) := p.nested then
    if literals.isEmpty && p.locals.equal.isEmpty then return nestedRows g prev p roots col op arity
  if p.nested.isNone then
    let last := roots.back!
    if last < lower || (covered.1 ≤ lower && covered.1 ≤ last && last < covered.2) then return {}
  if p.nested.isNone && literals.isEmpty then
    if let #[(.outer a, .outer b)] := p.locals.equal then
      let selected := roots.filter fun id =>
        id ≥ lower && g.nodes.arity id == p.arity && g.nodes.child id a == g.nodes.child id b
      return { roots := selected }
  let mut hits : Hits := {}
  for id in roots do
    if p.nested.isNone && (id < lower || (covered.1 ≤ id && id < covered.2)) then continue
    let row := g.nodes[id]!.args
    if row.size != p.arity then continue
    match p.nested with
    | none =>
      if id ≥ lower && accept row #[] then hits := { hits with roots := hits.roots.push id }
    | some (col, op, arity) =>
      for child in g.matchingNodes row[col]! op do
        if id < lower && child < lower then continue
        let args := g.nodes[child]!.args
        if args.size == arity && accept row args then
          hits := ⟨hits.roots.push id, hits.children.push child⟩
  return hits

@[specialize] private def foldHits (g : EGraph) (hits : Hits)
    (apply : EGraph → Id → Array Id → Array Id → EGraph) : EGraph := Id.run do
  let mut g := g
  for i in [:hits.roots.size] do
    let root := hits.roots[i]!
    let outer := g.nodes[root]!.args
    let inner := if hits.children.isEmpty then #[] else g.nodes[hits.children[i]!]!.args
    g := apply g root outer inner
  return g

-- Lower binary composition and distribution to straight-line instructions.
-- Both operators and operand positions come from the plan. The inner loop
-- reads three scalar operands before applying any writes.
@[inline] private def childAt (g : @& EGraph) (id col : Nat) : Nat :=
  g.nodes.child id col

private def composeBinary (g : EGraph) (hits : Hits) (outerPos a b op out : Nat) :
    EGraph := Id.run do
  let mut g := g
  for i in [:hits.roots.size] do
    let root := hits.roots[i]!
    let child := hits.children[i]!
    let x := childAt g root outerPos
    let y := childAt g child a
    let z := childAt g child b
    let (temp, next) := g.add2 op x y
    g := next.add2ToClass root out temp z
  return g

private def distributeBinary (g : EGraph) (hits : Hits) (outerPos a b f h out : Nat) :
    EGraph := Id.run do
  let mut g := g
  for i in [:hits.roots.size] do
    let root := hits.roots[i]!
    let child := hits.children[i]!
    let x := childAt g root outerPos
    let y := childAt g child a
    let z := childAt g child b
    let (u, next) := g.add2 f x y
    let (v, next) := next.add2 h x z
    g := next.add2ToClass root out u v
  return g

private def applyPlan (g : EGraph) (p : Plan) (hits : Hits) : EGraph := Id.run do
  if hits.roots.isEmpty then return g
  let mut g := g
  let mut consts := #[]
  for c in p.locals.constants do
    let (id, next) := instantiateBound c #[] g
    consts := consts.push id
    g := next
  if let .insert ⟨op, .two (.temp 0) (.inner b)⟩ := p.head then
    if let #[⟨f, .two (.outer x) (.inner a)⟩] := p.locals.calls then
      return composeBinary g hits x a b f op
  if let .insert ⟨op, .two (.temp 0) (.temp 1)⟩ := p.head then
    if let #[⟨f, .two (.outer x) (.inner a)⟩, ⟨h, .two (.outer x') (.inner b)⟩] := p.locals.calls then
      if x == x' then return distributeBinary g hits x a b f h op
  -- Tight projection loop for a flat binary head. Hoist the compiled head and
  -- its constants out of the row loop, avoiding a substitution array per match.
  if p.locals.calls.isEmpty && p.nested.isNone then
    if let .insert ⟨op, .two a b⟩ := p.head then
      if let .constRef k := a then
        if let .outer j := b then
          let c := consts[k]!
          for root in hits.roots do g := g.add2ToClass root op c (childAt g root j)
          return g
      if let .outer i := a then
        if let .constRef k := b then
          let c := consts[k]!
          for root in hits.roots do g := g.add2ToClass root op (childAt g root i) c
          return g
      if let .outer i := a then
        if let .outer j := b then
          for root in hits.roots do
            let x := childAt g root i
            let y := childAt g root j
            g := g.add2ToClass root op x y
          return g
      for root in hits.roots do
        let args := g.nodes[root]!.args
        g := g.add2ToClass root op (readSlot args #[] consts #[] a) (readSlot args #[] consts #[] b)
      return g
  if let .insert out := p.head then
    match p.locals.calls with
    | #[first] =>
      return foldHits g hits fun g root r n =>
        let (t0, g) := addCall g first r n consts 0 0
        finishCall g root out r n consts t0 0
    | #[first, second] =>
      return foldHits g hits fun g root r n =>
        let (t0, g) := addCall g first r n consts 0 0
        let (t1, g) := addCall g second r n consts t0 0
        finishCall g root out r n consts t0 t1
    | _ => pure ()
  for h in [:hits.roots.size] do
    let root := hits.roots[h]!
    let r := g.nodes[root]!.args
    let n := if hits.children.isEmpty then #[] else g.nodes[hits.children[h]!]!.args
    let mut temps := #[]
    for call in p.locals.calls do
      let (id, next) := match call.fields with
        | .two a b => g.add2 call.op (readSlot r n consts temps a) (readSlot r n consts temps b)
        | _ => g.add (mkNode call r n consts temps)
      g := next
      temps := temps.push id
    match p.head with
    | .unify slot => g := g.merge root (readSlot r n consts temps slot)
    | .insert call =>
      if let .two a b := call.fields then
        g := g.add2ToClass root call.op (readSlot r n consts temps a) (readSlot r n consts temps b)
      else g := g.addToClass root (mkNode call r n consts temps)
  return g

private inductive BinaryHead where
  | compose (f out : Nat)
  | distribute (f h out : Nat)
  deriving Inhabited

private structure FactorPlan where
  roots : Array Id
  rootArity : Nat
  joinCol : Nat
  outerCol : Nat
  first : Nat
  second : Nat
  inner : Array (Array Id)
  head : BinaryHead
  deriving Inhabited

private def binaryHead (p : Plan) : Option (Nat × Nat × Nat × BinaryHead) := do
  guard (p.locals.equal.isEmpty && p.locals.literals.isEmpty)
  match p.head, p.locals.calls with
  | .insert ⟨op, .two (.temp 0) (.inner b)⟩, #[⟨f, .two (.outer x) (.inner a)⟩] =>
    return (x, a, b, .compose f op)
  | .insert ⟨op, .two (.temp 0) (.temp 1)⟩,
      #[⟨f, .two (.outer x) (.inner a)⟩, ⟨h, .two (.outer x') (.inner b)⟩] =>
    guard (x == x')
    return (x, a, b, .distribute f h op)
  | _, _ => none

/-- Keep one bucket per inner result class. Many outer rows use the same
    bucket. Stream that Cartesian product into the actions once search ends. -/
private def factorPlans (g : @& EGraph) (plans : Array Plan) : Array (Option FactorPlan) := Id.run do
  if g.nodes.size < 32768 then return Array.replicate plans.size none
  let mut indices : Std.HashMap (Nat × Nat) (Array (Array Id)) := {}
  let mut factors := #[]
  for p in plans do
    let some (col, inner, arity) := p.nested | factors := factors.push none; continue
    let some (x, a, b, head) := binaryHead p | factors := factors.push none; continue
    let roots := g.opNodes p.op
    if roots.size < 1024 then factors := factors.push none; continue
    let mut index := #[]
    if let some old := indices.get? (inner, arity) then index := old
    else
      index := Array.replicate g.nodes.size #[]
      for id in g.opNodes inner do
        if g.nodes.arity id == arity then index := index.modify (g.find id) (·.push id)
      indices := indices.insert (inner, arity) index
    factors := factors.push (some ⟨roots, p.arity, col, x, a, b, index, head⟩)
  return factors

@[specialize] private def streamFactor (g : EGraph) (p : FactorPlan) (prev : Previous)
    (emit : EGraph → Nat → Nat → Nat → Nat → EGraph) : EGraph := Id.run do
  let mut g := g
  for root in p.roots do
    if g.nodes.arity root != p.rootArity then continue
    let target := g.nodes.child root p.joinCol
    let x := g.nodes.child root p.outerCol
    let old := if root < prev.nodes.size then some (prev.nodes.child root p.joinCol) else none
    for child in p.inner[target]! do
      if let some k := old then
        if child < prev.reps.size && prev.reps[child]! == k then continue
      let y := g.nodes.child child p.first
      let z := g.nodes.child child p.second
      g := emit g root x y z
  return g

private def applyFactor (g : EGraph) (p : FactorPlan) (prev : Previous) : EGraph :=
  match p.head with
  | .compose f op => streamFactor g p prev fun g root x y z =>
    let (u, g) := g.add2 f x y
    g.add2ToClass root op u z
  | .distribute f h op => streamFactor g p prev fun g root x y z =>
    let (u, g) := g.add2 f x y
    let (v, g) := g.add2 h x z
    g.add2ToClass root op u v

/-- Select matches on a round snapshot. Reuse previous matches only where their
    projected child values and class membership are still valid. -/
def run (initial : EGraph) (rules : Array Rewrite) (limit : Nat) : EGraph := Id.run do
  let mut plans := #[]
  let mut forks := #[]
  let mut fallback := #[]
  for rw in rules do
    if let some p := compile rw then plans := plans.push p
    else if let some p := ForkJoin.compile rw then forks := forks.push p
    else fallback := fallback.push rw
  let mut g := initial.rebuild
  let trackNested := plans.any (fun p => p.nested.isSome && p.locals.equal.isEmpty && p.locals.literals.isEmpty)
  let mut prev : Previous := {}
  let mut cutoff := 0
  let mut rekeyed := false
  let mut covered := Array.replicate plans.size (0, 0)
  for _ in [:limit] do
    let version := g.version
    let merges := g.mergeCount
    let factors := factorPlans g plans
    let batches := plans.mapIdx fun i p =>
      if factors[i]!.isSome then {} else searchPlan g prev p cutoff rekeyed covered[i]!
    let joined := forks.map (ForkJoin.search g prev.nodes prev.reps)
    let rest := searchRound g fallback
    cutoff := g.nodes.size
    let previous := prev
    if trackNested || !forks.isEmpty then prev := ⟨g.nodes, g.representatives⟩
    for i in [:plans.size] do
      let p := plans[i]!
      let first := g.nodes.size
      if let some plan := factors[i]! then g := applyFactor g plan previous
      else g := applyPlan g p batches[i]!
      if p.inverseCovered then covered := covered.set! i (first, g.nodes.size)
    for i in [:forks.size] do g := ForkJoin.fire g forks[i]! joined[i]!
    g := (applyRound g rest).rebuild
    if version == g.version then break
    rekeyed := g.mergeCount != merges
  return g

/-- Native search over the syntax checked by the semantic theorem of each rule. -/
def runVerified (g : EGraph) (rules : Array (VerifiedRule m)) (limit : Nat) : EGraph :=
  run g (rules.map (·.toRewrite)) limit

end Compiled
end Egglog
