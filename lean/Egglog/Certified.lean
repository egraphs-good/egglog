import Egglog.RelationalCompiler
import Egglog.GenericJoin

namespace Egglog.Certified

/-- Information visible to a user scheduling a round. Stored rows count
    interned terms, including multiple members of the same e-class. -/
structure ScheduleContext where
  round : Nat
  storedRows : Nat
  version : Nat

/-- A stateful policy may choose a subset of a rule's matches and postpone
    the rest. Substitutions expose representative syntax by rule variable ID. -/
structure MatchScheduler where
  State : Type
  initial : State
  choose : State → ScheduleContext → String → Array (List (Nat × Pat)) → State × Array Nat
  canStop : State → ScheduleContext → (hasPending : Bool) → Bool := fun _ _ pending => !pending

structure NamedRule (m : Model α) where
  name : String
  rule : GuardedRule m

def NamedRule.plain (name : String) (r : VerifiedRule m) : NamedRule m :=
  ⟨name, ⟨r.toRewrite, [], fun env _ => r.sound env⟩⟩

/-- A partial, exact e-class analysis. Each transfer equation states what a
    constructor denotes given the established values of its children. Equal
    analysis values justify merging their classes. -/
structure ValueAnalysis (m : Model α) (β : Type) where
  meaning : β → α
  transfer : Nat → List β → Option β
  sound : ∀ op xs y, transfer op xs = some y → m.interp op (xs.map meaning) = meaning y

/-- A schedule runs against one persistent graph, including across ruleset changes. -/
inductive Schedule (m : Model α) where
  | run (rules : Array (VerifiedRule m))
  | guarded (rules : Array (GuardedRule m))
  | relational (rules : Array (RelRule m))
  | analyze {β : Type} [BEq β] [Hashable β] [DecidableEq β] (analysis : ValueAnalysis m β)
  | seq (schedules : List (Schedule m))
  | repeat (n : Nat) (body : Schedule m)
  | saturate (fuel : Nat) (body : Schedule m)
  | fixpoint (body : Schedule m)
  | untilMatch (facts : List (Pat × Pat)) (body : Schedule m)
  | untilFacts (facts : List RelAtom) (body : Schedule m)
  | scheduled (rounds : Nat) (rules : Array (NamedRule m)) (policy : MatchScheduler)

private abbrev Same (m : Model α) (a b : Pat) := ∀ env, a.eval m env = b.eval m env
private abbrev Witness (m : Model α) (a : Pat) := { b : Pat // Same m b a }
private abbrev Args (m : Model α) (a : List Pat) :=
  { b : List Pat // ∀ env, Pat.evalArgs b m env = Pat.evalArgs a m env }

private def reflW (m : Model α) (a : Pat) : Witness m a := ⟨a, fun _ => rfl⟩

private def appW (op : Nat) (xs : Args m ps) : Witness m (.app op ps) :=
  ⟨.app op xs.val, fun e => congrArg (m.interp op) (xs.property e)⟩

private theorem cons_congr {a b : α} {as bs : List α} (h : a = b) (hs : as = bs) :
    a :: as = b :: bs := h ▸ hs ▸ rfl

private def consW (x : Witness m p) (xs : Args m ps) : Args m (p :: ps) :=
  ⟨x.val :: xs.val, fun e => cons_congr (x.property e) (xs.property e)⟩

mutual
  private def subst (σ : Nat → Pat) : Pat → Pat
    | .var i => σ i
    | .app op xs => .app op (substArgs σ xs)
  private def substArgs (σ : Nat → Pat) : List Pat → List Pat
    | [] => []
    | x :: xs => subst σ x :: substArgs σ xs
end

mutual
  private theorem eval_subst (p : Pat) (σ : Nat → Pat) (e : Nat → α) :
      (subst σ p).eval m e = p.eval m (fun i => (σ i).eval m e) := by
    cases p with
    | var i => rfl
    | app op xs => exact congrArg (m.interp op) (eval_substArgs xs σ e)
  private theorem eval_substArgs (xs : List Pat) (σ : Nat → Pat) (e : Nat → α) :
      Pat.evalArgs (substArgs σ xs) m e = Pat.evalArgs xs m (fun i => (σ i).eval m e) := by
    cases xs with
    | nil => rfl
    | cons x xs => exact cons_congr (eval_subst x σ e) (eval_substArgs xs σ e)
end

/-- An immutable node handle and its syntax. A graph operation can compare the
    syntax by pointer identity when both handles came from the same row. -/
private structure Ref where
  id : Nat
  term : Pat
  deriving Inhabited

private structure Located (m : Model α) (p : Pat) where
  ref : Ref
  valid : Same m ref.term p

private inductive Parent (m : Model α) where
  | root
  | link (source : Pat) (target : Located m source)

@[inline] private abbrev equalTerm (a b : Pat) := withPtrEqDecEq a b fun _ => decEq a b

private def locRefl (m : Model α) (r : Ref) : Located m r.term := ⟨r, fun _ => rfl⟩

private def locTrans (a : Located m p) (b : Located m a.ref.term) : Located m p :=
  ⟨b.ref, fun e => (b.valid e).trans (a.valid e)⟩

private inductive Shape
  | var (i : Nat)
  | app (op : Nat) (args : List Ref)
  deriving Inhabited

private def Shape.term : Shape → Pat
  | .var i => .var i
  | .app op args => .app op (args.map Ref.term)

-- Only IDs are hashed. Compare child syntax on a bucket hit to justify the
-- equality to the kernel. Shared child syntax takes the pointer fast path.
private def shapeHash : Shape → UInt64
  | .var i => i.toUInt64 * 1099511628211
  | .app op args =>
    args.foldl (fun h r => (h ^^^ (r.id + 1).toUInt64) * 1099511628211)
      ((op.toUInt64 + 14695981039346656037) * 1099511628211)

private def sameArgs : (xs ys : List Ref) → Option (PLift (xs.map Ref.term = ys.map Ref.term))
  | [], [] => some ⟨rfl⟩
  | a :: xs, b :: ys => do
    guard (a.id == b.id)
    match equalTerm a.term b.term with
    | isFalse _ => none
    | isTrue h =>
      let ⟨hs⟩ ← sameArgs xs ys
      return ⟨cons_congr h hs⟩
  | _, _ => none

private def sameShape : (a b : Shape) → Option (PLift (a.term = b.term))
  | .var i, .var j => if h : i = j then some ⟨h ▸ rfl⟩ else none
  | .app f xs, .app g ys => do
    if h : f = g then
      let ⟨args⟩ ← sameArgs xs ys
      return ⟨h ▸ congrArg (Pat.app f) args⟩
    else none
  | _, _ => none

private structure Row where
  shape : Shape
  ref : Ref
  valid : ref.term = shape.term

/-- The initial terms form a DAG before any union runs. Its immutable index
    stores only row numbers and stays valid as a base layer during saturation. -/
private abbrev SynRef (p : Pat) := { r : Ref // r.term = p }

private def seedProbe (rows : @& Array Row) (slots : @& Array Nat) (s : @& Shape)
    (mask : Nat) : Nat → Nat → Nat × Option (SynRef s.term)
  | 0, pos => (pos, none)
  | n+1, pos =>
    let i := slots[pos]!
    if i == 0 then (pos, none)
    else
      let next := (pos+1) &&& mask
      match rows[i-1]? with
      | none => (pos, none)
      | some row =>
        match sameShape row.shape s with
        | some ⟨h⟩ => (pos, some ⟨row.ref, row.valid.trans h⟩)
        | none => seedProbe rows slots s mask n next

@[inline] private def seedSlot (rows : @& Array Row) (slots : @& Array Nat) (s : @& Shape) (h : UInt64) :=
  let mask := slots.size - 1
  seedProbe rows slots s mask slots.size (h &&& mask.toUInt64).toNat

private structure Entry (m : Model α) where
  shape : Shape
  target : Located m shape.term
  hash : UInt64

private structure Table (m : Model α) where
  buckets : Array (List (Entry m)) := Array.replicate 16 []
  count : Nat := 0
  initial : Array Nat := #[]

private def bucket (hash : UInt64) (cap : Nat) := (hash &&& (cap.toUInt64 - 1)).toNat

private def overlay (table : Table m) (s : Shape) (h : UInt64) : Option (Located m s.term) :=
  (table.buckets.getD (bucket h table.buckets.size) []).findSome? fun e => do
    guard (e.hash == h)
    let ⟨p⟩ ← sameShape e.shape s
    return p ▸ e.target

private def lookup (rows : @& Array Row) (table : @& Table m) (s : @& Shape) (h : UInt64) :
    Option (Located m s.term) := Id.run do
  if table.count != 0 then
    if let some old := overlay table s h then return some old
  if table.initial.isEmpty then return none
  let (_, hit) := seedSlot rows table.initial s h
  let some r := hit | return none
  return some ⟨r.val, fun e => congrArg (fun t => t.eval m e) r.property⟩

private def put (table : Table m) (s : Shape) (h : UInt64) (target : Located m s.term) : Table m := Id.run do
  let mut bins := table.buckets
  if (table.count + 1) * 4 > bins.size * 3 then
    let cap := 2 * bins.size
    let mut bigger := Array.replicate cap []
    for bin in bins do
      for e in bin do bigger := bigger.modify (bucket e.hash cap) (e :: ·)
    bins := bigger
  let entry : Entry m := ⟨s, target, h⟩
  return ⟨bins.modify (bucket h bins.size) (entry :: ·), table.count + 1, table.initial⟩

private structure OpIndex where
  small : Array (Array Nat) := Array.replicate 64 #[]
  large : List (Nat × Array Nat) := []

private def OpIndex.get (ops : OpIndex) (op : Nat) : Array Nat :=
  if op < ops.small.size then ops.small.getD op #[]
  else (ops.large.lookup op).getD #[]

private def OpIndex.add (ops : OpIndex) (op id : Nat) : OpIndex :=
  if op < ops.small.size then { ops with small := ops.small.modify op (·.push id) }
  else { ops with large := (op, (ops.get op).push id) :: ops.large.eraseP (fun p => p.1 == op) }

private structure Cursor where
  rule : Rewrite
  premises : List (Pat × Pat)
  version : Nat

private structure RelRow (m : Model α) where
  name : String
  args : List Ref
  stamp : Nat
  sound : ∀ e, m.relation name (Pat.evalArgs (args.map Ref.term) m e)

private structure CellRow (m : Model α) where
  name : String
  keys : List Ref
  value : Ref
  stamp : Nat
  sound : ∀ e, m.CellHolds name (Pat.evalArgs (keys.map Ref.term) m e) (value.term.eval m e)

private structure CellTable (m : Model α) where
  entries : Std.HashMap (List Nat) (CellRow m) := {}
  order : Array (List Nat) := #[]

private structure RelCursor where
  name : Option String
  body : List RelAtom
  heads : List RelAtom
  version : Nat

private structure Graph (m : Model α) where
  rows : Array Row := #[]
  reps : Array (Parent m) := #[]
  classes : Array (Array Nat) := #[]
  uses : Array (Array Nat) := #[]
  byOp : OpIndex := {}
  table : Table m := {}
  dirty : Array Nat := #[]
  touched : Array Nat := #[]
  cursors : List Cursor := []
  relations : Std.HashMap String (Std.HashMap (List Nat) (RelRow m)) := {}
  congruentRows : Bool := false
  relationsDirty : Bool := false
  functions : Std.HashMap String (CellTable m) := {}
  functionsDirty : Bool := false
  relCursors : List RelCursor := []
  version : Nat := 0

private def root (g : @& Graph m) (r : Ref) : Located m r.term :=
  match g.reps.getD r.id .root with
  | .root => locRefl m r
  | .link source target =>
    match equalTerm source r.term with
    | isTrue h => h ▸ target
    | isFalse _ => locRefl m r

private def classId (g : @& Graph m) (r : @& Ref) : Nat :=
  match g.reps.getD r.id .root with
  | .root => r.id
  | .link _ target => target.ref.id

private def rebuildArgs (g : Graph m) : (rs : List Ref) →
    { ts : List Ref // ∀ e, Pat.evalArgs (ts.map Ref.term) m e = Pat.evalArgs (rs.map Ref.term) m e }
  | [] => ⟨[], fun _ => rfl⟩
  | r :: rs =>
    let x := root g r
    let xs := rebuildArgs g rs
    ⟨x.ref :: xs.val, fun e => cons_congr (x.valid e) (xs.property e)⟩

/-- Most operands already refer to their canonical class. Reuse the list until
    a union changes a representative. -/
private def canonicalArgs (g : Graph m) (rs : List Ref) :
    { ts : List Ref // ∀ e, Pat.evalArgs (ts.map Ref.term) m e = Pat.evalArgs (rs.map Ref.term) m e } :=
  if rs.any (fun r => classId g r != r.id) then rebuildArgs g rs else ⟨rs, fun _ => rfl⟩

private def canonical (g : Graph m) : (s : Shape) → { t : Shape // Same m t.term s.term }
  | .var i => ⟨.var i, fun _ => rfl⟩
  | .app op rs =>
    let xs := canonicalArgs g rs
    ⟨.app op xs.val, fun e => congrArg (m.interp op) (xs.property e)⟩

private def add (g : Graph m) (s : Shape) : Graph m × Located m s.term := Id.run do
  let key := canonical g s
  let h := shapeHash key.val
  if let some old := lookup g.rows g.table key.val h then
    let y := locTrans old (root g old.ref)
    return (g, ⟨y.ref, fun e => (y.valid e).trans (key.property e)⟩)
  let id := g.rows.size
  let ref : Ref := ⟨id, key.val.term⟩
  let target := locRefl m ref
  let mut uses := g.uses.push #[]
  let mut ops := g.byOp
  if let .app op children := key.val then
    ops := ops.add op id
    for r in children do uses := uses.modify r.id (·.push id)
  let g := { g with rows := g.rows.push ⟨key.val, ref, rfl⟩, reps := g.reps.push .root,
                    classes := g.classes.push #[], uses, byOp := ops,
                    touched := g.touched.push (g.version + 1),
                    table := put g.table key.val h target, version := g.version + 1 }
  return (g, ⟨ref, key.property⟩)

@[inline] private def memberIds (g : @& Graph m) (id : Nat) : Array Nat :=
  let ids := g.classes.getD id #[]
  if ids.isEmpty then #[id] else ids

private def mergeInto (g : Graph m) (big small : Ref) (eqv : Same m big.term small.term) : Graph m := Id.run do
  let moved := g.classes.getD small.id #[]
  let mut reps := g.reps
  let stamp := g.version + 1
  let mut touched := g.touched
  if moved.isEmpty then
    -- Singleton classes have no member vector or previous parent edge.
    reps := reps.set! small.id (.link small.term ⟨big, eqv⟩)
    touched := touched.set! small.id stamp
  else
    for id in moved do
      touched := touched.set! id stamp
      match reps.getD id .root with
      | .root => reps := reps.set! id (.link small.term ⟨big, eqv⟩)
      | .link source target =>
        match equalTerm target.ref.term small.term with
        | isFalse _ => pure ()
        | isTrue h =>
          let toBig : Located m source :=
            ⟨big, fun e => (eqv e).trans ((congrArg (fun t : Pat => t.eval m e) h.symm).trans (target.valid e))⟩
          reps := reps.set! id (.link source toBig)
  let parents := g.uses.getD small.id #[]
  touched := parents.foldl (fun ts id => ts.set! id stamp) touched
  -- Take the destination vector out of its slot before extending it. This lets
  -- repeated merges into one large class reuse its backing storage.
  let classes := (g.classes.set! small.id #[]).modify big.id fun members =>
    let members := if members.isEmpty then #[big.id] else members
    if moved.isEmpty then members.push small.id else members ++ moved
  return { g with reps, touched, classes, relationsDirty := !g.relations.isEmpty,
                  functionsDirty := !g.functions.isEmpty,
                  uses := (g.uses.set! small.id #[]).modify big.id (· ++ parents),
                  dirty := g.dirty ++ parents, version := g.version + 1 }

private def merge (g : Graph m) (a b : Ref) (eqv : Same m a.term b.term) : Graph m :=
  let x := root g a
  let y := root g b
  if x.ref.id == y.ref.id then g
  else
    have h : Same m x.ref.term y.ref.term := fun e => (x.valid e).trans ((eqv e).trans (y.valid e).symm)
    let weight := fun id => (g.classes.getD id #[]).size + (g.uses.getD id #[]).size
    if weight x.ref.id ≤ weight y.ref.id then
      mergeInto g y.ref x.ref (fun e => (h e).symm)
    else mergeInto g x.ref y.ref h

private def repair (g : Graph m) (id : Nat) : Graph m := Id.run do
  let some row := g.rows[id]? | return g
  let key := canonical g row.shape
  let h := shapeHash key.val
  have eqv : Same m row.ref.term key.val.term :=
    fun e => (congrArg (fun t : Pat => t.eval m e) row.valid).trans (key.property e).symm
  if let some old := lookup g.rows g.table key.val h then
    let g := if old.ref.id == row.ref.id then g else { g with congruentRows := true }
    return merge g row.ref old.ref (fun e => (eqv e).trans (old.valid e).symm)
  return { g with table := put g.table key.val h ⟨row.ref, eqv⟩ }

/-- Each additional dirty wave requires a successful class merge. -/
private def rebuildCongruences : Nat → Graph m → Graph m
  | 0, g => g
  | fuel + 1, g =>
    if g.dirty.isEmpty then g
    else
      let pending := g.dirty
      rebuildCongruences fuel (pending.foldl repair { g with dirty := #[] })

private abbrev Bindings := List (Nat × Ref)

private def assignment (σ : Bindings) (i : Nat) : Pat :=
  match σ.lookup i with
  | some r => r.term
  | none => .var i

mutual
  private def realize (g : Graph m) (σ : Bindings) (p : Pat) : Graph m × Located m (subst (assignment σ) p) :=
    match p with
    | .var i =>
      match h : σ.lookup i with
      | some r => (g, ⟨r, by simp [subst, assignment, h, Same]⟩)
      | none =>
        let (g, x) := add g (.var i)
        (g, ⟨x.ref, by simpa [assignment, h, subst, Shape.term] using x.valid⟩)
    | .app op ps =>
      let (g, ⟨rs, equiv⟩) := realizeArgs g σ ps
      let (g, x) := add g (.app op rs)
      (g, ⟨x.ref, fun e => (x.valid e).trans (congrArg (m.interp op) (equiv e))⟩)
  private def realizeArgs (g : Graph m) (σ : Bindings) (ps : List Pat) : Graph m ×
      { rs : List Ref // ∀ e, Pat.evalArgs (rs.map Ref.term) m e = Pat.evalArgs (substArgs (assignment σ) ps) m e } :=
    match ps with
    | [] => (g, ⟨[], fun _ => rfl⟩)
    | p :: ps =>
      let (g, x) := realize g σ p
      let (g, xs) := realizeArgs g σ ps
      (g, ⟨x.ref :: xs.val, fun e => cons_congr (x.valid e) (xs.property e)⟩)
end

private structure Seeds where
  rows : Array Row
  slots : Array Nat

private def addSeed (st : Seeds) (s : Shape) (p : Pat) (exact : s.term = p) :
    Seeds × SynRef p :=
  let (pos, hit) := seedSlot st.rows st.slots s (shapeHash s)
  match hit with
  | some r => (st, ⟨r.val, r.property.trans exact⟩)
  | none =>
    let ref : Ref := ⟨st.rows.size, p⟩
    (⟨st.rows.push ⟨s, ref, exact.symm⟩, st.slots.set! pos (ref.id + 1)⟩, ⟨ref, rfl⟩)

mutual
  private def seed (st : Seeds) (p : Pat) : Seeds × SynRef p :=
    match p with
    | .var i => addSeed st (.var i) (.var i) rfl
    | .app op ps =>
      let (st, kids) := seedArgs st ps
      addSeed st (.app op kids.val) (.app op ps) (congrArg (Pat.app op) kids.property)
  private def seedArgs (st : Seeds) (ps : List Pat) : Seeds × { rs : List Ref // rs.map Ref.term = ps } :=
    match ps with
    | [] => (st, ⟨[], rfl⟩)
    | p :: ps =>
      let (st, x) := seed st p
      let (st, xs) := seedArgs st ps
      (st, ⟨x.val :: xs.val, cons_congr x.property xs.property⟩)
end

mutual
  private def occurrences : Pat → Nat
    | .var _ => 1
    | .app _ ps => 1 + occurrencesArgs ps
  private def occurrencesArgs : List Pat → Nat
    | [] => 0
    | p :: ps => occurrences p + occurrencesArgs ps
end

private def emptySeeds (n : Nat) : Seeds :=
  ⟨Array.emptyWithCapacity n, Array.replicate (Nat.nextPowerOfTwo (2 * n + 1)) 0⟩

private def finishSeeds (st : Seeds) : Graph m := Id.run do
  let n := st.rows.size
  let mut uses := Array.replicate n #[]
  let mut byOp : OpIndex := {}
  for row in st.rows do
    if let .app op args := row.shape then
      byOp := byOp.add op row.ref.id
      for child in args do uses := uses.modify child.id (·.push row.ref.id)
  return { rows := st.rows, byOp, uses, classes := Array.replicate n #[], reps := Array.replicate n .root,
           table := { initial := st.slots }, version := n, touched := Array.replicate n n }

private abbrev Fits (m : Model α) (σ : Bindings) (e η : Nat → α) :=
  ∀ i r, σ.lookup i = some r → η i = r.term.eval m e

private theorem fitsAssignment (m : Model α) (σ : Bindings) (e : Nat → α) :
    Fits m σ e (fun i => (assignment σ i).eval m e) := by
  intro i r h
  simp [assignment, h]

private structure Hit (m : Model α) (base : Bindings) (goal : (Nat → α) → (Nat → α) → Prop) where
  bindings : Bindings
  fresh : Bool
  extendsEnv : ∀ e η, Fits m bindings e η → Fits m base e η
  valid : ∀ e η, Fits m bindings e η → goal e η

private abbrev PatternHit (m : Model α) (base : Bindings) (p : Pat) (r : Ref) :=
  Hit m base (fun e η => r.term.eval m e = p.eval m η)

private abbrev ArgsHit (m : Model α) (base : Bindings) (ps : List Pat) (rs : List Ref) :=
  Hit m base (fun e η => Pat.evalArgs (rs.map Ref.term) m e = Pat.evalArgs ps m η)

private theorem fitsTail (h : σ.lookup i = none) (fit : Fits m ((i, ref) :: σ) e η) : Fits m σ e η := by
  intro j t hj
  match decEq j i with
  | isTrue eq => exact False.elim (by subst j; rw [h] at hj; contradiction)
  | isFalse ne => exact fit j t (by rw [List.lookup_cons, beq_false_of_ne ne]; exact hj)

private theorem lookupHead (i : Nat) (r : Ref) (σ : Bindings) : ((i, r) :: σ).lookup i = some r := by
  rw [List.lookup_cons]
  change (match decide (i = i) with | true => some r | false => σ.lookup i) = some r
  rw [decide_eq_true rfl]

private def matchVar (key : Located m r.term) (i : Nat) (σ : Bindings) (fresh : Bool) :
    Option (PatternHit m σ (.var i) r) :=
  match h : σ.lookup i with
  | some old =>
    if old.id != key.ref.id then none else
    match equalTerm old.term key.ref.term with
    | isFalse _ => none
    | isTrue same => some ⟨σ, fresh, fun _ _ fit => fit, fun e _ fit =>
        (key.valid e).symm.trans ((congrArg (fun p => p.eval m e) same.symm).trans (fit i old h).symm)⟩
  | none =>
    some ⟨(i, key.ref) :: σ, fresh, fun _ _ fit => fitsTail h fit,
      fun e _ fit => (key.valid e).symm.trans (fit i key.ref (lookupHead i key.ref σ)).symm⟩

private def wrapApp (hit : ArgsHit m σ ps rs) (r : Ref)
    (same : Same m r.term (.app op (rs.map Ref.term))) : PatternHit m σ (.app op ps) r :=
  ⟨hit.bindings, hit.fresh, hit.extendsEnv,
   fun e η fit => (same e).trans (congrArg (m.interp op) (hit.valid e η fit))⟩

/-- A constructor with only variable children yields at most one substitution.
    Match it directly, including repeated variables and the projection skip. -/
private def matchVarArgs (g : Graph m) (fresh : Bool) (skip : Option (Nat × Nat))
    (ps : List Pat) (rs : List Ref) (σ : Bindings) : Option (ArgsHit m σ ps rs) :=
  match ps, rs with
  | [], [] => some ⟨σ, fresh, fun _ _ fit => fit, fun _ _ _ => rfl⟩
  | .var i :: ps, r :: rs => do
    let key := root g r
    guard (!skip.any (fun (v, id) => i == v && key.ref.id == id))
    let x ← matchVar key i σ fresh
    let xs ← matchVarArgs g fresh skip ps rs x.bindings
    return ⟨xs.bindings, fresh, fun e η fit => x.extendsEnv e η (xs.extendsEnv e η fit),
      fun e η fit => cons_congr (x.valid e η (xs.extendsEnv e η fit)) (xs.valid e η fit)⟩
  | _, _ => none

mutual
  private def matchClass (g : Graph m) (cutoff : Nat) (p : Pat) (r : Ref) (σ : Bindings) (fresh : Bool) (skip : Option (Nat × Nat) := none) (needDelta := false) :
      Array (PatternHit m σ p r) :=
    let key := root g r
    match p with
    | .var i =>
      if skip.any (fun (v, id) => i == v && key.ref.id == id) then #[]
      else (matchVar key i σ fresh).toArray
    | .app op ps =>
      (memberIds g key.ref.id).flatMap fun id => Id.run do
        let some row := (g.rows[id]? : Option Row) | return #[]
        match shape : row.shape with
        | .var _ => return #[]
        | .app f args =>
          if label : f = op then
            let changed := fresh || g.touched.getD id 0 > cutoff
            let onlyVars := ps.all (fun | .var _ => true | _ => false)
            if needDelta && !changed && onlyVars then return #[]
            let parent := root g row.ref
            match equalTerm parent.ref.term key.ref.term with
            | isFalse _ => return #[]
            | isTrue common =>
              have equiv : Same m r.term (.app op (args.map Ref.term)) := fun e =>
                (key.valid e).symm.trans ((congrArg (fun t : Pat => t.eval m e) common.symm).trans
                  ((parent.valid e).trans (congrArg (fun t : Pat => t.eval m e) (by simpa [shape, Shape.term, label] using row.valid))))
              if onlyVars then
                let some hit := matchVarArgs g changed skip ps args σ | return #[]
                return #[wrapApp hit r equiv]
              let hits := matchArgs g cutoff ps args σ changed skip needDelta
              return hits.map (fun hit => wrapApp hit r equiv)
          else return #[]
  private def matchArgs (g : Graph m) (cutoff : Nat) (ps : List Pat) (args : List Ref) (σ : Bindings) (fresh : Bool) (skip : Option (Nat × Nat) := none) (needDelta := false) :
      Array (ArgsHit m σ ps args) :=
    match ps, args with
    | [], [] => #[⟨σ, fresh, fun _ _ fit => fit, fun _ _ _ => rfl⟩]
    | p :: ps, a :: args =>
      (matchClass g cutoff p a σ fresh skip (needDelta && ps.isEmpty)).flatMap fun x =>
        (matchArgs g cutoff ps args x.bindings x.fresh skip needDelta).map fun xs =>
          ⟨xs.bindings, xs.fresh, fun e η fit => x.extendsEnv e η (xs.extendsEnv e η fit),
           fun e η fit => cons_congr (x.valid e η (xs.extendsEnv e η fit)) (xs.valid e η fit)⟩
    | _, _ => #[]
end

private abbrev Matched (m : Model α) (lhs : Pat) (σ : Bindings) (ref : Ref) :=
  ∀ e, ref.term.eval m e = lhs.eval m (fun i => (assignment σ i).eval m e)

private abbrev OpenMatch (m : Model α) (p : Pat) (σ : Bindings) (r : Ref) :=
  ∀ e η, Fits m σ e η → r.term.eval m e = p.eval m η

private theorem closeMatch (ok : OpenMatch m p σ r) : Matched m p σ r :=
  fun e => ok e _ (fitsAssignment m σ e)

private def emitRhs (g : Graph m) (p : Pat) (σ : Bindings) (lhs : Ref)
    (valid : Matched m p σ lhs) : Graph m :=
  let (g, r) := realize g σ p
  merge g lhs r.ref (fun e =>
    (valid e).trans ((eval_subst p (assignment σ) e).symm.trans (r.valid e).symm))

/- Probe a fully bound term in the snapshot. Its witness holds at every
   valuation compatible with the bindings, including later extensions. -/
mutual
  private def probeOpenPat (g : Graph m) (σ : Bindings) (p : Pat) :
      Option { r : Ref // OpenMatch m p σ r } :=
    match p with
    | .var i =>
      match h : σ.lookup i with
      | none => none
      | some r => some ⟨r, fun _ _ fit => (fit i r h).symm⟩
    | .app op ps => do
      let ⟨rs, equiv⟩ ← probeOpenArgs g σ ps
      let key := canonical g (.app op rs)
      let found ← lookup g.rows g.table key.val (shapeHash key.val)
      return ⟨found.ref, fun e η fit => (found.valid e).trans
        ((key.property e).trans (congrArg (m.interp op) (equiv e η fit)))⟩
  private def probeOpenArgs (g : Graph m) (σ : Bindings) (ps : List Pat) :
      Option { rs : List Ref // ∀ e η, Fits m σ e η →
        Pat.evalArgs (rs.map Ref.term) m e = Pat.evalArgs ps m η } :=
    match ps with
    | [] => some ⟨[], fun _ _ _ => rfl⟩
    | p :: ps => do
      let x ← probeOpenPat g σ p
      let xs ← probeOpenArgs g σ ps
      return ⟨x.val :: xs.val, fun e η fit => cons_congr (x.property e η fit) (xs.property e η fit)⟩
end

@[inline] private def probePat (g : Graph m) (σ : Bindings) (p : Pat) :
    Option (Located m (subst (assignment σ) p)) := do
  let ⟨r, ok⟩ ← probeOpenPat g σ p
  return ⟨r, fun e => (closeMatch ok e).trans (eval_subst p (assignment σ) e).symm⟩

@[inline] private def probeArgs (g : Graph m) (σ : Bindings) (ps : List Pat) :
    Option { rs : List Ref // ∀ e, Pat.evalArgs (rs.map Ref.term) m e =
      Pat.evalArgs (substArgs (assignment σ) ps) m e } := do
  let ⟨rs, ok⟩ ← probeOpenArgs g σ ps
  return ⟨rs, fun e => (ok e _ (fitsAssignment m σ e)).trans (eval_substArgs ps (assignment σ) e).symm⟩

/-- Resolve a closed rewrite destination once per rule if it is already present.
    An absent term is still created only after a successful match. -/
private def prepareRhs (initial : Graph m) (rhs : Pat) :
    Graph m → (σ : Bindings) → (lhs : Ref) → Matched m rhs σ lhs → Graph m := Id.run do
  if closed : rhs.variables = [] then
    if let some target := probePat initial [] rhs then
      return fun graph _ lhs valid => merge graph lhs target.ref (fun e =>
        (valid e).trans ((Pat.eval_closed rhs closed m _ _).trans
          ((eval_subst rhs (assignment []) e).symm.trans (target.valid e).symm)))
  return fun graph => emitRhs graph rhs

private def equalAt? (g : Graph m) (σ : Bindings) (a b : Pat) :
    Option (PLift (∀ e, a.eval m (fun i => (assignment σ i).eval m e) =
                        b.eval m (fun i => (assignment σ i).eval m e))) := do
  let l ← probePat g σ a
  let r ← probePat g σ b
  let x := locTrans l (root g l.ref)
  let y := locTrans r (root g r.ref)
  match equalTerm x.ref.term y.ref.term with
  | isFalse _ => none
  | isTrue h =>
    return ⟨fun e => (eval_subst a (assignment σ) e).symm.trans
      ((x.valid e).symm.trans ((congrArg (fun t : Pat => t.eval m e) h).trans
       ((y.valid e).trans (eval_subst b (assignment σ) e))))⟩

private def checkPremises (g : Graph m) (σ : Bindings) (ps : List (Pat × Pat)) :
    Option (PLift (∀ e, ∀ p ∈ ps, p.1.eval m (fun i => (assignment σ i).eval m e) =
                                  p.2.eval m (fun i => (assignment σ i).eval m e))) :=
  match ps with
  | [] => some ⟨by simp⟩
  | (a, b) :: ps => do
    let head ← equalAt? g σ a b
    let tail ← checkPremises g σ ps
    return ⟨by
      intro e pair mem
      rcases List.mem_cons.mp mem with h | h
      · cases h
        exact head.down e
      · exact tail.down e pair h⟩

private def applyGuarded (initial g : Graph m) (rule : GuardedRule m) (σ : Bindings) (lhs : Ref)
    (valid : Matched m rule.lhs σ lhs) : Graph m := Id.run do
  let some ok := checkPremises initial σ rule.premises | return g
  return emitRhs g rule.rhs σ lhs (fun e => (valid e).trans (rule.sound _ (ok.down e)))

/-- Most compiler rules have only variables and nullary constructors below
    the root. Compile those tests once per snapshot. Every argument list then
    produces at most one substitution, with no intermediate match arrays. -/
private inductive Leaf (m : Model α) : Pat → Type
  | var (i : Nat) : Leaf m (.var i)
  | fixed (op : Nat) (rep : Located m (.app op [])) : Leaf m (.app op [])

private inductive Flat (m : Model α) : List Pat → Type
  | nil : Flat m []
  | cons (x : Leaf m p) (xs : Flat m ps) : Flat m (p :: ps)

private def makeLeaf (g : Graph m) (cutoff : Nat) : (p : Pat) → Option (Leaf m p × Bool)
  | .var i => some (.var i, false)
  | .app op [] => (g.byOp.get op).findSome? fun id => do
    let row ← g.rows[id]?
    match shape : row.shape with
    | .app f [] =>
      if h : f = op then
        let rep := root g row.ref
        have valid : Same m rep.ref.term (.app op []) := fun e =>
          (rep.valid e).trans (congrArg (fun t : Pat => t.eval m e) (by simpa [shape, Shape.term, h] using row.valid))
        return (.fixed op ⟨rep.ref, valid⟩, g.touched.getD id 0 > cutoff)
      else none
    | _ => none
  | _ => none

private def makeFlat (g : Graph m) (cutoff : Nat) : (ps : List Pat) → Option (Flat m ps × Bool)
  | [] => some (.nil, false)
  | p :: ps => do
    let (x, xc) ← makeLeaf g cutoff p
    let (xs, xsc) ← makeFlat g cutoff ps
    return (.cons x xs, xc || xsc)

private def matchLeaf (g : Graph m) (leaf : Leaf m p) (r : Ref) (σ : Bindings) : Option (PatternHit m σ p r) :=
  let key := root g r
  match leaf with
  | .var i => matchVar key i σ true
  | .fixed _ rep => do
    guard (key.ref.id == rep.ref.id)
    match equalTerm key.ref.term rep.ref.term with
    | isFalse _ => none
    | isTrue same =>
      return ⟨σ, true, fun _ _ fit => fit, fun e _ _ =>
        (key.valid e).symm.trans ((congrArg (fun t : Pat => t.eval m e) same).trans (rep.valid e))⟩

private def matchFlat (g : Graph m) (plan : Flat m ps) (rs : List Ref) (σ : Bindings) :
    Option (ArgsHit m σ ps rs) :=
  match plan, rs with
  | .nil, [] => some ⟨σ, true, fun _ _ fit => fit, fun _ _ _ => rfl⟩
  | .cons x xs, r :: rs => do
    let a ← matchLeaf g x r σ
    let rest ← matchFlat g xs rs a.bindings
    return ⟨rest.bindings, true, fun e η fit => a.extendsEnv e η (rest.extendsEnv e η fit),
      fun e η fit => cons_congr (a.valid e η (rest.extendsEnv e η fit)) (rest.valid e η fit)⟩
  | _, _ => none

@[specialize] private def scanLhs (initial : Graph m) (cutoff : Nat) (lhs : Pat) (g : β)
    (apply : β → (σ : Bindings) → (ref : Ref) → OpenMatch m lhs σ ref → β) (rhsVar : Option Nat := none) : β := Id.run do
  if cutoff >= initial.version then return g
  let mut g := g
  match expr : lhs with
  | .app op ps =>
    let plan := makeFlat initial cutoff ps
    let projection := rhsVar.bind fun i => ps.idxOf? (.var i)
    let needRow := plan.all (fun (_, constantChanged) => constantChanged)
    for id in initial.byOp.get op do
      let fresh := initial.touched.getD id 0 > cutoff
      if !fresh && !needRow then continue
      let some row := (initial.rows[id]? : Option Row) | continue
      match shape : row.shape with
      | .var _ => pure ()
      | .app f args =>
        if label : f = op then
          if let some k := projection then
            if let some child := args[k]? then
              if classId initial child == classId initial row.ref then continue
          have same : Same m row.ref.term (.app op (args.map Ref.term)) := fun e =>
            congrArg (fun t : Pat => t.eval m e) (by simpa [shape, Shape.term, label] using row.valid)
          let fire := fun (g : β) (hit : ArgsHit m [] ps args) =>
            apply g hit.bindings row.ref (by
              rw [expr]
              intro e η fit
              exact (same e).trans (congrArg (m.interp op) (hit.valid e η fit)))
          match plan with
          | some (plan, constantChanged) =>
            if fresh || constantChanged then
              if let some hit := matchFlat initial plan args [] then g := fire g hit
          | none =>
            let skip := rhsVar.map (fun i => (i, classId initial row.ref))
            for hit in matchArgs initial cutoff ps args [] fresh skip true do
              if hit.fresh then g := fire g hit
  | .var i =>
    for row in initial.rows do
      if initial.touched.getD row.ref.id 0 > cutoff && (root initial row.ref).ref.id == row.ref.id then
        g := apply g [(i, row.ref)] row.ref (by
          rw [expr]
          intro e η fit
          exact (fit i row.ref (lookupHead i row.ref [])).symm)
  return g

private def sameCursor (r : Rewrite) (ps : List (Pat × Pat)) (c : Cursor) : Bool :=
  r == c.rule && ps == c.premises

private def cursor (g : Graph m) (r : Rewrite) (ps : List (Pat × Pat)) : Nat :=
  ((g.cursors.find? (sameCursor r ps)).map Cursor.version).getD 0

private def advance (g : Graph m) (r : Rewrite) (ps : List (Pat × Pat)) (v : Nat) : Graph m :=
  { g with cursors := ⟨r, ps, v⟩ :: g.cursors.eraseP (sameCursor r ps) }

/-- Canonical tuple keys are shared by indexed lookup, deduplication and join
    matching. On a merge, moved arguments give their row a new delta stamp. -/
private def normalizeRow (g : Graph m) (row : RelRow m) (changed : Bool) : RelRow m :=
  let canonical := canonicalArgs g row.args
  let stamp := if changed then row.args.foldl (fun n r => max n (g.touched.getD r.id 0)) row.stamp
    else row.stamp
  ⟨row.name, canonical.val, stamp, fun env => (canonical.property env).symm ▸ row.sound env⟩

private def relKey (args : List Ref) : List Nat := args.map Ref.id

private def indexRelations (g : Graph m) : Graph m := Id.run do
  if !g.relationsDirty then return g
  let mut tables := g.relations
  for (name, table) in g.relations do
    -- Unions can move a few keys in a much larger table. The existing keys
    -- still have their proof and timestamp if their representatives survived.
    let mut moved := #[]
    for (key, row) in table do
      if row.args.any (fun r => classId g r != r.id) then moved := moved.push (key, row)
    unless moved.isEmpty do
      let mut index := table
      for (old, row) in moved do
        index := index.erase old
        let row := normalizeRow g row true
        let key := relKey row.args
        -- A collision after union has both proofs; retain the most recent stamp.
        if (index[key]?.map RelRow.stamp).getD 0 ≤ row.stamp then index := index.insert key row
      tables := tables.insert name index
  return { g with relations := tables, relationsDirty := false }

/-- The join planner reads the same canonical table used for witness lookup. -/
private def relationRows (g : Graph m) (name : String) (arity : Nat) : Array (Nat × Array Nat) := Id.run do
  let some table := g.relations[name]? | return #[]
  let mut rows := #[]
  for (key, row) in table do
    if key.length == arity then rows := rows.push (row.stamp, key.toArray)
  return rows

private def storeRow (g : Graph m) (row : RelRow m) : Graph m :=
  let g := indexRelations g
  let row := normalizeRow g row false
  let key := relKey row.args
  if (g.relations[row.name]?).any (·.contains key) then g
  else
    let stamp := g.version + 1
    -- Taking the inner map out of its bucket first gives its next update
    -- exclusive ownership. A lookup followed by nested inserts copies it.
    let relations := g.relations.alter row.name fun table =>
      some ((table.getD {}).insert key {row with stamp})
    { g with version := stamp, relations }

/-- Even a faulty join index cannot establish a predicate: compare the name
    and witnessed arguments of the stored tuple before using its proof. -/
private def rowAt? (g : Graph m) (name : String) (rs : List Ref) :
    Option (PLift (∀ e, m.relation name (Pat.evalArgs (rs.map Ref.term) m e))) := do
  let query := canonicalArgs g rs
  let table ← g.relations[name]?
  let row ← table[relKey query.val]?
  if h : row.name = name then
    let ⟨same⟩ ← sameArgs row.args query.val
    return ⟨fun env => by
      rw [← query.property env, ← same, ← h]
      exact row.sound env⟩
  else none

private def normalizeCell (g : Graph m) (row : CellRow m) (changed : Bool) : CellRow m :=
  let keys := canonicalArgs g row.keys
  let value := root g row.value
  let stamp := if changed then row.keys.foldl (fun n r => max n (g.touched.getD r.id 0))
    (max row.stamp (g.touched.getD row.value.id 0)) else row.stamp
  ⟨row.name, keys.val, value.ref, stamp, fun e => by
    rw [keys.property e, value.valid e]
    exact row.sound e⟩

private def cellEvidence? (row : CellRow m) (name : String) (keys : List Ref) :
    Option (PLift (∀ e, m.CellHolds name (Pat.evalArgs (keys.map Ref.term) m e) (row.value.term.eval m e))) := do
  if h : row.name = name then
    let ⟨same⟩ ← sameArgs row.keys keys
    return ⟨fun e => by rw [← same, ← h]; exact row.sound e⟩
  else none

private def operand? (a b : Ref) (p : Pat) : Option (SynRef p) :=
  match equalTerm a.term p with
  | isTrue h => some ⟨a, h⟩
  | isFalse _ =>
    match equalTerm b.term p with
    | isTrue h => some ⟨b, h⟩
    | isFalse _ => none

private def operands? (a b : Ref) (ps : List Pat) : Option {rs : List Ref // rs.map Ref.term = ps} :=
  match ps with
  | [] => some ⟨[], rfl⟩
  | p :: ps => do
    let ⟨r, h⟩ ← operand? a b p
    let ⟨rs, hs⟩ ← operands? a b ps
    return ⟨r :: rs, cons_congr h hs⟩

/-- Common merges select an existing operand or construct one node from the
    operands. Their stored references avoid re-interning whole operand trees.
    A user-defined merge can also compute arbitrary fresh syntax. -/
private def realizeMerge (g : Graph m) (a b : Ref) (term : Pat) : Graph m × Located m term := Id.run do
  if let some ⟨r, h⟩ := operand? a b term then
    return (g, ⟨r, fun e => congrArg (fun p : Pat => p.eval m e) h⟩)
  match node : term with
  | .var _ => pure ()
  | .app op ps =>
    if let some ⟨rs, h⟩ := operands? a b ps then
      let (next, result) := add g (.app op rs)
      return (next, ⟨result.ref, fun e => by
        rw [node]
        exact (result.valid e).trans (congrArg (fun xs : List Pat => (Pat.app op xs).eval m e) h)⟩)
  let (g, r) := realize g [] term
  return (g, ⟨r.ref, fun e => (r.valid e).trans (eval_subst term (assignment []) e)⟩)

/-- Realize the user-defined merge as an ordinary term. The proof of the
    invariant follows from the two stored proofs and the declared merge law. -/
private def mergeCell (g : Graph m) (old new : CellRow m) : Graph m × CellRow m := Id.run do
  if old.value.id == new.value.id then return (g, old)
  let some ⟨prior⟩ := cellEvidence? old new.name new.keys | return (g, old)
  match h : m.functions new.name with
  | none => return (g, old)
  | some spec =>
    let term := spec.merge old.value.term new.value.term
    let (g, merged) := realizeMerge g old.value new.value term
    let value := locTrans merged (root g merged.ref)
    return (g, ⟨new.name, new.keys, value.ref, max old.stamp new.stamp, fun e => by
      have left := prior e
      have right := new.sound e
      simp only [Model.CellHolds, h] at left right ⊢
      refine ⟨right.1, ?_⟩
      rw [value.valid e]
      change spec.invariant _ (term.eval m e)
      rw [← Pat.denote_eq_eval]
      apply spec.merge_sound
      · simpa only [Pat.denote_eq_eval] using left.2
      · simpa only [Pat.denote_eq_eval] using right.2⟩)

private def writeCell (g : Graph m) (new : CellRow m) (asserting : Bool) : Graph m := Id.run do
  let key := relKey new.keys
  let old := g.functions[new.name]?.bind (fun t => t.entries[key]?)
  let mut (g, row) := (g, new)
  if let some old := old then
    let (updated, merged) := mergeCell g old new
    g := updated
    if old.value.id == merged.value.id && asserting then return g
    row := {merged with stamp := max merged.stamp new.stamp}
  let version := if asserting then g.version + 1 else g.version
  if asserting then row := {row with stamp := version}
  let fresh := old.isNone
  let functions := g.functions.alter new.name fun current =>
    let table := current.getD {}
    let order := if fresh then table.order.push key else table.order
    some ⟨table.entries.insert key row, order⟩
  return {g with version, functions}

/-- Re-key on e-class unions. Preserve resident rows as the old side of a
    merge, then replay moved keys in insertion order for noncommutative merges. -/
private def indexFunctions (g : Graph m) : Graph m := Id.run do
  if !g.functionsDirty then return g
  let mut out := {g with functionsDirty := false}
  for (name, table) in g.functions do
    if !table.entries.any (fun _ row => classId g row.value != row.value.id ||
        row.keys.any (fun r => classId g r != r.id)) then continue
    out := {out with functions := out.functions.erase name}
    let mut moved := #[]
    for key in table.order do
      let some old := table.entries[key]? | continue
      let row := normalizeCell g old true
      if key == relKey row.keys then out := writeCell out row false else moved := moved.push row
    for row in moved do out := writeCell out row false
  return out

private def rebuild (fuel : Nat) (g : Graph m) : Graph m :=
  indexFunctions (rebuildCongruences fuel g)

private def functionRows (g : Graph m) (name : String) (arity : Nat) : Array (Nat × Array Nat) := Id.run do
  let some table := g.functions[name]? | return #[]
  let mut rows := #[]
  for (key, row) in table.entries do
    if key.length == arity then rows := rows.push (row.stamp, key.toArray.push row.value.id)
  return rows

private def cellAt? (g : Graph m) (name : String) (keys : List Ref) (value : Ref) :
    Option (PLift (∀ e, m.CellHolds name (Pat.evalArgs (keys.map Ref.term) m e) (value.term.eval m e))) := do
  let canonical := canonicalArgs g keys
  let table ← g.functions[name]?
  let row ← table.entries[relKey canonical.val]?
  let ⟨existsCell⟩ ← cellEvidence? row name canonical.val
  let expected := root g value
  match equalTerm row.value.term expected.ref.term with
  | isFalse _ => none
  | isTrue h =>
    return ⟨fun e => by
      rw [← canonical.property e, ← expected.valid e, ← h]
      exact existsCell e⟩

/-- Full, old and new rows for one atom in a rule's query. -/
private structure Slice where
  all : GenericJoin.Input
  old : GenericJoin.Input
  new : GenericJoin.Input
  deriving Inhabited

private def slice (cols : Array Nat) (rows : Array (Nat × Array Nat)) (cutoff : Nat) : Slice :=
  let full := GenericJoin.Input.prepare cols (rows.map Prod.snd)
  if cutoff == 0 then ⟨full, ⟨cols, #[]⟩, full⟩
  else
    let (old, recent) := rows.partition (fun row => row.1 ≤ cutoff)
    ⟨full, GenericJoin.Input.prepare cols (old.map Prod.snd),
           GenericJoin.Input.prepare cols (recent.map Prod.snd)⟩

private inductive QuerySource where
  | constructor (op : Nat)
  | relation (name : String)
  | function (name : String)

private structure QueryAtom where
  source : QuerySource
  columns : Array Nat

private structure Query where
  names : List Nat
  next : Nat
  atoms : Array QueryAtom := #[]

private def queryRows (g : Graph m) (op arity : Nat) : Array (Nat × Array Nat) := Id.run do
  let mut rows := #[]
  for id in g.byOp.get op do
    if let some row := g.rows[id]? then
      if let .app _ args := row.shape then
        if args.length == arity then
          rows := rows.push (g.touched.getD id 0, (args.map (fun r => (root g r).ref.id)).toArray.push (root g row.ref).ref.id)
  return rows

mutual
  private def patternColumn (q : Query) : Pat → Query × Nat
    | .var i => (q, q.names.idxOf i)
    | .app op ps =>
      let (q, cols) := patternColumns q ps
      let v := q.next
      let atom : QueryAtom := ⟨.constructor op, cols.toArray.push v⟩
      ({ q with next := v + 1, atoms := q.atoms.push atom }, v)
  private def patternColumns (q : Query) : List Pat → Query × List Nat
    | [] => (q, [])
    | p :: ps =>
      let (q, v) := patternColumn q p
      let (q, vs) := patternColumns q ps
      (q, v :: vs)
end

/-- Equalities alias columns. Constructor rows and named relation tuples join
    on the same canonical class IDs, including inside nested arguments. -/
private def joinBindings (g : Graph m) (pattern : Option Pat) (premises : List (Pat × Pat))
    (names : List Nat) (cutoff : Nat) (relations : List (String × List Pat) := [])
    (cells : List (String × List Pat) := []) :
    Array Bindings := Id.run do
  let names := names.eraseDups
  let mut q : Query := ⟨names, names.length, #[]⟩
  if let some p := pattern then q := (patternColumn q p).1
  let mut equations := #[]
  for (lhs, rhs) in premises do
    let (left, x) := patternColumn q lhs
    if lhs == rhs then
      -- A rewrite's source pattern must exist. Joining it with a second
      -- copy of itself contributes no constraint or additional variable.
      q := left
    else
      let (right, y) := patternColumn left rhs
      q := right
      equations := equations.push (x, y)
  for (name, args) in relations do
    let (next, cols) := patternColumns q args
    q := { next with atoms := next.atoms.push ⟨.relation name, cols.toArray⟩ }
  for (name, args) in cells do
    let (next, columns) := patternColumns q args
    q := { next with atoms := next.atoms.push ⟨.function name, columns.toArray⟩ }
  let mut aliases := Array.range q.next
  for (x, y) in equations do
    let a := aliases[x]!
    let b := aliases[y]!
    if a != b then aliases := aliases.map fun n => if n == b then a else n

  let mut counts := Array.replicate q.next 0
  let atoms := q.atoms.map fun atom => { atom with columns := atom.columns.map (aliases[·]!) }
  for atom in atoms do
    for c in atom.columns do counts := counts.modify c (· + 1)
  let returned := (Array.range names.length).map (aliases[·]!)
  let mut inputs := #[]
  for atom in atoms do
    let rows := match atom.source with
      | .constructor op => queryRows g op (atom.columns.size - 1)
      | .relation name => relationRows g name atom.columns.size
      | .function name => functionRows g name (atom.columns.size - 1)
    -- A constructor result needed only to witness existence can be projected
    -- out. Repeated columns are retained for Input.prepare's equality test.
    let used := (Array.range atom.columns.size).filter fun i =>
      let c := atom.columns[i]!
      counts[c]! > 1 || returned.contains c
    if used.size == atom.columns.size then inputs := inputs.push (slice atom.columns rows cutoff)
    else
      let columns := used.map (atom.columns[·]!)
      let rows := rows.map fun (ts, row) => (ts, used.map (row[·]!))
      inputs := inputs.push (slice columns rows cutoff)
  let free := returned.toList.eraseDups.filter fun c => counts[c]! == 0
  if !free.isEmpty then
    let domain := g.rows.filterMap fun row =>
      let id := row.ref.id
      if (root g row.ref).ref.id == id then some (g.touched.getD id 0, #[id]) else none
    for x in free do inputs := inputs.push (slice #[x] domain cutoff)
  if inputs.any (fun input => input.all.rows.isEmpty) then return #[]
  let hits := if cutoff == 0 then GenericJoin.search (inputs.map Slice.all)
    else GenericJoin.searchDeltas (inputs.map (fun s => (s.all.vars, s.all.index))) fun i =>
      let s := inputs[i]!
      if s.new.rows.isEmpty then none else some (s.old.index, s.new.index)
  return hits.map fun env => Id.run do
    let mut bindings := []
    for name in names, i in [:names.length] do
      if let some id := env.getD aliases[i]! none then
        if let some row := g.rows[id]? then bindings := (name, row.ref) :: bindings
    return bindings

private structure Anchor (ps : List (Pat × Pat)) where
  bound : Pat
  pattern : Pat
  implies : ∀ (R : Pat → Pat → Prop), (∀ a b, R a b → R b a) → R bound pattern →
    ∀ p ∈ ps, R p.1 p.2

private def guardAnchor (ps : List (Pat × Pat)) (known : List Nat) : Option (Anchor ps) :=
  match ps with
  | [(left, right)] =>
    if left.variables.all known.contains then some ⟨left, right, by
      intro R symm eq p mem
      obtain rfl := List.mem_singleton.mp mem
      exact eq⟩
    else if right.variables.all known.contains then some ⟨right, left, by
      intro R symm eq p mem
      obtain rfl := List.mem_singleton.mp mem
      exact symm _ _ eq⟩
    else none
  | _ => none

/-- Equality premises can determine every variable before searching the lhs. -/
private def fixedBindings? (ps : List (Pat × Pat)) (names : List Nat) : Option (List (Nat × Pat)) :=
  let fixed := ps.filterMap fun
    | (.var i, p) | (p, .var i) => if p.variables.isEmpty then some (i, p) else none
    | _ => none
  if names.all (fun v => fixed.any (fun p => p.1 == v)) then some fixed else none

private def resolveFixed (g : Graph m) (fixed : List (Nat × Pat)) : Option Bindings :=
  fixed.mapM fun (i, p) => do return (i, (root g (← probePat g [] p).ref).ref)

private def applyBinding (initial g : Graph m) (rule : GuardedRule m) (σ : Bindings) : Graph m :=
  match probePat initial σ rule.lhs with
  | none => g
  | some lhs => applyGuarded initial g rule σ lhs.ref
    (fun e => (lhs.valid e).trans (eval_subst rule.lhs (assignment σ) e))

private def applyAnchor (initial g : Graph m) (rule : GuardedRule m) (a : Anchor rule.premises)
    (σ : Bindings) (lhs : Ref) (valid : OpenMatch m rule.lhs σ lhs) : Graph m :=
  match pat : a.bound with
  | .var i =>
    match found : σ.lookup i with
    | none => g
    | some bound =>
      (matchClass initial 0 a.pattern bound σ true).foldl (fun g hit =>
        emitRhs g rule.rhs hit.bindings lhs (fun e => by
          let η := fun j => (assignment hit.bindings j).eval m e
          have fit := fitsAssignment m hit.bindings e
          have old := hit.extendsEnv e η fit
          have eq : a.bound.eval m η = a.pattern.eval m η := by
            rw [pat]
            exact (old i bound found).trans (hit.valid e η fit)
          exact (valid e η old).trans (rule.sound η
            (a.implies (fun x y => x.eval m η = y.eval m η) (fun _ _ => Eq.symm) eq)))) g
  | .app _ _ =>
    match probePat initial σ a.bound with
    | none => g
    | some bound => (matchClass initial 0 a.pattern bound.ref σ true).foldl
      (fun g hit => applyBinding initial g rule hit.bindings) g

/-- An application can match only if its constructor occurs in this snapshot.
    Recheck the index every round so operators introduced by other rules become
    eligible immediately on the next snapshot. -/
private def hasRoot (g : @& Graph m) : Pat → Bool
  | .var _ => true
  | .app op _ => !(g.byOp.get op).isEmpty

private def guardedRound (initial : Graph m) (rules : Array (GuardedRule m)) : Graph m := Id.run do
  let mut g := initial
  for rule in rules do
    unless hasRoot initial rule.lhs do continue
    unless rule.premises.all (fun (a,b) => hasRoot initial a && hasRoot initial b) do continue
    let cutoff := cursor initial rule.toRewrite rule.premises
    if cutoff >= initial.version then continue
    let lhsVars := rule.lhs.variables
    let premiseVars := rule.premises.flatMap fun (a, b) => a.variables ++ b.variables
    if let some fixed := fixedBindings? rule.premises (lhsVars ++ premiseVars) then
      if let some σ := resolveFixed initial fixed then g := applyBinding initial g rule σ
    else if cutoff == 0 && premiseVars.all lhsVars.contains then
      g := scanLhs initial 0 rule.lhs g (fun g σ lhs valid => applyGuarded initial g rule σ lhs (closeMatch valid))
    else if let some anchor := if cutoff == 0 then guardAnchor rule.premises lhsVars else none then
      -- An anchored guard can only match a member of the known operand's class.
      -- Later delta rounds also account for changes on the guard side of the join.
      g := scanLhs initial 0 rule.lhs g (fun g => applyAnchor initial g rule anchor)
    else
      for σ in joinBindings initial (some rule.lhs) rule.premises (lhsVars ++ premiseVars) cutoff do
        g := applyBinding initial g rule σ
    g := advance g rule.toRewrite rule.premises initial.version
  return rebuild (g.rows.size + 1) g

/-- Reconstruct the evidence for each atom under the substitution produced
    by the generic join. Lookups use the round's fixed snapshot. -/
private def holdsAt? (g : Graph m) (σ : Bindings) (a : RelAtom) :
    Option (PLift (∀ e, a.Holds m (fun i => (assignment σ i).eval m e))) :=
  match a with
  | .eq l r => equalAt? g σ l r
  | .row name args => do
    let ⟨rs, valid⟩ ← probeArgs g σ args
    let ⟨existsRow⟩ ← rowAt? g name rs
    return ⟨fun e => by
      change m.relation name (Pat.evalArgs args m _)
      rw [← eval_substArgs args (assignment σ) e, ← valid e]
      exact existsRow e⟩

  | .cell name keys value => do
    let ⟨rs, valid⟩ ← probeArgs g σ keys
    let v ← probePat g σ value
    let ⟨existsCell⟩ ← cellAt? g name rs v.ref
    return ⟨fun e => by
      change m.CellHolds name (Pat.evalArgs keys m _) (value.eval m _)
      rw [← eval_substArgs keys (assignment σ) e, ← valid e,
          ← eval_subst value (assignment σ) e, ← v.valid e]
      exact existsCell e⟩

private def holdsBody? (g : Graph m) (σ : Bindings) (body : List RelAtom) :
    Option (PLift (∀ e, ∀ a ∈ body, a.Holds m (fun i => (assignment σ i).eval m e))) :=
  match body with
  | [] => some ⟨by simp⟩
  | a :: more => do
    let ⟨one⟩ ← holdsAt? g σ a
    let ⟨rest⟩ ← holdsBody? g σ more
    return ⟨by
      intro e b h
      rcases List.mem_cons.mp h with rfl | later
      · exact one e
      · exact rest e b later⟩

private def assertHead (g : Graph m) (σ : Bindings) (head : RelAtom)
    (proved : ∀ e, head.Holds m (fun i => (assignment σ i).eval m e)) : Graph m :=
  match head with
  | .row name args =>
    let (g, ⟨rs, valid⟩) := realizeArgs g σ args
    storeRow g ⟨name, rs, 0, fun e => by
      rw [valid e, eval_substArgs]
      exact proved e⟩
  | .eq a b =>
    let (g, x) := realize g σ a
    let (g, y) := realize g σ b
    merge g x.ref y.ref (fun e =>
      (x.valid e).trans ((eval_subst a _ e).trans ((proved e).trans
        ((eval_subst b _ e).symm.trans (y.valid e).symm))))

  | .cell name keys value =>
    let g := indexFunctions g
    let (g, ⟨rs, valid⟩) := realizeArgs g σ keys
    let (g, v) := realize g σ value
    let row : CellRow m := ⟨name, rs, v.ref, 0, fun e => by
      rw [valid e, eval_substArgs, v.valid e, eval_subst]
      exact proved e⟩
    writeCell g (normalizeCell g row false) true

private structure KeyLookup where
  name : String
  keys : List Pat
  valueVar : Nat

private def KeyLookup.atom (q : KeyLookup) := RelAtom.cell q.name q.keys (.var q.valueVar)

/-- A constructor scan binds the rewrite's operands. Each functional lookup
    may bind one more variable once its key is known. If these bindings cover
    every premise, relations and remaining equalities need only point probes. -/
private def lookupPlan? (anchor : Pat) (body : List RelAtom) : Option (List KeyLookup) := do
  let needed := body.flatMap RelAtom.variables
  let mut known := anchor.variables
  let mut lookups := []
  for _ in [:body.length + 1] do
    if needed.all known.contains then return lookups.reverse
    let next ← body.findSome? fun atom => match atom with
      | .cell name keys (.var v) =>
        if !known.contains v && (Pat.argsVariables keys).all known.contains then
          some (KeyLookup.mk name keys v)
        else none
      | _ => none
    lookups := next :: lookups
    known := next.valueVar :: known
  none

private def extendWithCells (g : Graph m) (σ : Bindings) : (lookups : List KeyLookup) →
    Option { full : Bindings // ∀ e η, Fits m full e η →
      Fits m σ e η ∧ ∀ a ∈ lookups.map KeyLookup.atom, a.Holds m η }
  | [] => some ⟨σ, fun _ _ fit => ⟨fit, by simp⟩⟩
  | q :: rest => do
    if fresh : σ.lookup q.valueVar = none then
      let ⟨keys, keyMatch⟩ ← probeOpenArgs g σ q.keys
      let canonical := canonicalArgs g keys
      let table ← g.functions[q.name]?
      let row ← table.entries[relKey canonical.val]?
      let ⟨evidence⟩ ← cellEvidence? row q.name canonical.val
      let ⟨full, ok⟩ ← extendWithCells g ((q.valueVar, row.value) :: σ) rest
      return ⟨full, fun e η fit => by
        obtain ⟨extended, later⟩ := ok e η fit
        have base := fitsTail fresh extended
        refine ⟨base, ?_⟩
        intro a ha
        rcases List.mem_cons.mp ha with rfl | tail
        · change m.CellHolds q.name (Pat.evalArgs q.keys m η) (η q.valueVar)
          rw [extended _ _ (lookupHead ..), ← keyMatch e η base, ← canonical.property e]
          exact evidence e
        · exact later a tail⟩
    else none  -- Only fresh outputs are selected by lookupPlan?.

/-- On its first round, an equality-producing rule may drive the match from
    its source constructor and read its guard tables by key. Later rounds
    use delta joins because a new guard row can enable an old source. -/
private def applyKeyedRewrite? (initial g : Graph m) (r : RelRule m) : Option (Graph m) :=
  match shape : r.head with
  | .eq lhs rhs => do
    guard r.extra.isEmpty
    guard ((match lhs with | .app .. => true | _ => false) == true)
    guard (r.body.contains (.eq lhs lhs))
    guard (r.body.any fun | .cell .. | .row .. => true | _ => false)
    let lookups ← lookupPlan? lhs r.body
    let covered := RelAtom.eq lhs lhs :: lookups.map KeyLookup.atom
    let guards := r.body.filter (fun a => decide (a ∉ covered))
    let emit := prepareRhs initial rhs
    return scanLhs initial 0 lhs g fun next σ ref source => Id.run do
      let some ⟨full, ok⟩ := extendWithCells initial σ lookups | return next
      let some ⟨hs⟩ := holdsBody? initial full guards | return next
      have matched : OpenMatch m lhs full ref := fun e η fit => source e η (ok e η fit).1
      return emit next full ref fun e => (closeMatch matched e).trans (by
        have sound := r.sound (fun i => (assignment full i).eval m e) (by
          intro a ha
          by_cases hit : a ∈ covered
          · rcases List.mem_cons.mp hit with rfl | cell
            · rfl
            · exact (ok e _ (fitsAssignment m full e)).2 a cell
          · exact hs e a (List.mem_filter.mpr ⟨ha, by simpa using hit⟩))
        rwa [shape] at sound)
  | _ => none

private def relationalMatches (g : Graph m) (body : List RelAtom) (cutoff : Nat) : Array Bindings :=
  if body.isEmpty then #[[]]
  else
    let eqs := body.filterMap fun | .eq l r => some (l, r) | _ => none
    let tables := body.filterMap fun | .row n args => some (n, args) | _ => none
    let cells := body.filterMap fun | .cell n keys value => some (n, keys ++ [value]) | _ => none
    let assignments := joinBindings g none eqs (body.flatMap RelAtom.variables) cutoff tables cells
    if cutoff == 0 || !g.congruentRows then assignments
    else Id.run do
      -- An old physical row and a recently rebuilt row may now denote one
      -- canonical tuple. In that case even different delta variants can emit
      -- the same assignment. A non-idempotent function merge observes repeats.
      let mut seen : Std.HashSet (List Nat) := {}
      let mut unique := #[]
      for σ in assignments do
        let key := σ.map (fun (_, r) => r.id)
        let (present, updated) := seen.containsThenInsert key
        seen := updated
        unless present do unique := unique.push σ
      return unique

private def fireRelHeads (g : Graph m) (σ : Bindings) (rule : RelRule m)
    (premises : ∀ e, ∀ a ∈ rule.body, a.Holds m (fun i => (assignment σ i).eval m e)) : Graph m :=
  rule.extra.foldl (fun graph action => assertHead graph σ action.head (fun e => action.sound _ (premises e)))
    (assertHead g σ rule.head (fun e => rule.sound _ (premises e)))

/-- For a body consisting of one constructor equality, the constructor index
    supplies both operands and result. Reuse the match proof for all head
    actions. Physical rows that became congruent may give the same binding. -/
@[specialize] private def foldRelEquality (initial : Graph m) (rule : RelRule m) (cutoff output : Nat)
    (pattern : Pat) (sufficesFor : ∀ η : Nat → _, η output = pattern.eval m η →
      ∀ a ∈ rule.body, a.Holds m η) (start : β) (dedup : Bool)
    (emit : β → (σ : Bindings) →
      (∀ e, ∀ a ∈ rule.body, a.Holds m (fun i => (assignment σ i).eval m e)) → β) : β := Id.run do
  let visit := fun (state : β × Std.HashSet (List Nat)) (σ : Bindings) (ref : Ref)
      (valid : OpenMatch m pattern σ ref) => Id.run do
    if h : σ.lookup output = none then
      let result := root initial ref
      let extended := (output, result.ref) :: σ
      let (acc, seen) := state
      let mut seen := seen
      -- Distinct physical rows can match the same assignment after rebuild
      -- discovers that their constructor keys have become congruent.
      if dedup && initial.congruentRows then
        let key := extended.map (fun (_, r) => r.id)
        let (present, updated) := seen.containsThenInsert key
        if present then return (acc, updated)
        seen := updated
      let next := emit acc extended (fun env => by
        let fit := fitsAssignment m extended env
        exact sufficesFor _ ((fit _ _ (lookupHead output result.ref σ)).trans
          ((result.valid env).trans (valid env _ (fitsTail h fit)))))
      return (next, seen)
    return state
  return (scanLhs initial cutoff pattern (start, {}) visit).1

private def scanRelEquality (initial g : Graph m) (rule : RelRule m) (cutoff output : Nat)
    (pattern : Pat) (sufficesFor : ∀ η : Nat → _, η output = pattern.eval m η →
      ∀ a ∈ rule.body, a.Holds m η) : Graph m := Id.run do
  match shape : rule.head with
  | .row name args =>
    let variables := output :: pattern.variables
    if rule.extra.isEmpty && args.all (fun | .var i => variables.contains i | _ => false) then
      -- Only the target table and clock change during this scan. Canonicalize
      -- against the working graph because an earlier rule may have merged keys.
      let g := indexRelations g
      let rows := (g.relations[name]?).getD {}
      let (stamp, rows) := foldRelEquality initial rule cutoff output pattern sufficesFor
        (g.version, rows) false fun (stamp, rows) σ premises => Id.run do
          let some ⟨refs, valid⟩ := probeArgs g σ args | return (stamp, rows)
          let canonical := canonicalArgs g refs
          let row : RelRow m := ⟨name, canonical.val, stamp + 1, fun e => by
            rw [canonical.property e, valid e, eval_substArgs]
            have h := rule.sound _ (premises e)
            rwa [shape] at h⟩
          let (present, rows) := rows.containsThenInsertIfNew (relKey row.args) row
          return (if present then stamp else stamp + 1, rows)
      if stamp == g.version then return g
      return { g with version := stamp, relations := g.relations.insert name rows }
  | _ => pure ()
  return foldRelEquality initial rule cutoff output pattern sufficesFor g true
    (fun graph σ => fireRelHeads graph σ rule)

private def applySingleEquality? (initial g : Graph m) (rule : RelRule m) (cutoff : Nat) :
    Option (Graph m) := do
  match body : rule.body with
  | [.eq (.var i) (.app op args)] =>
    guard (!(Pat.app op args).variables.contains i)
    return scanRelEquality initial g rule cutoff i (.app op args) fun _ h a ha => by
      rw [body, List.mem_singleton] at ha
      subst a
      exact h
  | [.eq (.app op args) (.var i)] =>
    guard (!(Pat.app op args).variables.contains i)
    return scanRelEquality initial g rule cutoff i (.app op args) fun _ h a ha => by
      rw [body, List.mem_singleton] at ha
      subst a
      exact h.symm
  | _ => none

private def relationalRound (initial : Graph m) (rules : Array (RelRule m)) : Graph m := Id.run do
  let initial := indexFunctions (indexRelations initial)
  let mut g := initial
  for rule in rules do
    let heads := rule.head :: rule.extra.map RelAction.head
    let sameRule := fun (c : RelCursor) => c.name == rule.name && c.body == rule.body && c.heads == heads
    let last := g.relCursors.find? sameRule
    if last.any (fun c => rule.body.isEmpty || c.version == initial.version) then continue
    let cutoff := (last.map RelCursor.version).getD 0
    if let some next := applySingleEquality? initial g rule cutoff then g := next
    else if let some next := if last.isNone then applyKeyedRewrite? initial g rule else none then g := next
    else
      for σ in relationalMatches initial rule.body cutoff do
        if let some ⟨premises⟩ := holdsBody? initial σ rule.body then
          g := fireRelHeads g σ rule premises
    g := { g with relCursors := ⟨rule.name, rule.body, heads, initial.version⟩ :: g.relCursors.eraseP sameRule }
  return rebuild (g.rows.size + 1) g

private def round (initial : Graph m) (rules : Array (VerifiedRule m)) : Graph m :=
  let g := rules.foldl (fun g rule =>
    if !hasRoot initial rule.lhs then g else
    let cutoff := cursor initial rule.toRewrite []
    if cutoff >= initial.version then g
    else
      let rhsVar := match rule.rhs with | .var i => some i | _ => none
      let emit := prepareRhs initial rule.rhs
      advance (scanLhs initial cutoff rule.lhs g (fun g σ lhs valid =>
        emit g σ lhs (fun e => (closeMatch valid e).trans (rule.sound _))) rhsVar)
        rule.toRewrite [] initial.version) initial
  rebuild (g.rows.size + 1) g

private def iterate (f : Graph m → Graph m) : Nat → Graph m → Graph m
  | 0, g => g
  | n + 1, g =>
    let version := g.version
    let next := f g
    if next.version == version then next else iterate f n next

private structure Delayed where
  cutoff : Nat := 0
  candidates : Array Bindings := #[]
  deriving Inhabited

private def scheduled (initial : Graph m) (limit : Nat) (rules : Array (NamedRule m))
    (policy : MatchScheduler) : Graph m := Id.run do
  let mut g := initial
  let mut state := policy.initial
  let mut delayed := Array.replicate rules.size ({} : Delayed)
  for iteration in [:limit] do
    let snapshot := g
    for entry in rules, i in [:rules.size] do
      let r := entry.rule
      let old := delayed[i]!
      let mut candidates := old.candidates
      if old.cutoff < snapshot.version then
        let vars := r.lhs.variables ++ r.premises.flatMap (fun (a,b) => a.variables ++ b.variables)
        candidates := candidates ++ joinBindings snapshot (some r.lhs) r.premises vars old.cutoff
      let views := candidates.map (fun σ => σ.map fun (v, x) => (v, (root g x).ref.term))
      let (next, selected) := policy.choose state ⟨iteration, g.rows.size, g.version⟩ entry.name views
      state := next
      let mut picked := Array.replicate candidates.size false
      for j in selected do
        if h : j < candidates.size then
          if !picked[j]! then
            picked := picked.set! j true
            g := applyBinding snapshot g r candidates[j]
      let waiting := (candidates.zip (Array.range candidates.size)).filterMap (fun (σ, j) => if picked[j]! then none else some σ)
      delayed := delayed.set! i ⟨snapshot.version, waiting⟩
    g := rebuild (g.rows.size + 1) g
    if g.version == snapshot.version &&
        policy.canStop state ⟨iteration, g.rows.size, g.version⟩ (delayed.any (!·.candidates.isEmpty)) then break
  return g

private partial def fixedPoint (step : Graph m → Graph m) (g : Graph m) : Graph m :=
  let next := step g
  if next.version == g.version then next else fixedPoint step next

private def factsHold (g : Graph m) (facts : List (Pat × Pat)) : Bool :=
  let names := facts.flatMap fun (l, r) => l.variables ++ r.variables
  if names.isEmpty then (checkPremises g [] facts).isSome
  else !(joinBindings g none facts names 0).isEmpty

private structure ValueFact {m : Model α} (a : ValueAnalysis m β) where
  ref : Ref
  value : β
  valid : ∀ env, ref.term.eval m env = a.meaning value

private def valueArgs (a : ValueAnalysis m β) (g : Graph m)
    (facts : Array (Option (ValueFact a))) (args : List Ref) :
    Option {xs : List β // ∀ env, Pat.evalArgs (args.map Ref.term) m env = xs.map a.meaning} :=
  match args with
  | [] => some ⟨[], fun _ => rfl⟩
  | r :: rs => do
    let k := root g r
    let some fact ← facts[k.ref.id]? | none
    let isTrue h := equalTerm k.ref.term fact.ref.term | none
    let tail ← valueArgs a g facts rs
    return ⟨fact.value :: tail.val, fun env => cons_congr
      ((k.valid env).symm.trans ((congrArg (fun t : Pat => t.eval m env) h).trans (fact.valid env)))
      (tail.property env)⟩

private def inferValue (a : ValueAnalysis m β) (g : Graph m)
    (facts : Array (Option (ValueFact a))) (row : Row) : Option (ValueFact a) := do
  match shapeEq : row.shape with
  | .var _ => none
  | .app op rs =>
    let xs ← valueArgs a g facts rs
    match ht : a.transfer op xs.val with
    | none => none
    | some v =>
      let k := root g row.ref
      return ⟨k.ref, v, fun env => (k.valid env).trans (by
        rw [row.valid, shapeEq]
        exact (congrArg (m.interp op) (xs.property env)).trans (a.sound _ _ _ ht))⟩

/-- Compute facts against one graph snapshot, then perform all justified merges.
    Parents that precede a newly available child are revisited in the next wave. -/
private def analyzeValues [BEq β] [Hashable β] [DecidableEq β] (a : ValueAnalysis m β)
    (g : Graph m) : Graph m := Id.run do
  let mut facts := Array.replicate g.rows.size (none : Option (ValueFact a))
  let mut pending := Array.range g.rows.size
  for _ in [:g.rows.size + 1] do
    if pending.isEmpty then break
    let mut next := #[]
    for i in pending do
      let some row := g.rows[i]? | continue
      let id := classId g row.ref
      if (facts.getD id none).isSome then continue
      if let some fact := inferValue a g facts row then facts := facts.set! id (some fact)
      else next := next.push i
    if next.size == pending.size then break
    pending := next
  let mut first : Std.HashMap β (ValueFact a) := {}
  let mut merged := g
  for entry in facts do
    let some fact := entry | continue
    if let some prior := first.get? fact.value then
      if h : fact.value = prior.value then
        merged := merge merged fact.ref prior.ref (fun env =>
          (fact.valid env).trans ((congrArg a.meaning h).trans (prior.valid env).symm))
    else first := first.insert fact.value fact
  return rebuild (merged.rows.size + 1) merged

mutual
  private def run (s : Schedule m) (g : Graph m) : Graph m :=
    match s with
    | .run rules => round g rules
    | .guarded rules => guardedRound g rules
    | .relational rules => relationalRound g rules
    | @Schedule.analyze _ _ β _eq _hash _dec a => analyzeValues (β := β) a g
    | .seq ss => runSeq ss g
    | .repeat n body => iterate (run body) n g
    | .saturate n body => iterate (run body) n g
    | .fixpoint body => fixedPoint (run body) g
    | .scheduled n rules policy => scheduled g n rules policy
    | .untilMatch facts body => if factsHold g facts then g else run body g
    | .untilFacts facts body =>
      let g := indexFunctions (indexRelations g)
      if (relationalMatches g facts 0).any (fun σ => (holdsBody? g σ facts).isSome)
      then g else run body g
  private def runSeq (ss : List (Schedule m)) (g : Graph m) : Graph m :=
    match ss with
    | [] => g
    | s :: ss => runSeq ss (run s g)
end

/-- Cost search selects enodes using only integer class IDs. The final walk
    checks the selected parent-child links and constructs the semantic witness
    for each reachable class once. Unselected alternatives allocate no terms. -/
private structure Choices where
  prices : Array Nat
  rows : Array Nat

/-- Zero is unknown. Finite prices are encoded as one plus the tree cost. -/
@[inline] private def price (g : @& Graph m) (prices : @& Array Nat) (cost : Nat → Nat)
    (s : Shape) : Nat := Id.run do
  match s with
  | .var _ => return 2
  | .app op args =>
    let mut n := cost op + 1
    for r in args do
      let k := prices.getD (classId g r) 0
      if k == 0 then return 0
      n := n + (k - 1)
    return n

private def relaxChoices (g : @& Graph m) (cost : Nat → Nat) (choices : Choices)
    (todo : Option (Array Nat)) : Array Nat × Choices := Id.run do
  let mut prices := choices.prices
  let mut rows := choices.rows
  let mut next := #[]
  let n := todo.elim g.rows.size Array.size
  for i in [:n] do
    let id := if let some ids := todo then ids[i]! else i
    let some row := g.rows[id]? | continue
    let r := classId g row.ref
    let n := price g prices cost row.shape
    let old := prices.getD r 0
    if n == 0 || (old != 0 && n >= old) then continue
    prices := prices.set! r n
    rows := rows.set! r id
    -- The first pass visits rows in insertion order. Only earlier parents
    -- could already have missed this improvement. Later waves wake all users.
    for parent in g.uses.getD r #[] do
      if todo.isSome || parent <= id then next := next.push parent
  return (next, ⟨prices, rows⟩)

private def selectChoices (g : Graph m) (cost : Nat → Nat) : Nat → Choices → Option (Array Nat) → Choices
  | 0, cs, _ => cs
  | n+1, cs, todo =>
    let (next, cs) := relaxChoices g cost cs todo
    if next.isEmpty then cs else selectChoices g cost n cs (some next)

private structure Chosen (m : Model α) where
  source : Pat
  tree : Witness m source

private abbrev Cache (m : Model α) := Array (Option (Chosen m))

@[specialize] private def recoverArgs (rec : (r : Ref) → StateM (Cache m) (Option (Witness m r.term))) :
    (rs : List Ref) → StateM (Cache m) (Option (Args m (rs.map Ref.term)))
  | [] => pure (some ⟨[], fun _ => rfl⟩)
  | x :: xs => do
    let some y ← rec x | return none
    let some ys ← recoverArgs rec xs | return none
    return some (consW y ys)

@[inline] private def recoverShape (rec : (r : Ref) → StateM (Cache m) (Option (Witness m r.term))) :
    (s : Shape) → StateM (Cache m) (Option (Witness m s.term))
  | .var v => pure (some (reflW m (.var v)))
  | .app op args => do
    let some children ← recoverArgs rec args | return none
    return some (appW op children)

private def recover (g : Graph m) (choices : Choices) : Nat →
    (r : Ref) → StateM (Cache m) (Option (Witness m r.term))
  | 0, _ => pure none
  | fuel + 1, r => do
    let key := root g r
    let id := key.ref.id
    -- A class with just one incoming use cannot save a second tree walk.
    let shared := (g.uses.getD id #[]).size > 1
    if shared then
      if let some hit := (← get).getD id none then
        if let isTrue eq := equalTerm hit.source key.ref.term then
          return some ⟨hit.tree.val, fun e => (hit.tree.property e).trans
            ((congrArg (fun p => p.eval m e) eq).trans (key.valid e))⟩
    if choices.prices.getD id 0 == 0 then return none
    let some node := g.rows[choices.rows.getD id g.rows.size]? | return none
    let repr := root g node.ref
    let isTrue eq := equalTerm repr.ref.term key.ref.term | return none
    let some t ← recoverShape (recover g choices fuel) node.shape | return none
    let best : Witness m key.ref.term := ⟨t.val, fun e => (t.property e).trans
      ((congrArg (fun p => p.eval m e) node.valid.symm).trans
        ((repr.valid e).symm.trans (congrArg (fun p => p.eval m e) eq)))⟩
    if shared then modify (·.set! id (some ⟨key.ref.term, best⟩))
    return some ⟨best.val, fun e => (best.property e).trans (key.valid e)⟩

private def extract (g : Graph m) (cost : Nat → Nat) (r : Ref) : Option (Witness m r.term) :=
  let n := g.rows.size
  let choices := selectChoices g cost (n + 1) ⟨Array.replicate n 0, Array.replicate n 0⟩ none
  (recover g choices (n + 1) r).run' (Array.replicate n none)

/-- Equality saturation and minimum additive tree-cost extraction. Object-level
    variables denote inputs. Rule variables range over entire e-classes. -/
def optimize (m : Model α) (input : Pat) (schedule : Schedule m)
    (cost : Nat → Nat := fun _ => 1) : { out : Pat // ∀ env, out.eval m env = input.eval m env } :=
  let (seeds, start) := seed (emptySeeds (occurrences input)) input
  let g := run schedule (finishSeeds seeds)
  match extract g cost start.val with
  | some t => ⟨t.val, fun e => (t.property e).trans (congrArg (fun p => p.eval m e) start.property)⟩
  | none => reflW m input

/-- Prove two inputs equivalent by saturating them in the same graph. -/
def proveEq? (m : Model α) (a b : Pat) (schedule : Schedule m) :
    Option (PLift (∀ env, a.eval m env = b.eval m env)) := do
  let (st, a') := seed (emptySeeds (occurrences a + occurrences b)) a
  let (st, b') := seed st b
  let g := run schedule (finishSeeds st)
  let x := a'.property ▸ root g a'.val
  let y := b'.property ▸ root g b'.val
  match equalTerm x.ref.term y.ref.term with
  | isTrue h => return ⟨fun e => (x.valid e).symm.trans ((congrArg (fun p => p.eval m e) h).trans (y.valid e))⟩
  | isFalse _ => none

/-- A persistent proof-carrying graph shared by any number of compiler roots. -/
structure Session (m : Model α) where
  private mk ::
  private graph : Graph m

/-- A root returned by `Session.add`, indexed by its original input. Handles
    remain usable as the originating session grows and its classes merge. -/
structure Handle (m : Model α) (input : Pat) where
  private mk ::
  private location : Located m input

def Session.empty (m : Model α) : Session m := ⟨{}⟩

def Session.add (s : Session m) (p : Pat) : Session m × Handle m p :=
  let (g, node) := realize s.graph [] p
  (⟨g⟩, ⟨node.ref, fun e => (node.valid e).trans (eval_subst p (assignment []) e)⟩)

def Session.run (s : Session m) (phases : Schedule m) : Session m :=
  ⟨Certified.run phases s.graph⟩

/-- Seed a positive relation with terms in the source language, requiring a
    proof for all valuations of their free inputs. -/
def Session.addFact (s : Session m) (name : String) (args : List Pat)
    (proof : ∀ env, m.relation name (Pat.evalArgs args m env)) : Session m :=
  ⟨assertHead s.graph [] (.row name args) proof⟩

/-- Retrieve semantic evidence for a currently known tuple modulo congruence. -/
def Session.fact? (s : Session m) (name : String) (args : List Pat) :
    Option (PLift (∀ env, m.relation name (Pat.evalArgs args m env))) := do
  let (g, ⟨rs, valid⟩) := realizeArgs s.graph [] args
  let ⟨known⟩ ← rowAt? (indexRelations g) name rs
  return ⟨fun e => by
    have eq := (valid e).trans (eval_substArgs args (assignment []) e)
    exact eq ▸ known e⟩

/-- Assert a verified function entry. Conflicting values execute the function's
    merge expression in the same graph as the surrounding compiler terms. -/
def Session.setCell (s : Session m) (name : String) (keys : List Pat) (value : Pat)
    (proof : ∀ e, m.CellHolds name (Pat.evalArgs keys m e) (value.eval m e)) : Session m :=
  ⟨assertHead s.graph [] (.cell name keys value) proof⟩

/-- Read the current function value together with its invariant for every
    assignment to the free source inputs. -/
def Session.getCell? (s : Session m) (name : String) (keys : List Pat) :
    Option { value : Pat // ∀ e, m.CellHolds name (Pat.evalArgs keys m e) (value.eval m e) } := do
  let (g, ⟨rs, located⟩) := realizeArgs (indexFunctions s.graph) [] keys
  let canonical := canonicalArgs g rs
  let table ← g.functions[name]?
  let row ← table.entries[relKey canonical.val]?
  let ⟨valid⟩ ← cellEvidence? row name canonical.val
  return ⟨row.value.term, fun e => by
    have eq := (canonical.property e).trans ((located e).trans (eval_substArgs keys (assignment []) e))
    exact eq ▸ valid e⟩

/-- Check both the slot and its original syntax. Two diverging snapshots can
    assign the same numeric slot to different terms. -/
private def owns (g : Graph m) (r : Ref) : Bool :=
  (g.rows[r.id]?).any (fun row => (equalTerm row.ref.term r.term).decide)

/-- Query an existing graph without rerunning its rules. -/
def Session.equivalent? (s : Session m) (a : Handle m p) (b : Handle m q) :
    Option (PLift (∀ env, p.eval m env = q.eval m env)) := do
  let x := a.location
  let y := b.location
  guard (owns s.graph x.ref && owns s.graph y.ref)
  let l := locTrans x (root s.graph x.ref)
  let r := locTrans y (root s.graph y.ref)
  match equalTerm l.ref.term r.ref.term with
  | isFalse _ => none
  | isTrue h => return ⟨fun e => (l.valid e).symm.trans ((congrArg (fun t : Pat => t.eval m e) h).trans (r.valid e))⟩

/-- Add a host-proved instance of a DSL fact using the same actions as a rule
    head. Equalities trigger rebuilding before the caller observes the session. -/
def Session.assertAtom (s : Session m) (a : RelAtom) (proof : ∀ env, a.Holds m env) : Session m :=
  let g := assertHead s.graph [] a proof
  ⟨rebuild (g.rows.size + 1) g⟩

/-- Check a ground DSL fact with its free variables interpreted as program
    inputs. A function lookup checks the current value up to congruence. -/
def Session.atom? (s : Session m) (a : RelAtom) : Option (PLift (∀ e, a.Holds m e)) :=
  match a with
  | .row name args => s.fact? name args
  | .eq a b =>
    let (s, x) := s.add a
    let (s, y) := s.add b
    s.equivalent? x y
  | .cell name keys value => do
    let ⟨current, valid⟩ ← s.getCell? name keys
    let (s, stored) := s.add current
    let (s, queried) := s.add value
    let ⟨same⟩ ← s.equivalent? stored queried
    return ⟨fun e => by
      change m.CellHolds name (Pat.evalArgs keys m e) (value.eval m e)
      rw [← same e]
      exact valid e⟩

/-- Stop between stages as soon as two previously inserted roots coincide.
    Handles avoid re-matching both ground patterns at each stop check. -/
def Session.runUntilEqual (s : Session m) (a : Handle m p) (b : Handle m q)
    (stages : List (Schedule m)) (rounds : Nat) : Session m := Id.run do
  let mut s := s
  if (s.equivalent? a b).isSome then return s
  for _ in [:rounds] do
    let version := s.graph.version
    for stage in stages do
      s := s.run stage
      if (s.equivalent? a b).isSome then return s
    if s.graph.version == version then break
  return s

private def Session.mergeOwned (s : Session m) (x : Located m p) (y : Located m q)
    (equiv : ∀ env, p.eval m env = q.eval m env) : Session m :=
  let g := merge s.graph x.ref y.ref (fun e => (x.valid e).trans ((equiv e).trans (y.valid e).symm))
  ⟨rebuild (g.rows.size + 1) g⟩

/-- Import a separately proved source equality, then repair all congruences. -/
def Session.merge? (s : Session m) (a : Handle m p) (b : Handle m q)
    (equiv : ∀ env, p.eval m env = q.eval m env) : Option (Session m) := do
  guard (owns s.graph a.location.ref && owns s.graph b.location.ref)
  return s.mergeOwned a.location b.location equiv

/-- A merge with foreign handles leaves the session alone. Putting the
ownership test before the update lets a unique session grow in place. -/
def Session.mergeIfOwned (s : Session m) (a : Handle m p) (b : Handle m q)
    (equiv : ∀ env, p.eval m env = q.eval m env) : Session m :=
  if owns s.graph a.location.ref && owns s.graph b.location.ref then
    s.mergeOwned a.location b.location equiv
  else s

/-- An offline proof hint. Bindings are terms in the source graph. The rule
    and all its premises are checked again when the hint is consumed. -/
structure GroundStep where
  rule : Nat
  bindings : List (Nat × Pat)
  deriving Repr, Inhabited

/-- Enumerate rule instances for one round of offline proof reconstruction.
    Only applications that add terms or merge classes enter the transcript.
    Normal saturation uses its existing specialized matchers. -/
def Session.recordRound (s : Session m) (rules : Array (GuardedRule m))
    (offset : Nat := 0) : Session m × Array GroundStep := Id.run do
  let snapshot := s.graph
  let mut g := snapshot
  let mut steps := #[]
  for r in rules, i in [:rules.size] do
    unless hasRoot snapshot r.lhs do continue
    let vars := r.lhs.variables ++ r.premises.flatMap (fun (l,h) => l.variables ++ h.variables)
    for σ in joinBindings snapshot (some r.lhs) r.premises vars 0 do
      let next := applyBinding snapshot g r σ
      if next.version != g.version then
        steps := steps.push ⟨offset + i, σ.map (fun (v,x) => (v, x.term))⟩
      g := next
  return (⟨rebuild (g.rows.size + 1) g⟩, steps)

/-- Apply concrete instances of proved rules without enumerating matches.
    An invalid rule index or unsatisfied premise contributes no equality. -/
def Session.replaySteps (s : Session m) (rules : Array (GuardedRule m))
    (steps : List GroundStep) : Session m := Id.run do
  let mut g := s.graph
  for step in steps do
    let some r := rules[step.rule]? | continue
    let mut bindings : Bindings := []
    for (v, term) in step.bindings do
      let (next, value) := realize g [] term
      bindings := (v, value.ref) :: bindings
      g := next
    -- Materializing a premise mentions its endpoints. Only the existing
    -- class equality between them can justify applying a guarded rule.
    for (a, b) in r.premises do
      g := (realize g bindings a).1
      g := (realize g bindings b).1
    let (g', lhs) := realize g bindings r.lhs
    let next := applyGuarded g' g' r bindings lhs.ref (fun e =>
      (lhs.valid e).trans (eval_subst _ _ e))
    g := rebuild (next.rows.size + 1) next
  return ⟨g⟩

/-- A cost model solved once for every class in a session snapshot. -/
structure Extractor (m : Model α) where
  private mk ::
  private graph : Graph m
  private choices : Choices

def Session.extractor (s : Session m) (cost : Nat → Nat := fun _ => 1) : Extractor m :=
  let g := s.graph
  let n := g.rows.size
  ⟨g, selectChoices g cost (n + 1) ⟨Array.replicate n 0, Array.replicate n 0⟩ none⟩

private def Extractor.recoverHandle (s : Extractor m) (a : Handle m p) :
    StateM (Cache m) (Option (Witness m p)) := do
  let x := a.location
  if !owns s.graph x.ref then return none
  let some t ← recover s.graph s.choices (s.graph.rows.size + 1) x.ref | return none
  return some ⟨t.val, fun e => (t.property e).trans (x.valid e)⟩

/-- Recover the selected finite tree for one handle using the shared cost solution. -/
def Extractor.extract? (s : Extractor m) (a : Handle m p) :
    Option { out : Pat // ∀ env, out.eval m env = p.eval m env } :=
  (s.recoverHandle a).run' (Array.replicate s.graph.rows.size none)

/-- Recover a series of roots on one fixed extractor. The cache contains
only witnesses from its own immutable graph and cost solution. -/
structure ExtractionBatch (m : Model α) where
  private mk ::
  private extractor : Extractor m
  private cache : Cache m

def Extractor.batch (s : Extractor m) : ExtractionBatch m :=
  ⟨s, Array.replicate s.graph.rows.size none⟩

def ExtractionBatch.extract? (batch : ExtractionBatch m) (a : Handle m p) :
    ExtractionBatch m × Option { out : Pat // ∀ env, out.eval m env = p.eval m env } :=
  let (out, cache) := (batch.extractor.recoverHandle a).run batch.cache
  (⟨batch.extractor, cache⟩, out)

end Egglog.Certified
