import Egglog.Functions
import Egglog.GenericJoin

namespace Egglog

namespace Relational

abbrev Row := Array Id

/-- A set-valued relation with a lookup index for each bound column. -/
structure Table where
  sorts : Array String
  rows : Array Row := #[]
  seen : Std.HashSet Row := {}
  indexed : Array Nat := #[]
  index : Std.HashMap (Nat × Id) (Array Row) := {}
  deriving Inhabited

def Table.insert (t : Table) (row : Row) : Table := Id.run do
  let (present, seen) := t.seen.containsThenInsert row
  if present then return { t with seen }
  let mut index := t.index
  for col in t.indexed do
    index := index.alter (col, row[col]!) (fun bucket => some ((bucket.getD #[]).push row))
  return { t with rows := t.rows.push row, seen, index }

private def Table.ensureIndex (t : Table) (col : Nat) : Table := Id.run do
  if t.indexed.contains col then return t
  let mut index := t.index
  for row in t.rows do
    index := index.alter (col, row[col]!) (fun bucket => some ((bucket.getD #[]).push row))
  return { t with indexed := t.indexed.push col, index }

/-- In a constructor atom, the optional last column holds the result class. -/
inductive Atom where
  | relation (name : String) (fields : Array Pat)
  | function (name : String) (args : Array Pat) (result : Option Pat)
  | constructor (op : Nat) (args : Array Pat) (result : Pat)
  deriving Inhabited

inductive Action where
  | insert (name : String) (args : Array Pat)
  | set (name : String) (key : Array Pat) (value : Pat)
  | union (lhs rhs : Pat)
  | attach (target op : Nat) (args : Array Pat) (result : Option Nat)
  | build (term : Pat)
  | bind (varId : Nat) (term : Pat)
  | changeCtor (op : Nat) (args : Array Pat) (subsume : Bool)
  | deleteFunction (name : String) (key : Array Pat)
  deriving Inhabited

structure Rule where
  body : Array Atom
  head : Array Action
  ruleset : String := ""
  guards : Array GuardTest := #[]
  deriving Inhabited

structure Frame where
  program : ProgramState := {}
  tables : Std.HashMap String Table := {}
  functions : Functions := {}
  rules : Array Rule := #[]
  /-- Counts rounds that changed at least one table or class. -/
  epoch : Nat := 0
  deriving Inhabited

structure Database extends Frame where
  stack : List Frame := []
  deriving Inhabited

abbrev Delta := Std.HashMap String (Array Row)

private inductive Source where
  | relation (name : String)
  | function (name : String)
  | constructor (op arity : Nat)
  deriving Inhabited, BEq

private structure AtomPlan where
  source : Source
  columns : Array Nat
  deriving Inhabited

private structure QueryPlan where
  atoms : Array AtomPlan := #[]
  nextVar : Nat := 0
  deriving Inhabited

private def fresh : StateM QueryPlan Nat := do
  let s ← get
  set { s with nextVar := s.nextVar + 1 }
  return s.nextVar

private def addAtom (source : Source) (columns : Array Nat) : StateM QueryPlan Unit :=
  modify fun s => { s with atoms := s.atoms.push ⟨source, columns⟩ }

/-- Translate an embedded pattern to constructor relations with a result column.
    The fresh variables are existential within a match. -/
private partial def flatten : Pat → StateM QueryPlan Nat
  | .var v => pure v
  | .app op fields => do
    let cols ← fields.mapM flatten
    let result ← fresh
    addAtom (.constructor op fields.length) (cols.toArray.push result)
    return result

private partial def nextAfter : Pat → Nat
  | .var v => v + 1
  | .app _ kids => kids.foldl (fun n p => max n (nextAfter p)) 0

private def compileQuery (body : Array Atom) : Array AtomPlan := Id.run do
  let fields := body.flatMap fun
    | .relation _ args => args
    | .function _ args result => if let some r := result then args.push r else args
    | .constructor _ args out => args.push out
  let action : StateM QueryPlan Unit := do
    for atom in body do
      match atom with
      | .relation name args => addAtom (.relation name) (← args.mapM flatten)
      | .function name args out =>
        let cols ← args.mapM flatten
        let r ← match out with | some p => flatten p | none => fresh
        addAtom (.function name) (cols.push r)
      | .constructor op args out =>
        let cols ← args.mapM flatten
        addAtom (.constructor op args.size) (cols.push (← flatten out))
  return (action.run { nextVar := fields.foldl (fun n p => max n (nextAfter p)) 0 }).2.atoms

private def sourceRows (db : Database) : Source → Array Row
  | .relation name => db.tables[name]!.rows
  | .function name => Id.run do
    let table := db.functions[name]!
    let mut rows := #[]
    for key in table.keys do
      let result := table.entries[key]!
      rows := rows.push ((key.map db.program.graph.find).push (db.program.graph.find result))
    return rows
  | .constructor op arity => Id.run do
    let g := db.program.graph
    let mut rows := #[]
    for id in g.opNodes op do
      let fields := g.nodes[id]!.args
      if fields.size == arity then rows := rows.push (fields.push (g.find id))
    return rows

/-- Source occurrences differing only by variable renaming share their partitioned
    rows. The layout identifies a repeated variable by its first column. -/
private structure SourceView where
  source : Source
  layout : Array Nat
  deriving BEq, Inhabited

private structure JoinAtom where
  view : Nat
  vars : Array Nat
  deriving Inhabited

private def assignView (atom : AtomPlan) : StateM (Array SourceView) JoinAtom := do
  let key := ⟨atom.source, atom.columns.map atom.columns.idxOf⟩
  let views ← get
  let view := views.idxOf key
  if view == views.size then set (views.push key)
  return ⟨view, (GenericJoin.Input.prepare atom.columns #[]).vars⟩

private structure Snapshot where
  full : GenericJoin.Index
  rowCount : Nat
  split : Option (GenericJoin.Index × GenericJoin.Index) := none
  deriving Inhabited

private def SourceView.snapshot (db : Database) (cutoff : Option Nat) (previous : Option Snapshot)
    (v : SourceView) : Snapshot := Id.run do
  if let some old := previous then
    if cutoff == some old.rowCount then
      let extent := match v.source with
        | .constructor op _ => db.program.graph.storedRowCount op
        | .relation name => db.tables[name]!.rows.size
        | .function name => db.functions[name]!.keys.size
      -- With an append-only source, this equality proves the old full index
      -- still covers every visible row, including its lazily built partitions.
      if extent == old.rowCount then return { old with split := none }
  let rows := sourceRows db v.source
  let prepare := fun rs => (GenericJoin.Input.prepare v.layout rs).index
  let full := prepare rows
  let some n := cutoff | return ⟨full, rows.size, none⟩
  if n >= rows.size then return ⟨full, rows.size, none⟩
  if n == 0 then return ⟨full, rows.size, some (prepare #[], full)⟩
  return ⟨full, rows.size, some (prepare (rows.extract 0 n), prepare (rows.extract n rows.size))⟩

/-- The first new source row distinguishes semi-naive variants. Earlier
    inputs range over old rows only. All searches use one immutable snapshot. -/
private def findMatches (sources : Array Snapshot) (plan : Array JoinAtom) (semiNaive : Bool) :
    Array Bindings := Id.run do
  let mut inputs := plan.map (fun a => (a.vars, sources[a.view]!.full))
  unless semiNaive do return GenericJoin.searchIndexed inputs
  let mut hits := #[]
  for h : i in [:plan.size] do
    let a := plan[i]
    let some (old, recent) := sources[a.view]!.split | continue
    hits := hits ++ GenericJoin.searchIndexed (inputs.set! i (a.vars, recent))
    inputs := inputs.set! i (a.vars, old)
  return hits

private def rebuildTables (tabs : Std.HashMap String Table) (g : EGraph) : Std.HashMap String Table := Id.run do
  let mut out := {}
  for (name, t) in tabs do
    let mut clean : Table := { sorts := t.sorts, indexed := t.indexed }
    for row in t.rows do clean := clean.insert (row.map g.find)
    out := out.insert name clean
  return out

@[inline] private def evalTerm (g : EGraph) (p : Pat) (env : Bindings) : Except String (Id × EGraph) :=
  instantiateArithmetic p env g

private def evalArgs (g : EGraph) (args : Array Pat) (env : Bindings) :
    Except String (Row × EGraph) := do
  let mut g := g
  let mut row := Array.emptyWithCapacity args.size
  for arg in args do
    let (id, next) ← evalTerm g arg env
    row := row.push (next.find id)
    g := next
  return (row, g)

private partial def lookupReadTerm (g : EGraph) (env : Bindings) : Pat → Option Id
  | .var i => do
    let id ← env[i]?.join
    guard (id < g.nodes.size)
    return g.find id
  | .app op args => do
    let xs ← args.mapM (lookupReadTerm g env)
    g.lookup? ⟨op, xs.toArray⟩

private def storeBinding (env : Bindings) (i id : Nat) : Bindings :=
  if i == env.size then env.push (some id)
  else if i < env.size then env.set! i (some id)
  else (env ++ Array.replicate (i - env.size) none).push (some id)

/-- Attach the RHS of a union directly to its target class. On a hash-cons miss
    no existing representative changes, so function keys and join cursors stay
    valid. Keep separate let bindings for heads that need pre-union delete keys. -/
private def compileUnions (actions : Array Action) : Array Action := Id.run do
  let readsOldKeys := actions.any fun | .changeCtor .. => true | _ => false
  let mut result := #[]
  let mut i := 0
  while i < actions.size do
    match actions[i]! with
    | .bind v (.app op args) =>
      if let some (.union (.var target) (.var other)) := actions[i+1]? then
        if !readsOldKeys && v == other && target != v then
          result := result.push (.attach target op args.toArray (some v))
          i := i + 2
          continue
    | .union (.var target) (.app op args) =>
      result := result.push (.attach target op args.toArray none)
      i := i + 1
      continue
    | _ => pure ()
    result := result.push actions[i]!
    i := i + 1
  return result

private def applyHead (db : Database) (actions : Array Action) (env : Bindings) (readGraph : EGraph) :
    Except String Database := do
  let mut g := db.program.graph
  let mut tables := db.tables
  let mut functions := db.functions
  let mut env := env
  for action in actions do
    match action with
    | .insert name args =>
      let (row, next) ← evalArgs g args env
      tables := tables.modify name (·.insert row)
      g := next
    | .set name args value =>
      let (key, next) ← evalArgs g args env
      let (id, next) ← evalTerm next value env
      functions := functions.insert name (← functions[name]!.set next key id)
      g := next
    | .union a b =>
      let (i, next) ← evalTerm g a env
      let (j, next) ← evalTerm next b env
      g := next.merge i j
    | .attach target op ps result =>
      let some target := env[target]?.join | throw "unbound union target"
      let (ids, next) ← evalArgs g ps env
      g := next.addToClass target ⟨op, ids⟩
      if let some v := result then env := storeBinding env v (g.find target)
    | .bind i p =>
      let (id, next) ← evalTerm g p env
      env := storeBinding env i id
      g := next
    | .changeCtor op ps subsume =>
      let (ids, next) ← evalArgs g ps env
      let keys := ps.mapIdx fun j p => (lookupReadTerm readGraph env p).getD ids[j]!
      g := next.changeRow ⟨op, keys⟩ subsume
    | .deleteFunction name ps =>
      let (key, next) ← evalArgs g ps env
      functions := functions.modify name (fun t => t.erase next key)
      g := next
    | .build p =>
      let (_, next) ← evalTerm g p env
      g := next
  return { db with program := { db.program with graph := g }, tables, functions }

/-- A single function head can update all matched keys before putting the
    function back in the map. This keeps the growing entries and key array
    uniquely owned while applying the batch. -/
private def applySets (db : Database) (name : String) (key : Array Pat) (rhs : Pat)
    (envs : Array Bindings) : Except String Database := do
  if envs.isEmpty then return db
  let mut t := db.functions[name]!
  let fs := db.functions.erase name
  let mut g := db.program.graph
  for env in envs do
    let (ids, next) ← evalArgs g key env
    let (value, next) ← evalTerm next rhs env
    t ← t.set next ids value
    g := next
  return { db with program := { db.program with graph := g }, functions := fs.insert name t }

/-- A pure relational head projects bound variables directly into one table.
    Accumulate the whole batch before reinserting the table in its map. -/
private def insertProjection? (head : Array Action) : Option (String × Array Nat) := do
  let #[.insert name fields] := head | none
  let columns ← fields.mapM fun p => match p with | .var v => some v | _ => none
  return (name, columns)

private def applyProjection (db : Database) (name : String) (columns : Array Nat)
    (envs : Array Bindings) : Except String Database := do
  if envs.isEmpty then return db
  let some initial := db.tables[name]? | throw s!"unknown relation {name}"
  let mut table := initial
  let g := db.program.graph
  for env in envs do
    let mut row := Array.emptyWithCapacity columns.size
    for v in columns do
      let some id := env[v]?.join | throw "unbound head variable"
      row := row.push (g.find id)
    table := table.insert row
  return { db with tables := db.tables.insert name table }

-- Flat joins can move IDs directly from source columns to the output tuple.
-- Names, variable occurrences and column projections are resolved once per run.
private inductive Column where
  | left (col : Nat) | right (col : Nat)
  deriving Inhabited

@[inline] private def Column.read (l r : @& Row) : Column → Id
  | .left i => l[i]!
  | .right i => r[i]!

private inductive Projection where
  | sameLeft | sameRight | two (a b : Column) | other (fields : Array Column)

@[inline] private def Projection.row (l r : @& Row) : Projection → Row
  | .sameLeft => l
  | .sameRight => r
  | .two a b => #[a.read l r, b.read l r]
  | .other fields => fields.map (Column.read l r)

private structure FlatPlan where
  left : String
  right : Option String
  key : Option (Nat × Nat)
  tests : Array (Column × Column)
  target : String
  projection : Projection

private def variables (fields : Array Pat) : Option (Array Nat) :=
  fields.mapM fun | .var v => some v | _ => none

private def joinProjection (left : String) (ls : Array Nat) (right : Option String)
    (rs : Array Nat) (target : String) (out : Array Nat) : Option FlatPlan := do
  let mut refs : Std.HashMap Nat Column := {}
  let mut tests := #[]
  let mut key := none
  for i in [:ls.size] do
    let var := ls[i]!
    if let some prior := refs.get? var then tests := tests.push (prior, .left i)
    else refs := refs.insert var (.left i)
  for i in [:rs.size] do
    let var := rs[i]!
    if let some prior := refs.get? var then
      if let .left j := prior then
        if key.isNone then
          key := some (j, i)
          continue
      tests := tests.push (prior, .right i)
    else refs := refs.insert var (.right i)
  let fields ← out.mapM refs.get?
  let projection := if out == ls then .sameLeft
    else if right.isSome && out == rs then .sameRight
    else match fields with | #[a, b] => .two a b | _ => .other fields
  return ⟨left, right, key, tests, target, projection⟩

private def compileFlat (rule : Rule) : Option FlatPlan := do
  guard rule.guards.isEmpty
  let #[.insert target args] := rule.head | none
  let some (Atom.relation left ls) := rule.body[0]? | none
  let ls ← variables ls
  let (right, rs) ← match rule.body with
    | #[_] => some (none, #[])
    | #[_, .relation right fields] => (variables fields).map (fun vs => (some right, vs))
    | _ => none
  joinProjection left ls right rs target (← variables args)

/-- Constructor occurrences in a rule can supply the class of a compound
    term in its head. Reuse only terms actually bound by that rule's query. -/
private partial def boundColumn (atoms : Array AtomPlan) : Pat → Option Nat
  | .var v => some v
  | .app op fields => do
    let args := (← fields.mapM (boundColumn atoms)).toArray
    let atom ← atoms.find? fun a =>
      a.source == .constructor op args.size && a.columns.pop == args
    atom.columns.back?

private structure ReadPlan where
  constructors : Array ((Nat × Nat) × String) := #[]
  joins : Array FlatPlan := #[]

private def readSource (db : Database) (a : AtomPlan) : StateT ReadPlan Option String := do
  match a.source with
  | .function _ => failure
  | .relation name => return name
  | .constructor op arity =>
    let state ← get
    if let some (_, name) := state.constructors.find? (fun (k, _) => k == (op, arity)) then
      return name
    let mut name := s!"#constructor-row-{op}-{arity}"
    for _ in [:db.tables.size + 1] do
      if !db.tables.contains name then break
      name := name ++ "#"
    set { state with constructors := state.constructors.push ((op, arity), name) }
    return name

/-- A graph-read-only stage can index constructors once and propagate
    relation deltas through one- and two-atom projection joins. -/
private def planReadStage (db : Database) (rules : Array Rule) : Option ReadPlan := do
  let action : StateT ReadPlan Option Unit := do
    for r in rules do
      guard r.guards.isEmpty
      let atoms := compileQuery r.body
      guard (atoms.size == 1 || atoms.size == 2)
      let left ← readSource db atoms[0]!
      let right ← atoms[1]?.mapM (readSource db)
      let rs := (atoms[1]?.map (·.columns)).getD #[]
      for head in r.head do
        let .insert target args := head | failure
        let some vs := args.mapM (boundColumn atoms) | failure
        let some plan := joinProjection left atoms[0]!.columns right rs target vs | failure
        modify fun st => {st with joins := st.joins.push plan}
  return (← action.run {}).2

@[inline] private def emitFlat (p : FlatPlan) (l r : @& Row) (out : Array Row) : Array Row :=
  if p.tests.all (fun (a, b) => a.read l r == b.read l r) then out.push (p.projection.row l r)
  else out

/-- Join a nonempty driving input (a full table or a delta) with the other
    snapshot table. Build the probed column index on its first actual use. -/
private abbrev Tables := Std.HashMap String Table

private def probe (tables : Tables) (p : FlatPlan) (drive : Array Row) (other : String)
    (backward : Bool) (out : Array Row) : Tables × Array Row := Id.run do
  let mut out := out
  if drive.isEmpty || tables[other]!.rows.isEmpty then return (tables, out)
  let (tables, cols) := match p.key with
    | none => (tables, none)
    | some (l, r) =>
      let (fromCol, toCol) := if backward then (r, l) else (l, r)
      let tables : Tables := if tables[other]!.indexed.contains toCol then tables else
        tables.modify other (·.ensureIndex toCol)
      (tables, some (fromCol, toCol))
  let t := tables[other]!
  for row in drive do
    let candidates := match cols with
      | none => t.rows
      | some (i, j) => t.index.getD (j, row[i]!) #[]
    for mate in candidates do
      out := if backward then emitFlat p mate row out else emitFlat p row mate out
  return (tables, out)

private def flatRows (tables : Tables) (p : FlatPlan) (delta : Option Delta) :
    Tables × Array Row := Id.run do
  let left := match delta with
    | none => tables[p.left]!.rows
    | some d => d.getD p.left #[]
  let some right := p.right | return (tables, left.foldl (fun out row => emitFlat p row #[] out) #[])
  let (tables, rows) := probe tables p left right false #[]
  let some d := delta | return (tables, rows)
  return probe tables p (d.getD right #[]) p.left true rows

private def flatBatches (tables : Tables) (plans : Array FlatPlan) (delta : Option Delta) :
    Tables × Array (String × Array Row) := Id.run do
  let mut tables := tables
  let mut batches := #[]
  for plan in plans do
    let (next, rows) := flatRows tables plan delta
    tables := next
    if !rows.isEmpty then batches := batches.push (plan.target, rows)
  return (tables, batches)

-- A relation-only round needs just the tables and the newly inserted tuples.
-- Keep the table map linear between rounds so row arrays and membership sets
-- can grow in place after all joins against the old snapshot have finished.
private partial def runFlat (initial : Database) (plans : Array FlatPlan) (limit : Option Nat)
    (stop : Database → Except String Bool) : Except String Database := do
  let mut tables := initial.tables
  let mut delta : Option Delta := none
  let mut epoch := initial.epoch
  let mut round := 0
  repeat
    if limit == some round then break
    if round != 0 then
      if ← stop { initial with tables, epoch } then break
    round := round + 1
    let (next, batches) := flatBatches tables plans delta
    tables := next
    let mut added : Delta := {}
    for (name, rows) in batches do
      let before := tables[name]!.seen.size
      tables := tables.modify name (fun t => rows.foldl Table.insert t)
      let after := tables[name]!.rows
      if after.size > before then
        let newRows := after.extract before after.size
        added := added.alter name fun prev => some ((prev.getD #[]) ++ newRows)
    if added.isEmpty then break
    delta := some added
    epoch := epoch + 1
  return { initial with tables, epoch }

private def runReadStage (db : Database) (plan : ReadPlan) (flat : Array FlatPlan)
    (limit : Option Nat) (stop : Database → Except String Bool) : Except String Database := do
  let mut tables := db.tables
  for ((op, arity), name) in plan.constructors do
    tables := tables.insert name {sorts := #[], rows := sourceRows db (.constructor op arity)}
  let visible := fun (st : Database) =>
    {st with tables := plan.constructors.foldl (fun ts (_, k) => ts.erase k) st.tables}
  let result ← runFlat {db with tables} (flat ++ plan.joins) limit (fun st => stop (visible st))
  return visible result

private partial def run (initial : Database) (name : String) (limit : Option Nat)
    (stop : Database → Except String Bool) : Except String Database := do
  let group ← initial.program.getGroup name
  if limit == some 0 then return initial
  -- A later schedule stage can arrive with its stopping condition already met.
  if ← stop initial then return initial
  let members := initial.program.ruleSetMembers name
  let mut flat := #[]
  let mut rules := #[]
  for rule in initial.rules do
    if members.contains rule.ruleset then
      if let some plan := compileFlat rule then flat := flat.push plan
      else rules := rules.push rule
  if group.plain.isEmpty && group.arithmetic.isEmpty && group.conditional.isEmpty then
    if rules.isEmpty then return ← runFlat initial flat limit stop
    if let some plan := planReadStage initial rules then
      return ← runReadStage initial plan flat limit stop
  let (queries, views) := (rules.mapM (fun (r : Rule) => do
    let atoms ← (compileQuery r.body).mapM assignView
    return (r.head, r.guards, atoms))).run #[]
  let mut db := initial
  let mut delta : Option Delta := none
  let mut cutoffs : Option (Array Nat) := none
  let mut previous : Array Snapshot := #[]
  let mut changedGuardTerms := false
  let mut round := 0
  repeat
    if limit == some round then break
    if round != 0 then
      if ← stop db then break
    round := round + 1
    let g := db.program.graph
    let version := g.version
    let merged := g.mergeCount
    let maskVersion := g.maskVersion
    let deletions := db.functions.map fun _ t => t.deletedVersion
    let versions := db.functions.map fun _ t => t.version
    let sizes := db.tables.map fun _ t => t.rows.size
    let (indexed, batches) := flatBatches db.tables flat delta
    db := { db with tables := indexed }
    let sources := views.mapIdx (fun i view => view.snapshot db (cutoffs.bind (·[i]?)) previous[i]?)
    let hits ← queries.mapM fun (head, guards, plan) => do
      let incremental := cutoffs.isSome && (guards.isEmpty || !changedGuardTerms)
      let envs := findMatches sources plan incremental
      if guards.isEmpty then return (head, envs)
      return (head, ← envs.filterM fun env => guards.allM (checkGuard g env))
    let rewrites := searchRound g group.plain
    let arithmetic ← searchArithmetic g group.arithmetic
    let guarded ← searchGuarded g group.conditional
    let needsReadKeys := hits.any fun (head, envs) =>
      !envs.isEmpty && head.any (fun | .changeCtor .. => true | _ => false)
    let readGraph := if needsReadKeys then g else EGraph.empty
    let graph := applyRound (applyRound (applyRound g rewrites) arithmetic) guarded
    db := { db with program := { db.program with graph } }
    for (target, rows) in batches do
      db := { db with tables := db.tables.modify target (fun t => rows.foldl Table.insert t) }
    for (head, envs) in hits do
      if let some (name, columns) := insertProjection? head then
        db ← applyProjection db name columns envs
      else if let #[.set name key rhs] := head then
        db ← applySets db name key rhs envs
      else
        for env in envs do db ← applyHead db head env readGraph
    let graph := db.program.graph.rebuild
    let changedTerms := graph.version != version
    let mergedClasses := graph.mergeCount != merged
    let tables := if mergedClasses then rebuildTables db.tables graph else db.tables
    let functions ← if mergedClasses then rebuildFunctions db.functions graph else pure db.functions
    let changedFunctions := functions.any fun name t => t.version != versions.getD name 0
    let invalidate := mergedClasses || graph.maskVersion != maskVersion ||
      functions.any (fun name t => t.deletedVersion != deletions.getD name 0)
    db := { db with program := { db.program with graph }, tables, functions }
    let mut added : Delta := {}
    for (name, t) in db.tables do
      let size := sizes.getD name 0
      if size < t.rows.size then added := added.insert name (t.rows.extract size t.rows.size)
    if !changedTerms && !changedFunctions && !invalidate && added.isEmpty then break
    -- Insertion preserves row order in every source. A replaced function
    -- result can occur before its suffix, so re-drive that entire source.
    -- A class merge can rekey rows anywhere and invalidates all cutoffs.
    cutoffs := if invalidate then none else some (views.mapIdx (fun i view =>
      let n := sources[i]!.rowCount
      match view.source with
      | .function name =>
        let t := functions[name]!
        if t.version - versions.getD name 0 > t.keys.size - n then 0 else n
      | _ => n))
    previous := if invalidate then #[] else sources.map fun s => { s with split := none }
    delta := if invalidate then none else some added
    changedGuardTerms := changedTerms
    db := { db with epoch := db.epoch + 1 }
  return db

private def app? : SExpr → Option (String × Array SExpr)
  | .list xs => do
    let .atom op ← xs[0]? | none
    return (op, xs.extract 1 xs.size)
  | _ => none

private def elabFields (s : ProgramState) (xs : Array SExpr) (sorts : Array String)
    (vars : PatternVars) (bind : Bool) : Except String (Array Pat × PatternVars) := do
  if xs.size != sorts.size then throw "wrong relation arity"
  let mut fields := #[]
  let mut vars := vars
  for x in xs, sort in sorts do
    let (p, _, next) ← elabPat s x (some sort) vars bind (!bind)
    fields := fields.push p
    vars := next
  return (fields, vars)

private def elabAtom (db : Database) (x : SExpr) (vars : PatternVars) :
    Except String (Atom × PatternVars) := do
  let some (name, args) := app? x | throw "expected a rule atom"
  if let some table := db.tables.get? name then
    let (ps, vars) ← elabFields db.program args table.sorts vars true
    return (.relation name ps, vars)
  if let some t := db.functions.get? name then
    let (ps, vars) ← elabFields db.program args t.inputs vars true
    return (.function name ps none, vars)
  if name == "=" && args.size == 2 then
    let useLeft := (app? args[0]!).any (fun (op, _) => db.functions.contains op) ||
                   (app? args[1]!).isNone
    let (result, term) := if useLeft then (args[1]!, args[0]!) else (args[0]!, args[1]!)
    let some (ctor, fields) := app? term | throw "expected a constructor application in equality"
    if let some t := db.functions.get? ctor then
      let (ps, vars) ← elabFields db.program fields t.inputs vars true
      let (p, _, vars) ← elabPat db.program result (some t.result) vars true
      return (.function ctor ps (some p), vars)
    let some c := db.program.constructors.get? ctor | throw s!"unknown constructor {ctor}"
    let (ps, vars) ← elabFields db.program fields c.inputs vars true
    let (p, _, vars) ← elabPat db.program result (some c.result) vars true
    return (.constructor c.op ps p, vars)
  if let some ctor := db.program.constructors.get? name then
    let (ps, vars) ← elabFields db.program args ctor.inputs vars true
    return (.constructor ctor.op ps (.var vars.next), { vars with next := vars.next + 1 })
  throw s!"unsupported rule atom {name}"

private def elabAction (db : Database) (x : SExpr) (vars : PatternVars) : Except String Action := do
  let some (name, args) := app? x | throw "expected a rule head action"
  if let some t := db.tables.get? name then
    let (ps, _) ← elabFields db.program args t.sorts vars false
    return .insert name ps
  if (name == "delete" || name == "subsume") && args.size == 1 then
    let some (fname, args) := app? args[0]! | throw s!"{name} expects a function application"
    if let some c := db.program.constructors.get? fname then
      let (ps, _) ← elabFields db.program args c.inputs vars false
      return .changeCtor c.op ps (name == "subsume")
    if let some t := db.functions.get? fname then
      if name == "subsume" then throw "cannot subsume a function with merge"
      let (ps, _) ← elabFields db.program args t.inputs vars false
      return .deleteFunction fname ps
    throw s!"expected a constructor or function in {name}"
  if name == "set" && args.size == 2 then
    let some (fname, fields) := app? args[0]! | throw "set requires a function application"
    let some t := db.functions.get? fname | throw s!"unknown function {fname}"
    let (ps, _) ← elabFields db.program fields t.inputs vars false
    let (value, _, _) ← elabPat db.program args[1]! (some t.result) vars false true
    return .set fname ps value
  if name == "union" && args.size == 2 then
    let (a, sort, _) ← elabPat db.program args[0]! none vars false true
    if primitive sort then throw "union requires an equality sort"
    let (b, _, _) ← elabPat db.program args[1]! (some sort) vars false true
    return .union a b
  let (p, _, _) ← elabPat db.program x none vars false true
  return .build p

private def groundRow (db : Database) (t : Table) (xs : Array SExpr) (insert : Bool) :
    Except String (Row × Database) := do
  if xs.size != t.sorts.size then throw "wrong relation arity"
  let mut s := db.program
  let mut ids := #[]
  for x in xs, expected in t.sorts do
    let (id, sort, next) ← groundWithFunctions s db.functions x insert
    unless expected == sort do throw s!"expected {expected}, got {sort}"
    ids := ids.push id
    s := next
  return (ids.map s.graph.find, { db with program := s })

private def freshName (db : Database) (name : String) : Except String Unit := do
  db.program.freshName name
  if db.functions.contains name || db.tables.contains name then throw s!"duplicate name {name}"

private def formatValue (s : ProgramState) (id : Id) : Except String String := do
  let some x := s.extract? id | throw "could not extract table value"
  return s.formatTree x.term

private def printTable (db : Database) (name : String) (limit : Option Nat) : Except String Database := do
  let s := db.program
  let mut lines := #["("]
  let render (args : Array Id) := do
    let values ← args.mapM (formatValue s)
    pure ("   (" ++ String.intercalate " " (name :: values.toList) ++ ")")
  if let some t := db.tables.get? name then
    for row in t.rows[:limit.getD t.rows.size] do lines := lines.push (← render row)
  else if let some t := db.functions.get? name then
    for (key, value) in t.entries do
      if lines.size > limit.getD t.entries.size then break
      lines := lines.push ((← render key) ++ " -> " ++ (← formatValue s value))
  else if let some c := s.constructors.get? name then
    let rows := s.graph.tableRows c.op
    for id in rows[:limit.getD rows.size] do
      lines := lines.push ((← render s.graph.nodes[id]!.args) ++ " -> " ++ (← formatValue s id))
  else throw s!"unknown table {name}"
  return { db with program := { s with output := s.output ++ lines.push ")" } }

private def rebuildIfMerged (db : Database) (s : ProgramState) : Except String Database := do
  if s.graph.mergeCount == db.program.graph.mergeCount then return { db with program := s }
  return { db with program := s, tables := rebuildTables db.tables s.graph,
                   functions := ← rebuildFunctions db.functions s.graph }

private def checkFacts (db : Database) (facts : Array SExpr) : Except String Unit := do
  for fact in facts do
    if let some (rel, xs) := app? fact then
      if let some t := db.tables.get? rel then
        let (row, _) ← groundRow db t xs false
        unless t.seen.contains row do throw "check failed"
        continue
      if db.functions.contains rel then
        let _ ← groundWithFunctions db.program db.functions fact false
        continue
      if (rel == "=" || rel == "!=") && xs.size == 2 && !db.functions.isEmpty then
        let (a, sort, s) ← groundWithFunctions db.program db.functions xs[0]! false
        let (b, other, _) ← groundWithFunctions s db.functions xs[1]! false
        if sort != other then throw s!"expected {sort}, found {other}"
        unless (a == b) == (rel == "=") do throw "check failed"
        continue
    let _ ← execCommand db.program (.list #[.atom "check", fact])

private def missingFact (err : String) := err == "check failed" ||
  err == "ground term is absent" || err.startsWith "no entry in function "

private def runConfig (db : Database) (cfg : RunConfig) (limit : Option Nat) : Except String Database :=
  run db cfg.name limit fun st => do
    let some facts := cfg.until | return false
    match checkFacts st facts with
    | .ok _ => return true
    | .error err => if missingFact err then return false else throw err

/-- A stage reaches a fixed point after an unbounded unconditional run or a
    positive bounded run with no changes. Only reuse the result at its exact
    database epoch within this one schedule (which cannot add rules). -/
private def scheduledStage (state : Database × Std.HashMap String Nat) (cfg : RunConfig)
    (limit : Option Nat) : Except String (Database × Std.HashMap String Nat) := do
  let (db, stable) := state
  if cfg.until.isNone && stable.get? cfg.name == some db.epoch then return state
  let next ← runConfig db cfg limit
  let fixedPoint := cfg.until.isNone && limit != some 0 && (limit.isNone || db.epoch == next.epoch)
  return (next, if fixedPoint then stable.insert cfg.name next.epoch else stable)

partial def exec (db : Database) (cmd : SExpr) : Except String Database := do
  let some (name, args) := app? cmd | throw "expected an egglog command"
  match name, args.toList with
  | "push", _ | "pop", _ =>
    let count ← match args.toList with
      | [] => pure 1
      | [.atom n] => match n.toNat? with | some v => pure v | none => throw "expected an unsigned stack count"
      | _ => throw "expected an unsigned stack count"
    let mut db := db
    for _ in [:count] do
      if name == "push" then db := { db with stack := db.toFrame :: db.stack }
      else
        let frame :: stack := db.stack | throw "pop exceeds stack depth"
        let output := db.program.output
        db := { frame with stack, program := { frame.program with output } }
    return db
  | "delete", [_] | "subsume", [_] =>
    let action ← elabAction db cmd {}
    let next ← applyHead db #[action] #[] db.program.graph
    return next
  | "function", _ =>
    let (fname, t) ← declareFun db.program args
    freshName db fname
    return { db with functions := db.functions.insert fname t }
  | "set", [app, rhs] =>
    let (program, functions) ← setGround db.program db.functions app rhs
    return { db with program, functions }
  | "let", [.atom n, term] =>
    if !db.functions.isEmpty then
      freshName db n
      let (id, sort, s) ← groundWithFunctions db.program db.functions term true
      return { db with program := { s with globals := s.globals.insert n (sort, id) } }
  | "extract", [term] =>
    if !db.functions.isEmpty then
      let (id, _, s) ← groundWithFunctions db.program db.functions term true
      return { db with program := { s with output := s.output.push (← formatValue s id) } }
  | "union", [a, b] =>
    if !db.functions.isEmpty then
      let (i, sort, s) ← groundWithFunctions db.program db.functions a true
      let (j, other, s) ← groundWithFunctions s db.functions b true
      if sort != other || primitive sort then throw "union requires matching equality sorts"
      return ← rebuildIfMerged db { s with graph := (s.graph.merge i j).rebuild }
  | "relation", [.atom rel, .list sorts] =>
    freshName db rel
    let inputs ← sorts.mapM fun
      | .atom s => if db.program.sorts.contains s then pure s else throw s!"unknown sort {s}"
      | _ => throw "expected a sort name"
    return { db with tables := db.tables.insert rel { sorts := inputs } }
  | "rule", .list body :: .list head :: opts =>
    let mut set := ""
    let mut options := opts
    while !options.isEmpty do
      match options with
      | .atom ":ruleset" :: .atom group :: rest => set := group; options := rest
      | .atom ":name" :: .atom label :: rest =>
        unless label.startsWith "\"" do throw "expected a quoted rule name"
        options := rest
      | _ => throw "expected :ruleset or :name"
    if db.program.combined.contains set then throw s!"cannot add a rule to combined ruleset {set}"
    let _ ← db.program.getGroup set
    let mut vars : PatternVars := {}
    let mut atoms := #[]
    let mut filters := #[]
    for fact in body do
      if let some (op, args) := app? fact then
        if ["<", "<=", ">", ">=", "!="].contains op then
          filters := filters.push fact
          continue
        if op == "=" then
          if let #[.atom label, term] := args then
            if let some (prim, _) := app? term then
              if (arithmeticOp? prim).isSome then
                let (pat, sort, _) ← elabPat db.program term none vars false true
                if vars.bindings.contains label || vars.derived.contains label then
                  filters := filters.push fact
                else
                  vars := { vars with derived := vars.derived.insert label (pat, sort) }
                  filters := filters.push (.list #[.atom "=", term, term])
                continue
      let (a, next) ← elabAtom db fact vars
      atoms := atoms.push a
      vars := next
    let mut guards := #[]
    for fact in filters do
      let some (op, operands) := app? fact | throw "expected a comparison"
      let mut terms := #[.atom op]
      for x in operands do
        if let some (fname, fields) := app? x then
          if let some t := db.functions.get? fname then
            let (ps, _) ← elabFields db.program fields t.inputs vars false
            let label := s!"#guard_{vars.next}"
            if vars.bindings.contains label then throw s!"reserved guard variable {label}"
            atoms := atoms.push (.function fname ps (some (.var vars.next)))
            vars := {vars with bindings := vars.bindings.insert label (vars.next, t.result), next := vars.next + 1 }
            terms := terms.push (.atom label)
            continue
        terms := terms.push x
      guards := guards.push (← compileGuard db.program vars (.list terms))
    let mut actions := #[]
    for x in head do
      if let some ("let", #[.atom localName, term]) := app? x then
        if vars.bindings.contains localName || vars.derived.contains localName then
          throw s!"local let shadows a rule variable {localName}"
        let (pat, sort, _) ← elabPat db.program term none vars false true
        actions := actions.push (.bind vars.next pat)
        vars := { vars with bindings := vars.bindings.insert localName (vars.next, sort), next := vars.next + 1 }
      else actions := actions.push (← elabAction db x vars)
    return { db with rules := db.rules.push ⟨atoms, compileUnions actions, set, guards⟩ }
  | "run", _ =>
    if !db.rules.isEmpty || !db.tables.isEmpty || !db.functions.isEmpty then
      let (cfg, n) ← parseBoundedRun args
      return ← runConfig db cfg n
  | "run-schedule", steps =>
    if !db.rules.isEmpty || !db.tables.isEmpty || !db.functions.isEmpty then
      let mut state : Database × Std.HashMap String Nat := (db, {})
      for expr in steps do
        state ← runScheduleWith (fun s cfg n => scheduledStage s cfg (some n)) (·.1.epoch) state expr
          (some (fun s cfg => scheduledStage s cfg none))
      return state.1
  | "check", _ =>
    checkFacts db args
    return db
  | "fail", [inner] =>
    match exec db inner with
    | .ok _ => throw "expected command to fail"
    | .error err =>
      if missingFact err || err == "conflicting values for :no-merge function" ||
          err == "i64 arithmetic overflow" || err == "undefined i64 operation" ||
          err == "pop exceeds stack depth" ||
          err.startsWith "unknown ruleset " || err.startsWith "cannot add a rule to combined ruleset " then return db
      throw err
  | "print-function", [.atom rel] =>
    return ← printTable db rel none
  | "print-function", [.atom rel, .atom n] =>
    let some limit := n.toNat? | throw "invalid row limit"
    return ← printTable db rel (some limit)
  | "print-size", [] =>
    let s := db.program
    let mut sizes := s.names.map fun n => (n, (s.tableSize? n).getD 0)
    for (n, t) in db.tables do sizes := sizes.push (n, t.rows.size)
    for (n, t) in db.functions do sizes := sizes.push (n, t.entries.size)
    let lines := (sizes.qsort fun a b => a.1 < b.1).map fun (n, size) => s!"({n} {size})"
    return { db with program := { s with output := s.output.push ("(" ++ String.intercalate "\n " lines.toList ++ ")") } }
  | "print-size", [.atom rel] =>
    if let some t := db.functions.get? rel then
      let s := db.program
      return { db with program := { s with output := s.output.push (toString t.entries.size) } }
    if let some t := db.tables.get? rel then
      let s := db.program
      return { db with program := { s with output := s.output.push (toString t.rows.size) } }
  | _, _ => pure ()
  if let some t := db.tables.get? name then
    let (row, db) ← groundRow db t args true
    return { db with tables := db.tables.insert name (t.insert row) }
  if db.functions.contains name || (!db.functions.isEmpty && db.program.constructors.contains name) then
    let (_, _, s) ← groundWithFunctions db.program db.functions cmd true
    return { db with program := s }
  if ["sort", "datatype", "let", "ruleset", "unstable-combined-ruleset"].contains name then
    if let some (SExpr.atom n) := args[0]? then freshName db n
  if name == "datatype*" then
    for group in args do
      if let some (sort, _) := app? group then freshName db sort
  let s ← execCommand db.program cmd
  if s.constructors.size != db.program.constructors.size then
    for (rel, _) in db.tables do
      if s.constructors.contains rel then throw s!"constructor name already used by relation {rel}"
    for (fname, _) in db.functions do
      if s.constructors.contains fname then throw s!"constructor name already used by function {fname}"
  rebuildIfMerged db s

def runSource (source : String) (db : Database := {}) : Except String Database := do
  (← parseSExprs source).foldlM exec db

end Relational
end Egglog
