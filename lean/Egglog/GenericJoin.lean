import Egglog.Rules

namespace Egglog.GenericJoin

/-- Each column names a logic variable. Rows sharing a variable must agree. -/
structure Input where
  vars : Array Nat
  rows : Array (Array Id)
  deriving Inhabited

/-- Remove repeated columns from an atom after enforcing their equalities. -/
def Input.prepare (vars : Array Nat) (rows : Array (Array Id)) : Input := Id.run do
  let mut distinct := #[]
  let mut keep := #[]
  let mut equal := #[]
  for h : i in [:vars.size] do
    let v := vars[i]
    if let some j := distinct.idxOf? v then
      equal := equal.push (keep[j]!, i)
    else
      keep := keep.push i
      distinct := distinct.push v
  if equal.isEmpty then return ⟨vars, rows⟩
  return ⟨distinct, rows.filterMap fun row =>
    if equal.all (fun (a, b) => row[a]! == row[b]!) then some (keep.map (fun i => row[i]!))
    else none⟩

private structure Occurrence where
  atom : Nat
  column : Nat
  deriving Inhabited

private structure Variable where
  name : Nat
  uses : Array Occurrence
  deriving Inhabited

private def variables (inputs : Array (Array Nat)) : Array Variable := Id.run do
  let mut vars : Array Variable := #[]
  for h : i in [:inputs.size] do
    let columns := inputs[i]
    for k : col in [:columns.size] do
      let name := columns[col]
      let loc : Occurrence := ⟨i, col⟩
      if let some pos := vars.findIdx? (·.name == name) then
        vars := vars.modify pos (fun v => { v with uses := v.uses.push loc })
      else vars := vars.push ⟨name, #[loc]⟩
  return vars

private abbrev Rows := Array (Array Id)

/-- Each column partition knows its maximum possible key count at construction.
    An integer-indexed table gives native and kernel execution the same lookup. -/
private structure Groups (α : Type) where
  slots : Array (Option (Id × α))
  keys : Array Id := #[]
  deriving Inhabited

private def findSlot (slots : @& Array (Option (Id × α))) (key : Id) : Nat → Nat → Nat
  | 0, i => i
  | n + 1, i =>
    match slots[i]? with
    | some (some (k, _)) =>
      if key == k then i
      else findSlot slots key n (if i + 1 == slots.size then 0 else i + 1)
    | _ => i

@[inline] private def Groups.slot (g : @& Groups α) (v : Id) : Nat :=
  findSlot g.slots v g.slots.size (v % g.slots.size)

private def Groups.get? (g : @& Groups α) (v : Id) : Option α := do
  let (_, rows) ← g.slots[g.slot v]?.join
  return rows

private def partition (rows : Rows) (col : Nat) : Groups Rows := Id.run do
  -- An odd capacity keeps power-of-two strides from collapsing into one bucket.
  let mut g : Groups Rows := ⟨Array.replicate (rows.size * 2 + 1) none, #[]⟩
  for row in rows do
    let value := row[col]!
    let i := g.slot value
    if (g.slots.getD i none).isSome then
      g := { g with slots := g.slots.modify i (fun slot =>
        slot.map (fun (k, hits) => (k, hits.push row))) }
    else
      g := ⟨g.slots.set! i (some (value, #[row])), g.keys.push value⟩
  return g

set_option genSizeOfSpec false in
/-- A snapshot's projected column index. Every key owns its child slice and
    that slice's lazy partitions. Distinct search prefixes reaching the same
    key in this relation reuse the child even when other relations differ. -/
inductive Index where
  | private slice (rows : Array (Array Id)) (columns : Array (Thunk (Groups Index)))

instance : Inhabited Index := ⟨.slice #[] #[]⟩

def Index.rows : Index → Array (Array Id)
  | .slice rs _ => rs

@[inline] private def Index.columns : Index → Array (Thunk (Groups Index))
  | .slice _ cs => cs

-- A query chooses each variable once. After enforcing repeated-column
-- equalities, at most the arity of this atom can be chosen along any path.
private def makeIndex (arity : Nat) : Nat → Rows → Index
  | 0, rows => .slice rows #[]
  | depth + 1, rows =>
    if rows.size ≤ 1 then .slice rows #[] else
      .slice rows ((Array.range arity).map fun col => Thunk.mk fun _ =>
        let g := partition rows col
        ⟨g.slots.map (·.map fun (key, rs) => (key, makeIndex arity depth rs)), g.keys⟩)

def Input.index (t : Input) : Index := makeIndex t.vars.size t.vars.size t.rows

private def bound (slices : Array Index) (v : Variable) : Nat :=
  v.uses.foldl (fun n atCol => min n slices[atCol.atom]!.rows.size) slices[v.uses[0]!.atom]!.rows.size

/-- At each prefix, the smallest incident slice bounds the extension count.
    Break ties in favor of variables constraining more atoms. -/
private def choose (slices : Array Index) (rest : Array Variable) : Nat := Id.run do
  let mut best := 0
  let mut cost := bound slices rest[0]!
  for i in [1:rest.size] do
    let v := rest[i]!
    let next := bound slices v
    if next < cost || (next == cost && v.uses.size > rest[best]!.uses.size) then
      best := i
      cost := next
  return best

private def Index.lookup (s : Index) (col : Nat) (v : Id) : Option Index :=
  if s.rows.size == 1 then
    if s.rows[0]![col]! == v then some s else none
  else s.columns[col]!.get.get? v

private structure Payload where
  atom : Nat
  fields : Array (Nat × Nat) -- (variable, column)
  deriving Inhabited

private def Payload.bind (p : Payload) (row : Array Id) (env : Bindings) : Bindings :=
  p.fields.foldl (fun e (v, col) => e.set! v (some row[col]!)) env

/-- After joining on shared variables, the remaining fields of a row can all
    be copied at once. Keep set semantics for callers supplying duplicate rows. -/
private def finish (slices : Array Index) : List Payload → Bindings → Array Bindings → Array Bindings
  | [], env, out => out.push env
  | p :: ps, env, out => Id.run do
    let rows := slices[p.atom]!.rows
    if let #[row] := rows then
      return finish slices ps (p.bind row env) out
    let mut seen : Std.HashSet (Array Id) := {}
    let mut out := out
    for row in rows do
      if seen.contains row then continue
      seen := seen.insert row
      out := finish slices ps (p.bind row env) out
    return out

/-- Intersect the incident projections before producing a partial binding.
    A singleton slice fixes the extension; other occurrences can probe their
    index for just that value. -/
private def extend (payload : List Payload) : Nat → Array Index → Array Variable → Bindings → Array Bindings → Array Bindings
  | 0, slices, rest, env, out => if rest.isEmpty then finish slices payload env out else out
  | fuel + 1, slices, rest, env, out => Id.run do
  if rest.isEmpty then return finish slices payload env out
  let pos := choose slices rest
  let v := rest[pos]!
  let rest := rest.eraseIdxIfInBounds pos
  let nextValues := if let some forced := v.uses.find? (fun c => slices[c.atom]!.rows.size == 1) then
    #[slices[forced.atom]!.rows[0]![forced.column]!]
  else Id.run do
    let mut smallest := v.uses[0]!
    for i in [1:v.uses.size] do
      let c := v.uses[i]!
      if slices[c.atom]!.columns[c.column]!.get.keys.size <
          slices[smallest.atom]!.columns[smallest.column]!.get.keys.size then smallest := c
    return slices[smallest.atom]!.columns[smallest.column]!.get.keys
  let mut out := out
  for value in nextValues do
    if rest.isEmpty && payload.isEmpty then
      if v.uses.all (fun c => ((slices[c.atom]!).lookup c.column value).isSome) then
        out := out.push (env.set! v.name (some value))
    else
      let mut next := slices
      let mut present := true
      for c in v.uses do
        let s := slices[c.atom]!
        let some child := s.lookup c.column value | present := false; break
        next := next.set! c.atom child
      if present then
        out := extend payload fuel next rest (env.set! v.name (some value)) out
  return out

def searchIndexed (inputs : Array (Array Nat × Index)) : Array Bindings := Id.run do
  if inputs.any (fun (_, t) => t.rows.isEmpty) then return #[]
  let vars := variables (inputs.map (·.1))
  let shared := vars.filter (·.uses.size > 1)
  let mut payload := #[]
  for i in [:inputs.size] do
    let fields := vars.filterMap fun v =>
      let c := v.uses[0]!
      if v.uses.size == 1 && c.atom == i then some (v.name, c.column) else none
    if !fields.isEmpty then payload := payload.push ⟨i, fields⟩
  -- Each variable is written exactly once. A singleton payload can pass this
  -- array directly to the next atom, without retaining a copy for later rows.
  let width := vars.foldl (fun n v => max n (v.name + 1)) 0
  return extend payload.toList shared.size (inputs.map (·.2)) shared (Array.replicate width none) #[]

def search (inputs : Array Input) : Array Bindings :=
  if inputs.any (fun t => t.rows.isEmpty) then #[]
  else searchIndexed (inputs.map (fun t => (t.vars, t.index)))

/-- Search each semi-naive variant against the same snapshot. The first atom
    using a new row owns the match: earlier atoms see old rows and later atoms
    see all rows. An unchanged atom has no split and retains its full index. -/
def searchDeltas (full : Array (Array Nat × Index)) (splitAt : Nat → Option (Index × Index)) :
    Array Bindings := Id.run do
  let mut inputs := full
  let mut hits := #[]
  for h : i in [:full.size] do
    if let some (old, recent) := splitAt i then
      let vars := full[i].1
      hits := hits ++ searchIndexed (inputs.set! i (vars, recent))
      inputs := inputs.set! i (vars, old)
  return hits

end Egglog.GenericJoin
