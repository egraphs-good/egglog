import Std.Data.HashMap
import Std.Data.HashSet

namespace Egglog

/-- A handle returned by `EGraph.add`. Handles stay valid through merges and rebuilds. -/
abbrev Id := Nat

structure ENode where
  op : Nat
  args : Array Id
  deriving BEq, Inhabited, Repr

instance : Hashable ENode where
  hash n := n.args.foldl (fun h id => mixHash h (hash id)) (hash n.op)

/-- Dense binary terms hold the constructor and both children in one machine
    word. The overflow arena also represents all other arities. -/
structure NodeRows where
  private data : Array Nat := #[]
  private other : Array ENode := #[]
  deriving Inhabited

namespace NodeRows

@[inline] def size (rows : @& NodeRows) : Nat := rows.data.size

@[inline] private def binary (v : Nat) : Bool := v.toUInt64 &&& 1 == 0
@[inline] private def wide (v : Nat) : Nat := v >>> 1

@[inline] private def fits (op a b : Nat) : Bool :=
  op < 64 && a < 268435456 && b < 268435456

@[inline] private def pack (op a b : Nat) : Nat :=
  ((op.toUInt64 <<< 57) ||| (a.toUInt64 <<< 29) ||| (b.toUInt64 <<< 1)).toNat

@[inline] private def left (v : Nat) : Nat := ((v.toUInt64 >>> 29) &&& 268435455).toNat
@[inline] private def right (v : Nat) : Nat := ((v.toUInt64 >>> 1) &&& 268435455).toNat
@[inline] private def symbol (v : Nat) : Nat := (v.toUInt64 >>> 57).toNat

@[inline] private def row (rows : @& NodeRows) (i : Nat) : ENode :=
  let v := rows.data[i]!
  if binary v then ⟨symbol v, #[left v, right v]⟩ else rows.other[wide v]!

/-- Scalar column access avoids reconstructing the application during matching. -/
@[inline] def child (rows : @& NodeRows) (i col : Nat) : Id :=
  let v := rows.data[i]!
  if binary v then
    if col == 0 then left v else if col == 1 then right v else panic! "child column out of range"
  else rows.other[wide v]!.args[col]!

@[inline] def arity (rows : @& NodeRows) (i : Nat) : Nat :=
  let v := rows.data[i]!
  if binary v then 2 else rows.other[wide v]!.args.size

@[inline] def op (rows : @& NodeRows) (i : Nat) : Nat :=
  let v := rows.data[i]!
  if binary v then symbol v else rows.other[wide v]!.op

@[inline] private def pushBinary (rows : NodeRows) (op a b : Nat) : NodeRows :=
  if fits op a b then { rows with data := rows.data.push (pack op a b) }
  else ⟨rows.data.push (rows.other.size * 2 + 1), rows.other.push ⟨op, #[a, b]⟩⟩

instance : GetElem NodeRows Nat ENode (fun xs i => i < xs.size) where
  getElem rows i _ := rows.row i
instance : GetElem? NodeRows Nat ENode (fun xs i => i < xs.size) where
  getElem? rows i := if i < rows.size then some (rows.row i) else none
  getElem! rows i := rows.row i

def toList (rows : NodeRows) : List ENode := (List.range rows.size).map rows.row

@[inline] private def push (rows : NodeRows) (n : ENode) : NodeRows :=
  if n.args.size == 2 && fits n.op n.args[0]! n.args[1]! then
    { rows with data := rows.data.push (pack n.op n.args[0]! n.args[1]!) }
  else ⟨rows.data.push (rows.other.size * 2 + 1), rows.other.push n⟩

@[noinline] private def set! (rows : NodeRows) (i : Nat) (n : ENode) : NodeRows :=
  let old := rows.data[i]!
  if n.args.size == 2 && fits n.op n.args[0]! n.args[1]! then
    let data := rows.data.set! i (pack n.op n.args[0]! n.args[1]!)
    let other := if binary old then rows.other else rows.other.set! (wide old) default
    ⟨data, other⟩
  else if binary old then
    ⟨rows.data.set! i (rows.other.size * 2 + 1), rows.other.push n⟩
  else { rows with other := rows.other.set! (wide old) n }

@[inline] private def sameBinary (rows : @& NodeRows) (i op a b : Nat) : Bool :=
  let v := rows.data[i]!
  if binary v then fits op a b && v == pack op a b else
    let n := rows.other[wide v]!
    n.op == op && n.args.size == 2 && n.args[0]! == a && n.args[1]! == b

@[inline] private def same (rows : @& NodeRows) (i : Nat) (n : @& ENode) : Bool :=
  if n.args.size == 2 then sameBinary rows i n.op n.args[0]! n.args[1]!
  else
    let v := rows.data[i]!
    !binary v && rows.other[wide v]! == n

@[inline] private def changed (rows : @& NodeRows) (i : Nat) (reps : @& Array Id) : Bool :=
  let v := rows.data[i]!
  if binary v then
    let a := left v
    let b := right v
    reps[a]! != a || reps[b]! != b
  else rows.other[wide v]!.args.any (fun a => reps[a]! != a)

end NodeRows

/-- Open addressing with linear probing. A small row ID occupies the low
    32 bits and a hash fingerprint occupies the next 31, all in one immediate
    Lean value on 64-bit targets. A zero low word selects a wide row ID.
    Fingerprint hits are always checked against the complete enode. -/
private structure NodeTable where
  slots : Array Nat := Array.replicate 16 0
  used : Nat := 0  -- includes tombstones
  live : Nat := 0
  deriving Inhabited

namespace NodeTable

@[inline] private def foldHash (h : UInt64) : UInt32 := (h ^^^ (h >>> 32)).toUInt32
@[inline] private def wordHash (op a b : Nat) : UInt32 :=
  let x := op.toUInt64 * 0x9e3779b185ebca87 +
           a.toUInt64 * 0x165667b19e3779f9 +
           b.toUInt64 * 0x85ebca77c2b2ae63
  (((x ^^^ (x >>> 29)) * 0x9e3779b185ebca87) >>> 33).toUInt32

@[inline] private def hash32 (n : @& ENode) : UInt32 :=
  if n.args.size == 2 then wordHash n.op n.args[0]! n.args[1]!
  else (foldHash (hash n)) &&& 0x7fffffff

@[inline] private def pack (key : UInt32) (id : Id) : Nat :=
  if id < 2147483647 then ((key.toUInt64 <<< 32) ||| (id.toUInt64 + 2)).toNat
  else ((id + 1) <<< 63) + (key.toNat <<< 32)
@[inline] private def fingerprint (entry : Nat) : UInt32 := (entry.toUInt64 >>> 32).toUInt32 &&& 0x7fffffff
@[inline] private def unpack (entry : Nat) : Id :=
  if entry.toUInt32 != 0 then (entry.toUInt32 - 2).toNat else (entry >>> 63) - 1

/-- Return a full-key hit or the first reusable position in the probe chain.
    Growth keeps at least one empty (zero) slot. One encodes a tombstone. -/
@[specialize] private partial def probe (slots : @& Array Nat) (same : Id → Bool)
    (key : UInt32) (mask pos firstFree : USize) : Nat :=
  let entry := slots[pos.toNat]!
  if entry == 0 then
    (if firstFree ≤ mask then firstFree else pos).toNat
  else if entry == 1 then
    probe slots same key mask ((pos + 1) &&& mask) (if firstFree ≤ mask then firstFree else pos)
  else if fingerprint entry == key && same (unpack entry) then pos.toNat
  else probe slots same key mask ((pos + 1) &&& mask) firstFree

@[inline] private def positionFor (t : @& NodeTable) (key : UInt32) (same : Id → Bool) : Nat :=
  let mask := t.slots.usize - 1
  probe t.slots same key mask (key.toUSize &&& mask) t.slots.usize

@[inline] private def position (t : @& NodeTable) (nodes : @& NodeRows)
    (n : @& ENode) (key : UInt32) : Nat :=
  positionFor t key (fun i => nodes.same i n)

private def get? (t : @& NodeTable) (nodes : @& NodeRows) (n : @& ENode) : Option Id :=
  let entry := t.slots[position t nodes n (hash32 n)]!
  if entry > 1 then some (unpack entry) else none

private partial def insertUnique (slots : Array Nat) (entry : Nat) (mask pos : USize) : Array Nat :=
  if slots[pos.toNat]! == 0 then slots.set! pos.toNat entry
  else insertUnique slots entry mask ((pos + 1) &&& mask)

private def rehash (t : NodeTable) (cap : Nat) : NodeTable := Id.run do
  let mask := cap.toUSize - 1
  let mut slots := Array.replicate cap 0
  for old in t.slots do
    if old > 1 then slots := insertUnique slots old mask ((fingerprint old).toUSize &&& mask)
  return ⟨slots, t.live, t.live⟩

@[inline] private def insertAt (t : NodeTable) (pos old : Nat) (key : UInt32) (id : Id) : NodeTable :=
  let entry := pack key id
  if old == 0 && (t.used + 1) * 4 > t.slots.size * 3 then
    let cap := if t.live * 2 < t.slots.size then t.slots.size else t.slots.size * 2
    let t := rehash t cap
    let mask := cap.toUSize - 1
    ⟨insertUnique t.slots entry mask (key.toUSize &&& mask), t.live + 1, t.live + 1⟩
  else ⟨t.slots.set! pos entry, t.used + (if old == 0 then 1 else 0), t.live + 1⟩

@[inline] private def getThenInsertIfNew? (t : NodeTable) (nodes : @& NodeRows) (n : @& ENode)
    (id : Id) : Option Id × NodeTable :=
  let key := hash32 n
  let pos := position t nodes n key
  let old := t.slots[pos]!
  if old > 1 then (some (unpack old), t) else (none, insertAt t pos old key id)

private def eraseOwner (t : NodeTable) (nodes : @& NodeRows) (n : @& ENode) (id : Id) :
    Bool × NodeTable :=
  let key := hash32 n
  let pos := position t nodes n key
  if t.slots[pos]! == pack key id then (true, { t with slots := t.slots.set! pos 1, live := t.live - 1 })
  else (false, t)

end NodeTable

private structure SmallTable where
  /-- Common constructor tables grow independently. -/
  byOp : Array (Std.HashMap (Array Id) Id) := Array.replicate 64 {}
  other : Std.HashMap ENode Id := {}
  size : Nat := 0
  deriving Inhabited

namespace SmallTable

@[inline] private def get? (t : @& SmallTable) (n : @& ENode) : Option Id :=
  if h : n.op < t.byOp.size then t.byOp[n.op].get? n.args else t.other.get? n

private def getThenInsertIfNew? (t : SmallTable) (n : ENode) (id : Id) : Option Id × SmallTable :=
  if h : n.op < t.byOp.size then
    let rows := t.byOp[n.op]
    let ops := t.byOp.set n.op {}
    let (old, rows) := rows.getThenInsertIfNew? n.args id
    (old, { t with byOp := ops.set! n.op rows, size := if old.isSome then t.size else t.size + 1 })
  else
    let (old, other) := t.other.getThenInsertIfNew? n id
    (old, { t with other, size := if old.isSome then t.size else t.size + 1 })

private def erase (t : SmallTable) (n : @& ENode) : SmallTable :=
  if n.op < t.byOp.size then { t with byOp := t.byOp.modify n.op (·.erase n.args), size := t.size - 1 }
  else { t with other := t.other.erase n, size := t.size - 1 }

end SmallTable

private inductive HashCons where
  | small (entries : SmallTable)
  | flat (entries : NodeTable)
  deriving Inhabited

namespace HashCons

private abbrev smallLimit := 32768

private def promote (entries : SmallTable) (estimated : Nat) : NodeTable := Id.run do
  let cap := Nat.nextPowerOfTwo ((max entries.size estimated * 4 + 2) / 3)
  let mask := cap.toUSize - 1
  let mut slots := Array.replicate cap 0
  for op in [:entries.byOp.size] do
    for (args, id) in entries.byOp[op]! do
      let key := NodeTable.hash32 ⟨op, args⟩
      slots := NodeTable.insertUnique slots (NodeTable.pack key id) mask (key.toUSize &&& mask)
  for (n, id) in entries.other do
    let key := NodeTable.hash32 n
    slots := NodeTable.insertUnique slots (NodeTable.pack key id) mask (key.toUSize &&& mask)
  return ⟨slots, entries.size, entries.size⟩

private def reserve (t : HashCons) (estimated : Nat) : HashCons :=
  match t with
  | .small m => if estimated < smallLimit then t else .flat (promote m estimated)
  | .flat m =>
    if estimated * 4 ≤ m.slots.size * 3 then t
    else .flat (m.rehash (Nat.nextPowerOfTwo ((estimated * 4 + 2) / 3)))

@[inline] private def get? (t : @& HashCons) (nodes : @& NodeRows) (n : @& ENode) : Option Id :=
  match t with
  | .small m => m.get? n
  | .flat m => m.get? nodes n

@[inline] private def getThenInsertIfNew? (t : HashCons) (nodes : @& NodeRows)
    (n : ENode) (id : Id) : Option Id × HashCons :=
  match t with
  | .small m =>
    let (prev, m) := m.getThenInsertIfNew? n id
    if m.size < smallLimit then (prev, .small m) else (prev, .flat (promote m (m.size * 2)))
  | .flat m =>
    let (prev, m) := m.getThenInsertIfNew? nodes n id
    (prev, .flat m)

private def eraseOwner (t : HashCons) (nodes : @& NodeRows) (n : @& ENode) (id : Id) :
    Bool × HashCons :=
  match t with
  | .small m =>
    if m.get? n == some id then (true, .small (m.erase n)) else (false, t)
  | .flat m =>
    let (found, m) := m.eraseOwner nodes n id
    (found, .flat m)

end HashCons

/-- Single-threaded congruence closure over uninterpreted, variadic constructor symbols.
    Child and class handles passed to the API must come from this graph. -/
structure EGraph where
  nodes : NodeRows := {}
  private reps : Array Id := #[]
  /-- A root with an empty entry is a singleton represented by its own ID. -/
  private classes : Array (Array Id) := #[]
  /-- Retired congruent handles still follow their class across future unions. -/
  private aliases : Array (Array Id) := #[]
  private uses : Array (Array Id) := #[]
  private smallOps : Array (Array Id) := Array.replicate 64 #[]
  private byOp : Std.HashMap Nat (Array Id) := {}
  private table : HashCons := .small {}
  private dirty : Array Id := #[]
  private rebuiltCount : Nat := 0
  private hidden : Std.HashSet Id := {}
  private deleted : Std.HashSet Id := {}
  /-- Visibility changes invalidate cached query row ranges. -/
  maskVersion : Nat := 0
  classCount : Nat := 0
  /-- Unions that changed an existing representative. A newly inserted row
      attached directly to a target leaves this counter unchanged. -/
  mergeCount : Nat := 0
  /-- Increases exactly once per new node and once per successful union. -/
  version : Nat := 0
  deriving Inhabited

namespace EGraph

def empty : EGraph := {}

/-- Reserve the hash-cons slots for a batch. Existing slots are kept whenever
    the expected total fits at the table's normal load factor. -/
def reserveNodes (g : EGraph) (estimatedTotal : Nat) : EGraph :=
  { g with table := g.table.reserve estimatedTotal }


/-- Smaller classes are relabeled eagerly during union, so find takes one array lookup. -/
@[inline] def find (g : @& EGraph) (id : Id) : Id := g.reps[id]!

/-- The rebuilt class of each handle at a round boundary. Later insertions and
    unions preserve this snapshot for delta joins against the previous round. -/
@[inline] def representatives (g : @& EGraph) : Array Id := g.reps

/-- Rust exposes the oldest handle as its canonical key for table merges. -/
def oldest (g : @& EGraph) (id : Id) : Id :=
  let root := g.find id
  let liveMin := g.classes[root]!.foldl min root
  if g.aliases.isEmpty then liveMin else g.aliases[root]!.foldl min liveMin

/-- Distinct application rows in the rebuilt class. Original handles remain in `nodes`. -/
@[inline] private def allMembers (g : @& EGraph) (id : Id) : Array Id :=
  let root := g.find id
  let ids := g.classes[root]!
  if ids.isEmpty then #[root] else ids

@[inline] private def allRows (g : @& EGraph) (op : Nat) : Array Id :=
  if h : op < g.smallOps.size then g.smallOps[op] else g.byOp.getD op #[]

/-- Stored row count, including masked rows. This upper bound is available
    without materializing the visible-row index. -/
@[inline] def storedRowCount (g : @& EGraph) (op : Nat) : Nat := (g.allRows op).size

/-- Subsumed rows stay in their function table and can still be addressed
    by the actions of an already matched rule. They do not support new matches. -/
@[inline] def opNodes (g : @& EGraph) (op : Nat) : Array Id :=
  if g.maskVersion == 0 then g.allRows op else (g.allRows op).filter (!g.hidden.contains ·)

@[inline] def members (g : @& EGraph) (id : Id) : Array Id :=
  if g.maskVersion == 0 then g.allMembers id else (g.allMembers id).filter (!g.hidden.contains ·)

def tableRows (g : @& EGraph) (op : Nat) : Array Id :=
  if g.deleted.isEmpty then g.allRows op else (g.allRows op).filter (!g.deleted.contains ·)

/-- Intersect the class and constructor indices by scanning the shorter one. -/
def matchingNodes (g : @& EGraph) (id : Id) (op : Nat) : Array Id :=
  let root := g.find id
  let inClass := g.allMembers root
  let withOp := g.allRows op
  if inClass.size ≤ withOp.size then
    inClass.filter (fun i => g.nodes.op i == op && (g.maskVersion == 0 || !g.hidden.contains i))
  else
    withOp.filter (fun i => g.find i == root && (g.maskVersion == 0 || !g.hidden.contains i))

/-- Materialize reverse edges for extraction or sparse congruence propagation.
    Capture stale children as well when indexing before a pending rebuild. -/
def indexParents (g : EGraph) : EGraph := Id.run do
  if g.uses.size == g.nodes.size then return g
  let mut uses : Array (Array Id) := Array.replicate g.nodes.size #[]
  let mut dirty := #[]
  for i in [:g.nodes.size] do
    if g.nodes.changed i g.reps then dirty := dirty.push i
    for col in [:g.nodes.arity i] do
      uses := uses.modify (g.find (g.nodes.child i col)) (·.push i)
  return { g with uses, dirty }


/-- One parent enode per occurrence as a child. Call `indexParents` once before
    iterating over many classes. -/
@[inline] def parentNodes (g : @& EGraph) (id : Id) : Array Id :=
  g.indexParents.uses[g.find id]!

@[inline] private def canonical (g : @& EGraph) (n : ENode) : ENode := Id.run do
  let mut args := n.args
  for i in [:args.size] do
    let old := args[i]!
    let root := g.find old
    if old != root then args := args.set! i root
  return { n with args }

/-- Query an application without inserting it. Rebuild first to see all congruences. -/
def lookup? (g : @& EGraph) (n : ENode) : Option Id :=
  (g.table.get? g.nodes (canonical g n)).map (fun id => g.find id)

/-- Subsumption leaves the constructor's hash-cons entry in place. Deletion
    frees its key so a later insertion creates a fresh class. -/
def changeRow (g : EGraph) (node : ENode) (subsume : Bool) : EGraph := Id.run do
  -- A rule can union two arguments before deleting or subsuming a row.
  -- Existing rows still use the read phase's keys until rebuild. Prefer
  -- that owner over an already canonical, distinct application.
  let current := canonical g node
  let owner := (g.table.get? g.nodes node).orElse fun _ =>
    (g.table.get? g.nodes current).orElse fun _ =>
      if g.rebuiltCount == g.mergeCount then none else
      (g.tableRows node.op).find? fun id => canonical g g.nodes[id]! == current
  let some id := owner | return g
  if subsume && g.hidden.contains id then return g
  let g := { g with hidden := g.hidden.insert id, maskVersion := g.maskVersion + 1, version := g.version + 1 }
  if subsume then return g
  let (_, table) := g.table.eraseOwner g.nodes g.nodes[id]! id
  return { g with table, deleted := g.deleted.insert id }

/-- Extraction must also exclude old duplicate row handles pointing at a
    now-subsumed surviving owner. -/
def extractable (g : @& EGraph) (id : Nat) : Bool :=
  g.maskVersion == 0 || (!g.hidden.contains id &&
    g.table.get? g.nodes (canonical g g.nodes[id]!) == some id)

def lookupActive? (g : @& EGraph) (n : ENode) : Option Id := do
  let owner ← g.table.get? g.nodes (canonical g n)
  guard (!g.hidden.contains owner)
  return g.find owner

/-- Register a freshly hash-consed row. Its class can be either a new singleton
    or an existing target when a rewrite creates its right-hand side. -/
@[specialize] private def register (g : EGraph) (root op : Id) (append : NodeRows → NodeRows) : EGraph := Id.run do
  let id := g.nodes.size
  let fresh := root == id
  let classes := if fresh then g.classes.push #[]
    else (g.classes.modify root (fun ids => if ids.isEmpty then #[root, id] else ids.push id)).push #[]
  let aliases := if g.aliases.isEmpty then g.aliases else g.aliases.push #[]
  let (smallOps, byOp) := if op < g.smallOps.size then
      (g.smallOps.modify op (·.push id), g.byOp)
    else (g.smallOps, g.byOp.alter op (fun ids => some ((ids.getD #[]).push id)))
  return { g with
    nodes := append g.nodes
    reps := g.reps.push root
    classes, aliases
    uses := #[], dirty := #[], smallOps, byOp
    classCount := if fresh then g.classCount + 1 else g.classCount
    version := g.version + (if fresh then 1 else 2) }

@[inline] private def insertRow (g : EGraph) (n : ENode) (root : Id) : EGraph :=
  register g root n.op (·.push n)

@[inline] private def insertPair (g : EGraph) (op a b root : Id) : EGraph :=
  register g root op (·.pushBinary op a b)

def add (g : EGraph) (n : ENode) : Id × EGraph :=
  let n := canonical g n
  let id := g.nodes.size
  let (previous, table) := g.table.getThenInsertIfNew? g.nodes n id
  let g := { g with table }
  match previous with
  | some old => (g.find old, g)
  | none => (id, insertRow g n id)

private def mergeInto (g : EGraph) (big small : Id) : EGraph := Id.run do
  let moved := g.allMembers small
  let mut reps := g.reps
  for id in moved do reps := reps.set! id big
  let mut aliases := g.aliases
  if !aliases.isEmpty then
    let old := aliases[small]!
    if !old.isEmpty then
      for id in old do reps := reps.set! id big
      aliases := (aliases.set! small #[]).modify big (· ++ old)
  let (uses, dirty) := if g.uses.isEmpty then (#[], #[]) else
    let parents := g.uses[small]!
    ((g.uses.set! small #[]).modify big (· ++ parents), g.dirty ++ parents)
  return { g with
    reps
    aliases
    classes := (g.classes.set! small #[]).modify big (fun ids => (if ids.isEmpty then #[big] else ids) ++ moved)
    uses, dirty
    classCount := g.classCount - 1
    mergeCount := g.mergeCount + 1
    version := g.version + 1 }

/-- Merge two classes. When a reverse index exists, queue the affected parents. -/
def merge (g : EGraph) (a b : Id) : EGraph :=
  let a := g.find a
  let b := g.find b
  if a == b then g
  else if max 1 g.classes[a]!.size + (if g.aliases.isEmpty then 0 else g.aliases[a]!.size) <
          max 1 g.classes[b]!.size + (if g.aliases.isEmpty then 0 else g.aliases[b]!.size) then mergeInto g b a
  else mergeInto g a b

/-- Add an application to an existing class. On a hash-cons miss the fresh
    application can join the target directly since it has no parents yet. -/
def addToClass (g : EGraph) (target : Id) (n : ENode) : EGraph :=
  let root := g.find target
  let n := canonical g n
  let (previous, table) := g.table.getThenInsertIfNew? g.nodes n g.nodes.size
  let g := { g with table }
  match previous with
  | some id => g.merge root id
  | none => insertRow g n root

/-- Keep the table separate from the graph while probing so the native array
    can be updated in place when the caller owns the graph uniquely. -/
private def insertBinary (t : HashCons) (nodes : @& NodeRows) (op a b id : Nat) :
    Option Id × HashCons :=
  match t with
  | .flat flat =>
    let key := NodeTable.wordHash op a b
    let pos := if NodeRows.fits op a b then
        let word := NodeRows.pack op a b
        flat.positionFor key (fun i => nodes.data[i]! == word)
      else flat.positionFor key (fun i => nodes.sameBinary i op a b)
    let old := flat.slots[pos]!
    if old > 1 then (some (NodeTable.unpack old), .flat flat)
    else (none, .flat (flat.insertAt pos old key id))
  | .small _ => t.getThenInsertIfNew? nodes ⟨op, #[a, b]⟩ id

/-- Binary application fast paths avoid constructing argument arrays for hits
    in the flat table. -/
@[inline] private def withBinary (g : EGraph) (op a b : Nat) (onHit : Id → EGraph → α)
    (onMiss : Id → Id → EGraph → α) : α :=
  let a := g.find a
  let b := g.find b
  let (previous, table) := insertBinary g.table g.nodes op a b g.nodes.size
  let g := { g with table }
  match previous with
  | some old => onHit old g
  | none => onMiss a b g

def add2 (g : EGraph) (op a b : Nat) : Id × EGraph :=
  withBinary g op a b (fun id g => (g.find id, g)) (fun a b g =>
    let id := g.nodes.size
    (id, insertPair g op a b id))

def add2ToClass (g : EGraph) (target op a b : Nat) : EGraph :=
  withBinary g op a b (fun id g => g.merge target id) (fun a b g => insertPair g op a b (g.find target))

private def compactRows (g : EGraph) (retired : Array Id) : EGraph := Id.run do
  if retired.isEmpty then return g
  let mut aliases := if g.aliases.isEmpty then Array.replicate g.nodes.size #[] else g.aliases
  let mut touched := #[]
  let mut seen := ByteArray.mk (Array.replicate g.nodes.size 0)
  -- Bit zero marks affected classes. Bit one marks retired rows.
  let mut ops : Std.HashSet Nat := {}
  let mut hidden := g.hidden
  for id in retired do
    if !hidden.isEmpty && hidden.contains id then
      if let some owner := g.table.get? g.nodes (canonical g g.nodes[id]!) then
        hidden := hidden.insert owner
    let root := g.find id
    aliases := aliases.modify root (·.push id)
    if seen[root]! &&& 1 == 0 then
      touched := touched.push root
      seen := seen.set! root (seen[root]! ||| 1)
    seen := seen.set! id (seen[id]! ||| 2)
    ops := ops.insert (g.nodes.op id)
  let live := fun id => seen[id]! &&& 2 == 0
  let mut classes := g.classes
  for root in touched do
    classes := classes.modify root (fun ids => (if ids.isEmpty then #[root] else ids).filter live)
  let mut byOp := g.byOp
  let mut smallOps := g.smallOps
  for op in ops do
    if op < smallOps.size then smallOps := smallOps.modify op (·.filter live)
    else byOp := byOp.modify op (·.filter live)
  return { g with classes, aliases, byOp, smallOps, hidden }

-- Supply the canonical row by value so its calculation releases the borrowed
-- graph before the hash-cons table is updated in place.
@[noinline] private def repairOne (g : EGraph) (id : Id) (old n : ENode) (retired : Array Id) :
    EGraph × Array Id := Id.run do
  let (removed, table) := g.table.eraseOwner g.nodes old id
  let g := { g with nodes := g.nodes.set! id n, table }
  if !removed then return (g, retired)
  let (previous, table) := g.table.getThenInsertIfNew? g.nodes n id
  let g := { g with table }
  if let some other := previous then return (g.merge id other, retired.push id)
  return (g, retired)

-- Probe the old slot by row ID. The new packed key then has an ordinary
-- full-word probe. Neither operation needs to allocate ENode or argument arrays.
@[noinline] private def rekeyPacked (flat : NodeTable) (nodes : NodeRows) (id old a b op : Nat)
    : NodeTable × NodeRows × Option Id := Id.run do
  let oldHash := NodeTable.wordHash op (NodeRows.left old) (NodeRows.right old)
  let oldPos := flat.positionFor oldHash (fun owner => owner == id)
  let owner := flat.slots[oldPos]! == NodeTable.pack oldHash id
  let word := NodeRows.pack op a b
  let nodes := { nodes with data := nodes.data.set! id word }
  if !owner then return (flat, nodes, none)
  let flat := { flat with slots := flat.slots.set! oldPos 1, live := flat.live - 1 }
  let hash := NodeTable.wordHash op a b
  let pos := flat.positionFor hash (fun i => nodes.data[i]! == word)
  let entry := flat.slots[pos]!
  if entry > 1 then return (flat, nodes, some (NodeTable.unpack entry))
  return (flat.insertAt pos entry hash id, nodes, none)

@[noinline] private def repairId (g : EGraph) (id : Id) (retired : Array Id) : EGraph × Array Id := Id.run do
  let raw := g.nodes.data[id]!
  if NodeRows.binary raw then
    if let .flat flat := g.table then
      let op := NodeRows.symbol raw
      let a := g.find (NodeRows.left raw)
      let b := g.find (NodeRows.right raw)
      if NodeRows.fits op a b then
        let (flat, nodes, hit) := rekeyPacked flat g.nodes id raw a b op
        let g := { g with nodes, table := .flat flat }
        if let some other := hit then return (g.merge id other, retired.push id)
        return (g, retired)
  let old := g.nodes[id]!
  return repairOne g id old (canonical g old) retired

/-- Repair densely growing batches by scanning the packed rows. Once congruences
    propagate through six successive sweeps, switch to an indexed parent worklist. -/
partial def rebuild (g : EGraph) : EGraph := Id.run do
  let mut g := g
  let mut retired := #[]
  let mut sweeps := 0
  while g.rebuiltCount != g.mergeCount do
    if !g.uses.isEmpty || sweeps ≥ 6 then
      -- Very deep, sparsely propagating congruences use the reverse-edge
      -- worklist. A scan would revisit every row once per level.
      g := g.indexParents
      while !g.dirty.isEmpty do
        let id := g.dirty.back!
        g := { g with dirty := g.dirty.pop }
        if g.nodes.changed id g.reps then
          let (next, dead) := repairId g id retired
          g := next
          retired := dead
      g := { g with rebuiltCount := g.mergeCount }
    else
      sweeps := sweeps + 1
      g := { g with rebuiltCount := g.mergeCount }
      for id in [:g.nodes.size] do
        if g.nodes.changed id g.reps then
          let (next, dead) := repairId g id retired
          g := next
          retired := dead
  return compactRows g retired

end EGraph
end Egglog
