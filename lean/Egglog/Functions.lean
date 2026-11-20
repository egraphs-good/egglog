import Egglog.Parser

namespace Egglog.Relational

inductive MergePolicy where
  | noMerge | old | new | min | max
  deriving Inhabited, BEq

/-- A partial function over canonical class IDs. Numeric merges compare the i64
    values represented by the result classes. -/
structure FunTable where
  inputs : Array String
  result : String
  merge : MergePolicy
  entries : Std.HashMap (Array Id) Id := {}
  keys : Array (Array Id) := #[]
  version : Nat := 0
  deletedVersion : Nat := 0
  deriving Inhabited

abbrev Functions := Std.HashMap String FunTable

private def application : SExpr → Except String (String × Array SExpr)
  | .list xs => do
    let some (SExpr.atom name) := xs[0]? | throw "expected an application"
    return (name, xs.extract 1 xs.size)
  | _ => throw "expected an application"

private def parseMerge (spec : List SExpr) (result : String) : Except String MergePolicy := do
  match spec with
  | [.atom ":no-merge"] => return .noMerge
  | [.atom ":merge", .atom "old"] => return .old
  | [.atom ":merge", .atom "new"] => return .new
  | [.atom ":merge", term] =>
    let (op, args) ← application term
    let [.atom a, .atom b] := args.toList | throw "expected a binary merge of old and new"
    unless (a == "old" && b == "new") || (a == "new" && b == "old") do
      throw "expected old and new as merge operands"
    unless result == "i64" do throw s!"{op} merge requires i64 output"
    match op with
    | "min" => return .min
    | "max" => return .max
    | _ => throw s!"unsupported merge operator {op}"
  | _ => throw "expected :no-merge or :merge with an explicit merge expression"

/-- Parse the argument list following the `function` keyword. The caller checks
    the name against the relational database's existing declarations. -/
def declareFun (s : ProgramState) (args : Array SExpr) : Except String (String × FunTable) := do
  let .atom name :: .list ins :: .atom out :: spec := args.toList | throw "invalid function declaration"
  let inputs ← ins.mapM fun
    | .atom name => pure name
    | _ => throw "expected a sort name"
  for sort in inputs.push out do
    unless s.sorts.contains sort do throw s!"unknown sort {sort}"
  return (name, { inputs, result := out, merge := ← parseMerge spec out })

private def readI64 (g : EGraph) (id : Id) : Except String Int :=
  match decodeInt? g.nodes[id]!.op with
  | some x => .ok x
  | none => .error "expected an i64 literal in merge"

private def mergeResult (policy : MergePolicy) (g : EGraph) (old new : Id) : Except String Id := do
  if old == new then return old
  match policy with
  | .noMerge => throw "conflicting values for :no-merge function"
  | .old => return old
  | .new => return new
  | .min => return if (← readI64 g old) ≤ (← readI64 g new) then old else new
  | .max => return if (← readI64 g old) ≥ (← readI64 g new) then old else new

def FunTable.set (t : FunTable) (g : EGraph) (key : Array Id) (value : Id) :
    Except String FunTable := do
  unless key.size == t.inputs.size do throw "wrong function arity"
  let key := key.map g.oldest
  let value := g.find value
  if let some prev := t.entries.get? key then
    let value ← mergeResult t.merge g (g.find prev) value
    if value == prev then return t
    return { t with entries := t.entries.insert key value, version := t.version + 1 }
  return { t with entries := t.entries.insert key value, keys := t.keys.push key, version := t.version + 1 }

def FunTable.erase (t : FunTable) (g : EGraph) (key : Array Id) : FunTable := Id.run do
  let key := key.map g.oldest
  if !t.entries.contains key then return t
  return { t with keys := t.keys.filter (· != key), entries := t.entries.erase key,
                  version := t.version + 1, deletedVersion := t.deletedVersion + 1 }

/-- Merges can collapse distinct keys or change result representatives. Apply
    the declared merge policy again to colliding rows. -/
def rebuildFunctions (fs : Functions) (g : EGraph) : Except String Functions := do
  let mut out : Functions := {}
  for (name, t) in fs do
    let mut clean := { t with entries := {}, keys := #[], version := 0 }
    let mut moved := #[]
    -- Preserve resident rows as the old side of a merge. Reinsert changed keys
    -- in row order so a hash map's traversal order cannot choose the result.
    for key in t.keys do
      if key.all (fun id => g.oldest id == id) then
        clean ← clean.set g key t.entries[key]!
      else moved := moved.push key
    for key in moved do clean ← clean.set g key t.entries[key]!
    out := out.insert name { clean with version := t.version }
  return out

private def groundArgs (eval : ProgramState → SExpr → Bool → Except String (Id × String × ProgramState))
    (s : ProgramState) (args : Array SExpr) (sorts : Array String) (insert : Bool) :
    Except String (Array Id × ProgramState) := do
  unless args.size == sorts.size do throw "wrong application arity"
  let mut s := s
  let mut ids := Array.emptyWithCapacity args.size
  for x in args, expected in sorts do
    let (id, sort, next) ← eval s x insert
    unless sort == expected do throw s!"expected {expected}, found {sort}"
    s := next
    ids := ids.push id
  return (ids, s)

/-- Lookups of partial functions require an existing row, including when they
    occur in a term whose constructor applications may be inserted. -/
partial def groundWithFunctions (s : ProgramState) (fs : Functions) (term : SExpr) (insert : Bool) :
    Except String (Id × String × ProgramState) := do
  if fs.isEmpty then return ← ground s term insert
  match term with
  | .atom _ => ground s term insert
  | .list _ =>
    let (name, args) ← application term
    let eval := fun st a ins => groundWithFunctions st fs a ins
    if let some t := fs.get? name then
      let (key, s) ← groundArgs eval s args t.inputs insert
      let some id := t.entries.get? (key.map s.graph.oldest) | throw s!"no entry in function {name}"
      return (s.graph.find id, t.result, s)
    if let some op := arithmeticOp? name then
      let (ids, s) ← groundArgs eval s args #["i64", "i64"] true
      let lit ← calculate op s.graph.nodes[ids[0]!]!.op s.graph.nodes[ids[1]!]!.op
      let node : ENode := ⟨lit, #[]⟩
      let (id, g) := s.graph.add node
      return (id, "i64", { s with graph := g })
    let some c := s.constructors.get? name | throw s!"unknown constructor or function {name}"
    let (ids, s) ← groundArgs eval s args c.inputs insert
    let n : ENode := ⟨c.op, ids⟩
    if insert then
      let (id, g) := s.graph.add n
      return (id, c.result, { s with graph := g })
    let some id := s.graph.lookup? n | throw "ground term is absent"
    return (id, c.result, s)

def setGround (s : ProgramState) (fs : Functions) (app rhs : SExpr) :
    Except String (ProgramState × Functions) := do
  let (name, args) ← application app
  let some t := fs.get? name | throw s!"unknown function {name}"
  let (key, s) ← groundArgs (fun st x ins => groundWithFunctions st fs x ins) s args t.inputs true
  let (value, sort, s) ← groundWithFunctions s fs rhs true
  unless t.result == sort do throw s!"function {name} returns {t.result}, found {sort}"
  return (s, fs.insert name (← t.set s.graph key value))

end Egglog.Relational
