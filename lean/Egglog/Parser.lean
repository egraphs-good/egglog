import Egglog.Compiled
import Egglog.ArithmeticPrim
import Egglog.Extract
import Egglog.FloatLiteral

namespace Egglog

inductive SExpr where
  | atom (s : String)
  | list (items : Array SExpr)
  deriving Inhabited, Repr

/-- Strings are atoms with their outer quotes preserved and their contents decoded. -/
def parseSExprs (src : String) : Except String (Array SExpr) := do
  let mut stack : List (Array SExpr) := []
  let mut items := #[]
  let mut word := ""
  let mut comment := false
  let mut quoted := false
  let mut escape := false
  for c in src.toList do
    if comment then
      if c == '\n' then comment := false
      continue
    if quoted then
      if escape then
        let decoded ← match c with
          | 'n' => pure '\n' | 't' => pure '\t' | '"' => pure '"' | '\\' => pure '\\'
          | _ => throw s!"unknown string escape {c}"
        word := word.push decoded
        escape := false
      else if c == '\\' then escape := true
      else if c == '"' then
        items := items.push (.atom (word.push '"'))
        word := ""
        quoted := false
      else word := word.push c
      continue
    if c == '(' || c == ')' || c == ';' || c == '"' || c.isWhitespace then
      if !word.isEmpty then
        items := items.push (.atom word)
        word := ""
      match c with
      | '(' => stack := items :: stack; items := #[]
      | ')' =>
        let outer :: rest := stack | throw "unexpected ')'"
        items := outer.push (.list items)
        stack := rest
      | ';' => comment := true
      | '"' => quoted := true; word := "\""
      | _ => pure ()
    else word := word.push c
  if quoted then throw "unterminated string"
  if !stack.isEmpty then throw "missing ')'"
  if !word.isEmpty then items := items.push (.atom word)
  return items

structure Constructor where
  op : Nat
  inputs : Array String
  result : String
  cost : Nat := 1
  deriving Inhabited

structure RuleGroup where
  plain : Array Rewrite := #[]
  arithmetic : Array Rewrite := #[]
  conditional : Array GuardedRewrite := #[]
  deriving Inhabited

structure ProgramState where
  graph : EGraph := {}
  rewrites : Array Rewrite := #[]
  arithmeticRewrites : Array Rewrite := #[]
  conditional : Array GuardedRewrite := #[]
  rulesets : Std.HashMap String RuleGroup := {}
  combined : Std.HashMap String (Array String) := {}
  sorts : Std.HashSet String := .ofList ["i64", "String", "f64", "bool"]
  constructors : Std.HashMap String Constructor := {}
  names : Array String := #[]
  globals : Std.HashMap String (String × Id) := {}
  output : Array String := #[]
  deriving Inhabited

structure PatternVars where
  bindings : Std.HashMap String (Nat × String) := {}
  derived : Std.HashMap String (Pat × String) := {}
  next : Nat := 0

private def atomName : SExpr → Except String String
  | .atom s => if s.startsWith "\"" then throw "expected a name" else pure s
  | _ => throw "expected a name"

private def application : SExpr → Except String (String × Array SExpr)
  | .list xs => do
    let some hd := xs[0]? | throw "empty application"
    return (← atomName hd, xs.extract 1 xs.size)
  | _ => throw "expected an application"

private def constructorFor (st : ProgramState) (op : String) (arity : Nat) :
    Except String Constructor := do
  let some c := st.constructors.get? op | throw s!"unknown constructor {op}"
  unless c.inputs.size == arity do throw s!"{op} takes {c.inputs.size} arguments"
  return c

private def checkSort (expected : Option String) (actual : String) : Except String Unit := do
  if let some s := expected then
    unless s == actual do throw s!"expected sort {s}, found {actual}"

def primitive (s : String) := s == "i64" || s == "String" || s == "f64" || s == "bool"

-- Literal operators occupy a separate namespace. An injective encoding lets the
-- read-only pattern elaborator resolve literals even before the first fact uses one.


private def literal? (s : String) : Except String (Option (Nat × String)) := do
  if s.startsWith "\"" then
    let chars := ((s.drop 1).dropEnd 1).toString
    let code := chars.toList.foldl (fun n c => n * 0x110000 + c.toNat) 1
    return some (literalBase + code * 4 + 1, "String")
  let normalized := if s.startsWith "+" then (s.drop 1).toString else s
  if let some i := normalized.toInt? then
    unless i64InRange i do throw s!"i64 out of range: {s}"
    return some (encodeInt i, "i64")
  if s == "true" || s == "false" then
    return some (literalBase + (if s == "true" then 15 else 7), "bool")
  if let some bits := FloatLiteral.bits? s then
    return some (literalBase + bits * 8 + 3, "f64")
  let unsigned := if normalized.startsWith "-" then (normalized.drop 1).toString else normalized
  if (unsigned.startsWith "." || unsigned.toList.head?.any (·.isDigit)) &&
     unsigned.toList.any (·.isDigit) && unsigned.toList.all (fun c => c.isDigit || "+-.Ee".toList.contains c) then
    throw s!"unsupported numeric literal {s}"
  return none

/-- Convert a primitive enode back to egglog literal syntax. -/
def displayLiteral? (op : Nat) : Option String := Id.run do
  if op < literalBase then return none
  let k := op - literalBase
  let mut n := k / 2
  if k % 2 == 0 then
    return some (if n % 2 == 0 then toString (n / 2) else "-" ++ toString (n / 2 + 1))
  if k % 8 == 7 then return some (if k == 15 then "true" else "false")
  if k % 8 == 3 then return some (FloatLiteral.render (k / 8))
  n := k / 4
  let mut chars := []
  while n > 1 do
    chars := Char.ofNat (n % 0x110000) :: chars
    n := n / 0x110000
  let str := String.ofList chars
  return some ("\"" ++ (str.replace "\\" "\\\\").replace "\"" "\\\"" ++ "\"")

private def groundApp (st : ProgramState) (node : ENode) (result : String) (insert : Bool) :
    Except String (Id × String × ProgramState) := do
  if insert || primitive result then
    let (id, g) := st.graph.add node
    return (id, result, { st with graph := g })
  let some id := st.graph.lookup? node | throw "ground term is absent"
  return (id, result, st)

/-- Evaluate or look up a typed ground constructor term. An unsuccessful lookup
never inserts new enodes. -/
partial def ground (st : ProgramState) (term : SExpr) (insert : Bool) :
    Except String (Id × String × ProgramState) := do
  match term with
  | .atom s =>
    if let some (op, sort) ← literal? s then
      return ← groundApp st ⟨op, #[]⟩ sort insert
    let some (sort, id) := st.globals.get? s | throw s!"unbound term {s}"
    return (st.graph.find id, sort, st)
  | .list _ =>
    let (name, args) ← application term
    if let some op := arithmeticOp? name then
      let [a, b] := args.toList | throw s!"{name} takes two i64 operands"
      let (aid, sa, st) ← ground st a true
      let (bid, sb, st) ← ground st b true
      checkSort (some "i64") sa
      checkSort (some "i64") sb
      let result ← calculate op st.graph.nodes[aid]!.op st.graph.nodes[bid]!.op
      return ← groundApp st ⟨result, #[]⟩ "i64" insert
    let ctor ← constructorFor st name args.size
    let mut st := st
    let mut ids := Array.emptyWithCapacity args.size
    for a in args, s in ctor.inputs do
      let (id, actual, next) ← ground st a insert
      checkSort (some s) actual
      ids := ids.push id
      st := next
    groundApp st ⟨ctor.op, ids⟩ ctor.result insert

private partial def quoteTree : Tree → Pat
  | .app op cs => .app op (cs.toList.map quoteTree)

partial def elabPat (st : ProgramState) (term : SExpr) (expected : Option String)
    (vars : PatternVars) (bindNew : Bool) (arith : Bool := false) : Except String (Pat × String × PatternVars) := do
  match term with
  | .atom s =>
    if s.startsWith "@" then throw "names beginning with @ are reserved"
    if let some (op, sort) ← literal? s then
      checkSort expected sort
      return (.app op [], sort, vars)
    if s == "_" then
      unless bindNew do throw "wildcards cannot be referenced in rule heads"
      let some sort := expected | throw "cannot infer wildcard sort"
      return (.var vars.next, sort, { vars with next := vars.next + 1 })
    if let some (pattern, sort) := vars.derived.get? s then
      checkSort expected sort
      return (pattern, sort, vars)
    if let some (i, sort) := vars.bindings.get? s then
      checkSort expected sort
      return (.var i, sort, vars)
    if let some (sort, id) := st.globals.get? s then
      checkSort expected sort
      let some witness := st.graph.extract id | throw s!"no ground witness for global {s}"
      return (quoteTree witness.term, sort, vars)
    if s.startsWith "$" then throw s!"unknown global {s}"
    unless bindNew do throw s!"unbound pattern variable {s}"
    if st.constructors.contains s || st.sorts.contains s || st.globals.contains ("$" ++ s) then
      throw s!"pattern variable shadows a declaration: {s}"
    let some sort := expected | throw s!"cannot infer sort of {s}"
    return (.var vars.next, sort, { vars with bindings := vars.bindings.insert s (vars.next, sort), next := vars.next + 1 })
  | .list _ =>
    let (name, args) ← application term
    if let some op := arithmeticOp? name then
      unless arith do throw "arithmetic is supported in ground terms and raw rule heads"
      checkSort expected "i64"
      let [a, b] := args.toList | throw s!"{name} takes two i64 operands"
      let (a, _, vars) ← elabPat st a (some "i64") vars bindNew true
      let (b, _, vars) ← elabPat st b (some "i64") vars bindNew true
      return (.app op [a, b], "i64", vars)
    let ctor ← constructorFor st name args.size
    checkSort expected ctor.result
    let mut vars := vars
    let mut kids := #[]
    for a in args, sort in ctor.inputs do
      let (p, _, next) ← elabPat st a (some sort) vars bindNew arith
      kids := kids.push p
      vars := next
    return (.app ctor.op kids.toList, ctor.result, vars)

def ProgramState.freshName (st : ProgramState) (name : String) : Except String Unit := do
  if name.startsWith "@" || (← literal? name).isSome || (arithmeticOp? name).isSome then
    throw s!"reserved or literal name {name}"
  if st.constructors.contains name || st.globals.contains name ||
      st.sorts.contains name || (st.rulesets.contains name || st.combined.contains name) then throw s!"duplicate name {name}"

private def addSort (st : ProgramState) (name : String) : Except String ProgramState := do
  st.freshName name
  return { st with sorts := st.sorts.insert name }

private def addConstructor (st : ProgramState) (name : String) (inputs : Array String)
    (result : String) (cost : Nat) : Except String ProgramState := do
  st.freshName name
  for s in inputs.push result do
    unless st.sorts.contains s do throw s!"unknown sort {s}"
  if primitive result then throw "constructors must return an equality sort"
  if st.names.size ≥ constructorLimit then throw "too many constructors"
  let ctor : Constructor := ⟨st.names.size, inputs, result, cost⟩
  return { st with names := st.names.push name, constructors := st.constructors.insert name ctor }

private def parseNat (term : SExpr) : Except String Nat := do
  let s ← atomName term
  let some n := s.toNat? | throw s!"expected a natural number, found {s}"
  return n

private def splitCost (xs : Array SExpr) : Except String (Array SExpr × Nat) := do
  if xs.size ≥ 2 then
    if let .atom ":cost" := xs[xs.size - 2]! then
      return (xs.extract 0 (xs.size - 2), ← parseNat xs.back!)
  return (xs, 1)

def ProgramState.ruleSetMembers (st : ProgramState) (name : String) : Array String :=
  st.combined.getD name #[name]

private def ProgramState.baseGroup (st : ProgramState) (name : String) : Except String RuleGroup := do
  if name.isEmpty then return ⟨st.rewrites, st.arithmeticRewrites, st.conditional⟩
  let some group := st.rulesets.get? name | throw s!"unknown ruleset {name}"
  return group

def ProgramState.getGroup (st : ProgramState) (name : String) : Except String RuleGroup := do
  let some members := st.combined.get? name | return ← st.baseGroup name
  let mut result : RuleGroup := {}
  for child in members do
    let group ← st.baseGroup child
    result := ⟨result.plain ++ group.plain, result.arithmetic ++ group.arithmetic,
               result.conditional ++ group.conditional⟩
  return result

def ProgramState.runGroup (st : ProgramState) (name : String) (limit : Nat) :
    Except String ProgramState := do
  let group ← st.getGroup name
  if group.arithmetic.isEmpty && group.conditional.isEmpty then
    return { st with graph := Compiled.run st.graph group.plain limit }
  let mut g := st.graph.rebuild
  for _ in [:limit] do
    let version := g.version
    let batch := searchRound g group.plain
    let arith ← searchArithmetic g group.arithmetic
    let guarded ← searchGuarded g group.conditional
    g := (applyRound (applyRound (applyRound g batch) arith) guarded).rebuild
    if g.version == version then break
  return { st with graph := g }

def compileGuard (st : ProgramState) (vars : PatternVars) (fact : SExpr) :
    Except String GuardTest := do
  let (name, args) ← application fact
  let some cmp := comparison? name | throw s!"unsupported condition {name}"
  let [lhs, rhs] := args.toList | throw "comparison needs two operands"
  let (a, sort, _) ← elabPat st lhs none vars false true
  let (b, _, _) ← elabPat st rhs (some sort) vars false true
  if name != "=" && name != "!=" && sort != "i64" then throw "ordered comparison requires i64"
  return ⟨cmp, a, b, primitive sort⟩

private def newGuardVar (st : ProgramState) (vars : PatternVars) (x : SExpr) :
    Except String (Option String) := do
  let .atom name := x | return none
  if vars.bindings.contains name || vars.derived.contains name || st.globals.contains name ||
      name.startsWith "$" || name == "_" || (← literal? name).isSome then return none
  return some name

-- An equality condition can bind a fresh variable to a ground expression in
-- previously bound variables. Its reflexive guard checks that the lookup or
-- partial primitive succeeds before substituting that expression in the head.
private def mkRewrite (st : ProgramState) (left right : SExpr) (facts : Array SExpr) (raw : Bool) :
    Except String (Rewrite × Array GuardTest) := do
  let (lhs, sort, initialVars) ← elabPat st left none {} true
  let mut vars := initialVars
  if primitive sort then throw "rewrites must act on an equality sort"
  let mut tests := #[]
  for fact in facts do
    let (op, args) ← application fact
    if op == "=" then
      if let #[a, b] := args then
        let va ← newGuardVar st vars a
        let vb ← newGuardVar st vars b
        if va.isSome || vb.isSome then
          let (name, term) := if let some v := va then (v, b) else (vb.get!, a)
          let (pat, s, _) ← elabPat st term none vars false raw
          let _ ← elabPat st (.atom name) (some s) vars true
          vars := { vars with derived := vars.derived.insert name (pat, s) }
          tests := tests.push ⟨.eq, pat, pat, primitive s⟩
          continue
    tests := tests.push (← compileGuard st vars fact)
  let (rhs, _, _) ← elabPat st right (some sort) vars false raw
  return (⟨lhs, rhs⟩, tests)

private def register (st : ProgramState) (name : String) (r : Rewrite)
    (conditions : Array GuardTest) : Except String ProgramState := do
  if st.combined.contains name then throw s!"cannot add a rule to combined ruleset {name}"
  let group ← st.baseGroup name
  let group := if !conditions.isEmpty then { group with conditional := group.conditional.push ⟨r, conditions⟩ }
    else if containsArithmetic r.rhs then { group with arithmetic := group.arithmetic.push r }
    else { group with plain := group.plain.push r }
  if name.isEmpty then
    return { st with rewrites := group.plain, arithmeticRewrites := group.arithmetic, conditional := group.conditional }
  return { st with rulesets := st.rulesets.insert name group }

private def ruleOptions (opts : List SExpr) : Except String (String × Array SExpr) := do
  match opts with
  | [] => return ("", #[])
  | .atom ":ruleset" :: .atom name :: rest =>
    let (_, guards) ← ruleOptions rest
    return (name, guards)
  | .atom ":when" :: .list facts :: rest =>
    let (name, _) ← ruleOptions rest
    return (name, facts)
  | .atom ":name" :: .atom label :: rest =>
    unless label.startsWith "\"" do throw "expected a quoted rule name"
    ruleOptions rest
  | _ => throw "unsupported rewrite option"

def ProgramState.tableSize? (st : ProgramState) (name : String) : Option Nat := do
  let ctor ← st.constructors.get? name
  return (st.graph.tableRows ctor.op).size

structure RunConfig where
  name : String := ""
  «until» : Option (Array SExpr) := none
  deriving Inhabited

private def runConfig (name : String) (opts : List SExpr) : Except String RunConfig := do
  match opts with
  | [] => return ⟨name, none⟩
  | .atom ":until" :: facts => return ⟨name, some facts.toArray⟩
  | _ => throw "expected :until followed by ground facts"

def parseBoundedRun (args : Array SExpr) : Except String (RunConfig × Nat) := do
  let head :: rest := args.toList | throw "run requires an iteration bound"
  if let .atom n :: opts := rest then
    if let some limit := n.toNat? then return (← runConfig (← atomName head) opts, limit)
  return (← runConfig "" rest, ← parseNat head)

private def scheduledRun (expr : SExpr) : Except String (Option RunConfig) := do
  if let .atom set := expr then return some ⟨set, none⟩
  let (op, args) ← application expr
  if op != "run" then return none
  match args.toList with
  | [] => return some {}
  | .atom ":until" :: _ => return some (← runConfig "" args.toList)
  | name :: opts => return some (← runConfig (← atomName name) opts)

/-- Evaluate a nested egglog schedule with the database's own round implementation. -/
partial def runScheduleWith [Inhabited σ] (step : σ → RunConfig → Nat → Except String σ)
    (version : σ → Nat) (initial : σ) (expr : SExpr)
    (saturateRun : Option (σ → RunConfig → Except String σ) := none) : Except String σ := do
  if let some run ← scheduledRun expr then return ← step initial run 1
  let (op, args) ← application expr
  let mut st := initial
  match op, args.toList with
  | "seq", steps =>
    for s in steps do st ← runScheduleWith step version st s saturateRun
  | "repeat", limit :: steps =>
    let n ← parseNat limit
    if let [single] := steps then
      if let some run ← scheduledRun single then return ← step st run n
    for _ in [:n] do
      let before := version st
      for s in steps do st ← runScheduleWith step version st s saturateRun
      if before == version st then break
  | "saturate", steps =>
    if let [single] := steps then
      if let some run ← scheduledRun single then
        if let some fixedPoint := saturateRun then return ← fixedPoint st run
    repeat
      let before := version st
      for s in steps do st ← runScheduleWith step version st s saturateRun
      if before == version st then break
  | _, _ => throw s!"unsupported schedule {op}"
  return st

partial def ProgramState.formatTree (st : ProgramState) : Tree → String
  | .app op cs =>
    if let some literal := displayLiteral? op then literal
    else
      let head := st.names[op]!
      "(" ++ String.intercalate " " (head :: cs.toList.map st.formatTree) ++ ")"

def ProgramState.extract? (st : ProgramState) (id : Id) : Option Extracted :=
  st.graph.extract id fun op =>
    match st.names[op]?.bind st.constructors.get? with
    | some c => c.cost
    | none => 1

/-- Typecheck a ground term independently of the current table contents. -/
partial def groundSort (st : ProgramState) (term : SExpr) : Except String String := do
  match term with
  | .atom s =>
    if let some (_, sort) ← literal? s then return sort
    let some (sort, _) := st.globals.get? s | throw s!"unbound term {s}"
    return sort
  | .list _ =>
    let (name, args) ← application term
    if (arithmeticOp? name).isSome then
      unless args.size == 2 do throw s!"{name} takes two i64 operands"
      for arg in args do checkSort (some "i64") (← groundSort st arg)
      return "i64"
    let c ← constructorFor st name args.size
    for a in args, s in c.inputs do checkSort (some s) (← groundSort st a)
    return c.result

private def validateFacts (st : ProgramState) (facts : Array SExpr) (checked : Bool) :
    Except String Unit := do
  for fact in facts do
    let (op, args) ← application fact
    let sort ← if (comparison? op).isSome then do
      let [a,b] := args.toList | throw s!"{op} requires two ground terms"
      let s ← groundSort st a
      checkSort (some s) (← groundSort st b)
      if op != "=" && op != "!=" then checkSort (some "i64") s
      pure s
    else groundSort st fact
    if checked && primitive sort then throw "primitive terms are outside the certified signature"

private def checkCommand (st : ProgramState) (facts : Array SExpr) (checked : Bool) :
    Except String Unit := do
  for fact in facts do
    let (op, args) ← application fact
    if let some cmp := comparison? op then
      let [a, b] := args.toList | throw s!"{op} requires two ground terms"
      let (x, s, localGraph) ← ground st a false
      let (y, t, localGraph) ← ground localGraph b false
      checkSort (some s) t
      if checked && primitive s then throw "primitive terms are outside the certified signature"
      let graph := localGraph.graph
      let ok ← match cmp with
        | .eq => pure (graph.find x == graph.find y)
        | .ne => pure (graph.find x != graph.find y)
        | _ => do
          checkSort (some "i64") s
          compareI64 cmp graph.nodes[graph.find x]!.op graph.nodes[graph.find y]!.op
      unless ok do throw "check failed"
    else
      let (_, s, _) ← ground st fact false
      if checked && primitive s then throw "primitive terms are outside the certified signature"

private def runUntil (st : ProgramState) (checked : Bool) (cfg : RunConfig) (limit : Nat) :
    Except String ProgramState := do
  let some facts := cfg.until | return ← st.runGroup cfg.name limit
  let _ ← st.getGroup cfg.name
  validateFacts st facts checked
  let mut st := st
  for _ in [:limit] do
    match checkCommand st facts checked with
    | .ok () => break
    | .error "check failed" | .error "ground term is absent" => pure ()
    | .error err => throw err
    let before := st.graph.version
    st ← st.runGroup cfg.name 1
    if before == st.graph.version then break
  return st

private def execute (st : ProgramState) (cmd : SExpr)
    (certify : Rewrite → Except String Rewrite) (checked : Bool) : Except String ProgramState := do
  let (op, args) ← application cmd
  match op, args.toList with
  | "datatype*", groups =>
    let decls ← groups.mapM application
    let mut st := st
    for (sort, _) in decls do st ← addSort st sort
    for (sort, ctors) in decls do
      for ctor in ctors do
        let (name, tail) ← application ctor
        let (inputs, cost) ← splitCost tail
        st ← addConstructor st name (← inputs.mapM atomName) sort cost
    return st
  | "datatype", sort :: ctors =>
    let result ← atomName sort
    let mut st ← addSort st result
    for c in ctors do
      let (name, tail) ← application c
      let (inputs, cost) ← splitCost tail
      st ← addConstructor st name (← inputs.mapM atomName) result cost
    return st
  | "sort", [s] => addSort st (← atomName s)
  | "constructor", _ =>
    let (params, cost) ← splitCost args
    let [n, .list sorts, r] := params.toList | throw "invalid constructor declaration"
    addConstructor st (← atomName n) (← sorts.mapM atomName) (← atomName r) cost
  | "let", [n, t] =>
    let name ← atomName n
    st.freshName name
    let (id, sort, st) ← ground st t true
    if checked && primitive sort then throw "primitive terms are outside the certified signature"
    return { st with globals := st.globals.insert name (sort, id) }
  | "union", [a, b] =>
    if checked then throw "explicit union has no rule certificate"
    let (x, s, st) ← ground st a true
    let (y, t, st) ← ground st b true
    checkSort (some s) t
    if primitive s then throw "cannot union primitive values"
    return { st with graph := (st.graph.merge x y).rebuild }
  | "ruleset", [.atom name] =>
    st.freshName name
    return { st with rulesets := st.rulesets.insert name {} }
  | "unstable-combined-ruleset", n :: children =>
    let name ← atomName n
    st.freshName name
    let mut members := #[]
    for child in children do
      let id ← atomName child
      let _ ← st.getGroup id
      for leaf in st.ruleSetMembers id do
        if !members.contains leaf then members := members.push leaf
    return { st with combined := st.combined.insert name members }
  | "rewrite", a :: b :: opts | "birewrite", a :: b :: opts =>
    let (set, facts) ← ruleOptions opts
    let registerDirection := fun (st : ProgramState) left right => do
      let (r, conditions) ← mkRewrite st left right facts (!checked)
      let proved ← certify r
      register st set proved conditions
    let st ← registerDirection st a b
    if op == "birewrite" then return ← registerDirection st b a
    return st
  | "run", _ =>
    let (cfg, limit) ← parseBoundedRun args
    runUntil st checked cfg limit
  | "run-schedule", cs =>
    let mut st := st
    for c in cs do st ← runScheduleWith (fun s => runUntil s checked) (·.graph.version) st c
    return st
  | "check", _ =>
    checkCommand st args checked
    return st
  | "fail", [inner] =>
    let ("check", facts) ← application inner | throw "only fail on ground checks is supported"
    validateFacts st facts checked
    match checkCommand st facts checked with
    | .ok () => throw "check succeeded under fail"
    | .error "check failed" | .error "ground term is absent" => return st
    | .error err => throw err
  | "extract", [t] =>
    let (id, sort, st) ← ground st t true
    if checked && primitive sort then throw "primitive terms are outside the certified signature"
    let some result := st.extract? id | throw "no finite term in this class"
    return { st with output := st.output.push (st.formatTree result.term) }
  | "print-size", [] =>
    let rows := st.names.qsort (· < ·) |>.map fun name =>
      s!"({name} {(st.tableSize? name).getD 0})"
    return { st with output := st.output.push ("(" ++ String.intercalate "\n " rows.toList ++ ")") }
  | "print-stats", [] =>
    let g := st.graph
    let stats := s!"enodes={g.nodes.size} classes={g.classCount} unions={g.version - g.maskVersion - g.nodes.size}"
    return { st with output := st.output.push stats }
  | "print-size", [n] =>
    let name ← atomName n
    let some size := st.tableSize? name | throw s!"unknown constructor {name}"
    return { st with output := st.output.push (toString size) }
  | _, _ =>
    let (_, sort, st) ← ground st cmd true
    if checked && primitive sort then throw "primitive terms are outside the certified signature"
    return st

def execCommand (st : ProgramState) (cmd : SExpr) : Except String ProgramState :=
  execute st cmd pure false

def runSource (src : String) (st : ProgramState := {}) : Except String ProgramState := do
  (← parseSExprs src).foldlM execCommand st

private def checkSignature (st : ProgramState) (signature : Array (String × Nat))
    (complete : Bool) : Except String Unit := do
  if st.names.size > signature.size || (complete && st.names.size != signature.size) then
    throw "constructor count differs from the certified signature"
  if (st.sorts.toList.filter (fun s => !primitive s)).length > 1 then
    throw "the certified signature requires a single equality sort"
  for name in st.names, (wanted, arity) in signature do
    let c := st.constructors[name]!
    unless name == wanted && c.inputs.size == arity && c.inputs.all (· == c.result) do
      throw s!"constructor {name} differs from the certified signature"

/-- Start from an empty graph and resolve each parsed rewrite against the supplied
certificates. Only a single equality sort with the declared constructor order is accepted. -/
def runCheckedSource (src : String) (signature : List (String × Nat))
    (rules : List (VerifiedRule m)) : Except String ProgramState := do
  let mut st : ProgramState := {}
  let sig := signature.toArray
  for cmd in ← parseSExprs src do
    st ← execute st cmd (fun rw => do
      let some proved := rw.certify? rules | throw "rewrite has no matching certificate"
      return proved.toRewrite) true
    if let .list xs := cmd then
      if let some (SExpr.atom name) := xs[0]? then
        if name == "datatype" || name == "datatype*" || name == "constructor" || name == "sort" then
          checkSignature st sig false
  checkSignature st sig true
  return st

end Egglog
