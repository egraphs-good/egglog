import Egglog.Examples.EggCC
import Lean

open Lean EggCC

namespace EggCC.Driver

inductive Ty where
  | int | bool | token | nil | pair (lhs rhs : Ty)
  deriving BEq, Inhabited, Repr

def Ty.parse : String → Except String Ty
  | "int" => .ok .int
  | "bool" => .ok .bool
  | "state" => .ok .token
  | name => .error s!"unsupported argument type {name}"

def Ty.show : Ty → String
  | .int => "int" | .bool => "bool" | .token => "state"
  | .nil => "nil" | .pair _ _ => "tuple"

abbrev Check := Except String

def same (a b : Ty) : Check Ty := do
  unless a == b do throw s!"type mismatch: {repr a} / {repr b}"
  return a

def checkType (env : Ty) : Expr → Check Ty
  | .arg => pure env
  | .nil => pure .nil
  | .num _ => pure .int
  | .yes | .no => pure .bool
  | .pair a b => return .pair (← checkType env a) (← checkType env b)
  | .head a => do
    let .pair hd _ ← checkType env a | throw "head of non-tuple"
    return hd
  | .tail a => do
    let .pair _ tl ← checkType env a | throw "tail of non-tuple"
    return tl
  | .add a b | .sub a b | .mul a b => do
    discard <| same (← checkType env a) .int
    same (← checkType env b) .int
  | .le a b | .lt a b => do
    discard <| same (← checkType env a) .int
    discard <| same (← checkType env b) .int
    return .bool
  | .eq a b => do
    let t ← same (← checkType env a) (← checkType env b)
    unless t == .int do throw "eggcc Eq expects integer operands"
    return .bool
  | .neg a => do same (← checkType env a) .bool
  | .branch p i a b => do
    discard <| same (← checkType env p) .bool
    let inner ← checkType env i
    same (← checkType inner a) (← checkType inner b)
  | .select p a b => do
    discard <| same (← checkType env p) .bool
    same (← checkType env a) (← checkType env b)
  | .bindArg i a => do checkType (← checkType env i) a
  | .output x s => do
    let t ← checkType env x
    unless t == .int || t == .bool do throw "print expects an int or bool"
    same (← checkType env s) .token

/-- Effects occur on the linear state path; each candidate scalar value may
    be speculated by a backend without replaying a print or triggering a trap. -/
def pureRegion : Expr → Bool
  | .output .. => false
  | .nil | .arg | .yes | .no | .num _ => true
  | .head a | .tail a | .neg a => pureRegion a
  | .pair a b | .add a b | .sub a b | .mul a b | .lt a b | .le a b | .eq a b |
    .bindArg a b => pureRegion a && pureRegion b
  | .select p a b => pureRegion p && pureRegion a && pureRegion b
  | .branch p i a b => pureRegion p && pureRegion i && pureRegion a && pureRegion b

def linearState : Expr → Bool
  | .output x s => pureRegion x && linearState s
  | e => pureRegion e

structure Program where
  inputs : Array Ty
  body : Expr
  inputNodes : Nat

def Program.validate (p : Program) : Check Unit := do
  unless (p.inputs.filter (· == .token)).size == 1 do throw "expected one input state"
  let arg := p.inputs.foldr .pair .nil
  discard <| same (← checkType arg p.body) (.pair .token .nil)
  let .pair last .nil := p.body | throw "expected a singleton output state"
  unless linearState last do throw "print effects must form one linear output chain"

def project (n : Nat) (x : Expr) : Expr := .head (Nat.rec x (fun _ acc => Expr.tail acc) n)

private def requireSome (opt : Option α) (msg : String) : Check α :=
  match opt with | some a => .ok a | none => .error msg

def readNode (pool : Array Expr) (node : Json) : Check Expr := do
  let op : String ← node.getObjValAs? String "op"
  let args ← match node.getObjVal? "args" with
    | .ok ids => do
        let indices ← fromJson? (α := List Nat) ids
        indices.mapM fun i => requireSome pool[i]? s!"missing earlier node {i}"
    | .error _ => pure []
  let index := node.getObjValAs? Nat "n"
  match op, args with
  | "getat", [] => return project (← index) .arg
  | "get", [a] => return project (← index) a
  | "single", [a] => return .pair a .nil
  | "parallel", args => return args.foldr .pair .nil
  | "int", [] =>
    let n ← node.getObjValAs? Int "n"
    unless -(2^63 : Int) ≤ n && n < 2^63 do throw "literal outside signed i64"
    return .num (BitVec.ofInt 64 n).toNat
  | "ttrue", [] => return .yes
  | "tfalse", [] => return .no
  | "tif", [p,i,a,b] => return .branch p i a b
  | "select", [p,a,b] => return .select p a b
  | "tprint", [a,b] => return .output a b
  | "add", [a,b] => return .add a b
  | "sub", [a,b] => return .sub a b
  | "mul", [a,b] => return .mul a b
  | "less_than", [a,b] => return .lt a b
  | "less_eq", [a,b] => return .le a b
  | "greater_than", [a,b] => return .lt b a
  | "greater_eq", [a,b] => return .le b a
  | "eq", [a,b] => return .eq a b
  | "not", [a] => return .neg a
  | _, _ => throw s!"unsupported operator/arity {op}/{args.length}"

def parse (text : String) : Check Program := do
  let json ← Json.parse text
  let inputs ← (← json.getObjValAs? (Array String) "inputs").mapM Ty.parse
  let nodes ← json.getObjValAs? (Array Json) "nodes"
  let mut xs := #[]
  for n in nodes do xs := xs.push (← readNode xs n)
  let root ← json.getObjValAs? Nat "root"
  let body ← requireSome xs[root]? "missing root"
  let p := ⟨inputs, body, nodes.size⟩
  Program.validate p
  return p

def Expr.children : Expr → List Expr
  | .arg | .num _ | .yes | .no | .nil => []
  | .head a | .tail a | .neg a => [a]
  | .pair a b | .add a b | .sub a b | .mul a b | .lt a b | .le a b | .eq a b |
    .output a b | .bindArg a b => [a,b]
  | .select p a b => [p,a,b]
  | .branch p i a b => [p,i,a,b]

deriving instance Hashable for EggCC.Expr

abbrev Visited := Std.HashSet EggCC.Expr

private partial def visit (s : Visited) (e : EggCC.Expr) : Visited := Id.run do
  if s.contains e then return s
  return (Expr.children e).foldl visit (s.insert e)

def stats (e : EggCC.Expr) : Nat × Nat × Nat × Nat := Id.run do
  let nodes := visit {} e
  let mut branches := 0
  let mut selects := 0
  let mut binds := 0
  for x in nodes do
    match x with
    | .branch .. => branches := branches + 1
    | .select .. => selects := selects + 1
    | .bindArg .. => binds := binds + 1
    | _ => pure ()
  return (nodes.size, branches, selects, binds)

def readInputs (types : Array Ty) (args : List String) : Check Value := do
  let mut rest := args
  let mut values := #[]
  for t in types do
    let v ← match t with
      | .token => pure (.state [])
      | _ => do
        let a :: more := rest | throw "missing program argument"
        rest := more
        match t with
        | .int =>
          let n ← requireSome a.toInt? s!"invalid signed i64 {a}"
          unless -(2^63 : Int) ≤ n && n < 2^63 do throw "argument outside signed i64"
          pure (.data (.word (BitVec.ofInt 64 n)))
        | .bool =>
          unless a == "true" || a == "false" do throw "expected true or false"
          pure (.data (.flag (a == "true")))
        | _ => throw "bad input type"
    values := values.push v
  unless rest.isEmpty do throw "too many arguments"
  return values.foldr .pair .nil

def traceOf (v : Value) : Check (Array String) := do
  let .pair (.state trace) .nil := v | throw "expected a singleton state result"
  return trace.reverse.toArray.map fun
    | .word w => toString w.toInt
    | .flag b => toString b

/-- Recover a direct argument register from an extracted tuple projection. -/
def asArg (e : EggCC.Expr) : Option Nat := do
  let .head t := e | none
  let rec tails : EggCC.Expr → Option Nat
    | .arg => some 0
    | .tail x => (tails x).map (· + 1)
    | _ => none
  tails t

structure Bril where
  lines : Array String := #[]
  names : Std.HashMap EggCC.Expr (String × Ty) := {}
  inputs : Array Ty := #[]

abbrev Emit := StateT Bril (Except String)

private def scalar (e : EggCC.Expr) : Emit (String × Ty) := do
  if let some result := (← getThe Bril).names[e]? then return result
  if let some i := asArg e then
    let some ty := (← getThe Bril).inputs[i]? | throw "unbound argument"
    unless ty == .int || ty == .bool do throw "state in a scalar expression"
    return (s!"arg{i}", ty)
  let instruction ← match e with
  | .num n => pure (s!"const {(BitVec.ofNat 64 n).toInt}", Ty.int)
  | .yes => pure ("const true", .bool)
  | .no => pure ("const false", .bool)
  | .neg a => do let x ← scalar a; pure ("not " ++ x.1, .bool)
  | .add a b | .sub a b | .mul a b | .lt a b | .le a b | .eq a b => do
    let x ← scalar a
    let y ← scalar b
    let (op,ty) : String × Ty := match e with
      | .add .. => ("add", .int) | .sub .. => ("sub", .int) | .mul .. => ("mul", .int)
      | .lt .. => ("lt", .bool) | .le .. => ("le", .bool) | _ => ("eq", .bool)
    pure (s!"{op} {x.1} {y.1}", ty)
  | .select p a b => do
    let c ← scalar p
    let x ← scalar a
    let y ← scalar b
    pure (s!"select {c.1} {x.1} {y.1}", x.2)
  | _ => throw "result still contains an unlowered region or tuple"
  let s ← getThe Bril
  let name := s!"v{s.names.size}"
  set {s with names := s.names.insert e (name, instruction.2),
              lines := s.lines.push s!"  {name}: {instruction.2.show} = {instruction.1};"}
  return (name, instruction.2)
termination_by e

private def emitState (e : EggCC.Expr) : Emit Unit := do
  match e with
  | .output x s => do
    emitState s
    let (name,_) ← scalar x
    modify fun acc => { acc with lines := acc.lines.push s!"  print {name};" }
  | _ =>
    let some i := asArg e | throw "result still contains an unlowered state"
    unless (← getThe Bril).inputs[i]? == some .token do throw "expected the input state token"

def toBril (p : Program) : Check String := do
  let .pair trace .nil := p.body | throw "expected a singleton output state"
  let (_, s) ← (emitState trace).run {inputs := p.inputs}
  let args := p.inputs.toList.zipIdx.filterMap fun (t,i) =>
    if t == .token then none else some s!"arg{i}: {t.show}"
  return "@main(" ++ String.intercalate ", " args ++ ") {\n" ++
    String.intercalate "\n" s.lines.toList ++ "\n}\n"

end EggCC.Driver

open EggCC.Driver

def main (args : List String) : IO Unit := do
  let (emit, args) := match args with | "--emit-bril" :: more => (true, more) | _ => (false, args)
  let path :: values := args | throw (IO.userError "usage: eggcc-certified [--emit-bril] FILE [ARGS...]")
  let p ← IO.ofExcept (parse (← IO.FS.readFile path))
  let result ← IO.mkRef p.body
  let before ← IO.monoNanosNow
  result.modify (fun x => (EggCC.optimize x).output)
  let after ← IO.monoNanosNow
  let out := {p with body := ← result.get}
  IO.ofExcept out.validate
  if emit then IO.print (← IO.ofExcept (toBril out))
  else
    let (size, branches, selects, binds) := stats p.body
    let (outsize, ob, os, obind) := stats out.body
    IO.println s!"solver_ns={after-before} input_dag={size} output_dag={outsize} input_if={branches} output_if={ob} input_select={selects} output_select={os} input_bind={binds} output_bind={obind}"
    if !values.isEmpty || (p.inputs.all (· == .token)) then
      let arg ← IO.ofExcept (readInputs p.inputs values)
      let old ← IO.ofExcept (traceOf (p.body.eval arg))
      let newTrace ← IO.ofExcept (traceOf (out.body.eval arg))
      unless old == newTrace do throw (IO.userError "trace disagreement")
      IO.println ("input_trace=" ++ (toJson old).compress)
      IO.println ("output_trace=" ++ (toJson newTrace).compress)
