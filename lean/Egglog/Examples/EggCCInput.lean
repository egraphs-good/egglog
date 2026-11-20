import Egglog.Examples.EggCCTypes
import Lean.Data.Json

open Lean EggCC

namespace EggCC.Driver

structure Program where
  inputs : Array Ty
  body : Expr
  inputNodes : Nat

def Program.straightLineState (p : Program) : Bool :=
  match p.body with
  | .pair last .nil => linearState last
  | _ => false

def Program.validate (p : Program) : Check Unit := do
  unless (p.inputs.filter (· == .token)).size == 1 do throw "expected one input state"
  let arg := p.inputs.foldr .pair .nil
  discard <| same (← checkType arg p.body) (.pair .token .nil)
  unless p.straightLineState do throw "print effects must form one linear output chain"

private def requireSome (opt : Option α) (msg : String) : Check α :=
  match opt with | some a => .ok a | none => .error msg

/-- References in the interchange DAG point strictly backwards. Child regions
    are rebound at their enclosing If or DoWhile by the structured importer. -/
structure DagNode where
  op : String
  children : Array Nat
  source : Json
  deriving Inhabited

structure InputDAG where
  private mk ::
  inputs : Array Ty
  nodes : Array DagNode
  root : Nat

def InputDAG.read (text : String) : Check InputDAG := do
  let json ← Json.parse text
  let inputs ← (← json.getObjValAs? (Array String) "inputs").mapM Ty.parse
  let source ← json.getObjValAs? (Array Json) "nodes"
  let mut nodes := #[]
  for node in source do
    let op ← node.getObjValAs? String "op"
    let children : Array Nat ← match node.getObjVal? "args" with
      | .ok array => fromJson? array
      | .error _ => pure #[]
    unless children.all (· < nodes.size) do throw s!"{op} references a missing earlier node"
    nodes := nodes.push ⟨op, children, node⟩
  let root ← json.getObjValAs? Nat "root"
  unless root < nodes.size do throw "missing root"
  return ⟨inputs, nodes, root⟩

def DagNode.asExpr (node : DagNode) (args : List Expr) (input : Expr) : Check Expr := do
  let op := node.op
  let index := node.source.getObjValAs? Nat "n"
  match op, args with
  | "getat", [] => return project (← index) input
  | "get", [a] => return project (← index) a
  | "single", [a] => return .pair a .nil
  | "parallel", args => return args.foldr .pair .nil
  | "int", [] =>
    let n ← node.source.getObjValAs? Int "n"
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
  | "and", [a,b] => return .select a b .no
  | "or", [a,b] => return .select a .yes b
  | _, _ => throw s!"unsupported operator/arity {op}/{args.length}"

def InputDAG.scalarTree (d : InputDAG) : Check Program := do
  let mut xs := #[]
  for n in d.nodes do
    let children ← n.children.toList.mapM fun i => requireSome xs[i]? "missing earlier node"
    xs := xs.push (← n.asExpr children .arg)
  let body ← requireSome xs[d.root]? "missing root"
  return ⟨d.inputs, body, d.nodes.size⟩

def parse (text : String) : Check Program := do
  let p ← (← InputDAG.read text).scalarTree
  Program.validate p
  return p

end EggCC.Driver
