import Egglog.Examples.EggCCInput
import Egglog.Examples.EggCCFlow

namespace EggCC.Driver

private def nodeAt (d : InputDAG) (i : Nat) : Check DagNode :=
  match d.nodes[i]? with | some n => .ok n | none => .error s!"missing Tree IR node {i}"

-- Arms and loop bodies have fresh region arguments. Visiting them in the
-- parent scope would hoist their computations, including prints and loops.
private def localChildren (n : DagNode) : Array Nat :=
  match n.op with
  | "dowhile" => n.children.extract 0 1
  | "tif" => n.children.extract 0 2
  | _ => n.children

private def boundInRegion (n : DagNode) : Bool :=
  n.op == "dowhile" || n.op == "tif" || n.op == "tprint"

private structure Reachable where
  seen : Std.HashSet Nat := {}
  bindings : Array Nat := #[]

private partial def discover (d : InputDAG) (i : Nat) (r : Reachable) : Check Reachable := do
  if r.seen.contains i then return r
  let n ← nodeAt d i
  let r ← (localChildren n).foldlM (fun acc j => discover d j acc) {r with seen := r.seen.insert i}
  return if boundInRegion n then {r with bindings := r.bindings.push i} else r

private abbrev Slots := Std.HashMap Nat Nat

private def dropSlots (n : Nat) : Expr :=
  Nat.rec .arg (fun _ e => .tail e) n

-- Slot zero in a letVal body is the newest computed value. After the last
-- slot lies the upstream region argument, which getat refers to.
private partial def term (d : InputDAG) (slots : Slots) (i : Nat) : Check Expr := do
  if let some pos := slots[i]? then return .head (dropSlots (slots.size - pos - 1))
  let n ← nodeAt d i
  if boundInRegion n then throw s!"unbound {n.op} in current region"
  let children ← n.children.toList.mapM (term d slots)
  n.asExpr children (dropSlots slots.size)

/-- Import one region of the shared upstream DAG. Each loop, conditional and
    print has a single binding in its own invocation of that region. Pure,
    total If arms go through the existing proof-returning scalar optimizer.
    An arm containing control or effects runs only when its predicate holds. -/
private partial def region (d : InputDAG) (root : Nat) : Check Flow := do
  let reachable ← discover d root {}
  let mut slots : Slots := {}
  let mut values := #[]
  for i in reachable.bindings do
    let n ← nodeAt d i
    let rhs ← match n.op, n.children.toList with
      | "dowhile", [initial, body] => do
        let init ← term d slots initial
        pure (Flow.seq (.pure init) (.loop (← region d body)))
      | "tif", [p, input, y, z] => do
        let pred ← term d slots p
        let args ← term d slots input
        let yes ← region d y
        let no ← region d z
        let branchFlow := Flow.branch pred (.seq (.pure args) yes) (.seq (.pure args) no)
        match yes, no with
        | .pure a, .pure b =>
          pure (if pureRegion a && pureRegion b then .pure (.branch pred args a b) else branchFlow)
        | _, _ => pure branchFlow
      | "tprint", [v, token] => do
        pure (Flow.pure (.output (← term d slots v) (← term d slots token)))
      | _, _ => throw s!"invalid {n.op} region node at {i}"
    if i == root then return values.foldr Flow.letVal rhs
    values := values.push rhs
    slots := slots.insert i slots.size
  return values.foldr Flow.letVal (.pure (← term d slots root))

def flowType (env : Ty) : Flow → Check Ty
  | .pure e => checkType env e
  | .seq a b => do flowType (← flowType env a) b
  | .letVal rhs body => do flowType (.pair (← flowType env rhs) env) body
  | .branch pred a b => do
    discard <| same (← checkType env pred) .bool
    same (← flowType env a) (← flowType env b)
  | .loop body => do
    unless env == .nil || (match env with | .pair .. => true | _ => false) do
      throw "DoWhile requires an initial tuple"
    discard <| same (← flowType env body) (.pair .bool env)
    return env

def InputDAG.toFlow (d : InputDAG) : Check Flow := do
  unless (d.inputs.filter (· == .token)).size == 1 do throw "expected one input state"
  let f ← region d d.root
  discard <| same (← flowType (d.inputs.foldr .pair .nil) f) (.pair .token .nil)
  return f

inductive CompilerInput where
  | scalar (p : Program)
  | structured (inputs : Array Ty) (nodes : Nat) (body : Flow)

/-- Loops and conditional effects use explicit control flow. Straight-line
    programs keep their existing expression-language interface and schedule. -/
def importProgram (text : String) : Check CompilerInput := do
  let d ← InputDAG.read text
  unless d.nodes.any (fun n => n.op == "dowhile") do
    let p ← d.scalarTree
    if p.straightLineState then
      p.validate
      return .scalar p
  return .structured d.inputs d.nodes.size (← d.toFlow)

end EggCC.Driver
