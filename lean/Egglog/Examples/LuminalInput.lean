import Egglog.Examples.Luminal
import Egglog.Parser

namespace Luminal

open Egglog

private inductive Dim where
  | num (value : Nat)
  | sym (name : String)
  deriving BEq, Repr

private inductive Info where
  | integer (n : Nat)
  | text (s : String)
  | dim (d : Dim)
  | dims (values : List Dim)
  | fpType
  | fpTensor
  | other
  deriving Inhabited

structure Binding where
  name : String
  rhs : SExpr
  /-- Exact direct operands of the source let. -/
  refs : List Nat
  deriving Inhabited

structure Imported where
  declarations : Array SExpr := #[]
  bindings : Array Binding := #[]
  program : Program := #[]
  roots : List Nat := []

private def atom? : SExpr → Option String
  | .atom s => some s
  | _ => none

private def app? : SExpr → Option (String × Array SExpr)
  | .list a => do return (← a[0]? >>= atom?, a.extract 1 a.size)
  | _ => none

private def infer (op : String) (args : List Nat) (infos : Array Info) : Info := Id.run do
  let info : Nat → Info := fun i => infos[i]?.getD .other
  match op, args.map info with
  | "MNum", [.integer n] => return .dim (.num n)
  | "MVar", [.text s] => return .dim (.sym s)
  | "ECons", [.dim d,.dims ds] => return .dims (d :: ds)
  | "ENil", [] => return .dims []
  | "F32", [] => return .fpType
  | "Input", [_,_,.fpType] | "Cast", [_,.fpType] | "CustomOpHLIR", [_,_,.fpType]
    | "Gather", [_,_,_,.fpTensor,_,_] | "RowRope", [_,.fpTensor,_,_,_]
    | "Sqrt", [_,.fpTensor,_,_] | "Recip", [_,.fpTensor,_,_] | "Exp2", [_,.fpTensor,_,_]
    | "Sum", [_,_,.fpTensor,_,_,_] | "Add", [_,.fpTensor,_,.fpTensor,_,_]
    | "Mul", [_,.fpTensor,_,.fpTensor,_,_] | "OrderedMatMul", [_,_,.fpTensor,.fpTensor]
    | "Constant", [_] => return .fpTensor
  | _, _ => return .other

/-- Decode only the [s,n,k] product with physical strides [k,0,1],
[0,k,1], [n*k,k,1], and the matching contiguous last-axis reduction.
Every other instruction retains its verbatim expression and operands. -/
private def decode? (op : String) (args : List Nat) (infos : Array Info) : Option Inst := do
  let get : Nat → Info := fun i => infos[i]?.getD .other
  let terms := args.map get
  match op, args, terms with
  | "Mul", [_,x,_,w,_,_], [.dims [.sym batch,.num n,.num k],.fpTensor,.dims sa,
        .fpTensor,.dims sb,.dims dst] =>
    guard (n > 0 && k > 0 && sa == [.num k,.num 0,.num 1] &&
      sb == [.num 0,.num k,.num 1] && dst == [.num (n*k),.num k,.num 1])
    return .times ⟨batch,n,k⟩ x w
  | "Sum", [_,_,x,_,_,_], [.dims [.sym batch,.num n],.dim (.num k),.fpTensor,.dims src,
        .dim (.num 1),.dims dst] =>
    guard (n > 0 && k > 0 && src == [.num (n*k),.num k] && dst == [.num n,.num 1])
    return .sum ⟨batch,n,k⟩ x
  | "OrderedMatMul", [_,_,a,b], [.dims [.sym batch,.num n],.dim (.num k),.fpTensor,.fpTensor] =>
    guard (n > 0 && k > 0)
    return .dot ⟨batch,n,k⟩ a b
  | _, _, _ => none

def importSource (source : String) : Except String Imported := do
  let mut data : Imported := {}
  let mut ids : Std.HashMap String Nat := {}
  let mut infos : Array Info := #[]
  for form in ← parseSExprs source do
    let some (op, fields) := app? form | throw "expected a top-level command"
    if op == "datatype*" || op == "datatype" || op == "sort" || op == "constructor" then
      data := {data with declarations := data.declarations.push form}
    if op != "let" then continue
    let #[.atom name, rhs] := fields | throw "invalid SSA binding"
    if ids.contains name then throw s!"duplicate binding {name}"
    let i := data.program.size
    let dataRow : Inst × Info × List Nat × Bool ← match rhs with
      | .atom s =>
        if let some n := s.toNat? then pure (.opaqueOp name [], .integer n, [], false)
        else if s.startsWith "\"" then pure (.opaqueOp name [], .text s, [], false)
        else if let some j := ids[s]? then pure (.opaqueOp name [j], .other, [j], false)
        else if s.toInt?.isSome || (s.toList.head?.any (fun c => c.isDigit || c == '-') && s.contains '.') then
          pure (.opaqueOp name [], .other, [], false)
        else throw s!"unknown SSA atom {s} in {name}"
      | .list _ => do
        let some (head, args) := app? rhs | throw s!"malformed expression at {name}"
        let refs ← args.toList.mapM fun field => do
          let some ref := atom? field | throw s!"expected flat SSA operand at {name}"
          let some j := ids[ref]? | throw s!"unknown operand {ref} in {name}"
          pure j
        let info := infer head refs infos
        pure ((decode? head refs infos).getD (.opaqueOp name refs), info, refs, head == "Output")
    let (ins, info, refs, root) := dataRow
    data := {data with
      program := data.program.push ins
      bindings := data.bindings.push ⟨name,rhs,refs⟩
      roots := if root then i :: data.roots else data.roots}
    infos := infos.push info
    ids := ids.insert name i
  if data.roots.isEmpty then throw "no Luminal Output in the imported graph"
  return data

partial def printSExpr : SExpr → String
  | .atom s => if s.startsWith "\"" then (s.drop 1 |>.dropEnd 1 |>.toString).quote else s
  | .list a => "(" ++ String.intercalate " " (a.toList.map printSExpr) ++ ")"

/-- Keep the original shape and reduction extent as target operands so the
emitted graph can be inspected/typechecked as egglog data. -/
private def inputs (model : Imported) (i : Nat) (optimized : Program) : List Nat :=
  match optimized[i]?, model.bindings[i]? with
  | some (.dot _ a b), some bd => (bd.refs.take 2) ++ [a,b]
  | _, some bd => bd.refs
  | _, _ => []

def live (model : Imported) (p : Program) : Std.HashSet Nat := Id.run do
  let mut seen := Std.HashSet.ofList model.roots
  for j in [:p.size] do
    let i := p.size - 1 - j
    if seen.contains i then
      for dep in inputs model i p do seen := seen.insert dep
  return seen

/-- Emit a data-only module with the original datatypes and reachable SSA
bindings. It can be loaded by the Rust egglog CLI independently. -/
def emit (model : Imported) (p : Program) : String := Id.run do
  let mut lines := model.declarations.map printSExpr
  let declaration := "(constructor OrderedMatMul (EList Expression IR IR) IR)"
  if !lines.contains declaration then lines := lines.push declaration
  let seen := live model p
  for i in [:model.bindings.size] do
    if !seen.contains i then continue
    let bd := model.bindings[i]!
    let value := match p[i]? with
      | some (.dot ..) => "(OrderedMatMul " ++ String.intercalate " "
          ((inputs model i p).map fun (j : Nat) => (model.bindings[j]! : Binding).name) ++ ")"
      | _ => printSExpr bd.rhs
    lines := lines.push s!"(let {bd.name} {value})"
  return String.intercalate "\n" lines.toList ++ "\n"

private def translate? (back : Array Nat) : Inst → Option Inst
  | .opaqueOp tag args => do return .opaqueOp tag (← args.mapM (back[·]?))
  | .times l a b => do return .times l (← back[a]?) (← back[b]?)
  | .sum l x => do return .sum l (← back[x]?)
  | .dot l a b => do return .dot l (← back[a]?) (← back[b]?)

/-- Validate the serialized boundary. Reindex each binding read from disk by
its retained source name and compare all decoded instructions, including
opaque operands, the original opaque syntax, and all output roots. -/
def checkEmission (model : Imported) (plan : Program) (emitted : Imported) : Bool := Id.run do
  let keep := live model plan
  let mut back := #[]
  for i in [:plan.size] do
    if !keep.contains i then continue
    let j := back.size
    let some source :=  model.bindings[i]? | return false
    let some dest :=  emitted.bindings[j]? | return false
    if source.name != dest.name then return false
    let some target := emitted.program[j]? | return false
    let some decoded := translate? back target | return false
    if some decoded != plan[i]? then return false
    if let .opaqueOp .. := decoded then
      if printSExpr source.rhs != printSExpr dest.rhs then return false
    back := back.push i
  return emitted.bindings.size == back.size && emitted.program.size == back.size &&
    (emitted.roots.mapM (back[·]?)) == some model.roots

/-- A portable egglog rendering of the verified projection workload. A sort per
layout gives the Rust searcher the same partitions as the certified searcher.
Register costs include Rust egglog's unit cost for the i64 payload. -/
def projectionWorkload (model : Imported) (p : Program) : String := Id.run do
  let mut shapes := #[]
  for ins in model.program do
    if let .sum l _ := ins then
      if !shapes.contains l then shapes := shapes.push l
  let mut setup := #[]
  let mut checks := #[]
  let mut queries := #[]
  for l in shapes, group in [:shapes.size] do
    let e := s!"Tensor{group}"
    let r := s!"Reg{group}"
    let m := s!"Times{group}"
    let s := s!"Sum{group}"
    let d := s!"Dot{group}"
    setup := setup.push s!"(sort {e})"
    setup := setup.push s!"(constructor {r} (i64) {e} :cost 0)"
    setup := setup.push s!"(constructor {m} ({e} {e}) {e})"
    setup := setup.push s!"(constructor {s} ({e}) {e} :cost 4)"
    setup := setup.push s!"(constructor {d} ({e} {e}) {e})"
    setup := setup.push s!"(rewrite ({s} ({m} a b)) ({d} a b))"
    for ins in model.program, i in [:model.program.size] do
      match ins with
      | .times l' a b =>
        if l == l' then setup := setup.push s!"(union ({r} {i}) ({m} ({r} {a}) ({r} {b})))"
      | .sum l' x =>
        if l == l' then
          setup := setup.push s!"(let $root{i} ({s} ({r} {x})))"
          if let some (.dot _ a b) := p[i]? then
            checks := checks.push s!"(check (= $root{i} ({d} ({r} {a}) ({r} {b}))))"
          queries := queries.push s!"(extract $root{i})"
      | _ => pure ()
  return String.intercalate "\n" ((setup.push "(run 1)" ++ checks ++ queries).toList) ++ "\n"

end Luminal
