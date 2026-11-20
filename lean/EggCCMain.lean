import Egglog.Examples.EggCCFlowInput
import Egglog.Examples.EggCCFlowBril
import Lean

open Lean EggCC

namespace EggCC.Driver

def Expr.children : Expr → List Expr
  | .arg | .num _ | .yes | .no | .nil => []
  | .head a | .tail a | .neg a => [a]
  | .pair a b | .add a b | .sub a b | .mul a b | .lt a b | .le a b | .eq a b |
    .output a b | .bindArg a b => [a,b]
  | .select p a b => [p,a,b]
  | .branch p i a b => [p,i,a,b]

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
          let some n := a.toInt? | throw s!"invalid signed i64 {a}"
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

def toBril (p : Program) : Check String :=
  flowToBril p.inputs (.pure p.body)

private def showTraces (before after : Value) : IO Unit := do
  let old ← IO.ofExcept (traceOf before)
  let newTrace ← IO.ofExcept (traceOf after)
  unless old == newTrace do throw (IO.userError "trace disagreement")
  IO.println ("input_trace=" ++ (toJson old).compress)
  IO.println ("output_trace=" ++ (toJson newTrace).compress)

def runAcyclic (p : Program) (values : List String) (emit : Bool) : IO Unit := do
  let result ← IO.mkRef (⟨p.body, fun _ => rfl⟩ : Egglog.Compiler.Result language p.body)
  let before ← IO.monoNanosNow
  result.set (EggCC.optimize p.body)
  let after ← IO.monoNanosNow
  let candidate ← result.get
  let checked ← IO.ofExcept (chooseBril p.inputs Flow.pure p.body candidate.output (fun _ => by
    funext v
    exact congrArg Flow.Observation.done
      (congrFun (candidate.sound (fun n _ => .data (.word (BitVec.ofNat 64 n)))) v)))
  let out := {p with body := checked.program}
  if emit then IO.print checked.text
  else
    let (size, branches, selects, binds) := stats p.body
    let (outsize, ob, os, obind) := stats out.body
    IO.println s!"solver_ns={after-before} input_dag={size} output_dag={outsize} input_if={branches} output_if={ob} input_select={selects} output_select={os} input_bind={binds} output_bind={obind}"
    if !values.isEmpty || (p.inputs.all (· == .token)) then
      let arg ← IO.ofExcept (readInputs p.inputs values)
      showTraces (p.body.eval arg) (out.body.eval arg)

private def loopCount : Flow → Nat
  | .pure _ => 0
  | .loop b => 1 + loopCount b
  | .seq a b | .letVal a b | .branch _ a b => loopCount a + loopCount b

private def observed (fuel : Nat) (f : Flow) (arg : Value) : Check Value :=
  match Flow.observe fuel f arg with
  | .done v => .ok v
  | .paused _ => .error s!"interpreter backedge budget {fuel} exhausted"
  | .stuck => .error "malformed loop continuation"

def runStructured (inputs : Array Ty) (nodes : Nat) (body : Flow)
    (values : List String) (emit : Bool) (fuel : Nat) : IO Unit := do
  let result ← IO.mkRef (⟨body, fun _ => rfl⟩ :
    {f : Flow // ∀ k, Flow.observe k f = Flow.observe k body})
  let before ← IO.monoNanosNow
  result.set ⟨body.optimize, body.optimize_observe⟩
  let after ← IO.monoNanosNow
  let candidate ← result.get
  let checked ← IO.ofExcept (chooseBril inputs id body candidate.val candidate.property)
  let out := checked.program
  if emit then IO.print checked.text
  else
    IO.println s!"solver_ns={after-before} input_nodes={nodes} input_loops={loopCount body} output_loops={loopCount out}"
    if !values.isEmpty || inputs.all (· == .token) then
      let args ← IO.ofExcept (readInputs inputs values)
      showTraces (← IO.ofExcept (observed fuel body args)) (← IO.ofExcept (observed fuel out args))

end EggCC.Driver

open EggCC.Driver

def main (args : List String) : IO Unit := do
  let (emit, args) := match args with | "--emit-bril" :: more => (true, more) | _ => (false, args)
  let (fuel, args) ← match args with
    | "--fuel" :: n :: more => do
      let some budget := n.toNat? | throw (IO.userError "expected a natural backedge budget")
      pure (budget, more)
    | _ => pure (10000, args)
  let path :: values := args | throw (IO.userError "usage: eggcc-certified [--emit-bril] [--fuel N] FILE [ARGS...]")
  match ← IO.ofExcept (importProgram (← IO.FS.readFile path)) with
  | .scalar p => runAcyclic p values emit
  | .structured inputs nodes body => runStructured inputs nodes body values emit fuel
