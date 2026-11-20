import Egglog.Examples.EggCCFlow
import Egglog.Examples.EggCCTypes

namespace EggCC.Driver

namespace FlowBril

/-- Tuples are carried as individual registers. State is represented by the version
    of Bril's implicit print stream that this wire denotes. Saved regions can
    contain aliases of earlier versions; consuming such an alias would fork or
    discard a visible effect. -/
private inductive Wire where
  | reg (name : String) (type : Ty)
  | token (version : Option Nat)
  | nil
  | pair (lhs rhs : Wire)
  deriving Inhabited

private structure Code where
  lines : Array String := #[]
  next : Nat := 1
  world : Nat := 0
  cache : Std.HashMap Expr Wire := {}

private abbrev Emit := StateT Code (Except String)

private def fresh : Emit Nat := do
  let s ← get
  set {s with next := s.next + 1}
  return s.next

private def line (s : String) : Emit Unit :=
  modify fun c => {c with lines := c.lines.push s}

private def freshWorld : Emit Nat := do
  let w ← fresh
  modify fun c => {c with world := w}
  return w

private def withLocalCache (action : Emit α) : Emit α := do
  let old := (← get).cache
  modify fun c => {c with cache := {}}
  let r ← action
  modify fun c => {c with cache := old}
  return r

private def scalar : Wire → Ty → Emit String
  | .reg x t, expected => do
    unless t == expected do throw "scalar type mismatch"
    return x
  | _, _ => throw "expected a scalar register"

private def instruction (t : Ty) (rhs : String) : Emit Wire := do
  let name := s!"v{← fresh}"
  line s!"  {name}: {t.show} = {rhs};"
  return .reg name t

private def freshLike : Wire → Emit Wire
  | .reg _ t => do return .reg s!"v{← fresh}" t
  | .pair a b => do return .pair (← freshLike a) (← freshLike b)
  | x => pure x

private def assign : Wire → Wire → Emit Unit
  | .reg dst a, .reg src b => do
    unless a == b do throw "mismatched scalar types at control-flow edge"
    if dst != src then line s!"  {dst}: {a.show} = id {src};"
  | .pair a b, .pair c d => do assign a c; assign b d
  | .nil, .nil | .token _, .token _ => pure ()
  | _, _ => throw "mismatched tuple shapes at control-flow edge"

private def overwritten (dst src : Wire) (names : Std.HashSet String) : Std.HashSet String :=
  match dst, src with
  | .reg x _, .reg y _ => if x == y then names else names.insert x
  | .pair a b, .pair c d => overwritten b d (overwritten a c names)
  | _, _ => names

/-- Read endangered source registers before the parallel assignment. A computed
    value from this iteration already has its own register and needs no copy. -/
private def saveReads (names : Std.HashSet String) : Wire → Emit Wire
  | .reg x t => if names.contains x then instruction t s!"id {x}" else pure (.reg x t)
  | .pair a b => do return .pair (← saveReads names a) (← saveReads names b)
  | x => pure x

/-- At a control-flow join, a returned token can authorize further effects
    only when it denotes the final stream on both paths. -/
private def joinTokens (leftWorld rightWorld newWorld : Nat) : Wire → Wire → Wire
  | .token a, .token b =>
      .token (if a == some leftWorld && b == some rightWorld then some newWorld else none)
  | .pair a b, .pair c d =>
      .pair (joinTokens leftWorld rightWorld newWorld a c) (joinTokens leftWorld rightWorld newWorld b d)
  | a, _ => a

private def sameTokenSlots (leftWorld rightWorld : Nat) : Wire → Wire → Bool
  | .token a, .token b => (a == some leftWorld) == (b == some rightWorld)
  | .pair a b, .pair c d =>
      sameTokenSlots leftWorld rightWorld a c && sameTokenSlots leftWorld rightWorld b d
  | _, _ => true -- scalar types and tuple shapes are checked by assign

private def conditional (guard : String) (onTrue onFalse : Emit Wire) : Emit Wire := do
  let id ← fresh
  let entry := (← get).world
  line s!"  br {guard} .yes{id} .no{id};"
  line s!".yes{id}:"
  let yes ← withLocalCache onTrue
  let yesWorld := (← get).world
  let result ← freshLike yes
  assign result yes
  line s!"  jmp .join{id};"
  line s!".no{id}:"
  modify fun c => {c with world := entry}
  let no ← withLocalCache onFalse
  let noWorld := (← get).world
  assign result no
  line s!"  jmp .join{id};"
  line s!".join{id}:"
  let world ← if yesWorld == entry && noWorld == entry then pure entry else freshWorld
  return joinTokens yesWorld noWorld world result no

private def choose (guard : String) : Wire → Wire → Emit Wire
  | .reg a ta, .reg b tb => do
    unless ta == tb do throw "different scalar types at select"
    if a == b then return .reg a ta
    instruction ta s!"select {guard} {a} {b}"
  | .pair a b, .pair c d => do return .pair (← choose guard a c) (← choose guard b d)
  | .token a, .token b => pure (.token (if a == b then a else none))
  | .nil, .nil => pure .nil
  | _, _ => throw "different tuple shapes at select"

private def arithmetic (op : String) (out : Ty) (a b : Wire) : Emit Wire := do
  instruction out s!"{op} {← scalar a .int} {← scalar b .int}"

private def expr (env : Wire) (e : Expr) : Emit Wire := do
  if let some wire := (← get).cache[e]? then return wire
  let wire ← match e with
    | .arg => pure env
    | .nil => pure .nil
    | .num n => instruction .int s!"const {(BitVec.ofNat 64 n).toInt}"
    | .yes => instruction .bool "const true"
    | .no => instruction .bool "const false"
    | .pair a b => do pure (.pair (← expr env a) (← expr env b))
    | .head a => do
      let .pair lhs _ ← expr env a | throw "head of a non-pair"
      pure lhs
    | .tail a => do
      let .pair _ rhs ← expr env a | throw "tail of a non-pair"
      pure rhs
    | .neg a => do instruction .bool s!"not {← scalar (← expr env a) .bool}"
    | .add a b => do arithmetic "add" .int (← expr env a) (← expr env b)
    | .sub a b => do arithmetic "sub" .int (← expr env a) (← expr env b)
    | .mul a b => do arithmetic "mul" .int (← expr env a) (← expr env b)
    | .lt a b => do arithmetic "lt" .bool (← expr env a) (← expr env b)
    | .le a b => do arithmetic "le" .bool (← expr env a) (← expr env b)
    | .eq a b => do arithmetic "eq" .bool (← expr env a) (← expr env b)
    | .select pred a b => do
      let c ← scalar (← expr env pred) .bool
      let before := (← get).world
      let lhs ← expr env a
      let rhs ← expr env b
      unless before == (← get).world do throw "cannot speculate print effects in a select"
      choose c lhs rhs
    | .branch pred inputs a b => do
      let c ← scalar (← expr env pred) .bool
      let child ← expr env inputs
      conditional c (expr child a) (expr child b)
    | .bindArg input body => do
      let child ← expr env input
      withLocalCache (expr child body)
    | .output x state => do
      let .token w ← expr env state | throw "print is missing its state token"
      let .reg value _ ← expr env x | throw "print expects a scalar value"
      unless w == some (← get).world do throw "print consumes a stale state token"
      line s!"  print {value};"
      pure (.token (some (← freshWorld)))
  modify fun c => {c with cache := c.cache.insert e wire}
  return wire
termination_by e

private def block (env : Wire) : Flow → Emit Wire
  | .pure e => withLocalCache (expr env e)
  | .seq a b => do block (← block env a) b
  | .letVal rhs body => do block (.pair (← block env rhs) env) body
  | .branch pred a b => do
    let c ← scalar (← withLocalCache (expr env pred)) .bool
    conditional c (block env a) (block env b)
  | .loop body => do
    let id ← fresh
    let entry := (← get).world
    let carried ← freshLike env
    assign carried env
    line s!"  jmp .loop{id};"
    line s!".loop{id}:"
    let .pair flag next ← block carried body | throw "loop body must return a flag and its next input"
    let cond ← scalar flag .bool
    let doneWorld := (← get).world
    unless sameTokenSlots entry doneWorld carried next do
      throw "loop backedge changes which slots carry the live print state"
    -- The flag and every next value may alias an incoming variable (including
    -- a swap of two carried registers). Read all of them before updating any.
    let writes := overwritten carried next {}
    let savedFlag ← saveReads writes (.reg cond .bool)
    let next ← saveReads writes next
    assign carried next
    line s!"  br {← scalar savedFlag .bool} .loop{id} .done{id};"
    line s!".done{id}:"
    return joinTokens entry doneWorld doneWorld carried next

private def mainArgs (inputs : Array Ty) : Except String (List String × Wire) := do
  unless (inputs.filter (· == .token)).size == 1 do throw "expected a single input state token"
  let mut args := []
  let mut wires := []
  for t in inputs, i in [:inputs.size] do
    let name := s!"arg{i}"
    match t with
    | .int | .bool =>
      args := s!"{name}: {t.show}" :: args
      wires := .reg name t :: wires
    | .token => wires := .token (some 0) :: wires
    | _ => throw "unsupported main argument type"
  return (args.reverse, wires.foldl (fun env w => Wire.pair w env) .nil)

end FlowBril

/-- Lower structured regions to Bril with explicit conditional edges and
    loop-carried registers. Validate the linear state path during emission. -/
def flowToBril (inputs : Array Ty) (body : Flow) : Check String := do
  let (args, env) ← FlowBril.mainArgs inputs
  let (out, code) ← (FlowBril.block env body).run {}
  let .pair (.token last) .nil := out | throw "main must return a singleton print state"
  unless last == some code.world do throw "main discarded a print state"
  return "@main(" ++ String.intercalate ", " args ++ ") {\n" ++
    String.intercalate "\n" (code.lines.push "  ret;").toList ++ "\n}\n"

/-- The selected IR preserves the source observations and passed the Bril emitter's
    state-token checks. The emission witness records which IR produced the text. -/
structure CheckedBril {α : Type} (inputs : Array Ty) (asFlow : α → Flow) (source : α) where
  program : α
  text : String
  preserves : ∀ k, Flow.observe k (asFlow program) = Flow.observe k (asFlow source)
  emitted : flowToBril inputs (asFlow program) = .ok text

theorem CheckedBril.evaluates_iff (r : CheckedBril inputs asFlow source) (i o : Value) :
    Flow.Evaluates (asFlow r.program) i o ↔ Flow.Evaluates (asFlow source) i o := by
  rw [← Flow.run_iff, ← Flow.run_iff]
  simp only [Flow.run, r.preserves]

/-- Use the cheaper extracted candidate if it can be emitted with a linear print
    state. Check the original as well: its functional semantics can discard an
    effect even when its dataflow cannot be executed on one imperative stream. -/
def chooseBril {α : Type} (inputs : Array Ty) (asFlow : α → Flow) (source candidate : α)
    (sound : ∀ k, Flow.observe k (asFlow candidate) = Flow.observe k (asFlow source)) :
    Check (CheckedBril inputs asFlow source) :=
  match original : flowToBril inputs (asFlow source) with
  | .error e => .error s!"source cannot be lowered to Bril: {e}"
  | .ok originalText =>
    match chosen : flowToBril inputs (asFlow candidate) with
    | .ok text => .ok ⟨candidate, text, sound, chosen⟩
    | .error _ => .ok ⟨source, originalText, fun _ => rfl, original⟩

def optimizeToBril (inputs : Array Ty) (source : Flow) : Check (CheckedBril inputs id source) :=
  chooseBril inputs id source source.optimize source.optimize_observe

end EggCC.Driver
