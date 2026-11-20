import Egglog.Examples.EggCCFlowSemantics

open Egglog Egglog.Compiler

namespace EggCC.Flow

/-- Match the upstream integer representation, including wrapping and signed
    comparisons. The dependent result records the guard's value at every input. -/
private def staticBool : (p : Expr) → Option {b : Bool // ∀ i, p.eval i = .data (.flag b)}
  | .yes => some ⟨true, fun _ => rfl⟩
  | .no => some ⟨false, fun _ => rfl⟩
  | .lt (.num a) (.num b) =>
    some ⟨(BitVec.ofNat 64 a).slt (BitVec.ofNat 64 b), fun _ => rfl⟩
  | .le (.num a) (.num b) =>
    some ⟨(BitVec.ofNat 64 a).sle (BitVec.ofNat 64 b), fun _ => rfl⟩
  | .eq (.num a) (.num b) =>
    some ⟨Atom.word (BitVec.ofNat 64 a) == Atom.word (BitVec.ofNat 64 b), fun _ => rfl⟩
  | _ => none

private def withStop (obs : Observation) : Observation :=
  obs.thenDo (fun v => .done (.pair (.data (.flag false)) v))

/-- The denotation of a structured region at every observation budget. -/
abbrev Process := Nat → Value → Observation

/-- Structured control with opaque scalar atoms. A branch predicate names a
    scalar atom from the same table. The extractor can introduce a false
    continuation around an arbitrary region by moving it past binders. -/
inductive Control (α : Type) where
  | atom (value : α)
  | identity
  | seq (first next : Control α)
  | letVal (rhs body : Control α)
  | branch (pred : α) (yes no : Control α)
  | loop (body : Control α)
  | withFalseExit (body : Control α)
  deriving Inhabited, DecidableEq, Repr

egg_signature ControlOps : Process where
  | identity => fun _ i => .done i
  | seq a b => fun k i => (a k i).thenDo (b k)
  | letVal a b => fun k i => (a k i).thenDo (fun v => b k (.pair v i))
  | branch p y n => fun k i => (p k i).thenDo (fun v => if v.test then y k i else n k i)
  | loop a costing 4 => fun k i => iterate (a k) k i
  | withFalseExit a => fun k i => withStop (a k i)
  otherwise fun _ _ => .stuck

namespace Control

egg_codec encode, decode : Control Nat where
  | [] .atom n <=> .var n
  | [] .identity <=> ControlOps.identity
  | [a,b] .seq a b <=> ControlOps.seq a b
  | [a,b] .letVal a b <=> ControlOps.letVal a b
  | [y,n] .branch p y n <=> ControlOps.branch (.var p) y n
  | [b] .loop b <=> ControlOps.loop b
  | [b] .withFalseExit b <=> ControlOps.withFalseExit b

def eval (env : α → Process) : Control α → Process
  | .atom x => env x
  | .identity => fun _ i => .done i
  | .seq a b => fun k i => (a.eval env k i).thenDo (b.eval env k)
  | .letVal a b => fun k i => (a.eval env k i).thenDo (fun v => b.eval env k (.pair v i))
  | .branch p y n => fun k i =>
    (env p k i).thenDo fun v => if v.test then y.eval env k i else n.eval env k i
  | .loop b => fun k i => iterate (b.eval env k) k i
  | .withFalseExit b => fun k i => withStop (b.eval env k i)

def language : Language (Control Nat) Process where
  model := ControlOps.model
  encode := encode
  decode := decode
  denote src env := src.eval env
  encode_sound src env := by
    induction src <;> simp_all [encode, eval, ControlOps.model, Pat.eval, Pat.evalArgs]

end Control

namespace ControlOps

/-- One post-tested body execution always happens, including its effects. -/
def eliminateLoop : VerifiedRule model :=
  egg_rule (b) : loop (withFalseExit b) ==> b by
    funext k i
    change iterate (fun v => withStop (b k v)) k i = b k i
    rw [iterate]
    cases b k i <;> rfl

private theorem withStop_thenDo (x : Observation) (k : Value → Observation) :
    x.thenDo (fun v => withStop (k v)) = withStop (x.thenDo k) := by cases x <;> rfl

def rules : Array (VerifiedRule model) := #[
  eliminateLoop,
  egg_rule (a,b) : seq a (withFalseExit b) ==> withFalseExit (seq a b) by
    funext k i; exact withStop_thenDo (a k i) (b k),
  egg_rule (a,b) : letVal a (withFalseExit b) ==> withFalseExit (letVal a b) by
    funext k i; exact withStop_thenDo (a k i) (fun v => b k (.pair v i)),
  egg_rule (p,a,b) : branch p (withFalseExit a) (withFalseExit b) ==> withFalseExit (branch p a b) by
    funext k i
    change (p k i).thenDo (fun v => if v.test then withStop (a k i) else withStop (b k i)) =
      withStop ((p k i).thenDo (fun v => if v.test then a k i else b k i))
    cases p k i with
    | done v => cases h : v.test <;> simp [Observation.thenDo, h]
    | stuck | paused _ => rfl,
  egg_rule (a) : seq identity a ==> a by rfl,
  egg_rule (a) : seq a identity ==> a by
    funext k i; change (a k i).thenDo Observation.done = a k i; cases a k i <;> rfl]

def schedule (rounds : Nat) : Certified.Schedule model := egg_schedule {
  saturate rounds { run rules }
}

end ControlOps

namespace Control

private def symbols : Control α → List α
  | .atom x => [x]
  | .identity => []
  | .seq a b | .letVal a b => a.symbols ++ b.symbols
  | .branch p a b => p :: (a.symbols ++ b.symbols)
  | .loop b | .withFalseExit b => b.symbols

private def index (idOf : α → Nat) : Control α → Control Nat
  | .atom x => .atom (idOf x)
  | .identity => .identity
  | .seq a b => .seq (a.index idOf) (b.index idOf)
  | .letVal a b => .letVal (a.index idOf) (b.index idOf)
  | .branch p y n => .branch (idOf p) (y.index idOf) (n.index idOf)
  | .loop b => .loop (b.index idOf)
  | .withFalseExit b => .withFalseExit (b.index idOf)

private theorem index_sound (c : Control α) (idOf : α → Nat) (vars : Nat → Process) (env : α → Process)
    (ok : ∀ x ∈ c.symbols, vars (idOf x) = env x) : (c.index idOf).eval vars = c.eval env := by
  induction c with
  | atom x => exact ok _ (by simp [symbols])
  | identity => rfl
  | seq a b ha hb | letVal a b ha hb =>
    have l := ha (fun v h => ok v (by simp [symbols, h]))
    have r := hb (fun v h => ok v (by simp [symbols, h]))
    simp only [index, eval, l, r]
  | branch p a b ha hb =>
    have l := ha (fun v h => ok v (by simp [symbols, h]))
    have r := hb (fun v h => ok v (by simp [symbols, h]))
    have guard := ok p (by simp [symbols])
    simp only [index, eval, guard, l, r]
  | loop b hb | withFalseExit b hb => simp [index, eval, hb ok]

private def size : Control α → Nat
  | .atom _ | .identity => 1
  | .seq a b | .letVal a b | .branch _ a b => 1 + a.size + b.size
  | .loop b | .withFalseExit b => 1 + b.size

private def exprProcess (e : Expr) : Process := fun _ i => .done (e.eval i)

private def appendFalseExit : Flow → Flow
  | .pure e => .pure (.pair .no e)
  | f => .seq f (.pure (.pair .no .arg))

private theorem appendFalseExit_sound (f : Flow) (fuel : Nat) (i : Value) :
    observe fuel (appendFalseExit f) i = withStop (observe fuel f i) := by cases f <;> rfl

private def emit (exprAt : α → Expr) : Control α → Flow
  | .atom x => .pure (exprAt x)
  | .identity => .pure .arg
  | .seq a b => .seq (a.emit exprAt) (b.emit exprAt)
  | .letVal a b => .letVal (a.emit exprAt) (b.emit exprAt)
  | .branch p y n => .branch (exprAt p) (y.emit exprAt) (n.emit exprAt)
  | .loop b => .loop (b.emit exprAt)
  | .withFalseExit b => appendFalseExit (b.emit exprAt)

private theorem emit_sound (c : Control α) (exprAt : α → Expr) (k : Nat) :
    observe k (c.emit exprAt) = c.eval (fun x => exprProcess (exprAt x)) k := by
  induction c <;> funext i <;>
    simp_all only [emit, eval, observe, exprProcess, Observation.thenDo, appendFalseExit_sound] <;> rfl

private def leaf (e : Expr) : Control Expr :=
  if e = .arg then .identity else .atom e

private theorem eval_leaf (e : Expr) : (leaf e).eval exprProcess = exprProcess e := by
  unfold leaf; split <;> subst_vars <;> rfl

/-- Reflect a proved statically false scalar continuation as an e-graph operator. -/
private def scalar : Expr → Control Expr
  | e@(.pair p tail) =>
    match staticBool p with
    | some ⟨false, _⟩ => .withFalseExit (leaf tail)
    | _ => .atom e
  | e => leaf e

private theorem eval_scalar (e : Expr) : (scalar e).eval exprProcess = exprProcess e := by
  unfold scalar
  split
  · split
    · rename_i sound _
      funext k i
      simp only [eval, eval_leaf, exprProcess, withStop, Observation.thenDo]
      change _ = Observation.done (.pair ((_ : Expr).eval i) _)
      rw [sound]; rfl
    · rfl
  · exact eval_leaf _

private def reify : Flow → Control Expr
  | .pure e => scalar (EggCC.optimize e).output
  | .seq a b => .seq (reify a) (reify b)
  | .letVal a b => .letVal (reify a) (reify b)
  | .branch p y n => .branch (EggCC.optimize p).output (reify y) (reify n)
  | .loop b => .loop (reify b)

private theorem reify_sound (f : Flow) :
    (reify f).eval exprProcess = (fun k => observe k f) := by
  induction f <;> funext k i <;>
    simp_all [reify, eval, eval_scalar, exprProcess, observe, Observation.thenDo, EggCC.optimize_correct]

end Control

open Control in
private def compile (f : Flow) : {out : Flow // ∀ fuel, observe fuel out = observe fuel f} := by
  let c := reify f
  let atoms := c.symbols
  let exprAt := fun (n : Nat) => atoms[n]?.getD Expr.arg
  let vars := fun n => exprProcess (exprAt n)
  let term := c.index atoms.idxOf
  let chosen := Control.language.optimize (ControlOps.schedule c.size) term ControlOps.cost
  exact ⟨emit exprAt chosen.output, fun fuel => by
    have lookupOk : ∀ x ∈ atoms, exprAt (atoms.idxOf x) = x := by
      intro x hx
      simp only [exprAt, List.getElem?_eq_getElem (List.idxOf_lt_length_of_mem hx), Option.getD_some]
      exact List.getElem_idxOf ..
    have equal := ((chosen.sound vars).trans
      ((index_sound c atoms.idxOf vars exprProcess (fun x hx => congrArg exprProcess (lookupOk x hx))).trans
        (reify_sound f)))
    exact (emit_sound chosen.output exprAt fuel).trans (congrFun equal fuel)⟩

/-- Optimize scalar regions and run one shared certified equality-saturation
    schedule over the complete structured program, followed by extraction. -/
def optimize (f : Flow) := (compile f).val

theorem optimize_observe (f : Flow) (fuel : Nat) :
    observe fuel (optimize f) = observe fuel f := (compile f).property fuel

theorem optimize_diverges (f : Flow) (i : Value) :
    Diverges (optimize f) i ↔ Diverges f i := by
  simp only [Diverges, optimize_observe]

theorem optimize_correct (f : Flow) (i o : Value) :
    Evaluates (optimize f) i o ↔ Evaluates f i o := by
  rw [← run_iff, ← run_iff]
  simp only [Flow.run, optimize_observe]

end EggCC.Flow
