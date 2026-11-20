import Egglog.Compiler

open Egglog Egglog.Compiler

namespace EggCC

/-- A state token denotes the output trace so far, in reverse chronological order.
    This IR fragment has integer/Boolean operations, tuples, regions and print. -/
inductive Atom where
  | word (bits : BitVec 64)
  | flag (b : Bool)
  deriving DecidableEq, Repr, Inhabited

inductive Value where
  | data (x : Atom)
  | state (trace : List Atom)
  | pair (a b : Value)
  | nil
  | invalid
  deriving DecidableEq, Repr, Inhabited

abbrev Region := Value → Value

namespace Value

def head : Value → Value
  | .pair a _ => a
  | _ => .invalid

def tail : Value → Value
  | .pair _ b => b
  | _ => .invalid

def test : Value → Bool
  | .data (.flag b) => b
  | _ => false

def put : Value → Value → Value
  | .data x, .state t => .state (x :: t)
  | _, _ => .invalid

def binary (op : BitVec 64 → BitVec 64 → Atom) : Value → Value → Value
  | .data (.word x), .data (.word y) => .data (op x y)
  | _, _ => .invalid

end Value

/-- The region argument is dynamically rebound by If. `bindArg` is the
    explicit context substitution used during if-conversion. -/
inductive Expr where
  | arg | nil | num (i : Nat) | yes | no
  | pair (x y : Expr) | head (x : Expr) | tail (x : Expr)
  | add (a b : Expr) | sub (a b : Expr) | mul (a b : Expr)
  | lt (a b : Expr) | le (a b : Expr) | eq (a b : Expr) | neg (x : Expr)
  | branch (p inputs thn els : Expr)
  | select (p thn els : Expr)
  | bindArg (inputs body : Expr)
  | output (x state : Expr)
  deriving DecidableEq, Repr, Inhabited

private def plus := Value.binary (fun x y => .word (x + y))
private def minus := Value.binary (fun x y => .word (x - y))
private def times := Value.binary (fun x y => .word (x * y))
private def less := Value.binary (fun x y => .flag (x.slt y))
private def lessEq := Value.binary (fun x y => .flag (x.sle y))

def equalVal : Value → Value → Value
  | .data a, .data b => .data (.flag (a == b))
  | _, _ => .invalid

def Expr.evalWith (constants : Nat → Region) : Expr → Region
  | .arg => id
  | .nil => fun _ => .nil
  | .num i => fun _ => constants i .nil
  | .yes => fun _ => .data (.flag true)
  | .no => fun _ => .data (.flag false)
  | .pair a b => fun v => .pair (a.evalWith constants v) (b.evalWith constants v)
  | .head a => fun v => (a.evalWith constants v).head
  | .tail a => fun v => (a.evalWith constants v).tail
  | .add a b => fun v => plus (a.evalWith constants v) (b.evalWith constants v)
  | .sub a b => fun v => minus (a.evalWith constants v) (b.evalWith constants v)
  | .mul a b => fun v => times (a.evalWith constants v) (b.evalWith constants v)
  | .lt a b => fun v => less (a.evalWith constants v) (b.evalWith constants v)
  | .le a b => fun v => lessEq (a.evalWith constants v) (b.evalWith constants v)
  | .eq a b => fun v => equalVal (a.evalWith constants v) (b.evalWith constants v)
  | .neg a => fun v => .data (.flag (!(a.evalWith constants v).test))
  | .branch p i a b => fun v =>
      if (p.evalWith constants v).test then a.evalWith constants (i.evalWith constants v)
      else b.evalWith constants (i.evalWith constants v)
  | .select p a b => fun v =>
      if (p.evalWith constants v).test then a.evalWith constants v else b.evalWith constants v
  | .bindArg i a => fun v => a.evalWith constants (i.evalWith constants v)
  | .output a b => fun v => (a.evalWith constants v).put (b.evalWith constants v)

def Expr.eval (x : Expr) : Region := x.evalWith (fun n _ => .data (.word (BitVec.ofNat 64 n)))

egg_signature Ops : Region where
  | arg => id
  | nil => fun _ => .nil
  | num x => fun _ => x .nil
  | yes => fun _ => .data (.flag true)
  | no => fun _ => .data (.flag false)
  | pair a b => fun v => .pair (a v) (b v)
  | head a => fun v => (a v).head
  | tail a => fun v => (a v).tail
  | add a b => fun v => plus (a v) (b v)
  | sub a b => fun v => minus (a v) (b v)
  | mul a b => fun v => times (a v) (b v)
  | lt a b => fun v => less (a v) (b v)
  | le a b => fun v => lessEq (a v) (b v)
  | eq a b => fun v => equalVal (a v) (b v)
  | neg a => fun v => .data (.flag (!(a v).test))
  | branch p i a b costing 4096 => fun v => if (p v).test then a (i v) else b (i v)
  | select p a b costing 2 => fun v => if (p v).test then a v else b v
  | bindArg i a costing 1 => fun v => a (i v)
  | output a b => fun v => (a v).put (b v)
  otherwise fun _ => .invalid

namespace Expr
egg_codec encode, decode : Expr using Ops where
  | num i <=> Ops.num (.var i)
end Expr

def language : Language Expr Region where
  model := Ops.model
  encode := Expr.encode
  decode := Expr.decode
  denote src env := src.evalWith env
  encode_sound src env := by
    induction src <;> funext v <;>
      simp_all [Expr.encode, Expr.evalWith, Ops.model, Pat.eval, Pat.evalArgs]

namespace Ops

/-- Rebind each selected arm to its region inputs before lowering the If.
    Replacing a branch-local Arg directly in the original environment is unsound. -/
def ifConversion : VerifiedRule model :=
  egg_rule (p,i,a,b) : branch p i a b ==> select p (bindArg i a) (bindArg i b) by rfl

def projection : Array (VerifiedRule model) := #[
  egg_rule (a,b) : head (pair a b) ==> a by rfl,
  egg_rule (a,b) : tail (pair a b) ==> b by rfl,
  egg_rule (p,a,b) : head (select p a b) ==> select p (head a) (head b) by
    funext v; change (if (p v).test then a v else b v).head = (if (p v).test then (a v).head else (b v).head)
    split <;> rfl,
  egg_rule (p,a,b) : tail (select p a b) ==> select p (tail a) (tail b) by
    funext v; change (if (p v).test then a v else b v).tail = (if (p v).test then (a v).tail else (b v).tail)
    split <;> rfl,
  egg_rule (p,a) : select p a a ==> a by
    funext v; change (if (p v).test then a v else a v) = a v; split <;> rfl,
  egg_rule (a,b) : select yes a b ==> a by rfl,
  egg_rule (a,b) : select no a b ==> b by rfl]

end Ops

/-- The e-graph explores region If and value Select. Branch bindings produced by
    the rule remain explicit until the extractor has selected an implementation. -/
def schedule : Certified.Schedule language.model := egg_schedule {
  run #[Ops.ifConversion],
  run Ops.projection
}

namespace Expr

private def pick (p a b : Expr) : Expr :=
  if a = b then a else match p with
  | .yes => a
  | .no => b
  | _ => .select p a b

private theorem pick_sound (p a b : Expr) (k : Nat → Region) (v : Value) :
    (pick p a b).evalWith k v = if (p.evalWith k v).test then a.evalWith k v else b.evalWith k v := by
  unfold pick
  split
  · rename_i h; rw [h]; simp
  · cases p <;> rfl

/-- A projection through an extracted select demands only the chosen column
    of each tuple. The selector can then disappear when both columns agree. -/
private def projectHead : Expr → Expr
  | .pair a _ => a
  | .select p a b => pick p (projectHead a) (projectHead b)
  | e => .head e

private def projectTail : Expr → Expr
  | .pair _ b => b
  | .select p a b => pick p (projectTail a) (projectTail b)
  | e => .tail e

private theorem head_sound (e : Expr) (k : Nat → Region) (v : Value) :
    (projectHead e).evalWith k v = (e.evalWith k v).head := by
  induction e <;> try rfl
  case select p a b _ ha hb =>
    simp only [projectHead, pick_sound, evalWith, ha, hb]
    split <;> rfl

private theorem tail_sound (e : Expr) (k : Nat → Region) (v : Value) :
    (projectTail e).evalWith k v = (e.evalWith k v).tail := by
  induction e <;> try rfl
  case select p a b _ ha hb =>
    simp only [projectTail, pick_sound, evalWith, ha, hb]
    split <;> rfl

/-- Materialize the region argument at its true binder. Nested If arms
    start a fresh context. This stage implements substitution on extracted
    syntax and never introduces an If-to-Select transformation. -/
def instantiate (body context : Expr) : Expr :=
  match body with
  | .arg => context
  | .nil => .nil
  | .num n => .num n
  | .yes => .yes
  | .no => .no
  | .pair a b => .pair (a.instantiate context) (b.instantiate context)
  | .head a => projectHead (a.instantiate context)
  | .tail a => projectTail (a.instantiate context)
  | .add a b => .add (a.instantiate context) (b.instantiate context)
  | .sub a b => .sub (a.instantiate context) (b.instantiate context)
  | .mul a b => .mul (a.instantiate context) (b.instantiate context)
  | .lt a b => .lt (a.instantiate context) (b.instantiate context)
  | .le a b => .le (a.instantiate context) (b.instantiate context)
  | .eq a b => .eq (a.instantiate context) (b.instantiate context)
  | .neg a => .neg (a.instantiate context)
  | .select p a b => pick (p.instantiate context) (a.instantiate context) (b.instantiate context)
  | .branch p i a b =>
      .branch (p.instantiate context) (i.instantiate context) (a.instantiate .arg) (b.instantiate .arg)
  | .bindArg i a => a.instantiate (i.instantiate context)
  | .output a b => .output (a.instantiate context) (b.instantiate context)

theorem instantiate_sound (body context : Expr) (k : Nat → Region) (v : Value) :
    (body.instantiate context).evalWith k v = body.evalWith k (context.evalWith k v) := by
  induction body generalizing context v <;>
    simp_all only [instantiate, evalWith, head_sound, tail_sound, pick_sound, id_eq]

end Expr

def optimize (x : Expr) : Result language x :=
  let chosen := language.optimize schedule x Ops.cost
  ⟨chosen.output.instantiate .arg, fun k => by
    funext v
    exact (Expr.instantiate_sound chosen.output .arg k v).trans (congrFun (chosen.sound k) v)⟩

theorem optimize_correct (x : Expr) (v : Value) :
    (optimize x).output.eval v = x.eval v := by
  exact congrFun ((optimize x).sound (fun n _ => .data (.word (BitVec.ofNat 64 n)))) v

end EggCC
