import Std

namespace Egglog

/-- First-order patterns. Operator IDs refer to a fixed constructor signature. -/
inductive Pat where
  | var (id : Nat)
  | app (op : Nat) (args : List Pat)
  deriving Repr, Inhabited

mutual
  private def Pat.decEq (a b : Pat) : Decidable (a = b) :=
    match a, b with
    | .var i, .var j =>
      if h : i = j then isTrue (h ▸ rfl) else isFalse (by intro e; cases e; exact h rfl)
    | .app f xs, .app g ys =>
      if h : f = g then
        match Pat.decEqArgs xs ys with
        | isTrue e => isTrue (h ▸ e ▸ rfl)
        | isFalse ne => isFalse (by intro e; cases e; exact ne rfl)
      else isFalse (by intro e; cases e; exact h rfl)
    | .var _, .app _ _ | .app _ _, .var _ => isFalse (by intro e; cases e)

  private def Pat.decEqArgs (xs ys : List Pat) : Decidable (xs = ys) :=
    match xs, ys with
    | [], [] => isTrue rfl
    | x :: xs, y :: ys =>
      match Pat.decEq x y, Pat.decEqArgs xs ys with
      | isTrue h, isTrue e => isTrue (h ▸ e ▸ rfl)
      | isFalse ne, _ => isFalse (by intro e; cases e; exact ne rfl)
      | _, isFalse ne => isFalse (by intro e; cases e; exact ne rfl)
    | [], _ :: _ | _ :: _, [] => isFalse (by intro e; cases e)
end

instance : DecidableEq Pat := Pat.decEq

mutual
  /-- Interpret a value term using only its constructors. Function schemas
      use this before a language's relations and functions are assembled. -/
  def Pat.denote (p : Pat) (interp : Nat → List α → α) (env : Nat → α) : α :=
    match p with
    | .var i => env i
    | .app op args => interp op (denoteArgs args interp env)

  def Pat.denoteArgs (ps : List Pat) (interp : Nat → List α → α) (env : Nat → α) : List α :=
    match ps with
    | [] => []
    | p :: ps => p.denote interp env :: denoteArgs ps interp env
end

/-- A partial function stores one value term per tuple of key e-classes.
    The ghost invariant constrains its meaning for every program input.
    A custom merge may compute a new term from both old and incoming values. -/
structure FunctionSpec (α : Type u) (interp : Nat → List α → α) where
  arity : Nat
  invariant : List α → α → Prop
  merge : Pat → Pat → Pat
  merge_sound : ∀ keys old new (env : Nat → α),
    invariant keys (old.denote interp env) → invariant keys (new.denote interp env) →
    invariant keys ((merge old new).denote interp env)

/-- An interpretation of the constructors of a single-sorted signature. -/
structure Model (α : Type u) where
  interp : Nat → List α → α
  /-- Meaning of the tuples asserted in a verified compiler relation. -/
  relation : String → List α → Prop := fun _ _ => False
  functions : String → Option (FunctionSpec α interp) := fun _ => none

def Model.CellHolds (m : Model α) (name : String) (keys : List α) (value : α) : Prop :=
  match m.functions name with
  | none => False
  | some f => keys.length = f.arity ∧ f.invariant keys value

mutual
  def Pat.eval (p : Pat) (m : Model α) (env : Nat → α) : α :=
    match p with
    | .var i => env i
    | .app op args => m.interp op (evalArgs args m env)

  def Pat.evalArgs (args : List Pat) (m : Model α) (env : Nat → α) : List α :=
    match args with
    | [] => []
    | p :: ps => p.eval m env :: evalArgs ps m env
end

mutual
  theorem Pat.denote_eq_eval (p : Pat) (m : Model α) (env : Nat → α) :
      p.denote m.interp env = p.eval m env := by
    cases p with
    | var => rfl
    | app op args => exact congrArg (m.interp op) (denoteArgs_eq_evalArgs args m env)

  theorem Pat.denoteArgs_eq_evalArgs (ps : List Pat) (m : Model α) (env : Nat → α) :
      denoteArgs ps m.interp env = evalArgs ps m env := by
    cases ps with
    | nil => rfl
    | cons p ps => simp only [denoteArgs, evalArgs, denote_eq_eval p m env, denoteArgs_eq_evalArgs ps m env]
end

mutual
  /-- Metavariable occurrences in a rule or a query pattern. -/
  def Pat.variables : Pat → List Nat
    | .var i => [i]
    | .app _ args => Pat.argsVariables args

  def Pat.argsVariables : List Pat → List Nat
    | [] => []
    | p :: ps => p.variables ++ Pat.argsVariables ps
end

structure Rewrite where
  lhs : Pat
  rhs : Pat
  deriving Repr, Inhabited, DecidableEq

/-- The proof is erased at compilation. Both evaluation and execution consume
    the two patterns stored in the inherited `Rewrite` field. -/
structure VerifiedRule (m : Model α) extends Rewrite where
  sound : ∀ env : Nat → α, lhs.eval m env = rhs.eval m env

/-- A conditional rewrite may rely on equalities already proved between
instantiated terms in the e-graph. All premises are interpreted in the same
assignment used for the rewrite. -/
structure GuardedRule (m : Model α) extends Rewrite where
  premises : List (Pat × Pat)
  sound : ∀ env : Nat → α,
    (∀ pair ∈ premises, pair.1.eval m env = pair.2.eval m env) →
    lhs.eval m env = rhs.eval m env

/-- Select the actual certified value; orientation and variable IDs matter. -/
def Rewrite.certify? (r : Rewrite) (rules : List (VerifiedRule m)) : Option (VerifiedRule m) :=
  rules.find? (fun candidate => candidate.toRewrite == r)

theorem Rewrite.certify?_exact {r : Rewrite} {rules : List (VerifiedRule m)}
    {checked : VerifiedRule m} (h : r.certify? rules = some checked) :
    checked.toRewrite = r := by
  have found : (checked.toRewrite == r) = true :=
    List.find?_some (p := fun candidate : VerifiedRule m => candidate.toRewrite == r) h
  exact eq_of_beq found

theorem Rewrite.sound_of_certify? {α : Type u} {m : Model α} {r : Rewrite} {rules : List (VerifiedRule m)}
    {checked : VerifiedRule m} (h : r.certify? rules = some checked) (env : Nat → α) :
    r.lhs.eval m env = r.rhs.eval m env := by
  rw [← certify?_exact h]
  exact checked.sound env

/-- Declaring a natural-number constructor binds its name, arity, and meaning. -/
inductive NatConstructor where
  | constant (name : String) (value : Nat)
  | unary (name : String) (eval : Nat → Nat)
  | binary (name : String) (eval : Nat → Nat → Nat)

def NatConstructor.signature (cs : List NatConstructor) : List (String × Nat) := cs.map fun
  | .constant name _ => (name, 0)
  | .unary name _ => (name, 1)
  | .binary name _ => (name, 2)

def NatConstructor.model (cs : List NatConstructor) : Model Nat where
  interp op args := match cs[op]?, args with
    | some (.constant _ value), [] => value
    | some (.unary _ eval), [x] => eval x
    | some (.binary _ eval), [x, y] => eval x y
    | _, _ => 0

namespace Arithmetic

private def constructors : List NatConstructor :=
  [.constant "Zero" 0, .constant "One" 1, .unary "Succ" Nat.succ,
   .binary "Add" Nat.add, .binary "Mul" Nat.mul]

def signature := NatConstructor.signature constructors

def model := NatConstructor.model constructors

private abbrev x : Pat := .var 0
private abbrev y : Pat := .var 1
private abbrev z : Pat := .var 2
private abbrev zero : Pat := .app 0 []
private abbrev one : Pat := .app 1 []
private abbrev succ (a : Pat) : Pat := .app 2 [a]
private abbrev add (a b : Pat) : Pat := .app 3 [a, b]
private abbrev mul (a b : Pat) : Pat := .app 4 [a, b]

def addZero : VerifiedRule model :=
  ⟨⟨add x zero, x⟩, fun env => Nat.add_zero (env 0)⟩

def mulOne : VerifiedRule model :=
  ⟨⟨mul x one, x⟩, fun env => Nat.mul_one (env 0)⟩

def mulZero : VerifiedRule model :=
  ⟨⟨mul x zero, zero⟩, fun env => Nat.mul_zero (env 0)⟩

def addComm : VerifiedRule model :=
  ⟨⟨add x y, add y x⟩, fun env => Nat.add_comm (env 0) (env 1)⟩

def mulComm : VerifiedRule model :=
  ⟨⟨mul x y, mul y x⟩, fun env => Nat.mul_comm (env 0) (env 1)⟩

def addAssoc : VerifiedRule model :=
  ⟨⟨add x (add y z), add (add x y) z⟩,
    fun env => (Nat.add_assoc (env 0) (env 1) (env 2)).symm⟩

def addSelf : VerifiedRule model :=
  ⟨⟨add x x, mul (succ one) x⟩, fun env => (Nat.two_mul (env 0)).symm⟩

def mulAdd : VerifiedRule model :=
  ⟨⟨mul x (add y z), add (mul x y) (mul x z)⟩,
    fun env => Nat.mul_add (env 0) (env 1) (env 2)⟩

def rules : List (VerifiedRule model) :=
  [addZero, mulOne, mulZero, addComm, mulComm, addAssoc, addSelf, mulAdd]

end Arithmetic
end Egglog
