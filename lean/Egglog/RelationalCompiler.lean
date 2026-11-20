import Egglog.Verified

namespace Egglog

/-- A positive relational premise or action. An equality in a premise matches
    existing e-classes. In a rule head it adds terms and unions their classes. -/
inductive RelAtom where
  | eq (left right : Pat)
  | row (name : String) (args : List Pat)
  | cell (name : String) (keys : List Pat) (value : Pat)
  deriving DecidableEq, Repr

/-- A partially applied table accessor. Looking up or setting its value uses
    the named function's invariant and merge implementation in the model. -/
structure FunctionKey where
  name : String
  keys : List Pat

def FunctionKey.cell (f : FunctionKey) (value : Pat) : RelAtom := .cell f.name f.keys value

def RelAtom.Holds (atom : RelAtom) (m : Model α) (env : Nat → α) : Prop :=
  match atom with
  | .eq left right => left.eval m env = right.eval m env
  | .row name args => m.relation name (Pat.evalArgs args m env)
  | .cell name keys value => m.CellHolds name (Pat.evalArgs keys m env) (value.eval m env)

def RelAtom.variables : RelAtom → List Nat
  | .eq left right => left.variables ++ right.variables
  | .row _ args => Pat.argsVariables args
  | .cell _ keys value => Pat.argsVariables keys ++ value.variables

/-- One action justified by the atoms matched at the start of the round. -/
structure RelAction (m : Model α) (body : List RelAtom) where
  head : RelAtom
  sound : ∀ env, (∀ a ∈ body, a.Holds m env) → head.Holds m env

/-- Successful joint matches justify asserting the head at every valuation of
    the free inputs. The witness proof erases in native compilation. -/
structure RelRule (m : Model α) where
  body : List RelAtom
  head : RelAtom
  sound : ∀ env, (∀ a ∈ body, a.Holds m env) → head.Holds m env
  extra : List (RelAction m body) := []
  /-- Distinguishes independently scheduled instances with identical patterns.
      The DSL supplies a declaration-site identifier. Programmatically assembled
      rules can set this field to give each copy its own semi-naive cursor. -/
  name : Option String := none

end Egglog
