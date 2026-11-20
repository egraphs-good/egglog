import Egglog.Examples.EggCC

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

def project (n : Nat) (x : Expr) : Expr := .head (Nat.rec x (fun _ acc => Expr.tail acc) n)

deriving instance Hashable for EggCC.Expr

end EggCC.Driver
