import Egglog.Compiler

open Egglog

namespace ParaBit

/-- Mathematical BWLang operators. Width expressions are integral, with the
    paper's nat conversion at shifts, truncations and powers. -/
def modulus (w : Int) : Int := 2 ^ w.toNat
def trunc (w x : Int) : Int := x % modulus w
def signedVal (w x : Int) : Int := trunc w (2 * trunc (w - 1) x) - trunc w x
def bitNot (a : Int) : Int := -a - 1
def bitAnd : Int → Int → Int
  | .ofNat a, .ofNat b => .ofNat (a &&& b)
  | .ofNat a, .negSucc b | .negSucc b, .ofNat a => .ofNat (a ^^^ (a &&& b))
  | .negSucc a, .negSucc b => .negSucc (a ||| b)
def bitOr (a b : Int) : Int := bitNot (bitAnd (bitNot a) (bitNot b))
def bitXor : Int → Int → Int
  | .ofNat a, .ofNat b | .negSucc a, .negSucc b => .ofNat (a ^^^ b)
  | .ofNat a, .negSucc b | .negSucc b, .ofNat a => .negSucc (a ^^^ b)
def truth (p : Prop) [Decidable p] : Int := if p then 1 else 0
def Fits (x w : Int) := 0 ≤ x ∧ x < modulus w
instance (x w : Int) : Decidable (Fits x w) := inferInstanceAs (Decidable (_ ∧ _))

egg_signature Op : Int where
  | zero => 0
  | one => 1
  | two => 2
  | bw w x => trunc w x
  | signed w x => signedVal w x
  | add a b => a + b
  | sub a b => a - b
  | neg a => -a
  | mul a b => a * b
  | div a b => a.fdiv b
  | pow a b => a ^ b.toNat
  | shl a b => a * modulus b
  | shr a b => a / modulus b
  | band a b => bitAnd a b
  | bor a b => bitOr a b
  | bxor a b => bitXor a b
  | bnot a => bitNot a
  | sel c a b => if c % 2 = 1 then a else b
  | le a b => truth (a ≤ b)
  | lt a b => truth (a < b)
  | fits x w => truth (Fits x w)
  otherwise 0

def node? : String → List Pat → Option Pat
  | "bw", [w,x] => some (Op.bw w x)
  | "signed", [w,x] => some (Op.signed w x)
  | "+", [a,b] => some (Op.add a b)
  | "-", [a,b] => some (Op.sub a b)
  | "-", [a] => some (Op.neg a)
  | "*", [a,b] => some (Op.mul a b)
  | "div", [a,b] => some (Op.div a b)
  | "^", [a,b] => some (Op.pow a b)
  | "<<", [a,b] => some (Op.shl a b)
  | ">>", [a,b] => some (Op.shr a b)
  | "and", [a,b] => some (Op.band a b)
  | "or", [a,b] => some (Op.bor a b)
  | "xor", [a,b] => some (Op.bxor a b)
  | "not", [a] => some (Op.bnot a)
  | "sel", [c,a,b] => some (Op.sel c a b)
  | "<=", [a,b] | ">=", [b,a] => some (Op.le a b)
  | "<", [a,b] | ">", [b,a] => some (Op.lt a b)
  | _, _ => none

def symbol (i : Nat) : Pat := .app (32 + 3*i) []
def lit : Int → Pat
  | 0 => Op.zero
  | 1 => Op.one
  | 2 => Op.two
  | .ofNat n => .app (33 + 3*n) []
  | .negSucc n => .app (34 + 3*n) []

def scalar (ρ : Nat → Int) : Model Int where
  interp op xs := if op < 32 then Op.model.interp op xs else
    let k := op - 32
    match k % 3 with
    | 0 => ρ (k / 3)
    | 1 => .ofNat (k / 3)
    | _ => .negSucc (k / 3)

def denote (ρ : Nat → Int) (p : Pat) : Int := p.eval (scalar ρ) (fun _ => 0)

@[simp] theorem denote_symbol (i : Nat) (ρ : Nat → Int) : denote ρ (symbol i) = ρ i := by
  simp [symbol, denote, Pat.eval, Pat.evalArgs, scalar, Nat.mul_comm 3 i] <;> omega
private theorem lit_pos (k : Nat) (ρ : Nat → Int) :
    denote ρ (Pat.app (33 + 3*k) []) = (k : Int) := by
  change (scalar ρ).interp (33 + 3*k) [] = _
  simp only [scalar]
  rw [if_neg (by omega), show (33 + 3*k - 32) % 3 = 1 by omega,
      show (33 + 3*k - 32) / 3 = k by omega]
  rfl
private theorem lit_neg (k : Nat) (ρ : Nat → Int) :
    denote ρ (Pat.app (34 + 3*k) []) = Int.negSucc k := by
  change (scalar ρ).interp (34 + 3*k) [] = _
  simp only [scalar]
  rw [if_neg (by omega), show (34 + 3*k - 32) % 3 = 2 by omega,
      show (34 + 3*k - 32) / 3 = k by omega]
  rfl
@[simp] theorem denote_lit (i : Int) (ρ : Nat → Int) : denote ρ (lit i) = i := by
  cases i with
  | ofNat n =>
    match n with
    | 0 | 1 | 2 => rfl
    | k + 3 => exact lit_pos (k+3) ρ
  | negSucc n => exact lit_neg n ρ

structure Query where
  lhs : Pat
  rhs : Pat
  assumptions : List Pat
  deriving Repr

def Query.Holds (q : Query) (ρ : Nat → Int) : Prop :=
  ∀ h ∈ q.assumptions, denote ρ h = 1

def Query.Equivalent (q : Query) (ρ : Nat → Int) : Prop :=
  denote ρ q.lhs = denote ρ q.rhs

/-- Search once over all valuations satisfying the query's premises. -/
def point {S : Type} (vars : S → Nat → Int) : Model (S → Int) where
  interp op xs e := (scalar (vars e)).interp op (xs.map (· e))

mutual
  theorem eval_point {S} (v : S → Nat → Int) (env : Nat → S → Int) (e : S) (p : Pat) :
      p.eval (point v) env e = p.eval (scalar (v e)) (fun i => env i e) := by
    cases p with
    | var => rfl
    | app op args => exact congrArg ((scalar (v e)).interp op) (evalArgs_point v env e args)
  theorem evalArgs_point {S} (v : S → Nat → Int) (env : Nat → S → Int) (e : S) (args : List Pat) :
      (Pat.evalArgs args (point v) env).map (· e) =
      Pat.evalArgs args (scalar (v e)) (fun i => env i e) := by
    cases args with
    | nil => rfl
    | cons p ps =>
      simp only [Pat.evalArgs, List.map_cons, eval_point, evalArgs_point]
end

end ParaBit
