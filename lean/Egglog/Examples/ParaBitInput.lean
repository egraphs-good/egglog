import Egglog.Examples.BWLang
import Egglog.Examples.ParaBit

namespace BWLang
open Egglog ParaBit

/-- Denotation of the source operator, before symbol interning or e-graph encoding. -/
def applyWord : String → List Int → Int
  | "bw", [p,x] => trunc p x
  | "signed", [p,x] => signedVal p x
  | "+", [a,b] => a+b | "-", [a,b] => a-b | "-", [a] => -a
  | "*", [a,b] => a*b | "div", [a,b] => a.fdiv b | "^", [a,b] => a ^ b.toNat
  | "<<", [a,b] => a * modulus b | ">>", [a,b] => a / modulus b
  | "and", [a,b] => bitAnd a b | "or", [a,b] => bitOr a b
  | "xor", [a,b] => bitXor a b | "not", [a] => bitNot a
  | "sel", [c,a,b] => if c % 2 = 1 then a else b
  | "<", [a,b] | ">", [b,a] => truth (a < b)
  | "<=", [a,b] | ">=", [b,a] => truth (a ≤ b)
  | _, _ => 0

mutual
  def Term.eval (ρ : String → Int) : Term → Int
    | .literal n => n
    | .symbol name => ρ name
    | .app op xs => applyWord op (Term.evalArgs ρ xs)
  def Term.evalArgs (ρ : String → Int) : List Term → List Int
    | [] => []
    | x :: xs => x.eval ρ :: Term.evalArgs ρ xs
end

private theorem node_sound (ρ : Nat → Int) (op : String) (xs : List Pat) (p : Pat)
    (h : node? op xs = some p) :
    denote ρ p = applyWord op (xs.map (denote ρ)) := by
  fun_cases node? op xs <;> simp_all [node?] <;> subst p <;> rfl

def Problem.premises (p : Problem) : List Term :=
  p.conditions ++ p.positiveWidths.map (fun w => .app ">" [w, .literal 0])

def Problem.Holds (p : Problem) (ρ : String → Int) : Prop :=
  ∀ t ∈ p.premises, t.eval ρ = 1

private abbrev liftNames (names : Array String) (ρ : String → Int) (i : Nat) := ρ names[i]!

private structure Encoded (names : Array String) (t : Term) where
  pat : Pat
  sound : ∀ ρ, denote (liftNames names ρ) pat = t.eval ρ

private structure EncodedArgs (names : Array String) (xs : List Term) where
  pats : List Pat
  sound : ∀ ρ, pats.map (denote (liftNames names ρ)) = Term.evalArgs ρ xs

mutual
  private def encode (names : Array String) (t : Term) : Except String (Encoded names t) :=
    match t with
    | .literal n => .ok ⟨lit n, fun _ => denote_lit n _⟩
    | .symbol s =>
      let i := names.toList.idxOf s
      if h : names[i]! = s then
        .ok ⟨symbol i, fun ρ => (denote_symbol i _).trans (congrArg ρ h)⟩
      else .error s!"unregistered variable {s}"
    | .app op xs => do
      let args ← encodeArgs names xs
      match h : node? op args.pats with
      | none => throw s!"unsupported ModIR operator {op}"
      | some p => return ⟨p, fun ρ => (node_sound _ _ _ _ h).trans (congrArg _ (args.sound ρ))⟩
  private def encodeArgs (names : Array String) (xs : List Term) : Except String (EncodedArgs names xs) :=
    match xs with
    | [] => .ok ⟨[], fun _ => rfl⟩
    | x :: xs => do
      let hd ← encode names x
      let tl ← encodeArgs names xs
      return ⟨hd.pat :: tl.pats, fun ρ => by simp only [List.map_cons, Term.evalArgs, hd.sound, tl.sound]⟩
end

mutual
  private def symbols : Term → List String
    | .symbol s => [s]
    | .app _ xs => symbolsArgs xs
    | .literal _ => []
  private def symbolsArgs : List Term → List String
    | [] => []
    | t :: ts => symbols t ++ symbolsArgs ts
end

/-- The raw search query, together with the proof that its result entails the
    original BWLang statement at every assignment to the named variables. -/
structure Prepared (p : Problem) where
  query : Query
  sound : (∀ v, query.Holds v → query.Equivalent v) →
    ∀ ρ, p.Holds ρ → p.lhs.eval ρ = p.rhs.eval ρ

def Problem.prepare (p : Problem) : Except String (Prepared p) := do
  let names := ((p.lhs :: p.rhs :: p.premises).flatMap symbols).eraseDups.toArray
  let lhs ← encode names p.lhs
  let rhs ← encode names p.rhs
  let conditions ← encodeArgs names p.premises
  let query : Query := ⟨lhs.pat, rhs.pat, conditions.pats⟩
  return ⟨query, fun found ρ hyp => by
    have allTrue : ∀ xs : List Term, (∀ t ∈ xs, Term.eval ρ t = 1) →
        ∀ i ∈ Term.evalArgs ρ xs, i = 1 := by
      intro xs
      induction xs with
      | nil => simp [Term.evalArgs]
      | cons x xs ih =>
        intro h i hi
        rcases List.mem_cons.mp hi with he | ht
        · exact he.trans (h x (by simp))
        · exact ih (fun t ht => h t (List.mem_cons_of_mem x ht)) i ht
    have holds : query.Holds (liftNames names ρ) := by
      intro t ht
      have hm := List.mem_map_of_mem (f := denote (liftNames names ρ)) ht
      change _ ∈ conditions.pats.map _ at hm
      rw [conditions.sound] at hm
      exact allTrue _ hyp _ hm
    exact (lhs.sound ρ).symm.trans ((found _ holds).trans (rhs.sound ρ))⟩

def Prepared.prove? {p : Problem} (s : Prepared p) (rounds : Nat := 6) :
    Option (PLift (∀ ρ, p.Holds ρ → p.lhs.eval ρ = p.rhs.eval ρ)) := do
  let proof ← s.query.prove? rounds
  return ⟨s.sound proof.down⟩

/-- Compile a named BWLang input and discharge the original semantic equality. -/
def Problem.prove? (p : Problem) (rounds : Nat := 6) :
    Option (PLift (∀ ρ, p.Holds ρ → p.lhs.eval ρ = p.rhs.eval ρ)) := do
  let prepared ← p.prepare.toOption
  prepared.prove? rounds

def Prepared.proveStaged? {p : Problem} (s : Prepared p) (rounds : Nat := 6) :
    Option (PLift (∀ ρ, p.Holds ρ → p.lhs.eval ρ = p.rhs.eval ρ)) := do
  let proof ← s.query.proveStaged? rounds
  return ⟨s.sound proof.down⟩

/-- Prove a named BWLang equality with the staged word/control schedule. -/
def Problem.proveStaged? (p : Problem) (rounds : Nat := 6) :
    Option (PLift (∀ ρ, p.Holds ρ → p.lhs.eval ρ = p.rhs.eval ρ)) := do
  let prepared ← p.prepare.toOption
  prepared.proveStaged? rounds

/-- Recheck a native proposal with the selected, semantically proved rules. -/
def Problem.replay? (p : Problem) (rounds : Nat) (hint : ReplayHint) :
    Option (PLift (∀ ρ, p.Holds ρ → p.lhs.eval ρ = p.rhs.eval ρ)) := do
  let prepared ← p.prepare.toOption
  let proof ← prepared.query.replay? rounds hint
  return ⟨prepared.sound proof.down⟩

end BWLang
