import Egglog.Examples.ParaBitBitwise
import Egglog.AnalysisEval

open Egglog
namespace ParaBit.Decision

/-- Each diagram denotes a Boolean function of the current bit of each atom. -/
private abbrev Assignment := Nat → Bool
private def model : Model (Assignment → Bool) where
  interp op args σ := match op, args with
    | 0, [] => false
    | 1, [] => true
    | 2, [a,b] => a σ && b σ
    | 3, [a,b] => a σ || b σ
    | 4, [a,b] => a σ ^^ b σ
    | i+5, [l,h] => if σ i then h σ else l σ
    | _, _ => false

private inductive Gate where | conj | disj | xor
private def Gate.apply : Gate → Bool → Bool → Bool
  | .conj => (· && ·) | .disj => (· || ·) | .xor => (· ^^ ·)
private def Gate.tag : Gate → Nat | .conj => 2 | .disj => 3 | .xor => 4
private def gate (g : Gate) (a b : Pat) : Pat := .app g.tag [a,b]
private def leaf (b : Bool) : Pat := .app (if b then 1 else 0) []
private def node (i : Nat) (l h : Pat) : Pat := .app (i+5) [l,h]
private theorem gate_eval (g a b env σ) : (gate g a b).eval model env σ =
    g.apply (a.eval model env σ) (b.eval model env σ) := by cases g <;> rfl
private theorem leaf_eval (b env σ) : (leaf b).eval model env σ = b := by cases b <;> rfl
private theorem node_eval (i l h env σ) : (node i l h).eval model env σ =
    (if σ i then h.eval model env σ else l.eval model env σ) := rfl

local macro "diagram_law" : tactic => `(tactic|
  (funext σ
   simp +zetaDelta only [gate_eval, node_eval, leaf_eval, Pat.eval] <;> grind))

private def collapse (i : Nat) : VerifiedRule model :=
  egg_rule (a) : node i a a ==> a by diagram_law

private def terminal (g : Gate) (a b : Bool) : VerifiedRule model :=
  egg_rule () : gate g (leaf a) (leaf b) ==> leaf (g.apply a b) by diagram_law

private def splitBoth (g : Gate) (i : Nat) : VerifiedRule model :=
  egg_rule (a,b,c,d) : gate g (node i a b) (node i c d) ==> node i (gate g a c) (gate g b d) by diagram_law

private def splitLowLeft (g : Gate) (i j : Nat) : VerifiedRule model :=
  egg_rule (a,b,c,d) : gate g (node i a b) (node j c d) ==>
    node i (gate g a (node j c d)) (gate g b (node j c d)) by diagram_law

private def splitLowRight (g : Gate) (i j : Nat) : VerifiedRule model :=
  egg_rule (a,b,c,d) : gate g (node j c d) (node i a b) ==>
    node i (gate g (node j c d) a) (gate g (node j c d) b) by diagram_law

private def terminalLeft (g : Gate) (b : Bool) (i : Nat) : VerifiedRule model :=
  egg_rule (l,h) : gate g (leaf b) (node i l h) ==> node i (gate g (leaf b) l) (gate g (leaf b) h) by diagram_law

private def terminalRight (g : Gate) (b : Bool) (i : Nat) : VerifiedRule model :=
  egg_rule (l,h) : gate g (node i l h) (leaf b) ==> node i (gate g l (leaf b)) (gate g h (leaf b)) by diagram_law

private def rules (atoms : Nat) : Array (VerifiedRule model) := Id.run do
  let mut out := #[]
  for i in [:atoms] do out := out.push (collapse i)
  for g in [Gate.conj, .disj, .xor] do
    for a in [false, true] do
      for b in [false, true] do out := out.push (terminal g a b)
    for i in [:atoms] do
      out := out.push (splitBoth g i)
      for b in [false, true] do out := (out.push (terminalLeft g b i)).push (terminalRight g b i)
      for j in [i+1:atoms] do out := (out.push (splitLowLeft g i j)).push (splitLowRight g i j)
  return out

private inductive Form where
  | const (b : Bool)
  | atom (p : Pat)
  | bin (g : Gate) (a b : Form)

private def Form.value (ρ : Nat → Int) (i : Nat) : Form → Bool
  | .const b => b
  | .atom p => digit (denote ρ p) i
  | .bin g a b => g.apply (a.value ρ i) (b.value ρ i)

private structure Reflected (p : Pat) where
  form : Form
  correct : ∀ ρ i, form.value ρ i = digit (denote ρ p) i

/-- Reflection follows the bitwise operators. Arithmetic subterms become independent
    atoms. Identical subterms in either endpoint share an atom. -/
private def reflect : (p : Pat) → Reflected p
  | .app 0 [] => ⟨.const false, fun _ i => (digit_zero i).symm⟩
  | .app 34 [] => ⟨.const true, fun _ i => (digit_ones i).symm⟩
  | .app 3 [w,x] =>
    let a := reflect x
    ⟨.bin .conj (.atom (Op.bw w (lit (-1)))) a.form, fun ρ i => by
      change (digit (trunc (denote ρ w) (-1)) i && a.form.value ρ i) = digit (trunc (denote ρ w) (denote ρ x)) i
      simp [digit_trunc, digit_ones, a.correct]⟩
  | .app 13 [x,y] =>
    let a := reflect x; let b := reflect y
    ⟨.bin .conj a.form b.form, fun ρ i => by
      change (a.form.value ρ i && b.form.value ρ i) = digit (bitAnd (denote ρ x) (denote ρ y)) i
      rw [digit_and, a.correct, b.correct]⟩
  | .app 14 [x,y] =>
    let a := reflect x; let b := reflect y
    ⟨.bin .disj a.form b.form, fun ρ i => by
      change (a.form.value ρ i || b.form.value ρ i) = digit (bitOr (denote ρ x) (denote ρ y)) i
      rw [digit_or, a.correct, b.correct]⟩
  | .app 15 [x,y] =>
    let a := reflect x; let b := reflect y
    ⟨.bin .xor a.form b.form, fun ρ i => by
      change (a.form.value ρ i ^^ b.form.value ρ i) = digit (bitXor (denote ρ x) (denote ρ y)) i
      rw [digit_xor, a.correct, b.correct]⟩
  | .app 16 [x] =>
    let a := reflect x
    ⟨.bin .xor a.form (.const true), fun ρ i => by
      change (a.form.value ρ i ^^ true) = digit (bitNot (denote ρ x)) i
      rw [digit_not, a.correct]; simp⟩
  | p => ⟨.atom p, fun _ _ => rfl⟩

private def Form.atoms : Form → List Pat
  | .const _ => [] | .atom p => [p] | .bin _ a b => a.atoms ++ b.atoms

private def assignment (basis : List Pat) (ρ : Nat → Int) (i j : Nat) : Bool :=
  (basis[j]?).any fun p => digit (denote ρ p) i
private def closed : Nat → Assignment → Bool := fun _ _ => false

private structure Encoded (basis : List Pat) (f : Form) where
  circuit : Pat
  correct : ∀ ρ i, circuit.eval model closed (assignment basis ρ i) = f.value ρ i

private def Form.encode (basis : List Pat) : (f : Form) → Option (Encoded basis f)
  | .const b => some ⟨leaf b, fun _ _ => leaf_eval ..⟩
  | .atom p => do
    let j := basis.idxOf p
    if h : basis[j]? = some p then
      return ⟨node j (leaf false) (leaf true), fun ρ i => by
        simp [node_eval, leaf_eval, assignment, h, Form.value]⟩
    else none
  | .bin g x y => do
    let a ← x.encode basis; let b ← y.encode basis
    return ⟨gate g a.circuit b.circuit, fun ρ i => by
      simp [gate_eval, a.correct, b.correct, Form.value]⟩

/-- A bit in a 64-bit analysis value stands for one assignment to six basis
    bits. A symbolic mask is one such basis word, regardless of its width. -/
private def assignmentRow (σ : Assignment) : BitVec 6 :=
  BitVec.ofBoolListLE [σ 0, σ 1, σ 2, σ 3, σ 4, σ 5]

private def column (i : Nat) : BitVec 64 :=
  match i with
  | 0 => 0xaaaaaaaaaaaaaaaa | 1 => 0xcccccccccccccccc
  | 2 => 0xf0f0f0f0f0f0f0f0 | 3 => 0xff00ff00ff00ff00
  | 4 => 0xffff0000ffff0000 | _ => 0xffffffff00000000

private theorem column_digit : ∀ (row : BitVec 6) (i : Fin 6),
    (column i).getLsbD row.toNat = row.getLsbD i := by decide +kernel

private theorem row_digit (σ : Assignment) (i : Nat) (hi : i < 6) :
    (assignmentRow σ).getLsbD i = σ i := by
  have h : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 ∨ i = 4 ∨ i = 5 := by omega
  unfold assignmentRow
  rw [BitVec.getLsbD_ofBoolListLE]
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl <;> rfl

private def lane (v : BitVec 64) (σ : Assignment) := v.getLsbD (assignmentRow σ).toNat

private def transfer : Nat → List (BitVec 64) → Option (BitVec 64)
  | 0, [] => some 0 | 1, [] => some (-1)
  | 2, [a,b] => some (a &&& b)
  | 3, [a,b] => some (a ||| b)
  | 4, [a,b] => some (a ^^^ b)
  | i+5, [l,h] => if i < 6 then some ((column i &&& h) ||| (~~~(column i) &&& l)) else none
  | _, _ => none

private theorem lane_node (σ : Assignment) (i : Nat) (hi : i < 6) (l h : BitVec 64) :
    (if σ i then lane h σ else lane l σ) =
    lane ((column i &&& h) ||| (~~~(column i) &&& l)) σ := by
  have hr : (assignmentRow σ).toNat < 64 := (assignmentRow σ).isLt
  have hc := (column_digit (assignmentRow σ) ⟨i, hi⟩).trans (row_digit σ i hi)
  unfold lane
  simp only [BitVec.getLsbD_or, BitVec.getLsbD_and, BitVec.getLsbD_not, hr, decide_true, Bool.true_and, hc]
  cases σ i <;> simp

private def laneAnalysis : Certified.ValueAnalysis model (BitVec 64) where
  meaning := lane
  transfer := transfer
  sound := by
    intro op xs y h
    fun_cases transfer op xs <;> simp_all [transfer, model, List.map] <;> (try (obtain ⟨hp, he⟩ := h; omega)) <;> subst y <;> funext σ
    · simp [lane]
    · have hr : (assignmentRow σ).toNat < 64 := (assignmentRow σ).isLt
      change true = (BitVec.allOnes 64).getLsbD (assignmentRow σ).toNat
      simp only [BitVec.getLsbD_allOnes, hr, decide_true]
    · simp [lane]
    · simp [lane]
    · simp [lane]
    · exact lane_node _ _ (by assumption) _ _

/-- The per-bit test represented by this circuit is true at every valuation
    allowed by the source problem. -/
private structure Feasible (q : Query) (basis : List Pat) where
  circuit : Pat
  valid : ∀ ρ, q.Holds ρ → ∀ i, circuit.eval model closed (assignment basis ρ i) = true

private theorem mask_bit (ρ : Nat → Int) (w : Pat) (i : Nat) :
    digit (denote ρ (Op.bw w (lit (-1)))) i = decide (i < (denote ρ w).toNat) := by
  change digit (trunc (denote ρ w) (-1)) i = _
  simp only [digit_trunc, digit_ones, Bool.and_true]

/-- If p ≤ w, a set bit of the p-bit low mask is also a set bit of the
    w-bit low mask. Use only masks already present in the reflected circuit. -/
private def orderedMask (q : Query) (basis : List Pat) (p w : Pat)
    (order : ∀ ρ, q.Holds ρ → denote ρ p ≤ denote ρ w) : Option (Feasible q basis) := do
  let low ← (Form.atom (Op.bw p (lit (-1)))).encode basis
  let high ← (Form.atom (Op.bw w (lit (-1)))).encode basis
  return ⟨gate .disj (gate .xor low.circuit (leaf true)) high.circuit, fun ρ hr i => by
    simp only [gate_eval, Gate.apply, leaf_eval, low.correct, high.correct, Form.value, mask_bit]
    have bound := Int.toNat_le_toNat (order ρ hr)
    by_cases h : i < (denote ρ p).toNat
    · simp [h, Nat.lt_of_lt_of_le h bound]
    · simp [h]⟩

private theorem true_condition {P : Prop} [Decidable P] (h : truth P = 1) : P := by
  by_cases hp : P
  · exact hp
  · simp [truth, hp] at h

private def orderGuard (q : Query) (basis : List Pat) (p : Pat) (hp : p ∈ q.assumptions) :
    Option (Feasible q basis) :=
  match p with
  | Op.lt Op.zero _ | Op.le Op.zero _ => none  -- the zero-bit mask is always empty
  | Op.le a b => orderedMask q basis a b fun _ hr => true_condition (hr _ hp)
  | Op.lt a b => orderedMask q basis a b fun _ hr => Int.le_of_lt (true_condition (hr _ hp))
  | _ => none

private def feasible (q : Query) (basis : List Pat) : Option (Feasible q basis) :=
  q.assumptions.attach.foldl (init := none) fun prior fact =>
    match orderGuard q basis fact.val fact.property with
    | none => prior
    | some next => match prior with
      | none => some next
      | some prev =>
        if prev.circuit == next.circuit then prior else
        some ⟨gate .conj prev.circuit next.circuit, fun ρ hr i => by
          simp [gate_eval, Gate.apply, prev.valid ρ hr i, next.valid ρ hr i]⟩

private def Feasible.restrictCircuit (guard : Option (Feasible q basis)) (c : Pat) : Pat :=
  match guard with
  | some g => gate .conj g.circuit c
  | none => c

private theorem Feasible.restrict_eval (guard : Option (Feasible q basis)) (c : Pat)
    (ρ : Nat → Int) (hr : q.Holds ρ) (i : Nat) :
    (Feasible.restrictCircuit guard c).eval model closed (assignment basis ρ i) =
      c.eval model closed (assignment basis ρ i) := by
  cases guard with
  | none => rfl
  | some g => simp [Feasible.restrictCircuit, gate_eval, Gate.apply, g.valid ρ hr i]

/-- A completed truth-table comparison can also exclude retrying the same
    Boolean abstraction with a decision diagram. Word-level rewrites can still
    prove equalities between independent abstract atoms. -/
inductive LaneResult (q : Query) where
  | proved (sound : ∀ ρ, q.Holds ρ → q.Equivalent ρ)
  | compared
  | needsDiagram

def compareLanes (q : Query) : LaneResult q := Id.run do
  let l := reflect q.lhs
  let r := reflect q.rhs
  let basis := (l.form.atoms ++ r.form.atoms).eraseDups
  if basis.length > 6 then return .needsDiagram
  let some a := l.form.encode basis | return .needsDiagram
  let some b := r.form.encode basis | return .needsDiagram
  let guard := feasible q basis
  let (s, x) := (Certified.Session.empty model).add (Feasible.restrictCircuit guard a.circuit)
  let (s, y) := s.add (Feasible.restrictCircuit guard b.circuit)
  let s := s.run (egg_schedule { analyze laneAnalysis })
  let some h := s.equivalent? x y | return .compared
  return .proved (fun ρ hr => eq_of_digits (fun i => by
    have same := congrFun (h.down closed) (assignment basis ρ i)
    rw [Feasible.restrict_eval guard _ ρ hr i, Feasible.restrict_eval guard _ ρ hr i] at same
    rw [← l.correct, ← r.correct, ← a.correct, ← b.correct, same]))

/-- Recheck a truth-table solution by folding the verified lane transfer over
    the two reflected circuits. This avoids rebuilding an e-graph in the
    kernel and retains the full named problem's width hypotheses. -/
def replayLanes? (q : Query) : Option (PLift (∀ ρ, q.Holds ρ → q.Equivalent ρ)) := do
  let l := reflect q.lhs
  let r := reflect q.rhs
  let basis := (l.form.atoms ++ r.form.atoms).eraseDups
  guard (basis.length ≤ 6)
  let a ← l.form.encode basis
  let b ← r.form.encode basis
  let validRows := feasible q basis
  let x ← laneAnalysis.eval? (Feasible.restrictCircuit validRows a.circuit)
  let y ← laneAnalysis.eval? (Feasible.restrictCircuit validRows b.circuit)
  if same : x.val = y.val then
    return ⟨fun ρ hr => eq_of_digits (fun i => by
      have sem := (x.property closed).trans ((congrArg lane same).trans (y.property closed).symm)
      have h := congrFun sem (assignment basis ρ i)
      rw [Feasible.restrict_eval validRows _ ρ hr i, Feasible.restrict_eval validRows _ ρ hr i] at h
      rw [← l.correct, ← r.correct, ← a.correct, ← b.correct, h])⟩
  else none

/-- Exact six-atom analysis for an unconditional source equivalence. -/
def byLanes? (lhs rhs : Pat) : Option (PLift (∀ ρ, denote ρ lhs = denote ρ rhs)) :=
  let q : Query := ⟨lhs, rhs, []⟩
  match compareLanes q with
  | .proved h => some ⟨fun ρ => h ρ (by intro p hp; cases hp)⟩
  | _ => none

/-- The certified rules describe the same ordered Shannon splits as the relational
    `combine` table. The equality check takes place in an e-graph of decision nodes. -/
def equivalent? (lhs rhs : Pat) (rounds : Nat) : Option (PLift (∀ ρ, denote ρ lhs = denote ρ rhs)) := do
  let l := reflect lhs; let r := reflect rhs
  let basis := (l.form.atoms ++ r.form.atoms).eraseDups
  let a ← l.form.encode basis; let b ← r.form.encode basis
  let (s, x) := (Certified.Session.empty model).add a.circuit
  let (s, y) := s.add b.circuit
  let rs := rules basis.length
  let schedule := egg_schedule { run rs }
  let s := s.run (.saturate rounds (.untilMatch [(a.circuit, b.circuit)] schedule))
  let h ← s.equivalent? x y
  return ⟨fun ρ => eq_of_digits (fun i => by
    rw [← l.correct, ← r.correct, ← a.correct, ← b.correct, h.down closed])⟩

end ParaBit.Decision
