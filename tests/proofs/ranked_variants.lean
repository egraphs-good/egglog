/- Replay with Lean 4.8: lean tests/proofs/ranked_variants.lean
   Model of the rank grid used by rank_class. No external libraries. -/
import Lean

namespace RankedVariants

-- The Rust loop can increment any coordinate up to and including the first
-- nonzero one. Coordinates before the increment must therefore be zero.
inductive Step : List Nat → List Nat → Prop where
  | here : Step (n :: rest) ((n+1) :: rest)
  | later : Step a b → Step (0 :: a) (0 :: b)

def parent : List Nat → Option (List Nat)
  | [] => none
  | 0 :: rest => (parent rest).map (0 :: ·)
  | (n+1) :: rest => some (n :: rest)

theorem step_iff_parent {a b} : Step a b ↔ parent b = some a := by
  constructor
  · intro h
    induction h with
    | here => rfl
    | later _ ih => simp [parent, ih]
  · induction b generalizing a with
    | nil => simp [parent]
    | cons n rest ih =>
      cases n with
      | succ n =>
        intro h
        simp [parent] at h
        subst a
        exact .here
      | zero =>
        intro h
        cases hp : parent rest with
        | none => simp [parent, hp] at h
        | some p =>
          simp [parent, hp] at h
          subst a
          exact .later (ih hp)

-- Every predecessor in a nonempty Cartesian product remains inside its bounds.
inductive Fits : List Nat → List Nat → Prop where
  | nil : Fits [] []
  | cons : x < n → Fits xs ns → Fits (x :: xs) (n :: ns)

theorem fits_parent (h : Step a b) (hb : Fits b bounds) : Fits a bounds := by
  induction h generalizing bounds with
  | here =>
    cases hb with
    | cons h t => exact .cons (by omega) t
  | later _ ih =>
    cases hb with
    | cons h t => exact .cons h (ih t)

def weight : List Nat → Nat
  | [] => 0
  | n :: rest => n + weight rest

theorem step_weight (h : Step a b) : weight b = weight a + 1 := by
  induction h with
  | here => simp [weight]; omega
  | later _ ih => simpa [weight] using ih

-- The only seed of a row is its all-zero tuple (also covers nullary rows).
theorem seed_iff (r : List Nat) : parent r = none ↔ ∀ x ∈ r, x = 0 := by
  induction r with
  | nil => simp [parent]
  | cons n xs ih =>
    cases n with
    | succ n => simp [parent]
    | zero =>
      cases h : parent xs <;> simp_all [parent] <;> assumption

-- Loop invariant: a live seed or the first unpopped child of a popped node
-- is in the heap. Consequently its minimum is globally minimal among all
-- unpopped tuples of every row, even when different tuples have equal costs.
theorem heap_min_global {Key : Type} (le : Key → Key → Prop)
    (trans : ∀ {x y z}, le x y → le y z → le x z)
    (key : Nat → List Nat → Key) (bounds : Nat → List Nat)
    (popped heap : Nat → List Nat → Prop) (best : Key)
    (minimum : ∀ row r, heap row r → le best (key row r))
    (seed : ∀ row r, Fits r (bounds row) → parent r = none → ¬popped row r → heap row r)
    (generated : ∀ row p r, Step p r → Fits r (bounds row) →
        popped row p → ¬popped row r → heap row r)
    (mono : ∀ row p r, Step p r → Fits r (bounds row) → le (key row p) (key row r))
    (row : Nat) (r : List Nat) (valid : Fits r (bounds row)) (live : ¬popped row r) :
    le best (key row r) := by
  cases h : parent r with
  | none => exact minimum row r (seed row r valid h live)
  | some p =>
    have edge := step_iff_parent.mpr h
    by_cases hp : popped row p
    · exact minimum row r (generated row p r edge valid hp live)
    · exact trans (heap_min_global le trans key bounds popped heap best minimum seed generated
          mono row p (fits_parent edge valid) hp) (mono row p r edge valid)
termination_by weight r
decreasing_by
  simp_wf
  have hdec := step_weight edge
  omega

-- Saturating additive default cost, with equal cost broken by ranks.
def sat (cap fixed child : Nat) := min cap (fixed + child)

inductive LexLt : List Nat → List Nat → Prop where
  | head : a < b → LexLt (a :: xs) (b :: ys)
  | tail : LexLt xs ys → LexLt (a :: xs) (a :: ys)

def Before (ca : Nat) (a : List Nat) (cb : Nat) (b : List Nat) :=
  ca < cb ∨ (ca = cb ∧ LexLt a b)

-- Advancing the heap: popping q removes q and adds exactly its live children.
-- This is the invariant used by heap_min_global, including the case of a
-- different row and the initially empty set of popped candidates.
def Frontier (done_ : Nat → List Nat → Prop) (row : Nat) (r : List Nat) :=
  ¬done_ row r ∧ (parent r = none ∨ ∃ p, Step p r ∧ done_ row p)

theorem frontier_after_pop (done_ : Nat → List Nat → Prop) (qrow row : Nat)
    (q r : List Nat) :
    Frontier (fun i x => done_ i x ∨ (i = qrow ∧ x = q)) row r ↔
      ¬(row = qrow ∧ r = q) ∧
      (Frontier done_ row r ∨ (¬done_ row r ∧ row = qrow ∧ Step q r)) := by
  constructor
  · rintro ⟨live, source⟩
    have fresh : ¬done_ row r := fun h => live (Or.inl h)
    refine ⟨fun h => live (Or.inr h), ?_⟩
    rcases source with seed | ⟨p,e,old | ⟨hr,hp⟩⟩
    · exact Or.inl ⟨fresh, Or.inl seed⟩
    · exact Or.inl ⟨fresh, Or.inr ⟨p,e,old⟩⟩
    · subst p; exact Or.inr ⟨fresh,hr,e⟩
  · rintro ⟨other, ⟨live, source⟩ | ⟨live,hr,e⟩⟩
    · refine ⟨fun h => h.elim live other, ?_⟩
      rcases source with seed | ⟨p,e,was⟩
      · exact Or.inl seed
      · exact Or.inr ⟨p,e,Or.inl was⟩
    · exact ⟨fun h => h.elim live other, Or.inr ⟨q,e,Or.inr ⟨hr,rfl⟩⟩⟩

def sumChildren : List (Nat → Nat) → List Nat → Nat
  | f :: fs, i :: is => f i + sumChildren fs is
  | _, _ => 0

theorem step_sum (e : Step a b) (fs : List (Nat → Nat))
    (sorted : ∀ f ∈ fs, ∀ i j, i ≤ j → f i ≤ f j) :
    sumChildren fs a ≤ sumChildren fs b := by
  induction e generalizing fs with
  | @here n rs =>
    cases fs with
    | nil => exact Nat.le_refl _
    | cons f fs =>
      have h := sorted f (by simp) n (n+1) (by omega)
      simp only [sumChildren]; omega
  | later _ ih =>
    cases fs with
    | nil => exact Nat.le_refl _
    | cons f fs =>
      have h := ih fs (fun g hg => sorted g (by simp [hg]))
      simp only [sumChildren]; omega

theorem step_lex (e : Step a b) : LexLt a b := by
  induction e with
  | here => exact .head (Nat.lt_succ_self _)
  | later _ ih => exact .tail ih

theorem step_key (e : Step a b) (cap head : Nat) (fs : List (Nat → Nat))
    (sorted : ∀ f ∈ fs, ∀ i j, i ≤ j → f i ≤ f j) :
    Before (sat cap head (sumChildren fs a)) a (sat cap head (sumChildren fs b)) b := by
  have monotone := step_sum e fs sorted
  unfold Before sat
  by_cases h : min cap (head + sumChildren fs a) < min cap (head + sumChildren fs b)
  · exact Or.inl h
  · exact Or.inr ⟨by omega,step_lex e⟩

-- Syntactic constructor application is injective in each ordered child slot.
theorem slot_injective {T : Type} (pre post : List T) {x y : T}
    (h : pre ++ x :: post = pre ++ y :: post) : x = y := by
  induction pre with
  | nil => exact (List.cons.inj h).1
  | cons a xs ih => exact ih (List.cons.inj h).2

-- A retained witness must remain inside the window even when several child
-- ranks overflow it. Reset the other slots to zero and vary the offending one.
inductive PointLE : List Nat → List Nat → Prop where
  | nil : PointLE [] []
  | cons : i ≤ j → PointLE a b → PointLE (i::a) (j::b)

def zeros (r : List Nat) := r.map (fun _ => 0)

theorem zeros_le (r : List Nat) : PointLE (zeros r) r := by
  induction r with
  | nil => exact .nil
  | cons n r ih => exact .cons (Nat.zero_le _) ih

theorem reset_slot (pre post : List Nat) (h : j < i) :
    PointLE (zeros pre ++ j :: zeros post) (pre ++ i :: post) ∧
    LexLt (zeros pre ++ j :: zeros post) (pre ++ i :: post) := by
  induction pre with
  | nil => exact ⟨.cons (Nat.le_of_lt h) (zeros_le post), .head h⟩
  | cons x pre ih =>
    refine ⟨.cons (Nat.zero_le _) ih.1, ?_⟩
    cases x with
    | zero => exact .tail ih.2
    | succ x => exact .head (Nat.zero_lt_succ _)

theorem point_fits (h : PointLE a b) (valid : Fits b bounds) : Fits a bounds := by
  induction h generalizing bounds with
  | nil => exact valid
  | cons le _ ih =>
    cases valid with
    | cons h t => exact .cons (Nat.lt_of_le_of_lt le h) (ih t)

theorem point_sum (h : PointLE a b) (fs : List (Nat → Nat))
    (sorted : ∀ f ∈ fs, ∀ i j, i ≤ j → f i ≤ f j) :
    sumChildren fs a ≤ sumChildren fs b := by
  induction h generalizing fs with
  | nil => exact Nat.le_refl _
  | cons le _ ih =>
    cases fs with
    | nil => exact Nat.le_refl _
    | cons f fs =>
      have h₁ := sorted f (by simp) _ _ le
      have h₂ := ih fs (fun g hg => sorted g (by simp [hg]))
      simp only [sumChildren]; omega

def Window (k : Nat) (r : List Nat) := ∀ x ∈ r, x < k

theorem reset_window (pre post : List Nat) (j : Fin k) :
    Window k (zeros pre ++ j.val :: zeros post) := by
  have hj := j.isLt
  induction pre with
  | nil =>
    intro x hx
    have h : x = j.val ∨ ∃ y ∈ post, (0 : Nat) = x := by
      simpa [zeros] using hx
    rcases h with h | ⟨_,_,h⟩ <;> omega
  | cons a pre ih =>
    intro x hx
    have h : x = 0 ∨ x ∈ zeros pre ++ j.val :: zeros post := List.mem_cons.mp hx
    exact h.elim (fun e => by omega) (ih x)

-- At least k distinct represented answers rank earlier than this derivation.
def Dominated {D T : Type} (k : Nat) (tree : D → T) (earlier : D → D → Prop)
    (allowed : D → Prop) (d : D) :=
  ∃ w : Fin k → D, (∀ j, allowed (w j) ∧ earlier (w j) d) ∧
    ∀ i j, tree (w i) = tree (w j) → i = j

-- Pruning does not promote a later derivation into the first k distinct
-- answers. Covers ambiguity: the injection is on output trees, not row ids.
theorem prune_preserves_rank {D T : Type} (k : Nat) (tree : D → T)
    (lt : D → D → Prop) (trans : ∀ {a b c}, lt a b → lt b c → lt a c)
    (full kept : D → Prop) (sub : ∀ d, kept d → full d)
    (safe : ∀ d, full d → ¬kept d → Dominated k tree lt kept d) (d : D) :
    Dominated k tree lt full d ↔ Dominated k tree lt kept d := by
  constructor
  · rintro ⟨w,earlier,inj⟩
    by_cases h : ∀ j, kept (w j)
    · exact ⟨w,fun j => ⟨h j,(earlier j).2⟩,inj⟩
    · open Classical in
      have ex : ∃ j, ¬kept (w j) := Classical.not_forall.mp h
      obtain ⟨j,miss⟩ := ex
      obtain ⟨v,hv,vi⟩ := safe _ (earlier j).1 miss
      exact ⟨v,fun i => ⟨(hv i).1,trans (hv i).2 (earlier j).2⟩,vi⟩
  · rintro ⟨w,earlier,inj⟩
    exact ⟨w,fun j => ⟨sub _ (earlier j).1,(earlier j).2⟩,inj⟩

-- Instantiate the safety premise for a row, at any offending child slot.
theorem outside_window_dominated {T : Type} (k i cap head : Nat) (miss : k ≤ i)
    (pre post bounds : List Nat) (valid : Fits (pre ++ i :: post) bounds)
    (fs : List (Nat → Nat)) (sorted : ∀ f ∈ fs, ∀ a b, a ≤ b → f a ≤ f b)
    (tree : List Nat → T)
    (slot : ∀ a b : Fin k, tree (zeros pre ++ a.val :: zeros post) =
        tree (zeros pre ++ b.val :: zeros post) → a = b) :
    Dominated k tree
      (fun a b => Before (sat cap head (sumChildren fs a)) a
                         (sat cap head (sumChildren fs b)) b)
      (fun r => Fits r bounds ∧ Window k r) (pre ++ i :: post) := by
  refine ⟨fun j => zeros pre ++ j.val :: zeros post, ?_, slot⟩
  intro j
  have lt := Nat.lt_of_lt_of_le j.isLt miss
  obtain ⟨le,lex⟩ := reset_slot pre post lt
  have h := point_sum le fs sorted
  refine ⟨⟨point_fits le valid, reset_window pre post j⟩, ?_⟩
  unfold Before sat
  have order : min cap (head + sumChildren fs (zeros pre ++ j.val :: zeros post)) ≤
      min cap (head + sumChildren fs (pre ++ i :: post)) := by omega
  exact (Nat.lt_or_eq_of_le order).elim
    Or.inl (fun eq => Or.inr ⟨eq,lex⟩)

-- A leading occurrence has fewer than k distinct earlier trees. When a tree
-- has several occurrences, its earliest one determines whether it is emitted.
-- Combining this theorem with outside_window_dominated justifies storing only
-- the child prefix at each class in the acyclic bottom-up traversal.
theorem leading_occurrences_preserved {D T : Type} (k : Nat) (tree : D → T)
    (lt : D → D → Prop) (trans : ∀ {a b c}, lt a b → lt b c → lt a c)
    (full kept : D → Prop) (sub : ∀ d, kept d → full d)
    (safe : ∀ d, full d → ¬kept d → Dominated k tree lt kept d) (d : D) :
    (full d ∧ ¬Dominated k tree lt full d) ↔
    (kept d ∧ ¬Dominated k tree lt kept d) := by
  have eqv := prune_preserves_rank k tree lt trans full kept sub safe d
  constructor
  · rintro ⟨valid,lead⟩
    have h : ¬Dominated k tree lt kept d := fun h => lead (eqv.mpr h)
    exact ⟨Classical.byContradiction (fun miss => h (safe d valid miss)),h⟩
  · rintro ⟨valid,lead⟩
    exact ⟨sub d valid,fun h => lead (eqv.mp h)⟩

#print axioms leading_occurrences_preserved
#print axioms prune_preserves_rank
#print axioms outside_window_dominated
#print axioms frontier_after_pop
#print axioms step_key
#print axioms heap_min_global
end RankedVariants
