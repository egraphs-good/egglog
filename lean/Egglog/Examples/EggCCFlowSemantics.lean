import Egglog.Examples.EggCC

namespace EggCC

/-- Structured Tree IR. A loop consumes its incoming tuple as the first body
    argument. The body returns a continuation flag followed by the next tuple. -/
inductive Flow where
  | pure (e : Expr)
  | seq (first next : Flow)
  | letVal (rhs body : Flow)
  | branch (pred : Expr) (yes no : Flow)
  | loop (body : Flow)
  deriving Repr, DecidableEq, Inhabited

namespace Flow

/-- Unbounded executions of a post-tested loop. The last body execution
    contributes its returned state even when the first test is false. -/
inductive Iterates (step : Value → Value → Prop) : Value → Value → Prop where
  | stop : step i (.pair (.data (.flag false)) o) → Iterates step i o
  | more : step i (.pair (.data (.flag true)) next) → Iterates step next o → Iterates step i o

/-- Relational semantics of a terminating evaluation, with no iteration bound.
    Print effects travel through the explicit state tokens in the value. -/
def Evaluates : Flow → Value → Value → Prop
  | .pure e, i, o => e.eval i = o
  | .seq f g, i, o => ∃ mid, Evaluates f i mid ∧ Evaluates g mid o
  | .letVal f g, i, o => ∃ mid, Evaluates f i mid ∧ Evaluates g (.pair mid i) o
  | .branch p y n, i, o => if (p.eval i).test then Evaluates y i o else Evaluates n i o
  | .loop b, i, o => Iterates (Evaluates b) i o

/-- A pause records the next carried tuple at a taken backedge, including its
    state token and ordered print history. Ill-formed loop results are stuck. -/
inductive Observation where
  | done (value : Value)
  | paused (carried : Value)
  | stuck
  deriving Repr, DecidableEq, Inhabited

def Observation.thenDo (obs : Observation) (k : Value → Observation) : Observation :=
  match obs with
  | .done v => k v
  | .paused v => .paused v
  | .stuck => .stuck

def iterate (step : Value → Observation) : Nat → Value → Observation
  | fuel, i =>
    (step i).thenDo fun v => match v with
      | .pair (.data (.flag false)) next => .done next
      | .pair (.data (.flag true)) next =>
        match fuel with
        | 0 => .paused next
        | n + 1 => iterate step n next
      | _ => .stuck

/-- Fuel limits the number of taken backedges per invocation of a loop. Even
    fuel zero executes its first body. Nested invocations receive the same
    bound and propagate their own paused state to the caller. -/
def observe (fuel : Nat) : Flow → Value → Observation
  | .pure e, i => .done (e.eval i)
  | .seq f g, i => (observe fuel f i).thenDo (observe fuel g)
  | .letVal f g, i => (observe fuel f i).thenDo fun mid => observe fuel g (.pair mid i)
  | .branch p y n, i => if (p.eval i).test then observe fuel y i else observe fuel n i
  | .loop b, i => iterate (observe fuel b) fuel i

def «run» (fuel : Nat) (f : Flow) (i : Value) : Option Value :=
  match observe fuel f i with | .done o => some o | _ => none

private theorem thenDo_done (x : Observation) (f : Value → Observation) (o : Value) :
    x.thenDo f = .done o ↔ ∃ m, x = .done m ∧ f m = .done o := by
  cases x <;> simp [Observation.thenDo]

/-- A divergent run takes backedges at every observation budget. A stuck
    execution is distinguished from divergence by an eventual stuck observation. -/
def Diverges (f : Flow) (i : Value) := ∀ fuel, ∃ carried, observe fuel f i = .paused carried

private theorem iterate_sound {R : Value → Value → Prop} {step : Value → Observation}
    (valid : ∀ i o, step i = .done o → R i o) (k : Nat) (i o : Value)
    (runStep : iterate step k i = .done o) : Iterates R i o := by
  induction k generalizing i with
  | zero =>
    rw [iterate, thenDo_done] at runStep
    obtain ⟨v, hv, h⟩ := runStep
    split at h <;> simp_all
    exact .stop (valid _ _ hv)
  | succ k ih =>
    rw [iterate, thenDo_done] at runStep
    obtain ⟨v, hv, h⟩ := runStep
    split at h
    · cases h
      exact .stop (valid _ _ hv)
    · exact .more (valid _ _ hv) (ih _ h)
    · contradiction

theorem observe_sound (f : Flow) (fuel : Nat) (i o : Value)
    (h : observe fuel f i = .done o) : Evaluates f i o := by
  induction f generalizing i o with
  | pure e => exact Observation.done.inj h
  | seq a b ha hb | letVal a b ha hb =>
    obtain ⟨v,hv,hb'⟩ := (thenDo_done ..).mp h
    exact ⟨v, ha _ _ hv, hb _ _ hb'⟩
  | branch p y n hy hn =>
    simp only [observe] at h
    simp only [Evaluates]
    split at h <;> simp_all
  | loop body hb => exact iterate_sound hb _ _ _ h

private theorem iterate_complete {R : Value → Value → Prop} {step : Nat → Value → Observation}
    (complete : ∀ i o, R i o → ∃ low, ∀ fuel ≥ low, step fuel i = .done o)
    {i o : Value} (h : Iterates R i o) :
    ∃ low, ∀ fuel ≥ low, ∀ k ≥ low, iterate (step fuel) k i = .done o := by
  induction h with
  | stop hbody =>
    obtain ⟨n, hn⟩ := complete _ _ hbody
    exact ⟨n, fun fuel hf k _ => by rw [iterate, hn fuel hf]; rfl⟩
  | more hbody _ ih =>
    obtain ⟨n, hn⟩ := complete _ _ hbody
    obtain ⟨m, hm⟩ := ih
    refine ⟨max n (m+1), fun fuel hf k hk => ?_⟩
    obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : k ≠ 0)
    rw [iterate, hn fuel (by omega)]
    exact hm fuel (by omega) k (by omega)

theorem observe_complete (f : Flow) (i o : Value) (h : Evaluates f i o) :
    ∃ low, ∀ fuel ≥ low, observe fuel f i = .done o := by
  induction f generalizing i o with
  | pure e => exact ⟨0, fun _ _ => congrArg Observation.done h⟩
  | seq a b ha hb | letVal a b ha hb =>
    obtain ⟨v, av, bv⟩ := h
    obtain ⟨m, am⟩ := ha _ _ av
    obtain ⟨n, bn⟩ := hb _ _ bv
    refine ⟨max m n, fun fuel hf => ?_⟩
    apply (thenDo_done ..).mpr
    exact ⟨v, am fuel (by omega), bn fuel (by omega)⟩
  | branch p y n hy hn =>
    simp only [Evaluates] at h
    split at h
    · obtain ⟨low,hl⟩ := hy _ _ h
      exact ⟨low, fun fuel hf => by simp [observe, *]⟩
    · obtain ⟨low,hl⟩ := hn _ _ h
      exact ⟨low, fun fuel hf => by simp [observe, *]⟩
  | loop body hb =>
    obtain ⟨n,hn⟩ := iterate_complete hb h
    exact ⟨n, fun fuel hf => hn fuel hf fuel hf⟩

theorem run_iff (f : Flow) (i o : Value) : (∃ fuel, Flow.run fuel f i = some o) ↔ Evaluates f i o := by
  constructor
  · rintro ⟨fuel, h⟩
    unfold Flow.run at h
    split at h <;> simp_all
    apply observe_sound <;> assumption
  · intro h
    obtain ⟨n,hn⟩ := observe_complete _ _ _ h
    exact ⟨n, by simp [Flow.run, hn n (Nat.le_refl n)]⟩

end Flow
end EggCC
