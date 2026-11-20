import Egglog
import Egglog.Examples.BoundsCheck

namespace RelationalSessionTests
open Egglog BoundsCheck

-- The host compiler's bounds analysis has already proved i ≤ i and i < i+1.
-- Import just those results, then fold the check using the ordinary pass.
#guard Id.run do
  let i := Expr.reg 7
  let n := Expr.next i
  let (s, index) := ir.session.add i
  let (s, len) := s.add n
  let (s, check) := s.add (.inBounds i n)
  let key := F.lower len.term
  let fact := R.atMost index.term index.term
  unless (s.fact? fact).isNone && (s.readValue? key).isNone do return false
  let s := s.assertFact fact (fun regs => Nat.le_refl (regs 7))
  let s := s.setValue key i (fun regs => ⟨rfl, Nat.lt_succ_self (regs 7)⟩)
  let some ⟨value, proof⟩ := s.readValue? key | return false
  have _lower : ∀ regs, value.eval regs < (n.eval regs) := fun regs => (proof regs).2
  unless value == i && (s.fact? fact).isSome && (s.fact? (key.cell index.term)).isSome do return false
  let s := s.run (egg_schedule { relational #[checkFromLower] })
  return (s.extractor.extract? check).map (·.output) == some .yes

-- An auxiliary value may belong to the model but have no AST constructor.
-- The low-level read still delivers its invariant; decoding reports the miss.
#guard Id.run do
  let (s, len) := ir.session.add (.next (.reg 0))
  let key := F.lower len.term
  let opaqueValue := Pat.app 99 []
  let s := s.assertFact (key.cell opaqueValue) (fun regs => ⟨rfl, Nat.zero_lt_succ (regs 0)⟩)
  return (s.readTerm? key).map Subtype.val == some opaqueValue && (s.readValue? key).isNone

-- The stop condition shares k across an equality, a predicate and a lookup.
-- Both table rows exist in each run. Only the first query joins them.
#guard Id.run do
  let i := Expr.yes
  let (s, index) := ir.session.add i
  let (s, len) := s.add (.next i)
  let (s, query) := s.add (.inBounds i (.next i))
  let s := s.assertFact (R.atMost index.term index.term) (fun _ => Nat.le_refl 1)
  let s := s.setValue (F.lower len.term) i (fun _ => ⟨rfl, Nat.lt_succ_self 1⟩)
  let skipped := s.run (egg_schedule {
    until (k) (k = index.term) (R.atMost index.term k) (F.lower len.term ↦ k) {
      relational #[checkFromLower]
    }
  })
  let executed := s.run (egg_schedule {
    until (k) (k = len.term) (R.atMost index.term k) (F.lower len.term ↦ k) {
      relational #[checkFromLower]
    }
  })
  return (skipped.extractor.extract? query).map (·.output) == some (.inBounds i (.next i)) &&
    (executed.extractor.extract? query).map (·.output) == some .yes

end RelationalSessionTests
