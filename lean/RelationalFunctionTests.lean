import Egglog.RelationalDSL

open Egglog

namespace FunctionCollisionTest

egg_signature Ops : Nat where
  | one => 1
  | two => 2
  | six => 6
  | copy x => x
  | maximum x y => max x y
  otherwise 0

open Ops

/-- These two independent lower bounds combine to a fresh expression. Its
    syntax is canonical regardless of which value arrives first. -/
private def combine (old new : Pat) : Pat :=
  if (old = one ∧ new = two) ∨ (old = two ∧ new = one) then maximum one two
  else if new = maximum one two then new else old

egg_functions F : Nat over Ops.model where
  | lower i ↦ v => v ≤ i
    merge old new => combine old new
    preserving ho hn by
      dsimp only [combine]
      split
      · rename_i h
        rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
        · exact Nat.max_le.mpr ⟨ho, hn⟩
        · exact Nat.max_le.mpr ⟨hn, ho⟩
      · split <;> first | exact hn | exact ho

egg_relations Mark : Nat where
  | seen k v => v ≤ k

private def m := F.model (Mark.model Ops.model)

private def left : RelRule m :=
  egg_rule () : set F.lower six := one by exact (by decide : 1 ≤ 6)

private def right : RelRule m :=
  egg_rule () : set F.lower (copy six) := two by exact (by decide : 2 ≤ 6)

-- The invariant rules out this proposed lower bound even with no premises.
example : True := by
  fail_if_success
    let _ : RelRule m := egg_rule () : set F.lower one := six by change 6 ≤ 1; decide
  trivial

private def readCells : RelRule m :=
  egg_rule (k, v) : when (bound : F.lower k ↦ v) yield Mark.seen k v by
    exact bound

private def copies : VerifiedRule m :=
  egg_rule (x) : copy x ==> x by rfl

private def lowerName := (F.lower six).name

private def seen (s : Certified.Session m) (key value : Pat) :=
  (s.fact? "FunctionCollisionTest.Mark.seen" [key, value]).isSome

private def writes (reverse : Bool) : Certified.Session m :=
  let rules := if reverse then #[right, left] else #[left, right]
  (Certified.Session.empty m).run (egg_schedule { relational rules })

-- Two equal-valued but as-yet unmerged keys hold different cells. A read of a
-- missing key cannot create a cell or a derived analysis fact.
#guard Id.run do
  let s := (writes false).run (egg_schedule { relational #[readCells] })
  return seen s six one && seen s (copy six) two &&
    !seen s six two && !seen s (copy six) one && !seen s one one

-- Both old values were present before the key equality. After union exactly
-- one value is readable under either key. No union of the value e-classes is
-- justified: 1 and 2 still denote different numbers.
private def collision (reverse : Bool) : Bool := Id.run do
  let s := (writes reverse).run (egg_schedule { run #[copies], saturate { relational #[readCells] } })
  let combined := maximum one two
  let (s, a) := s.add one
  let (s, b) := s.add two
  return (s.getCell? lowerName [six]).map Subtype.val == some combined &&
    (s.getCell? lowerName [copy six]).map Subtype.val == some combined &&
    (s.getCell? lowerName [one]).isNone &&
    seen s six combined && seen s (copy six) combined &&
    !seen s six one && !seen s six two && (s.equivalent? a b).isNone

#guard collision false
#guard collision true

-- Check a rule whose earlier lookup found no merged cell. Its next schedule
-- round must revisit the function row produced by canonicalizing the key.
private def readCombined : RelRule m :=
  egg_rule (k) : when (bound : F.lower k ↦ maximum one two)
    yield Mark.seen k (maximum one two) by exact bound

#guard Id.run do
  let s := (writes false).run (egg_schedule { relational #[readCombined] })
  if seen s six (maximum one two) then return false
  let s := s.run (egg_schedule { run #[copies], relational #[readCombined] })
  return seen s six (maximum one two)

-- Conflicts from writes to the same existing key follow the same merge policy.
-- Reasserting a weaker bound after reaching the fixed point must retain the merged cell.
#guard Id.run do
  let s := (Certified.Session.empty m).setCell lowerName [six] one (fun _ => ⟨rfl, (by decide : 1 ≤ 6)⟩)
  let s := s.run (egg_schedule { relational #[readCombined] })
  if seen s six (maximum one two) then return false
  let s := s.setCell lowerName [six] two (fun _ => ⟨rfl, (by decide : 2 ≤ 6)⟩)
  let s := s.run (egg_schedule { saturate { relational #[readCombined] } })
  if !seen s six (maximum one two) then return false
  let s := s.setCell lowerName [six] one (fun _ => ⟨rfl, (by decide : 1 ≤ 6)⟩)
  return (s.getCell? lowerName [six]).map Subtype.val == some (maximum one two) &&
    (s.getCell? lowerName []).isNone && (s.getCell? "unknown" [six]).isNone

-- An idempotent self-update after a key collision must reach a fixed point.
private def refresh : RelRule m :=
  egg_rule (k, v) : when (bound : F.lower k ↦ v) set F.lower k := v by exact bound

#guard Id.run do
  let s := (writes false).run (egg_schedule {
    run #[copies], saturate { relational #[left, right, refresh] }
  })
  return (s.getCell? lowerName [six]).map Subtype.val == some (maximum one two)

end FunctionCollisionTest

namespace GroundFunctionRuleTest

open FunctionCollisionTest.Ops

egg_functions Latest : Nat over FunctionCollisionTest.Ops.model where
  | value ↦ _v => True
    merge _old new => new
    preserving _ho hn by exact hn

private def m := Latest.model FunctionCollisionTest.Ops.model
private def seed : RelRule m := egg_rule () : set Latest.value := one by trivial

-- An empty query has one substitution, even if other writes happen later.
-- Rust egglog keeps the new value when a previously run ground rule runs again.
#guard Id.run do
  let s := (Certified.Session.empty m).run (egg_schedule { relational #[seed] })
  if (s.getCell? Latest.value.name []).map Subtype.val != some one then return false
  let s := s.setCell Latest.value.name [] two (fun _ => ⟨rfl, trivial⟩)
  let s := s.run (egg_schedule { relational #[seed] })
  return (s.getCell? Latest.value.name []).map Subtype.val == some two

-- A distinct declaration with the same query and actions has its own cursor.
-- Rust also runs two identical ground rules independently in separate rulesets.
private def secondSeed : RelRule m := egg_rule () : set Latest.value := one by trivial

#guard Id.run do
  let s := (Certified.Session.empty m).run (egg_schedule { relational #[seed] })
  let s := s.setCell Latest.value.name [] two (fun _ => ⟨rfl, trivial⟩)
  let s := s.run (egg_schedule { relational #[secondSeed] })
  return (s.getCell? Latest.value.name []).map Subtype.val == some one

end GroundFunctionRuleTest
