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

-- The source pattern binds i and n. Lookups are deliberately in reverse
-- dependency order: lower(n) supplies k, then lower(k) supplies j.
private def chain : RelRule m :=
  egg_rule (i, j, k, n) : when (jk : F.lower k ↦ j) (kn : F.lower n ↦ k)
    (ij : Mark.seen j i) rewrite maximum i n ==> n by
    exact Nat.max_eq_right (Nat.le_trans (show i ≤ j from ij)
      (Nat.le_trans (show j ≤ k from jk) (show k ≤ n from kn)))

private def chainMatch (late : Bool) : Bool := Id.run do
  let (s, expr) := (Certified.Session.empty m).add (maximum one six)
  let (s, tgt) := s.add six
  let phase := egg_schedule { relational #[chain] }
  let s := s.setCell lowerName [six] two (fun _ => ⟨rfl, (by decide : 2 ≤ 6)⟩)
  let s := s.setCell lowerName [two] (copy one) (fun _ => ⟨rfl, (by decide : 1 ≤ 2)⟩)
  let s := if late then s.run phase else s
  if (s.equivalent? expr tgt).isSome then return false
  let s := s.addFact "FunctionCollisionTest.Mark.seen" [copy one, one] (fun _ => Nat.le_refl 1)
  return ((s.run phase).equivalent? expr tgt).isSome

#guard chainMatch false
#guard chainMatch true

-- The compound-key lookup binds k. The other cell must contain the same
-- value even when its invariant also holds for a different stored value.
private def sameBound : RelRule m :=
  egg_rule (i, k, n) : when (_copy : F.lower (copy n) ↦ k) (bound : F.lower n ↦ k)
    (le : Mark.seen k i) rewrite maximum i n ==> n by
    exact Nat.max_eq_right (Nat.le_trans (show i ≤ k from le) (show k ≤ n from bound))

private def matchingCells (agree : Bool) : Bool := Id.run do
  let (s, source) := (Certified.Session.empty m).add (maximum one six)
  let (s, target) := s.add six
  let s := s.assertAtom ((F.lower (copy six)).cell two) (fun _ => ⟨rfl, (by decide : 2 ≤ 6)⟩)
  let s := if agree
    then s.assertAtom ((F.lower six).cell two) (fun _ => ⟨rfl, (by decide : 2 ≤ 6)⟩)
    else s.assertAtom ((F.lower six).cell one) (fun _ => ⟨rfl, (by decide : 1 ≤ 6)⟩)
  let s := s.assertAtom (Mark.seen two one) (fun _ => (by decide : 1 ≤ 2))
  return ((s.run (egg_schedule { relational #[sameBound] })).equivalent? source target).isSome

#guard matchingCells true
#guard !matchingCells false

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

namespace CongruentFunctionRuleTest
open FunctionCollisionTest.Ops

egg_functions History : Nat over FunctionCollisionTest.Ops.model where
  | value ↦ _v => True
    merge old new => maximum old new
    preserving ho _hn by exact ho

private def m := History.model FunctionCollisionTest.Ops.model
private def copies : VerifiedRule m := egg_rule (x) : copy x ==> x by rfl
private def record : RelRule m :=
  egg_rule (r, v) : when (_matched : r = maximum one v) set History.value := v by trivial

-- Rebuilding makes two physical constructor rows match the same substitution.
-- Firing twice would yield maximum (maximum one two) two.
#guard Id.run do
  let (s, _) := (Certified.Session.empty m).add (maximum one two)
  let (s, _) := s.add (maximum (copy one) two)
  let s := s.setCell History.value.name [] one (fun _ => ⟨rfl, trivial⟩)
  let s := s.run (egg_schedule { run #[copies], relational #[record] })
  return (s.getCell? History.value.name []).map Subtype.val == some (maximum one two)

end CongruentFunctionRuleTest

namespace GroupedFunctionWritesTest
open FunctionCollisionTest FunctionCollisionTest.Ops

-- The last input discards the previous merge term. A relation must still be
-- able to match that intermediate constructor after all three writes finish.
egg_functions Ordered : Nat over Ops.model where
  | value ↦ _v => True
    merge old new => if new = six then copy six else maximum old new
    preserving ho _hn by exact ho

private def m := Ordered.model (Mark.model Ops.model)
private def grouped : RelRule m := egg_rule () : do {
  set Ordered.value := one by trivial,
  set Ordered.value := two by trivial,
  set Ordered.value := six by trivial
}
private def first : RelRule m := egg_rule () : set Ordered.value := one by trivial
private def second : RelRule m := egg_rule () : set Ordered.value := two by trivial
private def third : RelRule m := egg_rule () : set Ordered.value := six by trivial

private def observe : RelRule m := egg_rule (r, x, y) : when (_h : r = maximum x y)
  yield Mark.seen r x by
    change r = max x y at _h
    change x ≤ r
    exact _h.symm ▸ Nat.le_max_left x y

private def writesAgree (together : Bool) : Bool := Id.run do
  let schedule := if together then egg_schedule { relational #[grouped] }
    else egg_schedule { relational #[first], relational #[second], relational #[third] }
  let s := ((Certified.Session.empty m).run schedule).run (egg_schedule { relational #[observe] })
  let seen := fun x y => (s.fact? "FunctionCollisionTest.Mark.seen" [x, y]).isSome
  return (s.getCell? Ordered.value.name []).map Subtype.val == some (copy six) &&
    seen (maximum one two) one && !seen (maximum two one) two

#guard writesAgree false
#guard writesAgree true

private def dropCopy : RelRule m :=
  egg_rule (r, x) : when (h : r = copy x) union r = x by exact h

-- Both rules match the same snapshot. The projected tuple's second key must
-- use the representative chosen by the first rule in the working graph.
#guard Id.run do
  let source := maximum (copy one) two
  let (s, _) := (Certified.Session.empty m).add source
  let s := s.run (egg_schedule { relational #[dropCopy, observe] })
  return (s.atom? (Mark.seen source one)).isSome && (s.atom? (Mark.seen source two)).isNone

end GroupedFunctionWritesTest

namespace RelationalDeltaOverlapTest
open FunctionCollisionTest.Ops CongruentFunctionRuleTest

egg_relations Active : Nat where
  | item _x => True

private def m := Active.model (History.model FunctionCollisionTest.Ops.model)
private def copies : VerifiedRule m := egg_rule (x) : copy x ==> x by rfl
private def record : RelRule m :=
  egg_rule (r, x, v) : when (_entry : r = maximum x v) (_active : Active.item x)
    set History.value := v by trivial

-- The old constructor row survives, and a second physical row acquires the
-- very same canonical key in the same interval that the relation gains a row.
-- Both delta variants find the assignment, but its merge must run only once.
#guard Id.run do
  let (s, _) := (Certified.Session.empty m).add (maximum one two)
  let (s, _) := s.add (maximum (copy one) two)
  let s := s.setCell History.value.name [] six (fun _ => ⟨rfl, trivial⟩)
  let s := s.run (egg_schedule { relational #[record], run #[copies] })
  let s := s.assertAtom (Active.item one) (fun _ => trivial)
  let s := s.run (egg_schedule { relational #[record] })
  return (s.getCell? History.value.name []).map Subtype.val == some (maximum six two)

end RelationalDeltaOverlapTest
