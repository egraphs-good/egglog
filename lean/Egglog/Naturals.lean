import Egglog.Verified

namespace Egglog.Naturals

/-- The signature declared incrementally in `tests/web-demo/naturals.egg`. -/
private def constructors : List NatConstructor :=
  [.constant "Z" 0, .unary "S" Nat.succ, .binary "Plus" Nat.add, .binary "Times" Nat.mul]

def signature := NatConstructor.signature constructors

def model := NatConstructor.model constructors

private abbrev x := Pat.var 0
private abbrev y := Pat.var 1
private abbrev zero := Pat.app 0 []
private abbrev succ (a : Pat) := Pat.app 1 [a]
private abbrev plus (a b : Pat) := Pat.app 2 [a, b]
private abbrev times (a b : Pat) := Pat.app 3 [a, b]

def plusZero : VerifiedRule model :=
  ⟨⟨plus zero x, x⟩, fun env => Nat.zero_add (env 0)⟩

def plusSucc : VerifiedRule model :=
  ⟨⟨plus (succ x) y, succ (plus x y)⟩, fun env => Nat.succ_add (env 0) (env 1)⟩

def timesZero : VerifiedRule model :=
  ⟨⟨times zero x, zero⟩, fun env => Nat.zero_mul (env 0)⟩

def timesSucc : VerifiedRule model :=
  ⟨⟨times (succ x) y, plus y (times x y)⟩, fun env =>
    (Nat.succ_mul (env 0) (env 1)).trans (Nat.add_comm _ _)⟩

def rules : List (VerifiedRule model) := [plusZero, plusSucc, timesZero, timesSucc]

end Egglog.Naturals
