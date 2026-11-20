import Egglog.Examples.Rlibm.NativeWord
import Egglog.Examples.Rlibm.Certificate

namespace Rlibm.NativeWord

/-- Check the actual UInt16 entry and exit, UInt64 normalization, signed
    reconstruction and rounding. The separate polynomial identity proof
    lets all three rearrangements share the cubic's full-domain check. -/
private def check (b : LogBase) (raw : Nat) : Bool :=
  let input := raw.toUInt16
  let reference := core b raw
  ((prepare b .source).eval input).toNat == reference &&
    ((prepare b .cells).eval input).toNat == reference

set_option Elab.async false
set_option maxRecDepth 8000
set_option maxHeartbeats 0

certify_bfloat16 (check .two) as two
certify_bfloat16 (check .ten) as ten

theorem returns_core (b : LogBase) (m : Mode) (x : UInt16) :
    ((prepare b m).eval x).toNat = core b x.toNat := by
  have ok : check b x.toNat = true := by
    cases b
    · exact two x.toFin
    · exact ten x.toFin
  have pair := Bool.and_eq_true_iff.mp ok
  by_cases cells : m = .cells
  · subst m
    simpa using eq_of_beq pair.2
  · rw [wide_eq_source b m cells x]
    simpa using eq_of_beq pair.1

theorem correctly_rounded (b : LogBase) (m : Mode) (x : UInt16) :
    CorrectLog b x.toNat ((prepare b m).eval x).toNat := by
  rw [returns_core]
  exact core_correct b x.toFin

end Rlibm.NativeWord
