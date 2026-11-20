import Egglog.Examples.Rlibm.Certificate
import Egglog.Examples.Rlibm.Upstream

namespace Rlibm.Upstream

private def check (raw : Nat) : Bool :=
  (if 0 < raw ∧ raw < 0x7f80 then decide (FrexpContract raw) else true) &&
  (replay .two raw == some (core .two raw)) &&
  (replay .ten raw == some (core .ten raw))

private theorem frexp_of_check (input : Fin 65536) (ok : check input.val = true) (positive : 0 < input.val) (finite : input.val < 0x7f80) :
    FrexpContract input.val := by
  have pos : 0 < input.val ∧ input.val < 0x7f80 := ⟨positive, finite⟩
  simp only [check, if_pos pos, Bool.and_eq_true, decide_eq_true_eq] at ok
  exact ok.1.1

/-- The strict binary64 Horner computation with the upstream constants has a
    defined result on every encoding and agrees bit for bit with the verified
    fixed-point routine, including every exceptional return. -/
private theorem output_of_check (base : LogBase) (input : Fin 65536) (ok : check input.val = true) :
    implementation base input.val = some (core base input.val) := by
  simp only [check, Bool.and_eq_true, beq_iff_eq, replay_eq] at ok
  cases base
  · exact ok.1.2
  · exact ok.2

set_option maxRecDepth 8000
set_option maxHeartbeats 0
set_option Elab.async false

certify_bfloat16 check as allWords

theorem frexp_contract (input : Fin 65536) (positive : 0 < input.val) (finite : input.val < 0x7f80) :
    FrexpContract input.val := frexp_of_check input (allWords input) positive finite

theorem implementation_eq (base : LogBase) (input : Fin 65536) :
    implementation base input.val = some (core base input.val) :=
  output_of_check base input (allWords input)

theorem correctly_rounded (base : LogBase) (input : Fin 65536) :
    ∃ result, implementation base input.val = some result ∧ CorrectLog base input.val result :=
  ⟨core base input.val, implementation_eq base input, core_correct base input⟩

end Rlibm.Upstream
