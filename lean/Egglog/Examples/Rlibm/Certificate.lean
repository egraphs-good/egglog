import Egglog.Examples.Rlibm.Polynomial
import Egglog.Examples.Rlibm.Spec
import Egglog.Examples.Rlibm.KernelPages

namespace Rlibm

private def checkCore (base : LogBase) (x : Nat) : Bool :=
  checkLog base x (core base x)

set_option maxRecDepth 8000
set_option maxHeartbeats 0
set_option Elab.async false

certify_bfloat16 (checkCore .two) as allTwo
certify_bfloat16 (checkCore .ten) as allTen

/-- Both piecewise cubic routines return the round-to-nearest, ties-to-even
    logarithm for every bfloat16 encoding, including the special encodings. -/
theorem core_correct (base : LogBase) (input : Fin 65536) :
    CorrectLog base input.val (core base input.val) := by
  apply checkLog_sound
  cases base
  · exact allTwo input
  · exact allTen input

end Rlibm
