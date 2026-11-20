import Egglog.Examples.Rlibm.Certificate
import Egglog.Examples.Rlibm.Optimizer

namespace Rlibm.Schedule

/-- Every extracted fixed-point reduction computes the correctly rounded
    bfloat16 logarithm for every input word, including both zeros and NaNs. -/
theorem optimized_correct (p : Profile) (base : LogBase) (input : Fin 65536) :
    CorrectLog base input.val (evalWith (prepare p).eval base input.val) := by
  rw [optimized_eq_core]
  exact core_correct base input

/-- The same rounding-cell guarantee for the 64-bit unsigned arithmetic IR
    consumed by the C emitter, with the final accumulator interpreted as signed. -/
theorem machine_correct (p : Profile) (base : LogBase) (input : Fin 65536) :
    CorrectLog base input.val (evalWith (machine p) base input.val) := by
  rw [machine_eq_core]
  exact core_correct base input

end Rlibm.Schedule
