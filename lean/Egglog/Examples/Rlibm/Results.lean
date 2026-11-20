import Egglog.Examples.Rlibm.Cells
import Egglog.Examples.Rlibm.VerifiedSchedules
import Egglog.Examples.Rlibm.UpstreamCertificate

namespace Rlibm.Cells

/-- The extracted change of reduction and approximation degree preserves the
    correctly rounded logarithm, including the upstream special-value policy. -/
theorem optimized_correct (b : LogBase) (x : Fin 65536) :
    CorrectLog b x ((prepare b).eval x) := by
  rw [prepare_eq_core]
  exact core_correct b x

def narrowLinear (b : LogBase) (raw : Nat) : Int :=
  let (e,m) := rangeReduce raw
  let c := quadratic b m
  linearForm e (exponentStep b) (m%8) c.c0 c.c1 c.c2 0

/-- The two LPs use different local coordinates and coefficients. Their
    linear forms must lie inside the same mathematically certified cell. -/
theorem lp_certificate (b : LogBase) (x : Fin 65536) (positive : 0 < x.val)
    (finite : x.val < 0x7f80) : CorrectLog b x (core b x) ∧
    inCell (core b x) (reduced linearForm b x) = true ∧
    inCell (core b x) (narrowLinear b x) = true := by
  have same : linearForm = accum := by
    funext e k t c0 c1 c2 c3
    exact (linear_eq e k t c0 c1 c2 c3).symm
  have small : narrowLinear b x = narrowValue b x := by
    simp only [narrowLinear, narrowValue, same, accum, Int.mul_zero, Int.add_zero]
  rw [same, small]
  exact ⟨core_correct b x, feasible b x positive finite⟩

end Rlibm.Cells
