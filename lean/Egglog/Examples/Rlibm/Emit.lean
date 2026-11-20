import Egglog.Examples.Rlibm.Optimizer

namespace Rlibm.Schedule

private def argNames := #["e", "k", "t", "c0", "c1", "c2", "c3"]

def WordExpr.toC : WordExpr → Except String String
  | .arg i => match argNames[i]? with
    | some name => .ok name
    | none => .error s!"unknown kernel argument {i}"
  | .imm z => pure s!"UINT64_C({(BitVec.ofInt 64 z).toNat})"
  | .add a b => return s!"({← a.toC} + {← b.toC})"
  | .mul a b => return s!"({← a.toC} * {← b.toC})"

/-- Emit a C11 function (using the seven-operand kernel ABI) by lowering the
    supplied AST. The full-kernel and per-schedule benchmarks share this printer. -/
def renderAccumulator (name : String) (tree : Expr) : Except String String := do
  let signature := String.intercalate ", " (argNames.toList.map ("uint64_t " ++ ·))
  let body ← tree.lower.toC
  return s!"static inline uint64_t {name}({signature}) \{\n  return {body};\n}\n"

end Rlibm.Schedule
