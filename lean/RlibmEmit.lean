import Egglog.Examples.Rlibm.Emit
import Egglog.Examples.Rlibm.Cells

/-! Emit the arithmetic actually extracted by Lean for every benchmark entry.
    The common kernel supplies bfloat16 classification, the coefficient table,
    and the final rounding. -/

open Rlibm.Schedule

private def kernels : List (String × Expr) :=
  [("source", source), ("horner", selected .horner), ("estrin", selected .estrin), ("split", selected .split)]

/-- Lower the selected whole-routine plan. Fail on unsupported trees so a
    renamed or unexpected optimizer output cannot silently change the benchmark. -/
private def renderCells : Rlibm.Cells.Expr → Except String String
  | .narrow2 (.input 0) => .ok "rlibm_run_quadratic(bits, 2)"
  | .narrow10 (.input 0) => .ok "rlibm_run_quadratic(bits, 10)"
  | .wide2 (.input 0) => .ok "rlibm_run(bits, 2, rlibm_accum_source)"
  | .wide10 (.input 0) => .ok "rlibm_run(bits, 10, rlibm_accum_source)"
  | other => .error s!"unsupported extracted logarithm plan: {repr other}"

def main (args : List String) : IO Unit := do
  let [directory] := args | throw (IO.userError "usage: lake env lean --run RlibmEmit.lean OUTPUT_DIRECTORY")
  let dir := System.FilePath.mk directory
  IO.FS.createDirAll dir
  let mut c := "#include \"kernel.h\"\n"
  let mut manifest := "function,schedule,symbol\n"
  for (name, tree) in kernels do
    c := c ++ "\n" ++ (← IO.ofExcept (renderAccumulator s!"rlibm_accum_{name}" tree))
    for base in [2, 10] do
      let symbol := s!"verified_log{base}_{name}"
      c := c ++ s!"uint16_t {symbol}(uint16_t bits) \{\n  return rlibm_run(bits, {base}, rlibm_accum_{name});\n}\n"
      manifest := manifest ++ s!"log{base},{name},{symbol}\n"
  for (base, radix) in [(Rlibm.LogBase.two, 2), (Rlibm.LogBase.ten, 10)] do
    let body ← IO.ofExcept (renderCells (Rlibm.Cells.selected base))
    let symbol := s!"verified_log{radix}_cells"
    c := c ++ s!"uint16_t {symbol}(uint16_t bits) \{\n  return {body};\n}\n"
    manifest := manifest ++ s!"log{radix},cells,{symbol}\n"
  IO.FS.writeFile (dir / "schedules.c") c
  IO.FS.writeFile (dir / "schedules.csv") manifest
