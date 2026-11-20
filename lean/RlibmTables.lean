import Egglog.Examples.Rlibm.Coefficients

open Rlibm

/-- Emit the exact coefficients imported by the Lean proof. Negative
    coefficients convert modulo 2^64 as specified by unsigned C arithmetic. -/
def main : IO Unit := do
  IO.println "#ifndef RLIBM_COEFFICIENTS_H"
  IO.println "#define RLIBM_COEFFICIENTS_H"
  IO.println "#include <stdint.h>"
  for (suffix,b) in [("2",LogBase.two), ("10",LogBase.ten)] do
    IO.println s!"static const uint64_t rlibm_step{suffix} = {exponentStep b};"
    IO.println ("static const uint64_t rlibm_coef" ++ suffix ++ "[8][4] = {")
    for c in coefficients b do
      IO.println ("  {" ++ String.intercalate ", " ([c.c0,c.c1,c.c2,c.c3].map toString) ++ "},")
    IO.println "};"
    IO.println ("static const uint64_t rlibm_quad" ++ suffix ++ "[16][3] = {")
    for c in quadratics b do
      IO.println ("  {" ++ String.intercalate ", " ([c.c0,c.c1,c.c2].map toString) ++ "},")
    IO.println "};"
  IO.println "#endif"
