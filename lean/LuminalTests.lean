import Egglog.Examples.LuminalInput

open Luminal

private def l : Layout := ⟨"s",2,3⟩
private def otherLayout : Layout := ⟨"other batch",2,3⟩
private def shared : Program := #[
  .opaqueOp "a" [], .opaqueOp "b" [], .times l 0 1, .sum l 2,
  .sum l 2, .sum otherLayout 2, .sum l 0]

-- One shared product, two consumers, an incompatible batch, and an unrelated sum.
#guard compile shared == #[.opaqueOp "a" [], .opaqueOp "b" [], .times l 0 1,
                          .dot l 0 1, .dot l 0 1, .sum otherLayout 2, .sum l 0]
#guard !checkSlice (compile shared) (fun i => i == 3)
#guard checkSlice (compile shared) (fun i => [0,1,3].contains i)

-- Two rows of activations and output features, using the concrete rounded
-- FP32 operators also for multiply, accumulation and zero.
private def buf (xs : Array Float32) : Buffer := fun i => (xs.getD i 0).toBits
private def a := buf #[1,2,3,4,5,6]
private def w := buf #[7,8,9,10,11,12]
#guard [0,1,2,3].map (orderedMatmul ⟨l,f32⟩ a w) == ([50,68,122,167] : List Float32).map Float32.toBits
#guard [0,1,2,3].map (reduce ⟨l,f32⟩ (product ⟨l,f32⟩ a w)) ==
  ([50,68,122,167] : List Float32).map Float32.toBits

private def fusions (p : Program) := p.foldl (fun n ins => match ins with | .dot .. => n + 1 | _ => n) 0

def main : IO Unit := do
  let src ← IO.FS.readFile "../tests/luminal-llama.egg"
  let model ← IO.ofExcept (importSource src)
  let out := compile model.program
  unless model.program.size == 2413 && fusions out == 225 do
    throw (IO.userError "unexpected archived model or contraction count")
  let emitted := emit model out
  let reopened ← IO.ofExcept (importSource emitted)
  unless checkEmission model out reopened && reopened.program.size == 2155 && fusions reopened.program == 225 &&
      emit reopened (compile reopened.program) == emitted do
    throw (IO.userError "lowered model did not round trip")
  let swapped := emitted.replace "(let t63 (OrderedMatMul t12 t8 t52 t56))"
    "(let t63 (OrderedMatMul t12 t8 t56 t52))"
  unless swapped != emitted do throw (IO.userError "missing fused q projection")
  let wrong ← IO.ofExcept (importSource swapped)
  if checkEmission model out wrong then throw (IO.userError "swapped contraction inputs accepted")
  -- t63 originally reads a contiguous product. Changing one reduction stride
  -- invalidates fusion at precisely that site; the unchanged sites remain eligible.
  let altered := src.replace "(let t63 (Sum t12 t8 t61 t62 t19 t29))"
    "(let t63 (Sum t12 t8 t61 t62 t22 t29))"
  unless altered != src do throw (IO.userError "missing q projection in fixture")
  let changed ← IO.ofExcept (importSource altered)
  unless fusions (compile changed.program) == 224 do throw (IO.userError "bad stride accepted")
