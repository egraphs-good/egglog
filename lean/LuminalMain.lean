import Egglog.Examples.LuminalInput

open Luminal

def main (args : List String) : IO Unit := do
  let (source, output, counterPart) ← match args with
    | [src] => pure (src, none, false)
    | ["--emit", src, dst] => pure (src, some dst, false)
    | ["--emit-projections", src, dst] => pure (src, some dst, true)
    | _ => throw (IO.userError "expected [--emit | --emit-projections] MODEL.egg [OUTPUT.egg]")
  let model ← IO.ofExcept <| importSource (← IO.FS.readFile source)
  let ref ← IO.mkRef model.program
  let before ← IO.monoNanosNow
  ref.modify compile
  let after ← IO.monoNanosNow
  let p ← ref.get
  let original := live model model.program
  let optimized := live model p
  unless checkSlice p optimized.contains && model.roots.all optimized.contains do
    throw (IO.userError "output slice omitted a dependency or root")
  let mut fusions := 0
  let mut perToken := 0
  for i in [:p.size] do
    if let .dot l .. := p[i]! then
      if optimized.contains i then
        fusions := fusions + 1
        perToken := perToken + l.n * l.k
  if let some path := output then
    if counterPart then IO.FS.writeFile path (projectionWorkload model p)
    else
      let module := emit model p
      let decoded ← IO.ofExcept (importSource module)
      unless checkEmission model p decoded do throw (IO.userError "SSA serialization mismatch")
      IO.FS.writeFile path module
  IO.println s!"bindings={p.size} outputs={model.roots.length} fused={fusions} live_before={original.size} live_after={optimized.size} product_words_per_token={perToken} compiler_ns={after-before}"
