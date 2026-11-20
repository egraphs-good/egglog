import Egglog.Examples.Rlibm.Optimizer
import Egglog.Examples.Rlibm.Polynomial
import Egglog.Examples.Rlibm.Cells

/-! Native Lean correctness and execution worker for the rlibm comparison. -/

open Rlibm

private def baseOf : String → Option LogBase
  | "log2" => some .two
  | "log10" => some .ten
  | _ => none

private def profileOf : String → Option Schedule.Profile
  | "horner" => some .horner
  | "estrin" => some .estrin
  | "split" => some .split
  | _ => none

private def prepare (b : LogBase) (schedule : String) : IO ((Nat → Nat) × Nat) := do
  if schedule == "source" then return (core b, 0)
  if schedule == "cells" then
    let slot ← IO.mkRef (core b)
    let start ← IO.monoNanosNow
    slot.set (Cells.prepare b).eval
    let stop ← IO.monoNanosNow
    return (← slot.get, stop - start)
  let some p := profileOf schedule | throw (IO.userError "unknown extraction profile")
  let slot ← IO.mkRef accum
  let start ← IO.monoNanosNow
  slot.set (Schedule.prepare p).eval
  let stop ← IO.monoNanosNow
  let kernel ← slot.get
  return (evalWith kernel b, stop - start)

private def loadWords (path : System.FilePath) (count : Nat) : IO (Array Nat) := do
  let bytes ← IO.FS.readBinFile path
  unless bytes.size == 2 * count do throw (IO.userError s!"expected {count} little-endian words")
  return (Array.range count).map fun i =>
    bytes[2*i]!.toNat + 256*bytes[2*i+1]!.toNat

private def validate (fn : Nat → Nat) (path : String) : IO Unit := do
  let oracle ← loadWords path 65536
  let mut hash := (14695981039346656037 : UInt64)
  for i in [:oracle.size] do
    let y := fn i
    unless y == oracle[i]! do throw (IO.userError s!"word {i}: expected {oracle[i]!}, got {y}")
    hash := (hash ^^^ y.toUInt64) * 1099511628211
  IO.println hash

@[noinline] private def sweep (fn : Nat → Nat) (xs : Array Nat) (sum : UInt64) : UInt64 :=
  xs.foldl (fun acc x => acc + (fn x).toUInt64) sum

private def runSweeps (fn : Nat → Nat) (xs : Array Nat) (n : Nat) : UInt64 := Id.run do
  let mut sum := 0
  for _ in [:n] do sum := sweep fn xs sum
  return sum

private def runChain (fn : Nat → Nat) (xs : Array Nat) (n : Nat) : UInt64 := Id.run do
  let mut sum := 0
  let mut step := (0 : UInt32)
  let mut previous := (0 : Nat)
  for _ in [:n] do
    for _ in [:xs.size] do
      step := step + 40503
      previous := fn xs[((previous.toUInt32 + step) &&& 65535).toNat]!
      sum := sum + previous.toUInt64
  return sum

def main (args : List String) : IO Unit := do
  let mode :: fname :: schedule :: path :: rest := args |
    throw (IO.userError "expected validate|time log2|log10 SCHEDULE WORDS [DISTRIBUTION SWEEPS]")
  let some b := baseOf fname | throw (IO.userError "unknown function")
  let (kernel, setup) ← prepare b schedule
  if mode == "validate" && rest.isEmpty then return ← validate kernel path
  let [distribution, count] := rest | throw (IO.userError "expected a distribution and sweep count")
  let some n := count.toNat? | throw (IO.userError "invalid sweep count")
  unless mode == "time" && n > 0 do throw (IO.userError "expected positive sweep count for time")
  unless ["all", "positive", "chain"].contains distribution do throw (IO.userError "unknown distribution")
  let xs ← loadWords path (if distribution == "positive" then 0x7f7f else 65536)
  let work := if distribution == "chain" then runChain kernel xs else runSweeps kernel xs
  let result ← IO.mkRef (work 8)
  unless (← result.get) != 0 do throw (IO.userError "empty warmup checksum")
  let start ← IO.monoNanosNow
  result.set (work n)
  let stop ← IO.monoNanosNow
  IO.println s!"{stop-start},{xs.size*n},{← result.get},{setup}"
