import Egglog.Examples.Rlibm.NativeWord

/-! Evaluate the compiled UInt16 kernel directly on machine-word inputs. -/

open Rlibm

private def modeOf : String → Option NativeWord.Mode
  | "source" => some .source
  | "horner" => some .horner
  | "estrin" => some .estrin
  | "split" => some .split
  | "cells" => some .cells
  | _ => none

private def load (path : String) (count : Nat) : IO (Array UInt16) := do
  let bytes ← IO.FS.readBinFile path
  unless bytes.size == 2 * count do throw (IO.userError s!"expected {count} little-endian words")
  return (Array.range count).map fun i =>
    bytes[2*i]!.toUInt16 ||| (bytes[2*i+1]!.toUInt16 <<< 8)

private def validate (fn : UInt16 → UInt16) (path : String) : IO Unit := do
  let oracle ← load path 65536
  let mut hash := (14695981039346656037 : UInt64)
  for i in [:oracle.size] do
    let y := fn i.toUInt16
    unless y == oracle[i]! do throw (IO.userError s!"word {i}: expected {oracle[i]!}, got {y}")
    hash := (hash ^^^ y.toUInt64) * 1099511628211
  IO.println hash

@[noinline] private def sweep (fn : UInt16 → UInt16) (xs : Array UInt16) (sum : UInt64) : UInt64 :=
  xs.foldl (fun acc x => acc + (fn x).toUInt64) sum

private def throughput (fn : UInt16 → UInt16) (xs : Array UInt16) (sweeps : Nat) : UInt64 := Id.run do
  let mut sum := 0
  for _ in [:sweeps] do sum := sweep fn xs sum
  return sum

private def chain (fn : UInt16 → UInt16) (xs : Array UInt16) (sweeps : Nat) : UInt64 := Id.run do
  let mut sum := 0
  let mut step := (0 : UInt16)
  let mut previous := (0 : UInt16)
  for _ in [:sweeps] do
    for _ in [:xs.size] do
      step := step + 40503
      previous := fn xs[(previous + step).toNat]!
      sum := sum + previous.toUInt64
  return sum

def main (args : List String) : IO Unit := do
  let action :: fname :: variant :: path :: rest := args |
    throw (IO.userError "expected validate|time log2|log10 SCHEDULE WORDS [DISTRIBUTION SWEEPS]")
  let some b := (match fname with | "log2" => some LogBase.two | "log10" => some .ten | _ => none) |
    throw (IO.userError "unknown function")
  let some m := modeOf variant | throw (IO.userError "unknown schedule")
  let slot ← IO.mkRef (fun (x : UInt16) => x)
  let before ← IO.monoNanosNow
  slot.set (NativeWord.prepare b m).eval
  let after ← IO.monoNanosNow
  let fn ← slot.get
  if action == "validate" && rest.isEmpty then return ← validate fn path
  let [kind, count] := rest | throw (IO.userError "expected a distribution and sweep count")
  let some n := count.toNat? | throw (IO.userError "invalid sweep count")
  unless action == "time" && n > 0 && ["positive", "all", "chain"].contains kind do
    throw (IO.userError "invalid workload")
  let xs ← load path (if kind == "positive" then 0x7f7f else 65536)
  let work := if kind == "chain" then chain fn xs else throughput fn xs
  let result ← IO.mkRef (work 8)
  unless (← result.get) != 0 do throw (IO.userError "empty warmup checksum")
  let start ← IO.monoNanosNow
  result.set (work n)
  let stop ← IO.monoNanosNow
  IO.println s!"{stop-start},{xs.size*n},{← result.get},{after-before}"
