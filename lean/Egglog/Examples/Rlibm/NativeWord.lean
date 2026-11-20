import Egglog.Examples.Rlibm.Optimizer
import Egglog.Examples.Rlibm.Cells

/-!
# Lowering extracted logarithm schedules to native Lean arithmetic

The elaborator translates the selected polynomial tree into an ordinary
seven-argument Lean function on unboxed UInt64 values. Selection and tree
traversal occur when building this module. The runtime kernel performs word
arithmetic and indexes a precomputed table.
-/
namespace Rlibm.NativeWord

inductive Mode where
  | source | horner | estrin | split | cells
  deriving DecidableEq, Repr

abbrev WordAccum := UInt64 → UInt64 → UInt64 → UInt64 → UInt64 → UInt64 → UInt64 → UInt64

open Lean Elab Term in
private def wordTerm (args : Array Ident) : Schedule.WordExpr → TermElabM Term
  | .arg i => do
    let some name := args[i]? | throwError "unknown coefficient register {i}"
    `($name:ident)
  | .imm z => pure ⟨Syntax.mkNumLit (toString (BitVec.ofInt 64 z).toNat)⟩
  | .add a b => do `($(← wordTerm args a) + $(← wordTerm args b))
  | .mul a b => do `($(← wordTerm args a) * $(← wordTerm args b))

open Lean Elab Term in
elab "compiled_polynomial% " profile:ident : term => do
  let tree ← match profile.getId with
    | `source => pure Schedule.source
    | `horner => pure (Schedule.selected .horner)
    | `estrin => pure (Schedule.selected .estrin)
    | `split => pure (Schedule.selected .split)
    | _ => throwErrorAt profile "unknown extraction profile"
  let args := #[`e, `k, `t, `c0, `c1, `c2, `c3].map mkIdent
  let body ← wordTerm args tree.lower
  elabTerm (← `(fun $args:ident* => $body)) (some (mkConst ``WordAccum))

@[inline] private def sourcePolynomial : WordAccum := compiled_polynomial% source
@[inline] private def hornerPolynomial : WordAccum := compiled_polynomial% horner
@[inline] private def estrinPolynomial : WordAccum := compiled_polynomial% estrin
@[inline] private def splitPolynomial : WordAccum := compiled_polynomial% split

/-- The generated word expressions implement one polynomial for *all*
    operands, independently of the bfloat16 input domain and LP coefficients. -/
private theorem horner_eq : hornerPolynomial = sourcePolynomial := by
  funext e k t a b c d
  simp only [hornerPolynomial, sourcePolynomial]
  ac_rfl

private theorem estrin_eq : estrinPolynomial = sourcePolynomial := by
  funext e k t a b c d
  simp only [estrinPolynomial, sourcePolynomial]
  grind only

private theorem split_eq : splitPolynomial = sourcePolynomial := by
  funext e k t a b c d
  simp only [splitPolynomial, sourcePolynomial]
  grind only

private structure Coeff where
  c0 : UInt64
  c1 : UInt64
  c2 : UInt64
  c3 : UInt64
  deriving Inhabited

private def word (i : Int) : UInt64 := ⟨BitVec.ofInt 64 i⟩

private def wideTable (b : LogBase) : Array Coeff :=
  (coefficients b).map fun c => ⟨word c.c0, word c.c1, word c.c2, word c.c3⟩
private def narrowTable (b : LogBase) : Array Coeff :=
  (quadratics b).map fun c => ⟨word c.c0, word c.c1, word c.c2, 0⟩

private def wide2 := wideTable .two
private def wide10 := wideTable .ten
private def narrow2 := narrowTable .two
private def narrow10 := narrowTable .ten

/-- Round the signed Q40 word to nearest-even bfloat16. The certified
    range of the accumulator keeps the exponent below overflow. -/
@[inline] private def roundWord (z : UInt64) : UInt16 := Id.run do
  if z == 0 then return 0
  let sign := z >>> 63
  let n := if sign == 0 then z else 0-z
  let s := n.log2
  let significand := if s < 7 then n <<< (7-s) else Id.run do
    let cut := s-7
    let unit : UInt64 := 1 <<< cut
    let high := n >>> cut
    let twice := 2 * (n &&& (unit-1))
    return high + (if twice > unit || (twice == unit && (high &&& 1) != 0) then 1 else 0)
  return ((sign <<< 15) + ((s+86) <<< 7) + significand).toUInt16

@[inline] private def kernel (table : @& Array Coeff) (bucketBits k : UInt64)
    (op : WordAccum) (bits : UInt16) : UInt16 := Id.run do
  if bits &&& 0x7fff == 0 then return 0xff80
  if bits == 0x7f80 then return bits
  if bits > 0x7f80 then return 0xffff
  let raw := bits.toUInt64
  let field := raw >>> 7
  let frac := raw &&& 127
  let shift := if field == 0 then 7-frac.log2 else 0
  let e := if field == 0 then 0-126-shift else field-127
  let m := if field == 0 then (frac <<< shift) - 128 else frac
  let c := table[(m >>> bucketBits).toNat]!
  let t := m &&& ((1 <<< bucketBits)-1)
  return roundWord (op e k t c.c0 c.c1 c.c2 c.c3)

@[inline] private def quadratic (e k t a b c _d : UInt64) : UInt64 := e*k + (a + t*(b + t*c))

/-- Resolve both table and exponent-compensation constant before the input
    loop. Each constant table is built once at module initialization. -/
@[inline] private def wide (base : LogBase) (op : WordAccum) (x : UInt16) : UInt16 :=
  match base with
  | .two => kernel wide2 4 (word (exponentStep .two)) op x
  | .ten => kernel wide10 4 (word (exponentStep .ten)) op x

@[inline] private def narrow (base : LogBase) (x : UInt16) : UInt16 :=
  match base with
  | .two => kernel narrow2 3 (word (exponentStep .two)) quadratic x
  | .ten => kernel narrow10 3 (word (exponentStep .ten)) quadratic x

open Lean Elab Term in
/-- Translate the extracted whole-routine constructor. Unexpected AST shapes
    are rejected during elaboration. -/
elab "compiled_log% " radix:num : term => do
  let base : LogBase ← match radix.getNat with
    | 2 => pure .two | 10 => pure .ten | _ => throwErrorAt radix "expected base 2 or 10"
  let body ← match Cells.selected base with
    | .narrow2 (.input 0) => `(fun x => narrow .two x)
    | .narrow10 (.input 0) => `(fun x => narrow .ten x)
    | .wide2 (.input 0) => `(fun x => wide .two sourcePolynomial x)
    | .wide10 (.input 0) => `(fun x => wide .ten sourcePolynomial x)
    | other => throwError "unsupported selected logarithm: {repr other}"
  elabTerm body (some (← elabType (← `(UInt16 → UInt16))))

private def cells2 : UInt16 → UInt16 := compiled_log% 2
private def cells10 : UInt16 → UInt16 := compiled_log% 10

structure Prepared where
  mode : Mode
  eval : UInt16 → UInt16

/-- Dispatch once when the caller chooses the routine. The closure contains
    compiled arithmetic and can be reused across arbitrary input batches. -/
@[noinline] def prepare (base : LogBase) (mode : Mode) : Prepared :=
  ⟨mode, match base, mode with
  | .two, .source => fun x => wide .two sourcePolynomial x
  | .ten, .source => fun x => wide .ten sourcePolynomial x
  | .two, .horner => fun x => wide .two hornerPolynomial x
  | .ten, .horner => fun x => wide .ten hornerPolynomial x
  | .two, .estrin => fun x => wide .two estrinPolynomial x
  | .ten, .estrin => fun x => wide .ten estrinPolynomial x
  | .two, .split => fun x => wide .two splitPolynomial x
  | .ten, .split => fun x => wide .ten splitPolynomial x
  | .two, .cells => cells2 | .ten, .cells => cells10⟩

theorem wide_eq_source (base : LogBase) (mode : Mode) (h : mode ≠ .cells) (x : UInt16) :
    (prepare base mode).eval x = (prepare base .source).eval x := by
  cases base <;> cases mode <;> simp_all [prepare, horner_eq, estrin_eq, split_eq]

end Rlibm.NativeWord
