import Init.Meta.Defs

namespace Egglog.FloatLiteral

/-- Egglog uses OrderedFloat keys. Both signs of zero have the same key, as do NaNs. -/
def bits? (token : String) : Option Nat := do
  match token with
  | "NaN" => return 0x7ff8000000000000
  | "inf" | "+inf" => return 0x7ff0000000000000
  | "-inf" => return 0xfff0000000000000
  | _ =>
    let negative := token.startsWith "-"
    let digits := if negative || token.startsWith "+" then (token.drop 1).toString else token
    let (mantissa, divide, exponent) ← Lean.Syntax.decodeScientificLitVal? digits
    let value := Float.ofScientific mantissa divide exponent
    if value == 0 then return 0
    return (if negative then -value else value).toBits.toNat

/-- Emit a round-trippable f64 literal, including values whose six-decimal
    Float.toString representation loses information. Every binary64 has an
    exact, finite decimal expansion. -/
def render (bits : Nat) : String := Id.run do
  let value := Float.ofBits bits.toUInt64
  if value.isNaN then return "NaN"
  if value.isInf then return if value < 0 then "-inf" else "inf"
  let short := toString value
  if bits? short == some bits then return short
  let exponent := bits / 2^52 % 2048
  let mantissa := bits % 2^52 + (if exponent == 0 then 0 else 2^52)
  let shift := max exponent 1
  let sign := if bits >= 2^63 then "-" else ""
  if shift >= 1075 then return s!"{sign}{mantissa * 2^(shift-1075)}.0"
  let places := 1075 - shift
  return s!"{sign}{mantissa * 5^places}.0e-{places}"

end Egglog.FloatLiteral
