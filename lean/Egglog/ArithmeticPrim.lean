import Egglog.Rules

namespace Egglog

-- User constructors, the interpreted operators and primitive literals have
-- disjoint IDs. These operators are interpreted only by the raw language runner.
def literalBase : Nat := 2 ^ 32
def constructorLimit : Nat := literalBase - 12

def arithmeticOp? : String → Option Nat
  | "+" => some constructorLimit
  | "-" => some (constructorLimit + 1)
  | "*" => some (constructorLimit + 2)
  | "/" => some (constructorLimit + 3)
  | "%" => some (constructorLimit + 4)
  | "&" => some (constructorLimit + 5)
  | "|" => some (constructorLimit + 6)
  | "^" => some (constructorLimit + 7)
  | "<<" => some (constructorLimit + 8)
  | ">>" => some (constructorLimit + 9)
  | "min" => some (constructorLimit + 10)
  | "max" => some (constructorLimit + 11)
  | _ => none

private def isArithmeticOp (op : Nat) : Bool := constructorLimit ≤ op && op ≤ 0xffffffff

def encodeInt (i : Int) : Nat := Id.run do
  let zigzag := match i with | .ofNat n => 2*n | .negSucc n => 2*n + 1
  return literalBase + zigzag * 2

def decodeInt? (op : Nat) : Option Int := do
  guard (op ≥ literalBase && op % 2 == 0)
  let i := (op - literalBase) / 2
  return if i % 2 == 0 then .ofNat (i / 2) else .negSucc (i / 2)

def i64InRange (n : Int) : Bool := -(2^63 : Int) ≤ n && n < (2^63 : Int)

/-- Evaluate a checked i64 operation and return the result's literal operator ID. -/
def calculate (op a b : Nat) : Except String Nat := do
  let some x := decodeInt? a | throw "expected an i64 operand"
  let some y := decodeInt? b | throw "expected an i64 operand"
  let a := Int64.ofInt x
  let b := Int64.ofInt y
  let n ← match op - constructorLimit with
    | 0 => pure (x + y)
    | 1 => pure (x - y)
    | 2 => pure (x * y)
    | 3 | 4 => do
      if y == 0 || (x == -(2^63 : Int) && y == -1) then throw "undefined i64 operation"
      pure (if op == constructorLimit + 3 then x.tdiv y else x.tmod y)
    | 5 => pure (a &&& b).toInt
    | 6 => pure (a ||| b).toInt
    | 7 => pure (a ^^^ b).toInt
    | 8 | 9 => do
      unless 0 ≤ y && y < 64 do throw "undefined i64 operation"
      pure (if op == constructorLimit + 8 then (a <<< b).toInt else (a >>> b).toInt)
    | 10 => pure (min x y)
    | 11 => pure (max x y)
    | _ => throw "unknown arithmetic operator"
  unless i64InRange n do throw "i64 arithmetic overflow"
  return encodeInt n

partial def containsArithmetic : Pat → Bool
  | .var _ => false
  | .app op kids => isArithmeticOp op || kids.any containsArithmetic

/-- Compute the operator ID for a primitive result without interning intermediate
    arithmetic values. Rule guards also use literal IDs independent of e-classes. -/
private partial def evalPrimitive (g : EGraph) (env : Bindings) : Pat → Except String Nat
  | .var v => do
    let some id := env[v]?.join | throw "unbound arithmetic operand"
    return g.nodes[g.find id]!.op
  | .app op kids => do
    if isArithmeticOp op then
      let [a, b] := kids | throw "expected two i64 operands"
      return ← calculate op (← evalPrimitive g env a) (← evalPrimitive g env b)
    unless kids.isEmpty do throw "expected an i64 operand"
    return op

private partial def foldArithmetic (g : EGraph) (env : Bindings) (p : Pat) : Except String Pat := do
  match p with
  | .var _ => return p
  | .app op kids =>
    if isArithmeticOp op then return .app (← evalPrimitive g env p) []
    return .app op (← kids.mapM (foldArithmetic g env))

/-- Evaluate a typed rule head in one pass. Constructor arguments are built
    directly, while interpreted arithmetic interns only its final literal. -/
partial def instantiateArithmetic (p : Pat) (env : Bindings) (g : EGraph) : Except String (Id × EGraph) :=
  match p with
  | .var v => do
    let some id := env[v]?.join | throw "unbound head variable"
    return (id, g)
  | .app op kids => do
    if isArithmeticOp op then return g.add ⟨← evalPrimitive g env p, #[]⟩
    let mut g := g
    let mut ids := #[]
    for kid in kids do
      let (id, next) ← instantiateArithmetic kid env g
      ids := ids.push id
      g := next
    return g.add ⟨op, ids⟩

def searchArithmetic (g : EGraph) (rules : Array Rewrite) : Except String MatchBatch := do
  let batch := searchRound g rules
  batch.filterHeadsM fun rhs env => some <$> foldArithmetic g env rhs

inductive Comparison where
  | eq | ne | lt | le | gt | ge
  deriving Inhabited

def comparison? : String → Option Comparison
  | "=" => some .eq | "!=" => some .ne | "<" => some .lt | "<=" => some .le
  | ">" => some .gt | ">=" => some .ge | _ => none

def compareI64 (cmp : Comparison) (x y : Nat) : Except String Bool := do
  let some a := decodeInt? x | throw "ordered comparison requires i64"
  let some b := decodeInt? y | throw "ordered comparison requires i64"
  match cmp with
  | .lt => return a < b | .le => return a ≤ b | .gt => return a > b | .ge => return a ≥ b
  | .eq => return a == b | .ne => return a != b

structure GuardTest where
  cmp : Comparison
  lhs : Pat
  rhs : Pat
  primitive : Bool
  deriving Inhabited

structure GuardedRewrite where
  rule : Rewrite
  conditions : Array GuardTest
  deriving Inhabited

private partial def lookupPattern (g : EGraph) (p : Pat) (env : Bindings) : Option Id := do
  match p with
  | .var v => return g.find (← env[v]?.join)
  | .app op kids => g.lookup? ⟨op, (← kids.mapM (fun k => lookupPattern g k env)).toArray⟩

def checkGuard (g : EGraph) (env : Bindings) (test : GuardTest) : Except String Bool := do
  let getValue := fun p => do
    let folded := foldArithmetic g env p
    if let .error err := folded then
      if err == "i64 arithmetic overflow" || err == "undefined i64 operation" then return none
    let p ← folded
    -- Primitive comparisons operate on the actual literal, even when no row
    -- in the graph mentions the literal appearing in the condition.
    if test.primitive then
      match p with
      | .var v => return (env[v]?.join).map fun id => g.nodes[g.find id]!.op
      | .app op [] => return some op
      | _ => return none
    return lookupPattern g p env
  let some x ← getValue test.lhs | return false
  let some y ← getValue test.rhs | return false
  match test.cmp with
  | .eq => return x == y
  | .ne => return x != y
  | order => compareI64 order x y

def searchGuarded (g : EGraph) (rules : Array GuardedRewrite) : Except String MatchBatch := do
  let mut all := searchRound g #[]
  for r in rules do
    let batch := searchRound g #[r.rule]
    let filtered ← batch.filterHeadsM fun rhs env => do
      for cond in r.conditions do unless ← checkGuard g env cond do return none
      return some (← foldArithmetic g env rhs)
    all := all.append filtered
  return all

end Egglog
