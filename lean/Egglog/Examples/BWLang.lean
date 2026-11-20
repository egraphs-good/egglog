import Egglog.Parser
import Lean.Data.Json

/-! The CAV ParaBit artifact's JSON format. Both solvers consume the same parsed
    syntax and hypotheses. All order reasoning and simplification happen in the
    e-graph, after parsing. -/
namespace BWLang
open Egglog

inductive Term where
  | literal (n : Int)
  | symbol (name : String)
  | app (op : String) (args : List Term)
  deriving Repr, BEq, Inhabited

private partial def decode : SExpr → Except String Term
  | .atom s => do
    if let some n := s.toInt? then return .literal n
    if s.isEmpty || s.startsWith "\"" then throw "expected a BWLang identifier"
    return .symbol s
  | .list xs => do
    let some (SExpr.atom op) := xs[0]? | throw "expected a BWLang operator"
    let args ← (xs.extract 1 xs.size).toList.mapM decode
    let valid := match args.length with
      | 1 => ["-", "not"].contains op
      | 2 => ["bw", "signed", "+", "-", "*", "div", "^", "<<", ">>", "and",
                "or", "xor", "<", ">", "<=", ">="].contains op
      | 3 => op == "sel"
      | _ => false
    unless valid do throw s!"unknown BWLang application {op}/{args.length}"
    return .app op args

def parseTerm (s : String) : Except String Term := do
  let #[one] ← parseSExprs s | throw "expected one BWLang term"
  decode one

def Term.children : Term → List Term
  | .app _ xs => xs
  | _ => []

/-- Mirror upstream get_bitwidth_exprs. At (bw w e), it visits the children
    of e. In particular the outer annotation does not imply positivity of an
    immediately nested annotation's width. -/
private partial def widths (term : Term) : List Term :=
  match term with
  | .app "bw" [w, e] => w :: e.children.flatMap widths
  | other => other.children.flatMap widths

structure Problem where
  lhs : Term
  rhs : Term
  conditions : List Term
  positiveWidths : List Term
  deriving Inhabited, Repr

def parse (src : String) : Except String Problem := do
  let j ← Lean.Json.parse src
  let lhs ← parseTerm (← j.getObjValAs? String "lhs")
  let rhs ← parseTerm (← j.getObjValAs? String "rhs")
  let pre ← j.getObjValAs? (List String) "preconditions"
  let conditions ← pre.mapM parseTerm
  for cond in conditions do
    let .app op [_, _] := cond | throw "expected a width comparison"
    unless ["<", "<=", ">", ">="].contains op do throw "expected a width comparison"
  return ⟨lhs, rhs, conditions, (widths lhs ++ widths rhs).eraseDups⟩

def call (op : String) (xs : List SExpr) : SExpr :=
  .list ((.atom op :: xs).toArray)

private partial def relational : Term → Except String SExpr
  | .literal n => pure (call "Num" [.atom (toString n)])
  | .symbol s => pure (call "Var" [.atom ("\"" ++ s ++ "\"")])
  | .app op xs => do
    let ctor ← match op with
      | "bw" => pure "Bw" | "signed" => pure "Signed"
      | "+" => pure "Add" | "-" => pure (if xs.length == 1 then "Neg" else "Sub")
      | "*" => pure "Mul" | "div" => pure "Div" | "^" => pure "Pow"
      | "<<" => pure "Shl" | ">>" => pure "Shr" | "not" => pure "Not"
      | "and" => pure "And" | "or" => pure "Or" | "xor" => pure "Xor"
      | "sel" => pure "Sel" | _ => throw s!"unsupported ModIR expression {op}"
    return call ctor (← xs.mapM relational)

/-- The decision-diagram rules need a common variable order for the two inputs.
    BWLang variables, opaque operations and symbolic low-bit masks get stable
    ranks. This traversal assigns names only. The rules compute the diagrams. -/
private partial def decisionAtoms (t : Term) : List Term := Id.run do
  let childAtoms := t.children.flatMap decisionAtoms
  match t with
  | .app "bw" [w, _] => return .app "bw" [w, .literal (-1)] :: childAtoms
  | .app "and" _ | .app "or" _ | .app "xor" _ | .app "not" _ => return childAtoms
  | .literal 0 | .literal (-1) => return []
  | _ => return t :: childAtoms

def Problem.commands (p : Problem) : Except String (List SExpr) := do
  let mut commands := [call "let" [.atom "$lhs", ← relational p.lhs],
                       call "let" [.atom "$rhs", ← relational p.rhs]]
  let atoms := ((decisionAtoms p.lhs) ++ (decisionAtoms p.rhs)).eraseDups
  for (atom, rank) in atoms.zipIdx do
    commands := call "atom-index" [← relational atom, .atom (toString rank)] :: commands
  for w in p.positiveWidths do
    commands := call "width" [← relational w] :: commands
  for cond in p.conditions do
    let .app op [a,b] := cond | throw "expected width comparison"
    let a ← relational a
    let b ← relational b
    let (a, b) := if op == ">" || op == ">=" then (b, a) else (a, b)
    let predicate := if op == "<" || op == ">" then "lt" else "le"
    commands := call predicate [a,b] :: commands
  return commands

end BWLang
