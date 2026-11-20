import Egglog.Certified
import Lean

namespace Egglog

/--
Declare fixed-arity constructors and give each constructor its meaning.
The namespace contains a `model` and a pattern builder for each constructor.
Its `cost` function assigns each operator unit cost or its declared `costing` value.
The fallback interprets ill-formed nodes. Object-language variables can use
`Pat.var` and take their meaning from the evaluation environment.
-/
syntax (name := eggSignature) "egg_signature " ident " : " term " where "
  ("| " ident ident* (" costing " term)? " => " term)+ "otherwise " term : command

open Lean Parser.Term in
macro_rules
  | `(egg_signature $sig:ident : $carrier where $[| $op:ident $arg:ident* $[costing $price]? => $body]* otherwise $fallback) => do
    let mut defs := #[]
    let mut clauses := #[]
    let mut prices : Array (TSyntax ``matchAlt) := #[]
    for i in [:op.size] do
      let name := op[i]!
      let args : Array Ident := arg[i]!
      let tag := Syntax.mkNumLit (toString i)
      let rhs := body[i]!
      let value := price[i]!.getD (← `(1))
      prices := prices.push (← `(matchAltExpr| | $tag => $value))
      let binders ← args.mapM fun a => `(bracketedBinder| ($a : Egglog.Pat))
      defs := defs.push (← `(command| @[match_pattern] abbrev $name $binders* : Egglog.Pat :=
        Egglog.Pat.app $tag [$[$args:ident],*]))
      clauses := clauses.push (← `(matchAltExpr| | $tag, [$[$args:ident],*] => $rhs))
    let modelName := mkIdent `model
    let modelDef ← `(command| def $modelName : Egglog.Model $carrier where
      interp op args := match op, args with
        $clauses:matchAlt*
        | _, _ => $fallback)
    let costName := mkIdent `cost
    let costDef ← `(command| def $costName (op : Nat) : Nat := match op with
      $prices:matchAlt*
      | _ => 1)
    let ns ← `(command| namespace $sig)
    let closeNs ← `(command| end $sig)
    return mkNullNode (#[ns.raw] ++ defs.map TSyntax.raw ++ #[modelDef.raw, costDef.raw, closeNs.raw])

private def rhsBound (r : Rewrite) (premises : List (Pat × Pat)) : Bool :=
  let available := r.lhs.variables ++ premises.flatMap fun (l, r) => l.variables ++ r.variables
  r.rhs.variables.all available.contains

private def checkedRule (r : VerifiedRule m) (_ : rhsBound r.toRewrite [] = true) : VerifiedRule m := r

private def checkedGuarded (r : GuardedRule m) (_ : rhsBound r.toRewrite r.premises = true) : GuardedRule m := r

/--
Bind rule metavariables by name. The attached proof is checked against precisely
the two patterns executed by the matcher. Inside the proof the bound names refer
to arbitrary semantic values, including for repeated or reordered variables.
A `when` clause matches additional terms and supplies a named semantic equality
for each premise.
-/
syntax "egg_rule " "(" ident,* ")" " : " term " ==> " term
  (" when " ("(" ident " : " term:51 " = " term ")")+)? " by " Lean.Parser.Tactic.tacticSeq : term

open Lean in
macro_rules
  | `(egg_rule ($names,*) : $lhs ==> $rhs $[when $[($hs:ident : $left = $right)]*]? by $tactics:tacticSeq) => do
    let names := names.getElems
    let env := mkIdent (← MonadQuotation.addMacroScope `assignment)
    let pre := mkIdent (← MonadQuotation.addMacroScope `premises)
    let mut proof ← `(by $tactics)
    for (name, i) in names.zipIdx.reverse do
      let tag := Syntax.mkNumLit (toString i)
      proof ← `(let $name := $env $tag; $proof)
    let mut rule ← match hs, left, right with
      | some labels, some lefts, some rights => do
        let pairs ← (lefts.zip rights).mapM fun (a, b) => `(($a, $b))
        for (label, i) in labels.zipIdx.reverse do
          let pair := pairs[i]!
          let mut member ← `(List.Mem.head _)
          for _ in [:i] do member ← `(List.Mem.tail _ $member)
          proof ← `(let $label := $pre $pair $member; $proof)
        `(checkedGuarded (Egglog.GuardedRule.mk ⟨$lhs, $rhs⟩ [$[$pairs],*] (fun $env $pre => $proof))
          (by rfl))
      | _, _, _ =>
        `(checkedRule (Egglog.VerifiedRule.mk ⟨$lhs, $rhs⟩ (fun $env => $proof)) (by rfl))
    for (name, i) in names.zipIdx.reverse do
      let tag := Syntax.mkNumLit (toString i)
      rule ← `(let $name : Egglog.Pat := Egglog.Pat.var $tag; $rule)
    return rule

declare_syntax_cat eggStage
syntax "run " term : eggStage
syntax "guarded " term : eggStage
syntax "analyze " term : eggStage
syntax "with_scheduler " term:max " with " term:max " for " term:max : eggStage
syntax "repeat " term:max " {" eggStage,* "}" : eggStage
syntax "saturate" "{" eggStage,* "}" : eggStage
syntax "until " "(" ident,* ")" ("(" term:51 " = " term ")")+ "{" eggStage,* "}" : eggStage
syntax "saturate " term:max " {" eggStage,* "}" : eggStage

/-- Run named rulesets in order on a shared e-graph. `saturate { ... }` runs to
a fixed point. `saturate n { ... }` also stops after `n` iterations.
`until (vars) (lhs = rhs) ... { ... }` skips its enclosed stages if there is
a simultaneous match for its equality facts at the beginning of the stage. -/
syntax "egg_schedule" "{" eggStage,* "}" : term

open Lean in
private partial def expandStage (stx : TSyntax `eggStage) : MacroM Term := do
  match stx with
  | `(eggStage| run $rules) => `(Egglog.Certified.Schedule.run $rules)
  | `(eggStage| with_scheduler $rules with $policy for $n) => `(Egglog.Certified.Schedule.scheduled $n $rules $policy)
  | `(eggStage| analyze $analysis) => `(Egglog.Certified.Schedule.analyze $analysis)
  | `(eggStage| guarded $rules) => `(Egglog.Certified.Schedule.guarded $rules)
  | `(eggStage| repeat $n { $stages,* }) =>
    let body ← stages.getElems.mapM expandStage
    `(Egglog.Certified.Schedule.repeat $n (.seq [$[$body],*]))
  | `(eggStage| saturate { $stages,* }) =>
    let body ← stages.getElems.mapM expandStage
    `(Egglog.Certified.Schedule.fixpoint (.seq [$[$body],*]))
  | `(eggStage| until ($names,*) $[($lhs = $rhs)]* { $stages,* }) =>
    let body ← stages.getElems.mapM expandStage
    let pairs ← (lhs.zip rhs).mapM fun (l, r) => `(($l, $r))
    let mut facts ← `([$[$pairs],*])
    for (name, i) in names.getElems.zipIdx.reverse do
      let tag := Syntax.mkNumLit (toString i)
      facts ← `(let $name : Egglog.Pat := Egglog.Pat.var $tag; $facts)
    `(Egglog.Certified.Schedule.untilMatch $facts (.seq [$[$body],*]))
  | `(eggStage| saturate $n { $stages,* }) =>
    let body ← stages.getElems.mapM expandStage
    `(Egglog.Certified.Schedule.saturate $n (.seq [$[$body],*]))
  | _ => Macro.throwUnsupported

macro_rules
  | `(egg_schedule { $stages,* }) => do
    let body ← stages.getElems.mapM expandStage
    `(Egglog.Certified.Schedule.seq [$[$body],*])

namespace Compiler

/-- On successful decoding, the source expression encodes the extracted tree. -/
abbrev Decoder {AST : Type u} (encode : AST → Pat) :=
  (term : Pat) → Option { src : AST // encode src = term }

/-- Adapt a conventional partial decoder by checking its output. A decoder
with a proved inverse can implement `Decoder` directly and erase this check. -/
def Decoder.checked (encode : AST → Pat) (read : Pat → Option AST) : Decoder encode :=
  fun term => do
    let src ← read term
    if h : encode src = term then return ⟨src, h⟩ else none

/--
Declare a bidirectional codec for an existing inductive IR. Each clause pairs
an IR constructor pattern with its pattern-language representation. List the
recursive children in square brackets. Other fields (for example a register
index) are copied directly. The generated decoder proves its inverse while
reading each node, so successful extraction needs only one decoding traversal.
-/
syntax "egg_codec " ident "," ident " : " term " where "
  ("| " "[" ident,* "]" term " <=> " term)+ : command

open Lean Parser.Term in
macro_rules
  | `(egg_codec $enc:ident, $dec:ident : $carrier where
      $[| [$kids:ident,*] $ir:term <=> $tree:term]*) => do
    let mut encAlts : Array (TSyntax ``matchAlt) := #[]
    let mut decAlts : Array (TSyntax ``matchAlt) := #[]
    for ((ast, pat), fields) in (ir.zip tree).zip kids do
      let children : Array Ident := fields.getElems
      let mut encoded := pat
      for child in children.reverse do
        encoded ← `(let $child:ident := $enc $child:ident; $encoded)
      encAlts := encAlts.push (← `(matchAltExpr| | $ast => $encoded))
      let mut decoded := ast
      let mut bindings : Array (Ident × Ident × Ident) := #[]
      for child in children do
        let name := mkIdent (← Macro.addMacroScope child.getId)
        let eq := mkIdent (← Macro.addMacroScope (child.getId.appendAfter "_inv"))
        decoded ← `(let $child:ident := $name; $decoded)
        bindings := bindings.push (child, name, eq)
      let pf ← if children.isEmpty then `(rfl) else `(by simp only [$enc:ident, *])
      let mut read ← `(some ⟨$decoded, $pf⟩)
      for (child, name, eq) in bindings.reverse do
        read ← `(($dec $child).bind fun ⟨$name, $eq⟩ => $read)
      decAlts := decAlts.push (← `(matchAltExpr| | $pat => $read))
    decAlts := decAlts.push (← `(matchAltExpr| | _ => none))
    let e ← `(command| def $enc (a : $carrier) : Egglog.Pat := match a with $encAlts:matchAlt*)
    let d ← `(command| def $dec : Egglog.Compiler.Decoder $enc := fun p => match p with $decAlts:matchAlt*)
    return mkNullNode #[e.raw, d.raw]

declare_syntax_cat eggCodecOverride
syntax "|" ident term:arg* "<=>" term : eggCodecOverride

/-- Derive the codec from the constructors of a non-indexed inductive. Ordinary
    constructors correspond to operators of the same name in the given signature.
    Override a constructor to map a payload, change operators, or split literals.
    The recursive fields are inferred in both default and overridden clauses. -/
syntax "egg_codec " ident "," ident " : " term " using " ident
  (" where " eggCodecOverride+)? : command

open Lean Meta Elab Command in
private def codecConstructors (carrier : Term) : CommandElabM (List (Name × Array Bool)) :=
  liftTermElabM do
    let type ← whnf (← Term.elabType carrier)
    let .const name levels := type.getAppFn
      | throwErrorAt carrier "expected an inductive type"
    let .inductInfo info ← getConstInfo name
      | throwErrorAt carrier "expected an inductive type"
    unless info.numIndices == 0 && info.all.length == 1 do
      throwErrorAt carrier "automatic codecs require a non-indexed, non-mutual inductive"
    info.ctors.mapM fun ctor => do
      let ctorType ← inferType (mkAppN (mkConst ctor levels) type.getAppArgs)
      forallTelescopeReducing ctorType fun fields _ => do
        let recursive ← fields.mapM fun field => do
          let decl ← field.fvarId!.getDecl
          unless decl.binderInfo.isExplicit do
            throwErrorAt carrier "automatic codecs require explicit constructor fields in '{ctor}'"
          isDefEq decl.type type
        return (ctor, recursive)

open Lean Elab Command in
elab_rules : command
  | `(egg_codec $enc:ident, $dec:ident : $carrier using $sig:ident $[where $cases:eggCodecOverride*]?) => do
    let ctors ← codecConstructors carrier
    let mut overrides : NameMap (Array (Array Term × Term)) := {}
    for c in cases.getD #[] do
      let `(eggCodecOverride| | $name:ident $args* <=> $pat) := c | throwUnsupportedSyntax
      unless ctors.any (fun (ctor, _) => ctor.getString! == name.getId.toString) do
        throwErrorAt name "unknown constructor '{name}' in codec for {carrier}"
      overrides := overrides.insert name.getId ((overrides.find? name.getId).getD #[] |>.push (args, pat))
    let mut asts := #[]
    let mut pats := #[]
    let mut kids := #[]
    for (ctor, recursive) in ctors do
      let ctorId := mkIdent ctor
      let name := Name.mkSimple ctor.getString!
      let clauses ← match overrides.find? name with
        | some cs => pure cs
        | none => do
          unless recursive.all id do
            throwErrorAt carrier "constructor '{ctor}' has a payload; supply a codec override for '{name}'"
          let args : Array Term ← recursive.mapM fun _ => do
            let arg := mkIdent (← liftCoreM (mkFreshUserName `child))
            `($arg:ident)
          let op := mkIdentFrom sig (sig.getId ++ name)
          pure #[(args, ← `($op:ident $args*))]
      for (args, pat) in clauses do
        unless args.size == recursive.size do
          throwErrorAt pat "constructor '{ctor}' expects {recursive.size} fields"
        let mut children : Array Ident := #[]
        for (arg, recField) in args.zip recursive do
          if recField then
            unless arg.raw.isIdent do
              throwErrorAt arg "expected a variable for this recursive field of '{ctor}'"
            children := children.push ⟨arg.raw⟩
        asts := asts.push (← `($ctorId:ident $args*))
        pats := pats.push pat
        kids := kids.push (Syntax.TSepArray.ofElems children)
    elabCommand (← `(egg_codec $enc, $dec : $carrier where
      $[| [$kids:ident,*] $asts:term <=> $pats:term]*))

/-- Connect an existing compiler IR and its semantics to the rewrite engine.
The environment can give arbitrary meanings to free object-language variables.
The semantic carrier may itself be a function, relation, or dependent package. -/
structure Language (AST : Type u) (α : Type v) where
  model : Model α
  encode : AST → Pat
  decode : Decoder encode
  denote : AST → (Nat → α) → α
  encode_sound : ∀ src env, (encode src).eval model env = denote src env

def Language.Equivalent (L : Language AST α) (a b : AST) : Prop :=
  ∀ env, L.denote a env = L.denote b env

/-- A compiler can use the output at runtime and its erased proof in Lean. -/
structure Result (L : Language AST α) (input : AST) where
  output : AST
  sound : L.Equivalent output input

/-- Execute the schedule, extract using the given additive operator cost, and
recover the source-language program. A decoder can reject an unrepresentable
operator or arity, which is reported here as `none`. -/
def Language.optimize? (L : Language AST α) (phases : Certified.Schedule L.model)
    (input : AST) (cost : Nat → Nat := fun _ => 1) : Option (Result L input) := do
  let ⟨tree, same⟩ := Certified.optimize L.model (L.encode input) phases cost
  let ⟨src, inverse⟩ ← L.decode tree
  return ⟨src, fun env => by
    rw [← L.encode_sound src env, inverse, same env, L.encode_sound input env]⟩

/-- A total transformation for a pass pipeline. If the decoder rejects the
extracted tree, retain the original IR with the reflexive proof. -/
@[irreducible] def Language.optimize (L : Language AST α) (phases : Certified.Schedule L.model)
    (input : AST) (cost : Nat → Nat := fun _ => 1) : Result L input :=
  (L.optimize? phases input cost).getD ⟨input, fun _ => rfl⟩

/-- Prove equivalence of two source programs by saturating both in one graph. -/
def Language.proveEq? (L : Language AST α) (phases : Certified.Schedule L.model)
    (a b : AST) : Option (PLift (L.Equivalent a b)) := do
  let ⟨same⟩ ← Certified.proveEq? L.model (L.encode a) (L.encode b) phases
  return ⟨fun env => (L.encode_sound a env).symm.trans
    ((same env).trans (L.encode_sound b env))⟩

/-- Discharge a semantic equivalence goal by kernel reduction of a closed
source-AST query. Free object-language variables still range over all inputs. -/
theorem Language.equivalent_of_query (L : Language AST α) (s : Certified.Schedule L.model)
    (a b : AST) (found : (L.proveEq? s a b).isSome) : L.Equivalent a b :=
  ((L.proveEq? s a b).get found).down

macro "egg_prove " lang:term:max " using " phases:term : tactic =>
  `(tactic| exact Egglog.Compiler.Language.equivalent_of_query $lang $phases _ _ (by decide +kernel))

end Compiler
end Egglog
