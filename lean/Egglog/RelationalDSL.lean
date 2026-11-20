import Egglog.Compiler
import Egglog.RelationalCompiler

namespace Egglog

/-- Declare fixed-arity named relations and their meanings on e-class values.
    `Schema.model Ops.model` equips the constructor model with these predicates. -/
syntax "egg_relations " ident " : " term " where "
  ("| " ident ident* " => " term)+ : command

open Lean Parser.Term Elab.Command in
elab_rules : command
  | `(egg_relations $sig:ident : $carrier where $[| $rel:ident $arg:ident* => $body]*) => do
    let schemaName := (← getCurrNamespace) ++ sig.getId
    let mut builders := #[]
    let mut cases : Array (TSyntax ``matchAlt) := #[]
    for i in [:rel.size] do
      let name : Ident := rel[i]!
      let key := Syntax.mkStrLit (schemaName ++ name.getId).toString
      let args := arg[i]!
      let pred := body[i]!
      let binders ← args.mapM fun a => `(bracketedBinder| ($a : Egglog.Pat))
      builders := builders.push (← `(command| abbrev $name $binders* : Egglog.RelAtom :=
        Egglog.RelAtom.row $key [$[$args:ident],*]))
      cases := cases.push (← `(matchAltExpr| | $key, [$[$args:ident],*] => $pred))
      cases := cases.push (← `(matchAltExpr| | $key, _ => False))
    let model := mkIdent `model
    let defn ← `(command| def $model (base : Egglog.Model $carrier) : Egglog.Model $carrier :=
      { base with relation := fun name args => match name, args with
          $cases:matchAlt*
          | name, args => base.relation name args })
    let ns ← `(command| namespace $sig)
    let closeNs ← `(command| end $sig)
    elabCommand (mkNullNode (#[ns.raw] ++ builders.map TSyntax.raw ++ #[defn.raw, closeNs.raw]))

declare_syntax_cat eggFunctionDecl
syntax "| " ident ident* " ↦ " ident " => " withPosition(term)
  &"merge" ident ident " => " withPosition(term)
  &"preserving" ident ident " by " Lean.Parser.Tactic.tacticSeq : eggFunctionDecl

/-- Define partial tables, each with a fixed key arity, a semantic invariant
    for its stored key/value entries, and a merge that preserves the invariant.
    The merger takes pattern terms. Its proof receives the semantic keys, the
    incoming terms, their interpretation environment `env`, and both invariants. -/
syntax "egg_functions " ident " : " term " over " term " where " eggFunctionDecl+ : command

open Lean Parser.Term Elab.Command in
elab_rules : command
  | `(egg_functions $sig:ident : $carrier over $base where $decls:eggFunctionDecl*) => do
    let schemaName := (← getCurrNamespace) ++ sig.getId
    let mut defs := #[]
    let mut cases : Array (TSyntax ``matchAlt) := #[]
    for decl in decls do
      let `(eggFunctionDecl| | $f:ident $ks:ident* ↦ $v:ident => $inv
          merge $old:ident $newVal:ident => $combine
          preserving $ho:ident $hn:ident by $tactics:tacticSeq) := decl
        | Elab.throwUnsupportedSyntax
      let key := Syntax.mkStrLit (schemaName ++ f.getId).toString
      let binders ← ks.mapM fun k => `(bracketedBinder| ($k : Egglog.Pat))
      defs := defs.push (← `(command| def $f:ident $binders* : Egglog.FunctionKey := ⟨$key, [$[$ks:ident],*]⟩))
      let arity := Syntax.mkNumLit (toString ks.size)
      let spec := mkIdent (← liftCoreM (mkFreshUserName `functionSpec))
      let env := mkIdentFrom decl `env
      let keys := mkIdent (← liftCoreM (mkFreshUserName `keyArgs))
      let impossible := mkIdent (← liftCoreM (mkFreshUserName `wrongArity))
      let mut proof ← `(tactic| cases $keys:ident with
        | nil => exact fun $ho:ident $hn:ident => by $tactics:tacticSeq
        | cons _ _ => exact fun $impossible:ident _ => False.elim $impossible:ident)
      for k in ks.reverse do
        proof ← `(tactic| cases $keys:ident with
          | nil => exact fun $impossible:ident _ => False.elim $impossible:ident
          | cons $k:ident $keys:ident => $proof:tactic)
      defs := defs.push (← `(command| private def $spec:ident : Egglog.FunctionSpec $carrier ($base).interp where
        arity := $arity
        invariant keys $v:ident := match keys with
          | [$[$ks:ident],*] => $inv
          | _ => False
        merge $old:ident $newVal:ident := $combine
        merge_sound $keys:ident $old:ident $newVal:ident $env:ident := by $proof:tactic))
      cases := cases.push (← `(matchAltExpr| | $key => some (same.symm ▸ $spec:ident)))
    let model := mkIdent `model
    let defn ← `(command| def $model (underlying : Egglog.Model $carrier)
        (same : underlying.interp = ($base).interp := by rfl) : Egglog.Model $carrier :=
      { underlying with functions := fun name => match name with
          $cases:matchAlt*
          | name => underlying.functions name })
    let ns ← `(command| namespace $sig)
    let closeNs ← `(command| end $sig)
    elabCommand (mkNullNode (#[ns.raw] ++ defs.map TSyntax.raw ++ #[defn.raw, closeNs.raw]))

declare_syntax_cat eggRelPremise
syntax "(" ident " : " term:51 " = " term ")" : eggRelPremise
syntax "(" ident " : " term:51 " ↦ " term ")" : eggRelPremise
syntax "(" ident " : " term:51 ")" : eggRelPremise

declare_syntax_cat eggRelAction
syntax "yield " term : eggRelAction
syntax "set " term " := " term : eggRelAction
syntax "rewrite " term " ==> " term : eggRelAction
open Lean Parser in
def unionToken := leading_parser nonReservedSymbol "union" true
syntax unionToken term:51 " = " term : eggRelAction

declare_syntax_cat eggRelClause
syntax eggRelAction " by " Lean.Parser.Tactic.tacticSeq : eggRelClause

private def safeHeads (r : RelRule m) : Bool :=
  let vars := r.body.flatMap RelAtom.variables
  (r.head :: r.extra.map RelAction.head).all fun a => a.variables.all vars.contains

private def checkedRelRule (r : RelRule m) (_ : safeHeads r = true) : RelRule m := r

/-- Each named premise supplies evidence to all head actions. A relational
    rewrite also matches its left side in the graph before adding the right side. -/
syntax "egg_rule " "(" ident,* ")" " : " ("when " eggRelPremise+)? eggRelClause : term
syntax "egg_rule " "(" ident,* ")" " : " ("when " eggRelPremise+)? "do" "{" eggRelClause,+ "}" : term

open Lean in
private def expandRelRule (names : Array Ident) (ps : Array (TSyntax `eggRelPremise))
    (clauses : Array (TSyntax `eggRelClause)) : MacroM Term := do
  let mut atoms : Array Term := #[]
  let mut labels : Array Ident := #[]
  let mut lookups := #[]
  for premise in ps do
    match premise with
    | `(eggRelPremise| ($h:ident : $l = $r)) =>
      labels := labels.push h
      atoms := atoms.push (← `(Egglog.RelAtom.eq $l $r))
    | `(eggRelPremise| ($h:ident : $lookup ↦ $value)) =>
      lookups := lookups.push atoms.size
      labels := labels.push h
      atoms := atoms.push (← `(Egglog.FunctionKey.cell $lookup $value))
    | `(eggRelPremise| ($h:ident : $row)) =>
      labels := labels.push h
      atoms := atoms.push row
    | _ => Macro.throwUnsupported
  let env := mkIdent (← MonadQuotation.addMacroScope `assignment)
  let premises := mkIdent (← MonadQuotation.addMacroScope `premises)
  let mut heads : Array (Term × Term) := #[]
  for clause in clauses do
    let `(eggRelClause| $act:eggRelAction by $tactics:tacticSeq) := clause
      | Macro.throwUnsupported
    let mut proof ← `(by $tactics)
    let head ← match act with
      | `(eggRelAction| yield $fact) => pure fact
      | `(eggRelAction| set $lookup := $value) => do
        proof ← `(⟨by rfl, $proof⟩)
        `(Egglog.FunctionKey.cell $lookup $value)
      | `(eggRelAction| union $lhs = $rhs) => `(Egglog.RelAtom.eq $lhs $rhs)
      | `(eggRelAction| rewrite $lhs ==> $rhs) => do
        atoms := atoms.push (← `(Egglog.RelAtom.eq $lhs $lhs))
        `(Egglog.RelAtom.eq $lhs $rhs)
      | _ => Macro.throwUnsupported
    for (name, i) in names.zipIdx.reverse do
      let tag := Syntax.mkNumLit (toString i)
      proof ← `(let $name := $env $tag; $proof)
    for (label, i) in labels.zipIdx.reverse do
      let atom := atoms[i]!
      let mut member ← `(List.Mem.head _)
      for _ in [:i] do member ← `(List.Mem.tail _ $member)
      let fact ← if lookups.contains i then `(($premises $atom $member).2) else `($premises $atom $member)
      proof ← `(let $label := $fact; $proof)
    heads := heads.push (head, ← `(fun $env $premises => $proof))
  let (head, proof) := heads[0]!
  let extra ← (heads.extract 1 heads.size).mapM fun (a,p) => `(Egglog.RelAction.mk $a $p)
  let name := Syntax.mkStrLit (← MonadQuotation.addMacroScope `egg_rule).toString
  let mut rule ← `(checkedRelRule
    (Egglog.RelRule.mk [$[$atoms],*] $head $proof [$[$extra],*] (some $name)) (by rfl))
  for (name, i) in names.zipIdx.reverse do
    let tag := Syntax.mkNumLit (toString i)
    rule ← `(let $name : Egglog.Pat := Egglog.Pat.var $tag; $rule)
  return rule

open Lean in
macro_rules
  | `(egg_rule ($names,*) : $[when $ps:eggRelPremise*]? $act:eggRelClause) =>
    expandRelRule names.getElems (ps.getD #[]) #[act]
  | `(egg_rule ($names,*) : $[when $ps:eggRelPremise*]? do { $acts:eggRelClause,* }) =>
    expandRelRule names.getElems (ps.getD #[]) acts.getElems

end Egglog
