import Egglog.Compiler

namespace Egglog

/-- Define a new first-order source AST. Each operand is a recursive child.
    The variable constructor carries its de Bruijn index or register number.
    One definition of each operator's meaning generates both the source
    evaluator and the e-graph signature. The generated `Expr.language` can
    use a relation/function schema with the same constructor interpretation. -/
syntax "egg_language " ident " using " ident " : " term " with_vars " ident " where "
  ("| " ident ident* (" costing " term)? " => " term)+ "otherwise " term : command

open Lean Parser.Term in
macro_rules
  | `(egg_language $ast:ident using $sig:ident : $carrier with_vars $reg:ident where
      $[| $op:ident $arg:ident* $[costing $price]? => $body]* otherwise $fallback) => do
    let mut binders := #[]
    let mut semantics := #[]
    let mut sigBodies := #[]
    let mut costs : Array (Option (TSyntax `term)) := #[]
    let mut alts : Array (TSyntax ``matchAlt) := #[]
    let named := fun (parent child : Ident) => mkIdentFrom child (parent.getId ++ child.getId)
    let eval := named ast (mkIdent `eval)
    let encode := named ast (mkIdent `encode)
    let decode := named ast (mkIdent `decode)
    let sound := named ast (mkIdent `encode_sound)
    let language := named ast (mkIdent `language)
    let base := named sig (mkIdent `model)
    let inputs := mkIdent (← MonadQuotation.addMacroScope `inputs)
    let index := mkIdent (← MonadQuotation.addMacroScope `index)
    let varCtor := named ast reg
    alts := alts.push (← `(matchAltExpr| | $varCtor:ident $index:ident => $inputs $index))
    for i in [:op.size] do
      let ctor := named ast op[i]!
      let args := arg[i]!
      binders := binders.push (← args.mapM fun a => `(bracketedBinder| ($a : $ast)))
      -- Elaborate meanings in the caller's namespace. An operator called
      -- `max` can use Lean's `max` without shadowing it with its own builder.
      let fn := mkIdent (← MonadQuotation.addMacroScope (Name.mkSimple s!"meaning_{i}"))
      let argTypes ← args.mapM fun a => `(bracketedBinder| ($a : $carrier))
      let rhs := body[i]!
      semantics := semantics.push (← `(command| @[inline] private def $fn:ident $argTypes:bracketedBinder* : $carrier := $rhs))
      sigBodies := sigBodies.push (← `($fn:ident $args:ident*))
      if let some cost := price[i]! then
        let localCost := mkIdent (← MonadQuotation.addMacroScope (Name.mkSimple s!"operatorCost_{i}"))
        semantics := semantics.push (← `(command| @[inline] private def $localCost:ident : Nat := $cost))
        costs := costs.push (some (← `($localCost:ident)))
      else
        costs := costs.push none
      let values ← args.mapM fun a => `($eval $inputs $a)
      alts := alts.push (← `(matchAltExpr| | $ctor:ident $args:ident* => $fn:ident $values*))
    let defaultMeaning := mkIdent (← MonadQuotation.addMacroScope `fallbackMeaning)
    semantics := semantics.push (← `(command| @[inline] private def $defaultMeaning:ident : $carrier := $fallback))
    let decl ← `(command| inductive $ast where
      | $reg:ident (index : Nat)
      $[| $op:ident $binders:bracketedBinder*]*
      deriving Repr, DecidableEq)
    let interp ← `(command| def $eval:ident ($inputs : Nat → $carrier) (src : $ast) : $carrier :=
      match src with $alts:matchAlt*)
    let signature ← `(command| egg_signature $sig : $carrier where
      $[| $op $arg* $[costing $costs]? => $sigBodies]* otherwise $defaultMeaning)
    let ns ← `(command| namespace $ast)
    let endNs ← `(command| end $ast)
    let enc := mkIdent `encode
    let dec := mkIdent `decode
    let codec ← `(command| egg_codec $enc, $dec : $ast using $sig where
      | $reg $index <=> .var $index)
    let theoremDecl ← `(command| theorem $sound:ident (src : $ast) (env : Nat → $carrier) :
        ($encode src).eval $base env = $eval env src := by
      induction src <;> simp_all only [$encode:ident, $eval:ident, $base:ident, Egglog.Pat.eval, Egglog.Pat.evalArgs])
    let model := mkIdentFrom ast `model
    let same := mkIdentFrom ast `same
    let lang ← `(command| def $language:ident ($model:ident : Egglog.Model $carrier := $base)
        ($same:ident : ($model).interp = ($base).interp := by rfl) : Egglog.Compiler.Language $ast $carrier where
      model := $model
      encode := $encode
      decode := $decode
      denote src env := $eval env src
      encode_sound src env := by
        rw [← Egglog.Pat.denote_eq_eval _ $model env, $same:ident, Egglog.Pat.denote_eq_eval _ $base env, $sound:ident])
    return mkNullNode (semantics.map TSyntax.raw ++
      #[decl.raw, interp.raw, signature.raw, ns.raw, codec.raw, endNs.raw, theoremDecl.raw, lang.raw])

end Egglog
