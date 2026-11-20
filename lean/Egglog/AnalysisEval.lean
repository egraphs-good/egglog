import Egglog.Certified

open Egglog

namespace Egglog.Certified.ValueAnalysis

mutual
  /-- Evaluate an exact analysis on one closed tree. The operation and its proof
    law are the ones used in the e-graph. This fold is convenient for replay
    in the kernel when both source terms can be evaluated independently. -/
  def eval? {m : Model α} (a : ValueAnalysis m β) (p : Pat) :
      Option {v : β // ∀ env, p.eval m env = a.meaning v} :=
    match p with
    | .var _ => none
    | .app op args => do
      let xs ← evalArgs? a args
      match h : a.transfer op xs.val with
      | none => none
      | some y => return ⟨y, fun env =>
          (congrArg (m.interp op) (xs.property env)).trans (a.sound op xs.val y h)⟩

  def evalArgs? {m : Model α} (a : ValueAnalysis m β) (args : List Pat) :
      Option {vs : List β // ∀ env, Pat.evalArgs args m env = vs.map a.meaning} :=
    match args with
    | [] => some ⟨[], fun _ => rfl⟩
    | x :: xs => do
      let head ← eval? a x
      let tail ← evalArgs? a xs
      return ⟨head.val :: tail.val, fun env => by
        simp only [Pat.evalArgs, List.map_cons, head.property env, tail.property env]⟩
end

end Egglog.Certified.ValueAnalysis
