import Egglog.Compiler

namespace Egglog.Compiler

/-- A source-language graph for a function, module, or compiler pass with several roots. -/
structure Session (L : Language AST α) where
  private mk ::
  private core : Certified.Session L.model

structure Handle (L : Language AST α) (input : AST) where
  private mk ::
  private core : Certified.Handle L.model (L.encode input)

def Language.session (L : Language AST α) : Session L := ⟨.empty L.model⟩

def Session.add {L : Language AST α} (s : Session L) (input : AST) : Session L × Handle L input :=
  let (g, root) := s.core.add (L.encode input)
  (⟨g⟩, ⟨root⟩)

def Session.run (s : Session L) (schedule : Certified.Schedule L.model) : Session L :=
  ⟨s.core.run schedule⟩

def Session.equivalent? (s : Session L) (a : Handle L input) (b : Handle L other) :
    Option (PLift (L.Equivalent input other)) := do
  let ⟨h⟩ ← s.core.equivalent? a.core b.core
  return ⟨fun env => (L.encode_sound input env).symm.trans ((h env).trans (L.encode_sound other env))⟩

def Session.merge? (s : Session L) (a : Handle L input) (b : Handle L other)
    (h : L.Equivalent input other) : Option (Session L) := do
  let g ← s.core.merge? a.core b.core (fun env => (L.encode_sound input env).trans
    ((h env).trans (L.encode_sound other env).symm))
  return ⟨g⟩

/-- A cost solution shared by all roots in one session snapshot. -/
structure Extractor (L : Language AST α) where
  private mk ::
  private core : Certified.Extractor L.model

def Session.extractor (s : Session L) (cost : Nat → Nat := fun _ => 1) : Extractor L :=
  ⟨s.core.extractor cost⟩

private def decodeWitness {L : Language AST α} {input : AST}
    (w : {tree : Pat // ∀ env, tree.eval L.model env = (L.encode input).eval L.model env}) :
    Option (Result L input) := do
  let ⟨tree, same⟩ := w
  let ⟨program, inverse⟩ ← L.decode tree
  return ⟨program, fun env => by
    rw [← L.encode_sound program env, inverse, same env, L.encode_sound input env]⟩

def Extractor.extract? (s : Extractor L) (a : Handle L input) : Option (Result L input) :=
  (s.core.extract? a.core).bind decodeWitness

/-- Amortize recovery of shared subtrees across a module's output roots. -/
structure ExtractionBatch (L : Language AST α) where
  private mk ::
  private core : Certified.ExtractionBatch L.model

def Extractor.batch (s : Extractor L) : ExtractionBatch L := ⟨s.core.batch⟩

def ExtractionBatch.extract? (s : ExtractionBatch L) (root : Handle L input) :
    ExtractionBatch L × Option (Result L input) :=
  let (next, hit) := s.core.extract? root.core
  (⟨next⟩, hit.bind decodeWitness)

end Egglog.Compiler
