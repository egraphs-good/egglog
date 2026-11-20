import Egglog.Compiler

namespace Luminal

open Egglog Egglog.Compiler

/-- Contiguous product [batch,n,k] and reduction [batch,n]. Batch may be
symbolic. The proved equality holds at every output address, for any batch. -/
structure Layout where
  batch : String
  n : Nat
  k : Nat
  deriving Repr, DecidableEq, Inhabited, Hashable

abbrev Buffer := Nat → UInt32

/-- Each scalar operation receives and returns the rounded word. Fusion may
eliminate a temporary tensor but may neither reassociate nor contract the
multiplication and addition. -/
structure Scalars where
  zero : UInt32
  add : UInt32 → UInt32 → UInt32
  mul : UInt32 → UInt32 → UInt32

def f32 : Scalars where
  zero := 0
  add x y := (Float32.ofBits x + Float32.ofBits y).toBits
  mul x y := (Float32.ofBits x * Float32.ofBits y).toBits

structure Context where
  layout : Layout
  scalars : Scalars

/-- Row-major flattening of broadcast inputs [batch,1,k] and [1,n,k].
The intermediate stores one rounded word per product. -/
def product (c : Context) (a b : Buffer) : Buffer := fun addr =>
  let out := addr / c.layout.k
  let i := addr % c.layout.k
  c.scalars.mul (a (out / c.layout.n * c.layout.k + i))
                (b (out % c.layout.n * c.layout.k + i))

private def sumTerms (s : Scalars) (term : Nat → UInt32) : Nat → UInt32
  | 0 => s.zero
  | k+1 => s.add (sumTerms s term k) (term k)

def reduce (c : Context) (x : Buffer) : Buffer := fun out =>
  sumTerms c.scalars (fun i => x (c.layout.k * out + i)) c.layout.k

/-- Fused kernel with direct activation and weight reads. -/
def orderedMatmul (c : Context) (a b : Buffer) : Buffer := fun out =>
  let activation := (out / c.layout.n) * c.layout.k
  let weight := (out % c.layout.n) * c.layout.k
  sumTerms c.scalars (fun i => c.scalars.mul (a (activation + i)) (b (weight + i))) c.layout.k

private theorem sumTerms_congr (s : Scalars) (f g : Nat → UInt32) (k : Nat)
    (h : ∀ i, i < k → f i = g i) : sumTerms s f k = sumTerms s g k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp only [sumTerms, ih (fun i hi => h i (Nat.lt_succ_of_lt hi)), h k (Nat.lt_succ_self k)]

theorem fuse_sound (c : Context) (a b : Buffer) :
    reduce c (product c a b) = orderedMatmul c a b := by
  funext out
  apply sumTerms_congr
  intro i hi
  have hk : 0 < c.layout.k := Nat.zero_lt_of_lt hi
  simp [product, Nat.mul_add_div hk, Nat.div_eq_of_lt hi, Nat.mod_eq_of_lt hi]

/-- Named instructions outside the modeled projection fragment can implement
any pure operator of their input buffers. The tag also identifies constants
and source input slots, whose meanings can be supplied by the caller. -/
inductive Inst where
  | opaqueOp (tag : String) (args : List Nat)
  | times (shape : Layout) (a b : Nat)
  | sum (shape : Layout) (x : Nat)
  | dot (shape : Layout) (a b : Nat)
  deriving Repr, DecidableEq, Inhabited

abbrev Oracle := String → List Buffer → Buffer

def Inst.eval (ops : Scalars) (other : Oracle) (env : Nat → Buffer) : Inst → Buffer
  | .opaqueOp tag args => other tag (args.map env)
  | .times l a b => product ⟨l,ops⟩ (env a) (env b)
  | .sum l x => reduce ⟨l,ops⟩ (env x)
  | .dot l a b => orderedMatmul ⟨l,ops⟩ (env a) (env b)

abbrev Program := Array Inst

/-- SSA equations include every original binding. A fused instruction makes its
product temporary dead when it was the only consumer. Other consumers can
continue reading it. -/
def Satisfies (p : Program) (ops : Scalars) (other : Oracle) (env : Nat → Buffer) :=
  ∀ i ins, p[i]? = some ins → env i = ins.eval ops other env


private structure Inputs (p : Program) where
  ops : Scalars
  env : Nat → Buffer
  productDefs : ∀ i l a b, p[i]? = some (.times l a b) →
    env i = product ⟨l,ops⟩ (env a) (env b)

private abbrev Value (p : Program) := Inputs p → Buffer

private def graphModel (p : Program) (l : Layout) : Model (Value p) where
  interp code args w := match code, args with
    | 0, [a,b] => product ⟨l,w.ops⟩ (a w) (b w)
    | 1, [x] => reduce ⟨l,w.ops⟩ (x w)
    | 2, [a,b] => orderedMatmul ⟨l,w.ops⟩ (a w) (b w)
    | n+3, [] => w.env n
    | _, _ => fun _ => 0

private def reg (id : Nat) : Pat := .app (id+3) []
private def times (a b : Pat) : Pat := .app 0 [a,b]
private def sum (x : Pat) : Pat := .app 1 [x]
private def dot (a b : Pat) : Pat := .app 2 [a,b]

private def fuse (p : Program) (l : Layout) : VerifiedRule (graphModel p l) :=
  egg_rule (a,b) : sum (times a b) ==> dot a b by
    funext w
    exact fuse_sound ⟨l,w.ops⟩ (a w) (b w)

private structure SumRoot (p : Program) (l : Layout) where
  src : Nat
  handle : Certified.Handle (graphModel p l) (sum (reg src))

private structure Proposal (p : Program) where
  layout : Layout
  src : Nat
  lhs : Nat
  rhs : Nat
  valid : ∀ w : Inputs p, orderedMatmul ⟨layout,w.ops⟩ (w.env lhs) (w.env rhs) =
    reduce ⟨layout,w.ops⟩ (w.env src)

private abbrev Proposals (p : Program) := Std.HashMap (Layout × Nat) (Proposal p)

private def proposal? {p : Program} (l : Layout) (x : Nat) (out : Pat)
    (h : ∀ env, out.eval (graphModel p l) env = (sum (reg x)).eval (graphModel p l) env) :
    Option (Proposal p) :=
  match out with
  | .app 2 [.app (a+3) [], .app (b+3) []] =>
    some ⟨l,x,a,b,fun w => congrFun (h (fun _ _ _ => 0)) w⟩
  | _ => none

/-- Seed every product once, including unused products. Its SSA equation
justifies identifying the result register with the product enode. A reduction
is inserted independently and only mentions its own source register. The
e-matcher must find products across the register aliases. -/
private def searchLayout (p : Program) (l : Layout) (previous : Proposals p) : Proposals p := Id.run do
  let mut s := Certified.Session.empty (graphModel p l)
  let mut roots : Array (SumRoot p l) := #[]
  for h : i in [:p.size] do
    match he : p[i] with
    | .times l' a b =>
      if hl : l' = l then
        let (s', defined) := s.add (times (reg a) (reg b))
        let (s', name) := s'.add (reg i)
        have valid (env) : (reg i).eval (graphModel p l) env =
            (times (reg a) (reg b)).eval (graphModel p l) env := by
          subst l'
          funext w
          exact w.productDefs i l a b (by simp [Array.getElem?_eq_getElem h.upper, he])
        s := s'.mergeIfOwned name defined valid
    | .sum l' x =>
      if l == l' then
        let (s', root) := s.add (sum (reg x))
        s := s'
        roots := roots.push ⟨x,root⟩
    | _ => pure ()
  let rules := #[fuse p l]
  let graph := s.run (egg_schedule {run rules})
  -- Input names have unit cost. Reading a separately materialized temporary
  -- through Sum is more expensive than the single fused operation.
  let mut batch := (graph.extractor (fun op => if op == 1 then 4 else 1)).batch
  let mut found := previous
  for root in roots do
    let (next, out) := batch.extract? root.handle
    batch := next
    if let some tree := out then
      if let some answer := proposal? l root.src tree.val tree.property then
        found := found.insert (l,root.src) answer
  return found

private def proposals (p : Program) : Proposals p := Id.run do
  let mut layouts := #[]
  for ins in p do
    if let .sum l _ := ins then
      if !layouts.contains l then layouts := layouts.push l
  return layouts.foldl (fun found l => searchLayout p l found) {}

private def lower (found : Proposals p) : Inst → Inst
  | .sum l x =>
    match found[(l,x)]? with
    | some r => if r.layout = l ∧ r.src = x then .dot l r.lhs r.rhs else .sum l x
    | none => .sum l x
  | ins => ins

def compile (p : Program) : Program := p.map (lower (proposals p))

private theorem lower_sound (p : Program) (found : Proposals p) (w : Inputs p) (other : Oracle)
    (ins : Inst) : (lower found ins).eval w.ops other w.env = ins.eval w.ops other w.env := by
  cases ins with
  | sum l x =>
    simp only [lower]
    split
    next r h =>
      split
      next hh => exact hh.1 ▸ hh.2 ▸ r.valid w
      next => rfl
    next => rfl
  | _ => rfl

/-- Whole-program fusion preserves exactly the satisfying SSA assignments,
including all original outputs, for every input and every surrounding pure
operator. No algebraic law on the floating-point operations is assumed. -/
theorem compile_correct (p : Program) (ops : Scalars) (other : Oracle) (env : Nat → Buffer) :
    Satisfies (compile p) ops other env ↔ Satisfies p ops other env := by
  constructor
  · intro h i ins hi
    have products : ∀ j l a b, p[j]? = some (.times l a b) →
        env j = product ⟨l,ops⟩ (env a) (env b) := by
      intro j l a b hj
      apply h j (.times l a b)
      simp [compile, Array.getElem?_map, hj, lower]
    rw [← lower_sound p (proposals p) ⟨ops,env,products⟩ other ins]
    apply h i (lower (proposals p) ins)
    simp [compile, Array.getElem?_map, hi]
  · intro h i ins hi
    simp only [compile, Array.getElem?_map] at hi
    cases hp : p[i]? with
    | none => simp [hp] at hi
    | some src =>
      have he : lower (proposals p) src = ins := by simpa [hp] using hi
      subst ins
      have products : ∀ j l a b, p[j]? = some (.times l a b) →
          env j = product ⟨l,ops⟩ (env a) (env b) := fun j l a b hj => h j (.times l a b) hj
      exact (h i src hp).trans (lower_sound p (proposals p) ⟨ops,env,products⟩ other src).symm

def Inst.dependencies : Inst → List Nat
  | .opaqueOp _ args => args
  | .sum _ x => [x]
  | .times _ a b | .dot _ a b => [a,b]

private theorem eval_congr (ins : Inst) (ops : Scalars) (other : Oracle) (a b : Nat → Buffer)
    (h : ∀ j ∈ ins.dependencies, a j = b j) : ins.eval ops other a = ins.eval ops other b := by
  cases ins with
  | opaqueOp tag args => exact congrArg (other tag) (List.map_congr_left h)
  | sum l x => simp only [Inst.eval, h x (by simp [Inst.dependencies])]
  | times l x y | dot l x y =>
    simp only [Inst.eval, h x (by simp [Inst.dependencies]), h y (by simp [Inst.dependencies])]

/-- Check the emitted slice is topologically ordered and closed under every
semantic input. Additional metadata needed only by the serializer may be kept. -/
def checkSlice (p : Program) (keep : Nat → Bool) :=
  (List.range p.size).all fun i => !keep i ||
    ((p[i]?.map Inst.dependencies).getD []).all (fun j => j < i && keep j)

def SatisfiesSlice (p : Program) (keep : Nat → Bool) (ops : Scalars)
    (other : Oracle) (env : Nat → Buffer) :=
  ∀ i ins, p[i]? = some ins → keep i = true → env i = ins.eval ops other env

/-- Any evaluation of the dataflow slice produced by the compiler agrees
with the original model at each retained output, including in the presence
of shared subgraphs. The predicate on the mask is checked during emission. -/
theorem slice_correct (p : Program) (keep : Nat → Bool) (ops : Scalars) (other : Oracle)
    (src dst : Nat → Buffer) (hsrc : Satisfies p ops other src)
    (hcheck : checkSlice (compile p) keep = true)
    (hdst : SatisfiesSlice (compile p) keep ops other dst) :
    ∀ i, i < (compile p).size → keep i = true → src i = dst i := by
  have allSrc := (compile_correct p ops other src).mpr hsrc
  intro i
  induction i using Nat.strongRecOn with
  | ind i ih =>
    intro hi hkeep
    let ins := (compile p)[i]
    have hnode : (compile p)[i]? = some ins := Array.getElem?_eq_getElem hi
    have deps := List.all_eq_true.mp hcheck i (List.mem_range.mpr hi)
    simp [hkeep, hnode] at deps
    rw [allSrc i ins hnode, hdst i ins hnode hkeep]
    apply eval_congr
    intro j hj
    obtain ⟨hlt, hjkeep⟩ := deps j hj
    exact ih j hlt (Nat.lt_trans hlt hi) hjkeep

end Luminal
