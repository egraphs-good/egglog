import Egglog.Examples.ParaBitPass

open Egglog ParaBit
namespace ParaBitPassTests

/-- The published Hydra opt_2750 statement. Width names are shared across its
    two roots and its premises. -/
private def hydra : BWLang.Problem :=
  let r := BWLang.Term.symbol "r"
  let s := BWLang.Term.symbol "s"
  let bw w x := BWLang.Term.app "bw" [w,x]
  let bor x y := BWLang.Term.app "or" [x,y]
  let x := bw r (.symbol "v0")
  let y := bw r (.symbol "newvar1")
  ⟨bw r (bw s (bor (bw s x) (bw s y))), bw r (bor x y),
    [.app "<" [r,s], .app ">" [s,r]], [r,s]⟩

/-- An independent open register-expression IR. The pass instantiates the two
    source words with an addition and a register, under an enclosing addition. -/
inductive IR where
  | reg (r : Nat)
  | low (x : IR)
  | wide (x : IR)
  | add (x y : IR)
  | bor (x y : IR)
  deriving BEq, Repr

namespace IR

def eval (env : Nat → Int) : IR → Int
  | .reg r => env r
  | .low x => trunc 1 (x.eval env)
  | .wide x => trunc 2 (x.eval env)
  | .add x y => x.eval env + y.eval env
  | .bor x y => bitOr (x.eval env) (y.eval env)

egg_codec encode, decode : IR where
  | [] .reg r <=> .var r
  | [x] .low x <=> Op.bw Op.one x
  | [x] .wide x <=> Op.bw Op.two x
  | [x,y] .add x y <=> Op.add x y
  | [x,y] .bor x y <=> Op.bor x y

abbrev model := scalar (fun _ => 0)

theorem encode_correct (src : IR) (env : Nat → Int) :
    src.encode.eval model env = src.eval env := by
  induction src <;> simp_all [encode, eval, Pat.eval, Pat.evalArgs, model, scalar, Op.model]

end IR

private def source (narrow wide : IR → IR) : IR :=
  let a := narrow (.add (.reg 91) (.reg 37))
  let b := narrow (.reg 52)
  .add (.reg 70) (narrow (wide (.bor (wide a) (wide b))))

private def shorter : IR :=
  .add (.reg 70) (.low (.bor (.low (.add (.reg 91) (.reg 37))) (.low (.reg 52))))

/-- Concrete width facts, independent of the artifact and input registers. -/
private def widths : Array (VerifiedRule IR.model) := #[
  egg_rule () : Op.lt Op.zero Op.one ==> Op.one by rfl,
  egg_rule () : Op.lt Op.zero Op.two ==> Op.one by rfl,
  egg_rule () : Op.lt Op.one Op.two ==> Op.one by rfl]

def optimize (rule : GuardedRule IR.model) (input : IR) :
    Option {out : IR // ∀ env, out.eval env = input.eval env} := do
  let (s, root) := (Certified.Session.empty IR.model).add input.encode
  let s := [Op.lt Op.zero Op.one, Op.lt Op.zero Op.two, Op.lt Op.one Op.two].foldl
    (fun graph p => (graph.add p).1) s
  let s := s.run (egg_schedule { run widths, guarded #[rule] })
  let term ← s.extractor.extract? root
  let out ← IR.decode term.val
  return ⟨out.val, fun env => by
    rw [← IR.encode_correct, out.property, term.property, IR.encode_correct]⟩

private def rewritten (p : IR) (staged : Bool := false) : Option IR := do
  let rule ← hydra.certifyRule? (staged := staged)
  return (← optimize rule p).val

#guard rewritten (source IR.low IR.wide) == some shorter
#guard rewritten (source IR.wide IR.low) == some (source IR.wide IR.low)
#guard rewritten (source IR.low IR.wide) (staged := true) == some shorter

end ParaBitPassTests
