import Egglog.Core
import Egglog.Verified

namespace Egglog

/-- One application in an exported graph. All IDs are class representatives. -/
structure ClassRow where
  op : Nat
  args : List Id
  result : Id
  deriving Repr, DecidableEq

abbrev Snapshot := List ClassRow

/-- Export the equations currently represented by the graph for optional checking. -/
def EGraph.snapshot (g : EGraph) : Snapshot :=
  g.nodes.toList.zipIdx.map fun (node, id) =>
    ⟨node.op, node.args.toList.map g.find, g.find id⟩

namespace Snapshot

/-- A semantic value for each class satisfies every constructor equation. -/
def consistent [DecidableEq α] (rows : Snapshot) (m : Model α) (value : Id → α) : Bool :=
  rows.all fun row => decide (value row.result = m.interp row.op (row.args.map value))

private def lookup? (rows : Snapshot) (op : Nat) (args : List Id) : Option Id :=
  (rows.find? (fun row => row.op == op && row.args == args)).map ClassRow.result

mutual
  /-- Resolve a ground term by matching constructor applications in the snapshot. -/
  def resolve? (rows : Snapshot) (p : Pat) : Option Id :=
    match p with
    | .var _ => none
    | .app op ps => (resolveArgs? rows ps).bind (lookup? rows op)

  def resolveArgs? (rows : Snapshot) (ps : List Pat) : Option (List Id) :=
    match ps with
    | [] => some []
    | p :: ps => do return (← resolve? rows p) :: (← resolveArgs? rows ps)
end

private theorem lookup?_sound {rows : Snapshot} {op : Nat} {args : List Id} {id : Id}
    {m : Model α} {value : Id → α} [DecidableEq α] (valid : consistent rows m value)
    (found : lookup? rows op args = some id) :
    m.interp op (args.map value) = value id := by
  obtain ⟨row, hrow, rfl⟩ := Option.map_eq_some_iff.mp found
  have parts := List.find?_some (p := fun row : ClassRow => row.op == op && row.args == args) hrow
  obtain ⟨hop, hargs⟩ := Bool.and_eq_true_iff.mp parts
  have sem := List.all_eq_true.mp valid row (List.mem_of_find?_eq_some hrow)
  simpa [eq_of_beq hop, eq_of_beq hargs] using (of_decide_eq_true sem).symm

section Soundness
variable {rows : Snapshot} {m : Model α} {value : Id → α} [DecidableEq α]
variable (env : Nat → α)

mutual
  theorem resolve?_sound (valid : consistent rows m value) (p : Pat) (id : Id) (found : resolve? rows p = some id) :
      p.eval m env = value id := by
    cases p with
    | var i => simp [resolve?] at found
    | app op ps =>
      obtain ⟨ids, hids, h⟩ := Option.bind_eq_some_iff.mp found
      exact (congrArg (m.interp op) (resolveArgs?_sound valid ps ids hids)).trans (lookup?_sound valid h)

  theorem resolveArgs?_sound (valid : consistent rows m value) (ps : List Pat) (ids : List Id)
      (found : resolveArgs? rows ps = some ids) :
      Pat.evalArgs ps m env = ids.map value := by
    cases ps with
    | nil => simp [resolveArgs?] at found; simp [← found, Pat.evalArgs]
    | cons p ps =>
      obtain ⟨id, hp, rest⟩ := Option.bind_eq_some_iff.mp found
      obtain ⟨is, hps, e⟩ := Option.bind_eq_some_iff.mp rest
      cases e
      change p.eval m env :: Pat.evalArgs ps m env = value id :: is.map value
      rw [resolve?_sound valid p id hp, resolveArgs?_sound valid ps is hps]
end

/-- A final graph is sufficient evidence for this ground query whenever a class
    interpretation passes the finite check. The theorem trusts neither the search
    procedure that produced the graph nor the procedure that proposed the values. -/
theorem query_sound (valid : consistent rows m value) {p q : Pat} {id : Id}
    (hp : resolve? rows p = some id) (hq : resolve? rows q = some id) :
    p.eval m env = q.eval m env :=
  (resolve?_sound env valid p id hp).trans (resolve?_sound env valid q id hq).symm

end Soundness

/-- Propose class values from ground constructors, revisiting unresolved rows
    until no more classes obtain a value. The checker independently validates the
    result, including every row that shares a class or participates in a cycle. -/
def inferValues (rows : Snapshot) (m : Model α) (fallback : α) : Id → α :=
  let step (known : Std.TreeMap Id α compare) (row : ClassRow) :=
    if known.contains row.result then known
    else match row.args.mapM known.get? with
      | none => known
      | some args => known.insert row.result (m.interp row.op args)
  let rec loop (fuel : Nat) (known : Std.TreeMap Id α compare) :=
    match fuel with
    | 0 => known
    | n+1 =>
      let oldSize := known.size
      let known := rows.foldl step known
      if known.size == oldSize then known else loop n known
  let known := loop rows.length {}
  fun id => known.getD id fallback

/-- Independently check a ground equality reported by the e-graph. -/
def checkEq [DecidableEq α] (rows : Snapshot) (m : Model α) (value : Id → α)
    (p q : Pat) : Bool :=
  consistent rows m value && match resolve? rows p, resolve? rows q with
    | some a, some b => a == b
    | _, _ => false

theorem checkEq_sound [DecidableEq α] {rows : Snapshot} {m : Model α}
    {value : Id → α} {p q : Pat} (h : checkEq rows m value p q) (env : Nat → α) :
    p.eval m env = q.eval m env := by
  obtain ⟨valid, common⟩ := Bool.and_eq_true_iff.mp h
  split at common
  · rename_i a b hp hq
    have eq : a = b := eq_of_beq common
    exact query_sound env valid (hp.trans (congrArg some eq)) hq
  · contradiction

end Snapshot
end Egglog
