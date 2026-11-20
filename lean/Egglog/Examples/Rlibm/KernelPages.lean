import Lean.Elab.Command

namespace Rlibm.KernelPages

/-- Check a power-of-two range without constructing a list of all words. -/
def checkSpan (test : Nat → Bool) (start : Nat) : Nat → Bool
  | 0 => test start
  | d+1 => checkSpan test start d && checkSpan test (start + 2^d) d

theorem span_sound {test start input} (d : Nat)
    (h : checkSpan test start d = true) (lo : start ≤ input) (hi : input < start + 2^d) :
    test input = true := by
  induction d generalizing start with
  | zero =>
    have eq : input = start := by
      simp only [Nat.pow_zero] at hi
      omega
    subst input
    exact h
  | succ d ih =>
    simp only [checkSpan, Bool.and_eq_true] at h
    rw [Nat.pow_succ] at hi
    by_cases cut : input < start + 2^d
    · exact ih h.1 lo cut
    · exact ih h.2 (by omega) (by omega)

theorem joinSpan {test s d} (l : checkSpan test s d = true)
    (r : checkSpan test (s + 2^d) d = true) : checkSpan test s (d+1) = true := by
  simp [checkSpan, l, r]

open Lean Elab Command in
/-- Certify an executable Boolean predicate for every bfloat16 input encoding.
    Each opaque leaf checks 128 encodings. This bounds the kernel's reduction
    cache without trusting native execution or generated tables. -/
elab "certify_bfloat16 " test:term " as " root:ident : command => do
  let rec emit (s d : Nat) (decl : Ident) : CommandElabM Unit := do
    let start := quote s
    let depth := quote d
    match d with
    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 =>
      elabCommand (← `(private theorem $decl : checkSpan $test $start $depth = true := by decide +kernel))
    | childDepth + 1 =>
      let name := fun offset => mkIdent (root.getId.appendAfter s!"_{childDepth}_{offset}")
      let halfway := s + 2^childDepth
      let l := name s
      let r := name halfway
      emit s childDepth l
      emit halfway childDepth r
      elabCommand (← `(private theorem $decl : checkSpan $test $start $depth = true := joinSpan $l $r))
  let all := mkIdent (root.getId.appendAfter "_all")
  emit 0 16 all
  elabCommand (← `(private theorem $root (x : Fin 65536) : ($test : Nat → Bool) x.val = true :=
    span_sound (start := 0) 16 $all (Nat.zero_le _) x.isLt))

end Rlibm.KernelPages
