import Egglog.Relations

open Egglog

#guard !(Relational.runSource "(relation Local ()) (datatype* (First (A)) (Local (B)))").isOk

-- Private fields can be copied from a matching row once the shared key is
-- fixed. Duplicate input rows and nullary atoms retain set semantics.
private def payloadMatches := GenericJoin.search #[
  .prepare #[0,1,1] #[#[2,7,7], #[2,7,7], #[2,8,8], #[2,9,0]],
  .prepare #[0,2] #[#[2,3], #[2,4], #[1,0]], .prepare #[] #[#[], #[]]]
#guard payloadMatches.size == 4
#guard (#[7,8] : Array Nat).all fun x =>
  (#[3,4] : Array Nat).all fun y => payloadMatches.contains #[some 2, some x, some y]

private def signature :=
  "(datatype E (Zero) (One) (Succ E) (Add E E) (Mul E E))\n"

private def checked (src : String) :=
  runCheckedSource src Arithmetic.signature Arithmetic.rules

-- Alpha-renaming a pattern preserves its certificate; editing either term does not.
#guard (checked (signature ++ "(rewrite (Add renamed (Zero)) renamed)")).isOk
#guard !(checked (signature ++ "(rewrite (Add renamed (Zero)) (One))")).isOk
#guard !(checked (signature ++ "(union (One) (Zero))")).isOk
#guard !(checked (signature ++ "(+ 1 1)")).isOk
#guard !(checked (signature ++ "(extract 3)")).isOk
#guard !(checked (signature ++ "(fail (check (Unknown)))")).isOk
#guard !(checked (signature ++ "(fail (check (Zero) (Unknown)))")).isOk
#guard !(checked "(datatype E (One) (Zero) (Succ E) (Add E E) (Mul E E))").isOk
#guard !(checked "(datatype E (Zero) (One) (Succ i64) (Add E E) (Mul E E))").isOk

-- Rule guards see the round's bindings. Skipped overflowing heads must not run.
#guard (runSource "
  (datatype E (N i64) (Inc E))
  (ruleset increment)
  (rewrite (Inc (N i)) (N (+ i 1)) :ruleset increment
           :when ((< i 9223372036854775807)))
  (let $big (Inc (N 9223372036854775807)))
  (let $small (Inc (N 3)))
  (run 1)
  (fail (check (= $small (N 4))))
  (run-schedule (repeat 2 (run increment)))
  (check (= $small (N 4)))
  (fail (check (= $big (N -9223372036854775808))))").isOk

#guard !(runSource "
  (datatype E (N i64) (Inc E))
  (rewrite (Inc (N i)) (N (+ i 1)))
  (Inc (N 9223372036854775807))
  (run 1)").isOk

-- An already true :until condition skips even the first iteration.
#guard (runSource "
  (datatype E (A) (B))
  (let $a (A))
  (rewrite (A) (B))
  (run 10 :until (= $a (A)))
  (fail (check (B)))
  (run-schedule (saturate (run :until (= $a (B)))))
  (check (= $a (B)))").isOk

#guard (runSource "(check (= (+ 2 2) 4) (!= 1 2))").isOk

-- Each anonymous variable matches independently.
#guard (runSource "
  (datatype E (F E E) (A) (B))
  (rewrite (F _ _) (A))
  (let $term (F (A) (B)))
  (run 1)
  (check (= $term (A)))").isOk

#guard !(runSource "(datatype E (E))").isOk
#guard !(runSource "(datatype E (A)) (let 3 (A))").isOk
#guard (runSource "(check (< -2 1) (<= 0 0) (> 4 3) (>= 8 7))").isOk

-- A partial primitive in the body may bind a head variable. Overflow makes the
-- match fail before the head runs.
#guard (runSource "
  (datatype E (N i64) (Inc E))
  (rewrite (Inc (N i)) (N j) :when ((= j (+ i 1))))
  (let $low (Inc (N 2)))
  (let $high (Inc (N 9223372036854775807)))
  (run 1)
  (check (= $low (N 3)))
  (fail (check (= $high (N 3))))").isOk
