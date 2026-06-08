//! End-to-end tests for container support in the term/proof encoding.

use egglog::EGraph;

/// Term-only: a `(Vec Math)` column should canonicalize its elements during
/// rebuilding. After unioning two Math terms, the two vecs `(vec-of A)` and
/// `(vec-of B)` canonicalize to the same vec, so the constructors holding them
/// become congruent.
#[test]
fn vec_rebuild_term_only() {
    let mut egraph = EGraph::new_with_term_encoding();
    egraph
        .parse_and_run_program(
            None,
            r#"
            (sort Math)
            (constructor A () Math)
            (constructor B () Math)
            (sort MathVec (Vec Math))
            (constructor Holds (MathVec) Math)
            (Holds (vec-of (A)))
            (Holds (vec-of (B)))
            (union (A) (B))
            (run 1)
            (check (= (Holds (vec-of (A))) (Holds (vec-of (B)))))
            "#,
        )
        .unwrap();
}

/// Term-only: a collapsing `(Set Math)` column. After unioning A and B, the
/// set `{A, B}` collapses to a singleton, matching `{B}` (a one-element set),
/// so the two holders become congruent.
#[test]
fn set_rebuild_collapse_term_only() {
    let mut egraph = EGraph::new_with_term_encoding();
    egraph
        .parse_and_run_program(
            None,
            r#"
            (sort Math)
            (constructor A () Math)
            (constructor B () Math)
            (sort MathSet (Set Math))
            (constructor Holds (MathSet) Math)
            (Holds (set-of (A) (B)))
            (Holds (set-of (B)))
            (union (A) (B))
            (run 1)
            (check (= (Holds (set-of (A) (B))) (Holds (set-of (B)))))
            "#,
        )
        .unwrap();
}

/// Term-only: a `(Map Math Math)` column canonicalizes keys and values.
#[test]
fn map_rebuild_term_only() {
    let mut egraph = EGraph::new_with_term_encoding();
    egraph
        .parse_and_run_program(
            None,
            r#"
            (sort Math)
            (constructor A () Math)
            (constructor B () Math)
            (sort MathMap (Map Math Math))
            (constructor Holds (MathMap) Math)
            (Holds (map-insert (map-empty) (A) (A)))
            (Holds (map-insert (map-empty) (B) (B)))
            (union (A) (B))
            (run 1)
            (check (= (Holds (map-insert (map-empty) (A) (A)))
                      (Holds (map-insert (map-empty) (B) (B)))))
            "#,
        )
        .unwrap();
}

/// Term-only: a `(MultiSet Math)` column canonicalizes elements.
#[test]
fn multiset_rebuild_term_only() {
    let mut egraph = EGraph::new_with_term_encoding();
    egraph
        .parse_and_run_program(
            None,
            r#"
            (sort Math)
            (constructor A () Math)
            (constructor B () Math)
            (sort MathMS (MultiSet Math))
            (constructor Holds (MathMS) Math)
            (Holds (multiset-of (A)))
            (Holds (multiset-of (B)))
            (union (A) (B))
            (run 1)
            (check (= (Holds (multiset-of (A))) (Holds (multiset-of (B)))))
            "#,
        )
        .unwrap();
}

/// Term-only: a nested `(Vec (Vec Math))` column. Canonicalizing the deep
/// element must propagate through the inner vec to the outer vec.
#[test]
fn nested_vec_rebuild_term_only() {
    let mut egraph = EGraph::new_with_term_encoding();
    egraph
        .parse_and_run_program(
            None,
            r#"
            (sort Math)
            (constructor A () Math)
            (constructor B () Math)
            (sort MathVec (Vec Math))
            (sort MathVecVec (Vec MathVec))
            (constructor Holds (MathVecVec) Math)
            (Holds (vec-of (vec-of (A))))
            (Holds (vec-of (vec-of (B))))
            (union (A) (B))
            (run 1)
            (check (= (Holds (vec-of (vec-of (A)))) (Holds (vec-of (vec-of (B))))))
            "#,
        )
        .unwrap();
}

/// Proof mode: a nested `(Vec (Vec Math))` column, with proof checking.
#[test]
fn nested_vec_rebuild_proof_mode() {
    let mut egraph = EGraph::new_with_proofs().with_proof_testing();
    egraph
        .parse_and_run_program(
            None,
            r#"
            (sort Math)
            (constructor A () Math)
            (constructor B () Math)
            (sort MathVec (Vec Math))
            (sort MathVecVec (Vec MathVec))
            (constructor Holds (MathVecVec) Math)
            (Holds (vec-of (vec-of (A))))
            (Holds (vec-of (vec-of (B))))
            (union (A) (B))
            (run 1)
            (check (= (Holds (vec-of (vec-of (A)))) (Holds (vec-of (vec-of (B))))))
            "#,
        )
        .unwrap();
}

/// Proof mode: a `(Pair Math Math)` column. The container rebuild produces a
/// `Congr` proof, and `with_proof_testing` extracts and checks the proof for
/// the `check`.
#[test]
fn pair_rebuild_proof_mode() {
    let mut egraph = EGraph::new_with_proofs().with_proof_testing();
    egraph
        .parse_and_run_program(
            None,
            r#"
            (sort Math)
            (constructor A () Math)
            (constructor B () Math)
            (sort MathPair (Pair Math Math))
            (constructor Holds (MathPair) Math)
            (Holds (pair (A) (A)))
            (Holds (pair (B) (B)))
            (union (A) (B))
            (run 1)
            (check (= (Holds (pair (A) (A))) (Holds (pair (B) (B)))))
            "#,
        )
        .unwrap();
}

/// Proof mode: a `(Vec Math)` column.
#[test]
fn vec_rebuild_proof_mode() {
    let mut egraph = EGraph::new_with_proofs().with_proof_testing();
    egraph
        .parse_and_run_program(
            None,
            r#"
            (sort Math)
            (constructor A () Math)
            (constructor B () Math)
            (sort MathVec (Vec Math))
            (constructor Holds (MathVec) Math)
            (Holds (vec-of (A)))
            (Holds (vec-of (B)))
            (union (A) (B))
            (run 1)
            (check (= (Holds (vec-of (A))) (Holds (vec-of (B)))))
            "#,
        )
        .unwrap();
}

/// Term-only: a `(Pair Math Math)` column should canonicalize both elements.
#[test]
fn pair_rebuild_term_only() {
    let mut egraph = EGraph::new_with_term_encoding();
    egraph
        .parse_and_run_program(
            None,
            r#"
            (sort Math)
            (constructor A () Math)
            (constructor B () Math)
            (sort MathPair (Pair Math Math))
            (constructor Holds (MathPair) Math)
            (Holds (pair (A) (A)))
            (Holds (pair (B) (B)))
            (union (A) (B))
            (run 1)
            (check (= (Holds (pair (A) (A))) (Holds (pair (B) (B)))))
            "#,
        )
        .unwrap();
}
