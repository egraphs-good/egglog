use crate::ast::FunctionSubtype;
use crate::termdag::{TermDag, TermId};
use crate::util::{HashMap, HashSet};
use crate::{ArcSort, EGraph, Value};

/// Root-directed extraction for proof terms.
///
/// Unlike the public extractor, this does not compute globally optimal costs
/// for the whole e-graph. It searches for any reconstructable term for the
/// requested root, ignoring `:unextractable` and hidden constructor flags, and
/// skips view tables so proof terms use their original constructor names.
struct RootExtractor {
    cache: HashMap<(Value, String), Option<TermId>>,
    active: HashSet<(Value, String)>,
}

impl RootExtractor {
    fn new() -> Self {
        Self {
            cache: Default::default(),
            active: Default::default(),
        }
    }

    fn extract(
        &mut self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: &ArcSort,
    ) -> Option<TermId> {
        let key = (value, sort.name().to_owned());
        if let Some(term) = self.cache.get(&key) {
            return *term;
        }
        if !self.active.insert(key.clone()) {
            return None;
        }

        let mut term = self.extract_exact(egraph, termdag, value, sort);
        if term.is_none() {
            let canonical = find_canonical(egraph, value, sort);
            if canonical != value {
                term = self.extract(egraph, termdag, canonical, sort);
            }
        }

        self.active.remove(&key);
        self.cache.insert(key, term);
        term
    }

    fn extract_exact(
        &mut self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: &ArcSort,
    ) -> Option<TermId> {
        if sort.is_container_sort() {
            self.extract_container(egraph, termdag, value, sort)
        } else if sort.is_eq_sort() {
            self.extract_eq(egraph, termdag, value, sort)
        } else {
            Some(sort.reconstruct_termdag_base(egraph.backend.base_values(), value, termdag))
        }
    }

    fn extract_container(
        &mut self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: &ArcSort,
    ) -> Option<TermId> {
        let elements = sort.inner_values(egraph.backend.container_values(), value);
        let mut ch_terms = Vec::with_capacity(elements.len());
        for (sort, value) in elements {
            ch_terms.push(self.extract(egraph, termdag, value, &sort)?);
        }
        Some(sort.reconstruct_termdag_container(
            egraph.backend.container_values(),
            value,
            termdag,
            ch_terms,
        ))
    }

    fn extract_eq(
        &mut self,
        egraph: &EGraph,
        termdag: &mut TermDag,
        value: Value,
        sort: &ArcSort,
    ) -> Option<TermId> {
        for func in egraph.functions.values() {
            if func.decl.subtype != FunctionSubtype::Constructor
                || func.extraction_output_sort().name() != sort.name()
                || func.decl.term_constructor.is_some()
            {
                continue;
            }

            let output_idx = func.extraction_output_index();
            let mut matching_rows = Vec::new();
            egraph
                .backend
                .for_each(func.backend_id, |row: egglog_bridge::ScanEntry| {
                    if !row.subsumed && row.vals[output_idx] == value {
                        matching_rows.push(row.vals.to_vec());
                    }
                });

            for row in matching_rows {
                let num_children = func.extraction_num_children();
                let mut ch_terms = Vec::with_capacity(num_children);
                let mut valid = true;
                for (value, sort) in row.iter().take(num_children).zip(func.schema.input.iter()) {
                    match self.extract(egraph, termdag, *value, sort) {
                        Some(term) => ch_terms.push(term),
                        None => {
                            valid = false;
                            break;
                        }
                    }
                }

                if valid {
                    return Some(termdag.app(func.extraction_term_name().to_string(), ch_terms));
                }
            }
        }

        None
    }
}

pub(crate) fn extract_root(
    egraph: &EGraph,
    termdag: &mut TermDag,
    value: Value,
    sort: ArcSort,
) -> Option<TermId> {
    RootExtractor::new().extract(egraph, termdag, value, &sort)
}

fn find_canonical(egraph: &EGraph, value: Value, sort: &ArcSort) -> Value {
    let Some(uf_name) = egraph.proof_state.uf_parent.get(sort.name()) else {
        return value;
    };

    let Some(uf_func) = egraph.functions.get(uf_name) else {
        return value;
    };

    let mut canonical = value;
    egraph
        .backend
        .for_each(uf_func.backend_id, |row: egglog_bridge::ScanEntry| {
            // The generated UF parent table is a normal custom function with
            // can_subsume=false, so there are no subsumed UF rows to filter.
            // This matches the public extractor's one-hop canonical scan.
            // UF table has (child, parent) as inputs.
            if row.vals[0] == value {
                canonical = row.vals[1];
            }
        });
    canonical
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_output(egraph: &EGraph, function_name: &str) -> Value {
        let func = egraph.functions.get(function_name).unwrap();
        let mut value = None;
        egraph.backend.for_each_while(func.backend_id, |row| {
            value = Some(row.vals[func.extraction_output_index()]);
            false
        });
        value.unwrap()
    }

    fn first_input(egraph: &EGraph, function_name: &str, input: usize) -> Value {
        let func = egraph.functions.get(function_name).unwrap();
        let mut value = None;
        egraph.backend.for_each_while(func.backend_id, |row| {
            value = Some(row.vals[input]);
            false
        });
        value.unwrap()
    }

    fn extract_to_string(egraph: &EGraph, value: Value, sort_name: &str) -> String {
        let mut termdag = TermDag::default();
        let term = extract_root(
            egraph,
            &mut termdag,
            value,
            egraph.get_sort_by_name(sort_name).unwrap().clone(),
        )
        .unwrap();
        termdag.to_string(term)
    }

    #[test]
    fn extracts_direct_constructor_root() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort Expr)
                (constructor Target () Expr)
                (Target)
                "#,
            )
            .unwrap();

        assert_eq!(
            extract_to_string(&egraph, first_output(&egraph, "Target"), "Expr"),
            "(Target)"
        );
    }

    #[test]
    fn canonicalizes_when_exact_root_is_subsumed() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort Expr :internal-uf UF_Expr)
                (function UF_Expr (Expr Expr) Unit :merge old :internal-hidden)
                (constructor Alias () Expr)
                (constructor Target () Expr)

                (let $alias (Alias))
                (let $target (Target))
                (set (UF_Expr $alias $target) ())
                (subsume (Alias))
                "#,
            )
            .unwrap();

        assert_eq!(
            extract_to_string(&egraph, first_output(&egraph, "Alias"), "Expr"),
            "(Target)"
        );
    }

    #[test]
    fn extracts_container_root() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort Expr)
                (sort ExprPair (Pair Expr Expr))
                (constructor Leaf () Expr)
                (constructor Box (ExprPair) Expr)
                (Box (pair (Leaf) (Leaf)))
                "#,
            )
            .unwrap();

        assert_eq!(
            extract_to_string(&egraph, first_input(&egraph, "Box", 0), "ExprPair"),
            "(pair (Leaf) (Leaf))"
        );
    }

    #[test]
    fn active_roots_break_cycles() {
        let mut egraph = EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                r#"
                (sort Expr)
                (constructor Target () Expr)
                (Target)
                "#,
            )
            .unwrap();

        let value = first_output(&egraph, "Target");
        let sort = egraph.get_sort_by_name("Expr").unwrap().clone();
        let mut extractor = RootExtractor::new();
        let mut termdag = TermDag::default();
        extractor.active.insert((value, sort.name().to_owned()));

        assert_eq!(extractor.extract(&egraph, &mut termdag, value, &sort), None);
    }
}
