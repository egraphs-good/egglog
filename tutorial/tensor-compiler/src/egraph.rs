use egglog::sort::{SetContainer, VecContainer};
use egglog::{
    sort::{SetSort, Sort, StringSort, VecSort},
    *,
};
use std::collections::BTreeSet;
use std::sync::Arc;

fn egglog_defn() -> String {
    include_str!("defn.egg").into()
}

fn egglog_main() -> String {
    include_str!("main.egg").into()
}

struct VecToSet;

fn new_egraph() -> EGraph{
    let mut egraph = EGraph::default();

    // Add definitions
    egraph
        .parse_and_run_program(Some("defn.egg".into()), &egglog_defn())
        .unwrap();

    // Register external functions
    let string_vec_sort =
        egraph.get_sort_by(|s: &Arc<VecSort>| s.element().name() == StringSort.name());
    let string_set_sort =
        egraph.get_sort_by(|s: &Arc<SetSort>| s.element().name() == StringSort.name());

    // TODO: add_primitives! is still very raw, requires the user to import a ton of stuff
    {
        use crate::sort::FromSort;
        use egglog::ast::Span;
        use egglog::ast::Symbol;
        use egglog::core_relations;
        use egglog::sort::IntoSort;
        add_primitive!(&mut egraph, "string-vec->set" = |s: @VecContainer<Value> (string_vec_sort)| -> @SetContainer<Value> (string_set_sort) { {
            let data = s.data.iter().cloned().collect::<BTreeSet<Value>>();
            SetContainer {data, do_rebuild: false }
            }
        });
        
    }
    // TODO: test string-vec->set

    // Add rewrite rules
    egraph
        .parse_and_run_program(Some("main.egg".into()), &egglog_main())
        .unwrap();
    
    egraph
}
