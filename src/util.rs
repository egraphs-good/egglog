use crate::{ast::ResolvedVar, core::ResolvedCall};

pub(crate) type BuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;
pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasher>;
pub(crate) type HashSet<K> = hashbrown::HashSet<K, BuildHasher>;
pub(crate) type HEntry<'a, A, B> = hashbrown::hash_map::Entry<'a, A, B, BuildHasher>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasher>;
pub type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

pub use egglog_ast::generic_ast_helpers::INTERNAL_SYMBOL_PREFIX;

/// Generates fresh symbols for internal use during typechecking and flattening.
/// Symbols have the form `<reserved prefix><hint>_<id>`, with one monotonically
/// increasing numeric ID shared across all hints and input types. The numeric
/// suffix is always present, including for ID zero. The final underscore makes
/// names unique even when hints contain underscores or end in digits.
/// A reserved prefix prevents collisions with user symbols.
///
/// Generating a symbol after the counter is exhausted panics rather than reusing
/// an earlier ID.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolGen {
    next_id: usize,
    reserved_string: String,
}

impl SymbolGen {
    /// Create a new symbol generator with the given reserved prefix.
    pub fn new(reserved_string: String) -> Self {
        Self {
            next_id: 0,
            reserved_string,
        }
    }

    /// Check if this symbol generator has been used to generate any symbols.
    pub fn has_been_used(&self) -> bool {
        self.next_id != 0
    }

    /// Get the reserved prefix used by this symbol generator.
    pub fn reserved_prefix(&self) -> &str {
        &self.reserved_string
    }

    /// Check if the given symbol is reserved (i.e., starts with the reserved prefix).
    pub fn is_reserved(&self, symbol: &str) -> bool {
        !self.reserved_string.is_empty() && symbol.starts_with(&self.reserved_string)
    }
}

/// This trait lets us statically dispatch between `fresh` methods for generic structs.
pub trait FreshGen<Head: ?Sized, Leaf> {
    fn fresh(&mut self, name_hint: &Head) -> Leaf;
}

impl FreshGen<str, String> for SymbolGen {
    fn fresh(&mut self, name_hint: &str) -> String {
        let id = self.next_id;
        self.next_id = id.checked_add(1).expect("fresh symbol counter exhausted");
        format!("{}{name_hint}_{id}", self.reserved_string)
    }
}

impl FreshGen<String, String> for SymbolGen {
    fn fresh(&mut self, name_hint: &String) -> String {
        self.fresh(name_hint.as_str())
    }
}

impl FreshGen<ResolvedCall, ResolvedVar> for SymbolGen {
    fn fresh(&mut self, name_hint: &ResolvedCall) -> ResolvedVar {
        ResolvedVar {
            name: self.fresh(name_hint.name()),
            sort: name_hint.output().clone(),
            // fresh variables are never global references, since globals
            // are desugared away by `remove_globals`
            is_global_ref: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Context, EGraph,
        ast::{FunctionSubtype, Span},
        prelude::BaseSort,
        sort::I64Sort,
        typechecking::FuncType,
    };
    use std::{panic::AssertUnwindSafe, sync::Arc};

    #[test]
    fn symbol_gen_is_unique_across_hints_and_input_types() {
        let mut generator = SymbolGen::new("@".into());
        assert!(!generator.has_been_used());
        assert_eq!(generator.reserved_prefix(), "@");
        assert!(generator.is_reserved("@anything"));
        assert!(!generator.is_reserved("anything"));
        assert!(!SymbolGen::new(String::new()).is_reserved("anything"));

        let mut names = HashSet::default();
        for hint in ["", "_", "__", "x", "x1", "x11", "x_0", "x_1", "_0_", "λ_12"] {
            let owned_hint = hint.to_owned();
            let call = ResolvedCall::Func(Arc::new(FuncType {
                name: owned_hint.clone(),
                subtype: FunctionSubtype::Custom,
                input: vec![],
                output: I64Sort.to_arcsort(),
            }));
            for _ in 0..3 {
                let from_str = generator.fresh(hint);
                let from_string = generator.fresh(&owned_hint);
                let from_call = generator.fresh(&call);
                assert_eq!(from_call.sort.name(), I64Sort.name());
                assert!(!from_call.is_global_ref);
                for name in [from_str, from_string, from_call.name] {
                    let (_, id) = name.rsplit_once('_').unwrap();
                    assert_eq!(id.parse::<usize>().unwrap(), names.len());
                    assert!(names.insert(name));
                }
            }
        }
        let egraph = EGraph::default();
        let primitive = ResolvedCall::from_resolution(
            "+",
            &[
                I64Sort.to_arcsort(),
                I64Sort.to_arcsort(),
                I64Sort.to_arcsort(),
            ],
            &egraph.type_info,
            Context::Pure,
            &Span::Panic,
        )
        .unwrap();
        let from_primitive = generator.fresh(&primitive);
        assert_eq!(from_primitive.name, format!("@+_{}", names.len()));
        assert_eq!(from_primitive.sort.name(), I64Sort.name());
        assert!(!from_primitive.is_global_ref);
        assert!(generator.has_been_used());
    }

    #[test]
    fn symbol_gen_exhaustion_does_not_wrap_or_reuse_names() {
        let mut generator = SymbolGen::new("@".into());
        generator.next_id = usize::MAX - 1;
        assert_eq!(generator.fresh("x"), format!("@x_{}", usize::MAX - 1));
        for _ in 0..2 {
            assert!(std::panic::catch_unwind(AssertUnwindSafe(|| generator.fresh("x"))).is_err());
            assert_eq!(generator.next_id, usize::MAX);
        }
    }
}
