use crate::{ast::ResolvedVar, core::ResolvedCall};

pub(crate) type BuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;
pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasher>;
pub(crate) type HashSet<K> = hashbrown::HashSet<K, BuildHasher>;
pub(crate) type HEntry<'a, A, B> = hashbrown::hash_map::Entry<'a, A, B, BuildHasher>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasher>;
pub type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

pub use egglog_ast::generic_ast_helpers::INTERNAL_SYMBOL_PREFIX;

/// Generates fresh symbols for internal use during typechecking and flattening.
/// These are guaranteed not to collide with the
/// user's symbols because they use a reserved prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolGen {
    hint_to_count: HashMap<String, usize>,
    used: HashSet<String>,
    reserved_string: String,
    leave_off_zero: bool,
}

impl SymbolGen {
    /// Create a new symbol generator with the given reserved prefix.
    pub fn new(reserved_string: String) -> Self {
        Self {
            hint_to_count: HashMap::default(),
            used: HashSet::default(),
            reserved_string,
            leave_off_zero: true,
        }
    }

    /// By default, the first symbol generated with a given hint
    /// does not have a numeric suffix (e.g., "var" instead of "var0").
    /// This method changes that behavior.
    pub fn include_zero(&mut self, include: bool) {
        self.leave_off_zero = !include;
    }

    /// Check if this symbol generator has been used to generate any symbols.
    pub fn has_been_used(&self) -> bool {
        !self.hint_to_count.is_empty()
    }

    /// Get the reserved prefix used by this symbol generator.
    pub fn reserved_prefix(&self) -> &str {
        &self.reserved_string
    }

    /// Check if the given symbol is reserved (i.e., starts with the reserved prefix).
    pub fn is_reserved(&self, symbol: &str) -> bool {
        !self.reserved_string.is_empty() && symbol.starts_with(&self.reserved_string)
    }

    /// Prevent future generated symbols from using this exact name.
    /// Returns whether the name was newly reserved.
    pub fn reserve(&mut self, symbol: &str) -> bool {
        self.used.insert(symbol.to_owned())
    }
}

/// This trait lets us statically dispatch between `fresh` methods for generic structs.
pub trait FreshGen<Head: ?Sized, Leaf> {
    fn fresh(&mut self, name_hint: &Head) -> Leaf;
}

impl FreshGen<str, String> for SymbolGen {
    fn fresh(&mut self, name_hint: &str) -> String {
        let entry = self.hint_to_count.entry(name_hint.to_string()).or_insert(0);
        loop {
            let count_before = *entry;
            *entry += 1;
            let name = format!(
                "{}{}{}",
                self.reserved_string,
                name_hint,
                if self.leave_off_zero && count_before == 0 {
                    "".to_string()
                } else {
                    count_before.to_string()
                }
            );
            if self.used.insert(name.clone()) {
                return name;
            }
        }
    }
}

impl FreshGen<String, String> for SymbolGen {
    fn fresh(&mut self, name_hint: &String) -> String {
        self.fresh(name_hint.as_str())
    }
}

impl FreshGen<ResolvedCall, ResolvedVar> for SymbolGen {
    fn fresh(&mut self, name_hint: &ResolvedCall) -> ResolvedVar {
        let name = self.fresh(&name_hint.to_string());
        let sort = match name_hint {
            ResolvedCall::Func(f) => f.output.clone(),
            ResolvedCall::Primitive(prim) => prim.output().clone(),
        };
        ResolvedVar {
            name,
            sort,
            // fresh variables are never global references, since globals
            // are desugared away by `remove_globals`
            is_global_ref: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_symbols_skip_emitted_and_reserved_names_across_hints() {
        let mut symbols = SymbolGen::new("@".into());
        assert_eq!(symbols.fresh("Call"), "@Call");
        assert_eq!(symbols.fresh("Call"), "@Call1");
        assert_eq!(symbols.fresh("Call"), "@Call2");
        assert_eq!(symbols.fresh("Call2"), "@Call21");
        assert!(symbols.reserve("@Other"));
        assert!(!symbols.reserve("@Other"));
        assert_eq!(symbols.fresh("Other"), "@Other1");
        let mut cloned = symbols.clone();
        assert_eq!(symbols.fresh("Call2"), cloned.fresh("Call2"));
        symbols.include_zero(true);
        assert_eq!(symbols.fresh("Zero"), "@Zero0");
    }

    #[test]
    fn resolved_and_string_symbols_share_reservations() {
        let call = ResolvedCall::Func(std::sync::Arc::new(crate::typechecking::FuncType {
            name: "Call".into(),
            subtype: crate::ast::FunctionSubtype::Constructor,
            is_relation: false,
            input: vec![],
            output: std::sync::Arc::new(crate::sort::EqSort {
                name: "Output".into(),
            }),
        }));
        let mut symbols = SymbolGen::new("@".into());
        assert_eq!(symbols.fresh("Call2"), "@Call2");
        assert_eq!(symbols.fresh(&call).name, "@Call");
        assert_eq!(symbols.fresh(&call).name, "@Call1");
        assert_eq!(symbols.fresh(&call).name, "@Call3");
        symbols.reserve("@Call4");
        let variable = symbols.fresh(&call);
        assert_eq!(variable.name, "@Call5");
        assert_eq!(variable.sort.name(), "Output");
        assert!(!variable.is_global_ref);
    }
}
