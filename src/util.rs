#![allow(unused)]

use crate::core::SpecializedPrimitive;
#[allow(unused_imports)]
use crate::*;

pub(crate) type BuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;
pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasher>;
pub(crate) type HashSet<K> = hashbrown::HashSet<K, BuildHasher>;
pub(crate) type HEntry<'a, A, B> = hashbrown::hash_map::Entry<'a, A, B, BuildHasher>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasher>;
pub type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

/// Generates fresh symbols for internal use during typechecking and flattening.
/// These are guaranteed not to collide with the
/// user's symbols because they use $.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolGen {
    count: usize,
    reserved_string: String,
}

impl SymbolGen {
    pub fn new(reserved_string: String) -> Self {
        Self {
            count: 0,
            reserved_string,
        }
    }

    pub fn has_been_used(&self) -> bool {
        self.count > 0
    }
}

/// This trait lets us statically dispatch between `fresh` methods for generic structs.
pub trait FreshGen<Head: ?Sized, Leaf> {
    fn fresh(&mut self, name_hint: &Head) -> Leaf;
}

impl FreshGen<str, String> for SymbolGen {
    fn fresh(&mut self, name_hint: &str) -> String {
        let s = format!("{}{}{}", self.reserved_string, name_hint, self.count);
        self.count += 1;
        s
    }
}

impl FreshGen<String, String> for SymbolGen {
    fn fresh(&mut self, name_hint: &String) -> String {
        self.fresh(name_hint.as_str())
    }
}

impl FreshGen<ResolvedCall, ResolvedVar> for SymbolGen {
    fn fresh(&mut self, name_hint: &ResolvedCall) -> ResolvedVar {
        let name = format!("{}{}{}", self.reserved_string, name_hint, self.count);
        self.count += 1;
        let sort = match name_hint {
            ResolvedCall::Func(f) => f.output.clone(),
            ResolvedCall::Primitive(SpecializedPrimitive { output, .. }) => output.clone(),
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
