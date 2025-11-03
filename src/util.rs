use crate::{
    ast::ResolvedVar,
    core::{ResolvedCall, SpecializedPrimitive},
};
use std::borrow::Cow;

pub const INTERNAL_SYMBOL_PREFIX: &str = "@";

/// Gets rid of internal symbol prefixes for printing.
/// This allows us to test parsing of desugared programs.
pub fn sanitize_internal_name(name: &str) -> Cow<'_, str> {
    if name.starts_with(INTERNAL_SYMBOL_PREFIX) {
        Cow::Owned(format!("_{}", &name[INTERNAL_SYMBOL_PREFIX.len()..]))
    } else {
        Cow::Borrowed(name)
    }
}

pub(crate) type BuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;
pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasher>;
pub(crate) type HashSet<K> = hashbrown::HashSet<K, BuildHasher>;
pub(crate) type HEntry<'a, A, B> = hashbrown::hash_map::Entry<'a, A, B, BuildHasher>;
pub type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasher>;
pub type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

/// Generates fresh symbols for internal use during typechecking and flattening.
/// These are guaranteed not to collide with the
/// user's symbols because they use a reserved prefix.
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

    pub fn reserved_prefix(&self) -> &str {
        &self.reserved_string
    }

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
