#![allow(unused)]

use std::fmt::Display;

use crate::core::SpecializedPrimitive;
#[allow(unused_imports)]
use crate::*;

pub(crate) type BuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;

pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasher>;
pub(crate) type HashSet<K> = hashbrown::HashSet<K, BuildHasher>;

pub type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasher>;
pub type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

pub(crate) fn concat_vecs<T>(to: &mut Vec<T>, mut from: Vec<T>) {
    if to.len() < from.len() {
        std::mem::swap(to, &mut from)
    }
    to.extend(from);
}

pub(crate) struct ListDisplay<'a, TS>(pub TS, pub &'a str);

impl<'a, TS> Display for ListDisplay<'a, TS>
where
    TS: Clone + IntoIterator,
    TS::Item: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut did_something = false;
        for item in self.0.clone().into_iter() {
            if did_something {
                f.write_str(self.1)?;
            }
            Display::fmt(&item, f)?;
            did_something = true;
        }
        Ok(())
    }
}

pub(crate) struct ListDebug<'a, TS>(pub TS, pub &'a str);

impl<'a, TS> Debug for ListDebug<'a, TS>
where
    TS: Clone + IntoIterator,
    TS::Item: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut did_something = false;
        for item in self.0.clone().into_iter() {
            if did_something {
                f.write_str(self.1)?;
            }
            Debug::fmt(&item, f)?;
            did_something = true;
        }
        Ok(())
    }
}

/// Generates fresh symbols for internal use during typechecking and flattening.
/// These are guaranteed not to collide with the
/// user's symbols because they use $.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolGen {
    gen: usize,
    reserved_string: String,
    special_reserved: HashMap<Symbol, String>,
}

impl SymbolGen {
    pub fn new(reserved_string: String) -> Self {
        Self {
            gen: 0,
            reserved_string,
            special_reserved: HashMap::default(),
        }
    }

    pub fn has_been_used(&self) -> bool {
        self.gen > 0
    }
}

/// This trait lets us statically dispatch between `fresh` methods for generic structs.
pub trait FreshGen<Head, Leaf> {
    fn fresh(&mut self, name_hint: &Head) -> Leaf;
}

impl FreshGen<Symbol, Symbol> for SymbolGen {
    fn fresh(&mut self, name_hint: &Symbol) -> Symbol {
        let s = format!("{}{}{}", self.reserved_string, name_hint, self.gen);
        self.gen += 1;
        Symbol::from(s)
    }
}

impl FreshGen<ResolvedCall, ResolvedVar> for SymbolGen {
    fn fresh(&mut self, name_hint: &ResolvedCall) -> ResolvedVar {
        let s = format!("{}{}{}", self.reserved_string, name_hint, self.gen);
        self.gen += 1;
        let sort = match name_hint {
            ResolvedCall::Func(f) => f.output.clone(),
            ResolvedCall::Primitive(SpecializedPrimitive { output, .. }) => output.clone(),
        };
        ResolvedVar {
            name: s.into(),
            sort,
            // fresh variables are never global references, since globals
            // are desugared away by `remove_globals`
            is_global_ref: false,
        }
    }
}

// This is a convenient for `for<'a> impl Into<Symbol> for &'a T`
pub(crate) trait SymbolLike {
    fn to_symbol(&self) -> Symbol;
}

impl SymbolLike for Symbol {
    fn to_symbol(&self) -> Symbol {
        *self
    }
}
