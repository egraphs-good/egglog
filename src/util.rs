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
pub(crate) trait FreshGen<Head, Leaf> {
    fn fresh(&mut self, name_hint: &Head) -> Leaf;
}

pub(crate) struct SymbolGen {
    gen: usize,
}

impl SymbolGen {
    pub(crate) fn new() -> Self {
        Self { gen: 0 }
    }
}

impl FreshGen<Symbol, Symbol> for SymbolGen {
    fn fresh(&mut self, name_hint: &Symbol) -> Symbol {
        let s = format!("__{}{}", name_hint, self.gen);
        self.gen += 1;
        Symbol::from(s)
    }
}

pub(crate) struct ResolvedGen {
    gen: usize,
}

impl ResolvedGen {
    pub(crate) fn new() -> Self {
        Self { gen: 0 }
    }
}

impl FreshGen<ResolvedCall, ResolvedVar> for ResolvedGen {
    fn fresh(&mut self, name_hint: &ResolvedCall) -> ResolvedVar {
        let s = format!("__{}{}", name_hint, self.gen);
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
