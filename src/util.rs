#![allow(unused)]

use std::fmt::Display;

#[allow(unused_imports)]
use crate::*;

pub(crate) type BuildHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;

pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, BuildHasher>;
pub(crate) type HashSet<K> = hashbrown::HashSet<K, BuildHasher>;

pub(crate) type IndexMap<K, V> = indexmap::IndexMap<K, V, BuildHasher>;
pub(crate) type IndexSet<K> = indexmap::IndexSet<K, BuildHasher>;

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
