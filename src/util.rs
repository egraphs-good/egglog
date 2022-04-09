#![allow(unused)]

use std::fmt::Display;

#[allow(unused_imports)]
use crate::*;

pub(crate) type BuildHasher = ahash::RandomState;
pub(crate) const BUILD_HASHER: BuildHasher = BuildHasher::with_seeds(0, 0, 0, 0);

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

pub(crate) struct ListDisplay<'a, T>(pub &'a [T]);

impl<'a, T> Display for ListDisplay<'a, T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut did_something = false;
        for item in self.0 {
            if did_something {
                write!(f, ", ")?;
            }
            Display::fmt(&item, f)?;
            did_something = true;
        }
        Ok(())
    }
}
