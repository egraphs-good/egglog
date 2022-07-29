use std::collections::HashMap;

use symbol_table::*;

#[derive(Debug, Clone)]
pub struct IdxName(Symbol); // todo

// 0 represents a null pointer
#[derive(Debug, Clone, Copy, Eq, Hash)]
pub struct DenseValue(u32);

impl DenseValue {
    pub fn null() -> Self {
        DenseValue(0)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Trie(pub HashMap<DenseValue, Self>);

impl Trie {
    pub fn len(&self) -> usize {
        self.0.len()
    }
}
