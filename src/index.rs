//! Column-level indexes on values from a common sort.
use smallvec::SmallVec;
use symbol_table::GlobalSymbol;

use crate::{util::HashMap, Value};

pub(crate) struct ColumnIndex {
    sort: GlobalSymbol,
    ids: HashMap<u64, SmallVec<[usize; 3]>>,
}

impl ColumnIndex {
    pub(crate) fn new(sort: GlobalSymbol) -> ColumnIndex {
        ColumnIndex {
            sort,
            ids: Default::default(),
        }
    }
    pub(crate) fn add(&mut self, v: Value, i: usize) {
        assert_eq!(v.tag, self.sort);
        self.ids.entry(v.bits).or_default().push(i);
    }

    pub(crate) fn get_indexes(&self, v: &Value) -> &[usize] {
        assert_eq!(v.tag, self.sort);
        self.ids.get(&v.bits).map(|x| x.as_slice()).unwrap_or(&[])
    }
}
