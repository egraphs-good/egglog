use smallvec::SmallVec;
use symbol_table::GlobalSymbol;

use crate::{ast::Id, util::IndexMap, Value};

pub(crate) struct ColumnIndex {
    sort: GlobalSymbol,
    ids: IndexMap<u64, SmallVec<[usize; 4]>>,
}

impl ColumnIndex {
    pub(crate) fn add(&mut self, v: Value, i: usize) {
        assert_eq!(v.tag, self.sort);
        self.ids.entry(v.bits).or_default().push(i);
    }
}
