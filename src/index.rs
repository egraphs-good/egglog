//! Column-level indexes on values from a common sort.
use smallvec::SmallVec;
use symbol_table::GlobalSymbol;

use crate::{unionfind::UnionFind, util::HashMap, Value};

#[derive(Clone)]
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

    pub(crate) fn clear(&mut self) {
        self.ids.clear()
    }

    fn get_indexes_for_bits(&self, bits: u64) -> &[usize] {
        self.ids.get(&bits).map(|x| x.as_slice()).unwrap_or(&[])
    }

    pub(crate) fn to_canonicalize<'a>(
        &'a self,
        uf: &'a UnionFind,
    ) -> impl Iterator<Item = usize> + '_ {
        uf.dirty_ids(self.sort).flat_map(|x| {
            self.get_indexes_for_bits(usize::from(x) as u64)
                .iter()
                .copied()
        })
    }
}
