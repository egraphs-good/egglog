//! Column-level indexes on values from a common sort.
use smallvec::SmallVec;
use symbol_table::GlobalSymbol;

use crate::{unionfind::UnionFind, util::HashMap, Value};

pub(crate) type Offset = u32;

#[derive(Clone, Debug)]
pub(crate) struct ColumnIndex {
    sort: GlobalSymbol,
    ids: HashMap<u64, SmallVec<[Offset; 8]>>,
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
        self.ids.entry(v.bits).or_default().push(i as Offset);
    }

    pub(crate) fn clear(&mut self) {
        self.ids.clear()
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn get(&self, v: &Value) -> Option<&[Offset]> {
        self.get_indexes_for_bits(v.bits)
    }

    fn get_indexes_for_bits(&self, bits: u64) -> Option<&[Offset]> {
        self.ids.get(&bits).map(|x| x.as_slice())
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (Value, &[Offset])> + '_ {
        self.ids.iter().map(|(bits, v)| {
            (
                Value {
                    tag: self.sort,
                    bits: *bits,
                },
                v.as_slice(),
            )
        })
    }

    pub(crate) fn to_canonicalize<'a>(
        &'a self,
        uf: &'a UnionFind,
    ) -> impl Iterator<Item = usize> + '_ {
        uf.dirty_ids(self.sort).flat_map(|x| {
            self.get_indexes_for_bits(usize::from(x) as u64)
                .unwrap_or(&[])
                .iter()
                .copied()
                .map(|x| x as usize)
        })
    }
}
