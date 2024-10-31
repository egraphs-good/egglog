//! Column-level indexes on values from a common sort.
use smallvec::SmallVec;

use crate::{unionfind::UnionFind, util::HashMap, Value};

pub(crate) type Offset = u32;

#[derive(Clone, Debug)]
pub(crate) struct ColumnIndex {
    sort: String,
    ids: HashMap<u64, SmallVec<[Offset; 8]>>,
}

impl ColumnIndex {
    pub(crate) fn new(sort: String) -> ColumnIndex {
        ColumnIndex {
            sort,
            ids: Default::default(),
        }
    }

    pub(crate) fn sort(&self) -> String {
        self.sort.clone()
    }

    pub(crate) fn add(&mut self, v: Value, i: usize) {
        #[cfg(debug_assertions)]
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
                    #[cfg(debug_assertions)]
                    tag: self.sort.clone(),
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
        uf.dirty_ids(self.sort.clone()).flat_map(|x| {
            self.get_indexes_for_bits(x)
                .unwrap_or(&[])
                .iter()
                .copied()
                .map(|x| x as usize)
        })
    }
}
#[derive(Clone, Debug)]
pub(crate) struct CompositeColumnIndex(SmallVec<[ColumnIndex; 2]>);

impl CompositeColumnIndex {
    pub(crate) fn new() -> CompositeColumnIndex {
        CompositeColumnIndex(SmallVec::new())
    }

    pub(crate) fn add(&mut self, s: String, v: Value, i: usize) {
        if let Some(index) = self.0.iter().position(|index| index.sort() == s) {
            (self.0)[index].add(v, i);
        } else {
            let mut index = ColumnIndex::new(s);
            index.add(v, i);
            self.0.push(index);
        }
    }

    pub(crate) fn clear(&mut self) {
        for index in self.0.iter_mut() {
            index.clear();
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &ColumnIndex> {
        self.0.iter()
    }
}
