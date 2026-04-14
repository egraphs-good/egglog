use std::{cmp, cmp::Ordering, fmt, mem};

use crate::numeric_id::{NumericId, define_id};

use crate::{
    Pool,
    pool::{Clear, Pooled, with_pool_set},
};

define_id!(pub RowId, u32, "a numeric offset into a table");

#[cfg(test)]
mod tests;

/// A trait for types that represent a sequence of sorted offsets into a table.
///
/// NB: this trait may have outlived its usefulness. We may want to just get rid
/// of it.
pub(crate) trait Offsets {
    // A half-open range enclosing the offsets in this sequence.
    fn bounds(&self) -> (RowId, RowId);
    fn is_empty(&self) -> bool {
        let (lo, hi) = self.bounds();
        lo == hi
    }
    fn offsets(&self, f: impl FnMut(RowId));
}

#[derive(PartialEq, Eq, Debug, Clone, Copy, Hash)]
pub struct OffsetRange {
    pub(crate) start: RowId,
    pub(crate) end: RowId,
}

impl Offsets for OffsetRange {
    fn bounds(&self) -> (RowId, RowId) {
        (self.start, self.end)
    }

    fn offsets(&self, f: impl FnMut(RowId)) {
        RowId::range(self.start, self.end).for_each(f)
    }
}

impl OffsetRange {
    pub fn new(start: RowId, end: RowId) -> OffsetRange {
        assert!(
            start <= end,
            "attempting to create malformed range {start:?}..{end:?}"
        );
        OffsetRange { start, end }
    }
    pub(crate) fn size(&self) -> usize {
        self.end.index() - self.start.index()
    }
}

#[derive(Default, Clone, PartialEq, Eq, Debug, Hash)]
pub struct SortedOffsetVector(Vec<RowId>);

impl SortedOffsetVector {
    pub(crate) fn slice(&self) -> &SortedOffsetSlice {
        // SAFETY: self.0 is sorted.
        unsafe { SortedOffsetSlice::new_unchecked(&self.0) }
    }

    pub(crate) fn push(&mut self, offset: RowId) {
        assert!(self.0.last().is_none_or(|last| last <= &offset));
        // SAFETY: we just checked the invariant
        unsafe { self.push_unchecked(offset) }
    }

    pub(crate) unsafe fn push_unchecked(&mut self, offset: RowId) {
        self.0.push(offset)
    }

    pub(crate) fn retain(&mut self, mut f: impl FnMut(RowId) -> bool) {
        self.0.retain(|off| f(*off))
    }

    pub(crate) fn extend_nonoverlapping(&mut self, other: &SortedOffsetSlice) {
        if other.inner().is_empty() {
            return;
        }
        if self.0.is_empty() {
            self.0.extend(other.iter());
            return;
        }
        if self.0.last().unwrap() <= other.inner().first().unwrap() {
            self.0.extend(other.iter());
            return;
        }
        panic!("attempting to extend with overlapping offsets")
    }

    /// Overwrite the contents of the current vector with those of the offset range.
    pub(crate) fn fill_from_dense(&mut self, range: &OffsetRange) {
        self.0.clear();
        self.0
            .extend((range.start.index()..range.end.index()).map(RowId::from_usize));
    }
}

impl Clear for SortedOffsetVector {
    fn clear(&mut self) {
        self.0.clear()
    }
    fn reuse(&self) -> bool {
        self.0.capacity() > 0
    }
    fn bytes(&self) -> usize {
        self.0.capacity() * mem::size_of::<RowId>()
    }
}

impl Offsets for SortedOffsetVector {
    fn bounds(&self) -> (RowId, RowId) {
        self.slice().bounds()
    }

    fn offsets(&self, f: impl FnMut(RowId)) {
        self.slice().offsets(f)
    }
}

#[derive(PartialEq, Eq)]
#[repr(transparent)]
pub struct SortedOffsetSlice([RowId]);

impl fmt::Debug for SortedOffsetSlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.0.iter()).finish()
    }
}

impl SortedOffsetSlice {
    pub(crate) unsafe fn new_unchecked(slice: &[RowId]) -> &SortedOffsetSlice {
        debug_assert!(
            slice.windows(2).all(|w| w[0] <= w[1]),
            "slice is not sorted: {slice:?}"
        );
        // SAFETY: SortedOffsetSlice is repr(transparent), so the two layouts are compatible.
        unsafe { mem::transmute::<&[RowId], &SortedOffsetSlice>(slice) }
    }
    fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = RowId> + '_ {
        self.0.iter().copied()
    }

    pub(crate) fn inner(&self) -> &[RowId] {
        &self.0
    }

    pub(crate) fn subslice(&self, lo: usize, hi: usize) -> &SortedOffsetSlice {
        // Safety: any subslice of a sorted slice is sorted.
        unsafe { SortedOffsetSlice::new_unchecked(&self.inner()[lo..hi]) }
    }

    /// Return the index of the first offset in the slice that is greater than or equal to `target`.
    pub(crate) fn binary_search_by_id(&self, target: RowId) -> usize {
        self.binary_search_from(0, target)
    }
    fn binary_search_from(&self, start: usize, target: RowId) -> usize {
        match self.inner()[start..].binary_search(&target) {
            Ok(mut found) => {
                found += start;
                // This is O(n), but offset slices probably won't have duplicates at all.
                while found > 0 && self.inner()[found - 1] == target {
                    found -= 1;
                }
                found
            }
            Err(x) => start + x,
        }
    }

    fn scan_for_offset(&self, start: usize, target: RowId) -> Result<usize, usize> {
        let i = self.binary_search_from(start, target);
        if i < self.len() && self.inner()[i] == target {
            Ok(i)
        } else {
            Err(i)
        }
    }

    pub(crate) fn contains(&self, row: RowId) -> bool {
        self.scan_for_offset(0, row).is_ok()
    }
}

impl Offsets for SortedOffsetSlice {
    fn bounds(&self) -> (RowId, RowId) {
        match (self.0.first(), self.0.last()) {
            (Some(&lo), Some(&hi)) => (lo, RowId::from_usize(hi.index() + 1)),
            _ => (RowId::new(0), RowId::new(0)),
        }
    }

    fn offsets(&self, f: impl FnMut(RowId)) {
        self.0.iter().copied().for_each(f)
    }
}

impl Offsets for &'_ SortedOffsetSlice {
    fn bounds(&self) -> (RowId, RowId) {
        (*self).bounds()
    }

    fn offsets(&self, f: impl FnMut(RowId)) {
        self.0.iter().copied().for_each(f)
    }
}

/// A bitvector representation for a subset, using 256-bit blocks.
///
/// Each block covers 256 consecutive row IDs starting at a block-aligned offset.
/// The block_starts array holds the start position of each block (always a multiple of 256),
/// and blocks holds the corresponding 256-bit bitvectors as four u64 words.
#[derive(Default, Clone, PartialEq, Eq, Debug, Hash)]
pub struct BitVecSubset {
    block_starts: Vec<u32>,
    blocks: Vec<[u64; 4]>,
    size: usize,
}

impl Offsets for BitVecSubset {
    fn bounds(&self) -> (RowId, RowId) {
        let lo = 'search: {
            for (start, block) in self.block_starts.iter().zip(self.blocks.iter()) {
                for (wi, &word) in block.iter().enumerate() {
                    if word != 0 {
                        let bit = word.trailing_zeros() as usize;
                        break 'search RowId::from_usize(*start as usize + wi * 64 + bit);
                    }
                }
            }
            return (RowId::new(0), RowId::new(0));
        };
        let hi = {
            let mut result = RowId::new(0);
            'outer: for (start, block) in self.block_starts.iter().zip(self.blocks.iter()).rev() {
                for (wi, &word) in block.iter().enumerate().rev() {
                    if word != 0 {
                        let bit = 63 - word.leading_zeros() as usize;
                        result = RowId::from_usize(*start as usize + wi * 64 + bit + 1);
                        break 'outer;
                    }
                }
            }
            result
        };
        (lo, hi)
    }

    fn offsets(&self, mut f: impl FnMut(RowId)) {
        for (start, block) in self.block_starts.iter().zip(self.blocks.iter()) {
            for (wi, &word) in block.iter().enumerate() {
                let mut w = word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    f(RowId::from_usize(*start as usize + wi * 64 + bit));
                    w &= w - 1;
                }
            }
        }
    }
}

impl BitVecSubset {
    fn block_start_for(row: RowId) -> u32 {
        (row.index() as u32 / 256) * 256
    }

    /// Set a bit for `row`, which must be >= all previously set bits (sorted insertion).
    fn push_sorted(&mut self, row: RowId) {
        let start = Self::block_start_for(row);
        if self.block_starts.last().copied() != Some(start) {
            self.block_starts.push(start);
            self.blocks.push([0u64; 4]);
        }
        let i = self.blocks.len() - 1;
        let bit_idx = row.index() - start as usize;
        self.blocks[i][bit_idx / 64] |= 1u64 << (bit_idx % 64);
        self.size += 1;
    }

    pub(crate) fn contains(&self, row: RowId) -> bool {
        let start = Self::block_start_for(row);
        match self.block_starts.binary_search(&start) {
            Ok(i) => {
                let bit_idx = row.index() - start as usize;
                (self.blocks[i][bit_idx / 64] >> (bit_idx % 64)) & 1 == 1
            }
            Err(_) => false,
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    fn retain(&mut self, mut f: impl FnMut(RowId) -> bool) {
        for (start, block) in self.block_starts.iter().zip(self.blocks.iter_mut()) {
            for (wi, word) in block.iter_mut().enumerate() {
                let mut w = *word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    let row = RowId::from_usize(*start as usize + wi * 64 + bit);
                    if !f(row) {
                        *word &= !(1u64 << bit);
                    }
                    w &= w - 1;
                }
            }
        }
        let mut keep = 0;
        let mut new_size = 0;
        for i in 0..self.blocks.len() {
            let block_size: usize = self.blocks[i].iter().map(|w| w.count_ones() as usize).sum();
            if block_size > 0 {
                self.block_starts.swap(keep, i);
                self.blocks.swap(keep, i);
                new_size += block_size;
                keep += 1;
            }
        }
        self.block_starts.truncate(keep);
        self.blocks.truncate(keep);
        self.size = new_size;
    }

    /// AND this bitvec with `other`, keeping only bits present in both.
    fn intersect_with(&mut self, other: &BitVecSubset) {
        let mut keep = 0;
        let mut si = 0;
        let mut oi = 0;
        let mut new_size = 0;
        while si < self.block_starts.len() && oi < other.block_starts.len() {
            match self.block_starts[si].cmp(&other.block_starts[oi]) {
                Ordering::Equal => {
                    let mut new_block = [0u64; 4];
                    #[allow(clippy::needless_range_loop)]
                    for k in 0..4 {
                        new_block[k] = self.blocks[si][k] & other.blocks[oi][k];
                    }
                    let block_size: usize = new_block.iter().map(|w| w.count_ones() as usize).sum();
                    if block_size > 0 {
                        self.block_starts[keep] = self.block_starts[si];
                        self.blocks[keep] = new_block;
                        new_size += block_size;
                        keep += 1;
                    }
                    si += 1;
                    oi += 1;
                }
                Ordering::Less => {
                    si += 1;
                }
                Ordering::Greater => {
                    oi += 1;
                }
            }
        }
        self.block_starts.truncate(keep);
        self.blocks.truncate(keep);
        self.size = new_size;
    }

    pub(crate) fn from_sorted_slice(slice: &SortedOffsetSlice) -> BitVecSubset {
        let mut bv = BitVecSubset::default();
        for row in slice.iter() {
            bv.push_sorted(row);
        }
        bv
    }
}

const BITVEC_DENSITY_THRESHOLD: f64 = 0.05;

#[derive(Copy, Clone)]
pub enum SubsetRef<'a> {
    Dense(OffsetRange),
    Sparse(&'a SortedOffsetSlice),
    Bitvec(&'a BitVecSubset),
}

impl Offsets for SubsetRef<'_> {
    fn bounds(&self) -> (RowId, RowId) {
        match self {
            SubsetRef::Dense(r) => r.bounds(),
            SubsetRef::Sparse(s) => s.bounds(),
            SubsetRef::Bitvec(bv) => bv.bounds(),
        }
    }
    fn offsets(&self, f: impl FnMut(RowId)) {
        match self {
            SubsetRef::Dense(r) => r.offsets(f),
            SubsetRef::Sparse(s) => s.offsets(f),
            SubsetRef::Bitvec(bv) => bv.offsets(f),
        }
    }
}

impl SubsetRef<'_> {
    pub(crate) fn size(&self) -> usize {
        match self {
            SubsetRef::Dense(range) => range.size(),
            SubsetRef::Sparse(vec) => vec.0.len(),
            SubsetRef::Bitvec(bv) => bv.size(),
        }
    }

    pub(crate) fn to_owned(self, pool: &Pool<SortedOffsetVector>) -> Subset {
        match self {
            SubsetRef::Dense(r) => Subset::Dense(r),
            SubsetRef::Sparse(s) => {
                let mut vec = pool.get();
                vec.extend_nonoverlapping(s);
                Subset::Sparse(vec)
            }
            SubsetRef::Bitvec(bv) => Subset::Bitvec(bv.clone()),
        }
    }

    /// Get the underlying slice of a sparse subset. Used for debugging.
    pub(crate) fn _slice(&self) -> &[RowId] {
        match self {
            SubsetRef::Dense(_) => panic!("getting slice from dense subset"),
            SubsetRef::Sparse(slc) => slc.inner(),
            SubsetRef::Bitvec(_) => panic!("getting slice from bitvec subset"),
        }
    }
    pub(crate) fn iter_bounded(
        self,
        start: usize,
        end: usize,
        mut f: impl FnMut(RowId),
    ) -> Option<usize> {
        match self {
            SubsetRef::Dense(r) => {
                let mut cur = start;
                for row in (r.start.index() + start.index())
                    ..cmp::min(r.start.index().saturating_add(end), r.end.index())
                {
                    f(RowId::new(row as _));
                    cur += 1;
                }
                if cur + r.start.index() < r.end.index() {
                    Some(cur)
                } else {
                    None
                }
            }
            SubsetRef::Sparse(vec) => {
                let end = cmp::min(vec.0.len(), end);
                let next = if end == vec.0.len() { None } else { Some(end) };
                vec.0[start..end].iter().copied().for_each(f);
                next
            }
            SubsetRef::Bitvec(bv) => {
                let total = bv.size();
                let end = cmp::min(total, end);
                let has_more = end < total;
                let mut count = 0;
                'outer: for (block_start, block) in bv.block_starts.iter().zip(bv.blocks.iter()) {
                    // Skip entire blocks that fall before `start`.
                    let block_count = block.iter().map(|w| w.count_ones() as usize).sum::<usize>();
                    if count + block_count <= start {
                        count += block_count;
                        continue;
                    }
                    for (wi, &word) in block.iter().enumerate() {
                        let mut w = word;
                        while w != 0 {
                            let bit = w.trailing_zeros() as usize;
                            if count >= start {
                                if count >= end {
                                    break 'outer;
                                }
                                f(RowId::from_usize(*block_start as usize + wi * 64 + bit));
                            }
                            count += 1;
                            w &= w - 1;
                        }
                    }
                }
                if has_more { Some(end) } else { None }
            }
        }
    }
}

/// Either an offset range, a sorted offset vector, or a bitvector.
#[derive(Debug, Hash, PartialEq, Eq)]
pub enum Subset {
    Dense(OffsetRange),
    Sparse(Pooled<SortedOffsetVector>),
    Bitvec(BitVecSubset),
}

impl Offsets for Subset {
    fn bounds(&self) -> (RowId, RowId) {
        match self {
            Subset::Dense(r) => r.bounds(),
            Subset::Sparse(s) => s.slice().bounds(),
            Subset::Bitvec(bv) => bv.bounds(),
        }
    }
    fn offsets(&self, f: impl FnMut(RowId)) {
        match self {
            Subset::Dense(r) => r.offsets(f),
            Subset::Sparse(s) => s.slice().offsets(f),
            Subset::Bitvec(bv) => bv.offsets(f),
        }
    }
}

impl Clone for Subset {
    fn clone(&self) -> Self {
        match self {
            Subset::Dense(r) => Subset::Dense(*r),
            Subset::Sparse(s) => Subset::Sparse(Pooled::cloned(s)),
            Subset::Bitvec(bv) => Subset::Bitvec(bv.clone()),
        }
    }
}

// TODO: consider making Subset::Sparse an Rc, so copies are shallow?

impl Subset {
    /// The size of the subset.
    pub fn size(&self) -> usize {
        match self {
            Subset::Dense(range) => range.size(),
            Subset::Sparse(vec) => vec.0.len(),
            Subset::Bitvec(bv) => bv.size(),
        }
    }

    /// The density of the subset: ratio of elements to the span of its bounds.
    /// Returns 0.0 for empty subsets.
    pub fn density(&self) -> f64 {
        let (lo, hi) = self.bounds();
        let range = hi.index().saturating_sub(lo.index());
        if range == 0 {
            return 0.0;
        }
        self.size() as f64 / range as f64
    }

    pub(crate) fn is_dense(&self) -> bool {
        matches!(self, Subset::Dense(_))
    }

    pub fn as_ref(&self) -> SubsetRef<'_> {
        match self {
            Subset::Dense(r) => SubsetRef::Dense(*r),
            Subset::Sparse(s) => SubsetRef::Sparse(s.slice()),
            Subset::Bitvec(bv) => SubsetRef::Bitvec(bv),
        }
    }

    pub(crate) fn retain(&mut self, mut filter: impl FnMut(RowId) -> bool) {
        match self {
            Subset::Dense(offs) => {
                let mut res = Subset::empty();
                offs.offsets(|row| {
                    if filter(row) {
                        res.add_row_sorted(row);
                    }
                });
                *self = res;
            }
            Subset::Sparse(offs) => offs.retain(filter),
            Subset::Bitvec(bv) => bv.retain(filter),
        }
    }

    /// Remove any elements of the current subset not present in `other`.
    pub(crate) fn intersect(&mut self, other: SubsetRef, pool: &Pool<SortedOffsetVector>) {
        if self.is_empty() || other.is_empty() {
            *self = Subset::Dense(OffsetRange::new(RowId::new(0), RowId::new(0)));
            return;
        }

        match (self, other) {
            (Subset::Dense(cur), SubsetRef::Dense(other)) => {
                let resl = cmp::max(cur.start, other.start);
                let resr = cmp::min(cur.end, other.end);
                if resl >= resr {
                    *cur = OffsetRange::new(resl, resl);
                } else {
                    *cur = OffsetRange::new(resl, resr);
                }
            }
            (x @ Subset::Dense(_), SubsetRef::Sparse(sparse)) => {
                let (low, hi) = x.bounds();
                let mut res = pool.get();
                let l = sparse.binary_search_by_id(low);
                let r = sparse.binary_search_by_id(hi);
                let subslice = sparse.subslice(l, r);
                res.extend_nonoverlapping(subslice);
                *x = Subset::Sparse(res);
            }
            (x @ Subset::Dense(_), SubsetRef::Bitvec(other)) => {
                let (low, hi) = x.bounds();
                let mut result = BitVecSubset::default();
                // TODO See below, use bounds to avoid enumerating rows outside the range.
                other.offsets(|row| {
                    if row >= low && row < hi {
                        result.push_sorted(row);
                    }
                });
                *x = Subset::Bitvec(result);
            }
            (Subset::Sparse(sparse), SubsetRef::Dense(dense)) => {
                let r = sparse.slice().binary_search_by_id(dense.end);
                sparse.0.truncate(r);
                sparse.retain(|row| row >= dense.start);
            }
            (Subset::Sparse(cur), SubsetRef::Sparse(other)) => {
                if cur.0.len() < other.inner().len() {
                    let mut other_off = 0;
                    cur.retain(|rowid| match other.scan_for_offset(other_off, rowid) {
                        Ok(found) => {
                            other_off = found + 1;
                            true
                        }
                        Err(next_off) => {
                            other_off = next_off;
                            false
                        }
                    })
                } else {
                    let mut cur_off = 0;
                    let mut cur_idx = 0;
                    other.inner().iter().copied().for_each(|rowid| {
                        match cur.slice().scan_for_offset(cur_off, rowid) {
                            Ok(found) => {
                                cur_off = found + 1;
                                cur.0[cur_idx] = rowid;
                                cur_idx += 1;
                            }
                            Err(next_off) => {
                                cur_off = next_off;
                            }
                        }
                    });
                    cur.0.truncate(cur_idx);
                }
            }
            (Subset::Sparse(sparse), SubsetRef::Bitvec(bv)) => {
                sparse.retain(|row| bv.contains(row));
            }
            (Subset::Bitvec(bv), SubsetRef::Dense(dense)) => {
                // TODO Make retain takes a bound so that it only enumerates rows within the range.
                bv.retain(|row| row >= dense.start && row < dense.end);
            }
            (Subset::Bitvec(bv), SubsetRef::Sparse(sparse)) => {
                // TODO save the result as a Sparse instead of BitVec because the result
                // must be more sparse than Sparse.
                bv.retain(|row| sparse.contains(row));
            }
            (Subset::Bitvec(bv_self), SubsetRef::Bitvec(other)) => {
                bv_self.intersect_with(other);
            }
        }
    }

    /// Append the given row id to the Subset.
    ///
    /// # Panics
    /// The row id in question must be greater than or equal to the upper bound
    /// of the subset. This method will panic if it is not.
    pub(crate) fn add_row_sorted(&mut self, row: RowId) {
        match self {
            Subset::Dense(range) => {
                if range.end == range.start {
                    range.start = row;
                    range.end = row.inc();
                    return;
                }
                if range.end == row {
                    range.end = row.inc();
                    return;
                }
                let mut vec = with_pool_set(|pool_set| pool_set.get::<SortedOffsetVector>());
                vec.fill_from_dense(range);
                vec.push(row);
                *self = Subset::Sparse(vec);
            }
            Subset::Sparse(s) => {
                s.push(row);
                // TODO: Don't switch to bitvec here, because the bounds can be an underestimate.
                let (lo, hi) = s.slice().bounds();
                let range_len = hi.index().saturating_sub(lo.index());
                if range_len > 0 && s.0.len() as f64 / range_len as f64 > BITVEC_DENSITY_THRESHOLD {
                    let bv = BitVecSubset::from_sorted_slice(s.slice());
                    *self = Subset::Bitvec(bv);
                }
            }
            Subset::Bitvec(bv) => {
                bv.push_sorted(row);
            }
        }
    }

    pub(crate) fn empty() -> Subset {
        Subset::Dense(OffsetRange::new(RowId::new(0), RowId::new(0)))
    }
}
