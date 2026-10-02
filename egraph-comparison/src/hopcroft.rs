//! Dirty-state partition refinement (Jacobs–Wißmann, Algorithm 6).
//! A largest piece retains its ID; only predecessors of smaller pieces are
//! invalidated. Clean members of a block have equal current signatures.
use crate::ids::{BlockId, ClassId};
use crate::{refine::Partition, signatures::Signatures};
use egglog_numeric_id::NumericId;
use std::collections::VecDeque;

/// Members occupy [start, end), with dirty members in [clean, end).
struct Block {
    start: usize,
    clean: usize,
    end: usize,
}

struct Worklist {
    members: Vec<ClassId>,
    position: Vec<usize>,
    blocks: Vec<Block>,
    // FIFO lets pending child splits accumulate before a parent is revisited.
    pending: VecDeque<BlockId>,
    // Compressed reverse edges: predecessors of c are parents[offset[c]..offset[c+1]].
    offset: Vec<usize>,
    parents: Vec<ClassId>,
    steps: usize,
    #[cfg(test)]
    evaluations: usize,
    #[cfg(test)]
    moves: Vec<usize>,
}

impl Worklist {
    fn new(partition: &Partition<'_>) -> Self {
        let n = partition.blocks.len();
        let count = partition
            .blocks
            .iter()
            .map(|b| b.index() + 1)
            .max()
            .unwrap_or(0);
        let mut sizes = vec![0; count];
        for b in &partition.blocks {
            sizes[b.index()] += 1;
        }
        let mut start = 0;
        let blocks: Vec<_> = sizes
            .iter()
            .map(|&size| {
                let block = Block {
                    start,
                    clean: start,
                    end: start + size,
                };
                start += size;
                block
            })
            .collect();
        let mut cursor: Vec<_> = blocks.iter().map(|b| b.start).collect();
        let mut members = vec![ClassId::from_usize(0); n];
        let mut position = vec![0; n];
        for (i, b) in partition.blocks.iter().enumerate() {
            position[i] = cursor[b.index()];
            members[position[i]] = ClassId::from_usize(i);
            cursor[b.index()] += 1;
        }

        let mut offset = vec![0; n + 1];
        for nodes in partition.left.nodes.iter().chain(&partition.right.nodes) {
            for child in nodes.iter().flat_map(|node| &node.children) {
                offset[child.index()] += 1;
            }
        }
        let mut total = 0;
        for entry in &mut offset {
            let degree = *entry;
            *entry = total;
            total += degree;
        }
        let mut parents = vec![ClassId::from_usize(0); total];
        cursor.clear();
        cursor.extend_from_slice(&offset[..n]);
        for (i, nodes) in partition
            .left
            .nodes
            .iter()
            .chain(&partition.right.nodes)
            .enumerate()
        {
            for child in nodes.iter().flat_map(|node| &node.children) {
                parents[cursor[child.index()]] = ClassId::from_usize(i);
                cursor[child.index()] += 1;
            }
        }
        Self {
            members,
            position,
            blocks,
            pending: (0..count).map(BlockId::from_usize).collect(),
            offset,
            parents,
            steps: 0,
            #[cfg(test)]
            evaluations: 0,
            #[cfg(test)]
            moves: vec![0; n],
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        self.members.swap(a, b);
        self.position[self.members[a].index()] = a;
        self.position[self.members[b].index()] = b;
    }

    fn dirty(&mut self, id: ClassId, partition: &Partition<'_>) {
        let b = partition.blocks[id.index()];
        let block = &mut self.blocks[b.index()];
        let pos = self.position[id.index()];
        if pos >= block.clean {
            return; // Already dirty and queued, including repeated child edges.
        }
        if block.clean == block.end {
            self.pending.push_back(b);
        }
        block.clean -= 1;
        let boundary = block.clean;
        self.swap(pos, boundary);
    }

    fn finish(&mut self, partition: &mut Partition<'_>) {
        let mut signatures = Signatures::default();
        let mut nodes = Vec::new();
        let mut assignments = Vec::new();
        let mut sizes = Vec::new();
        let mut starts = Vec::new();
        let mut cursor = Vec::new();
        while let Some(b) = self.pending.pop_front() {
            self.steps += 1;
            let Block { start, clean, end } = self.blocks[b.index()];
            signatures.clear();
            assignments.clear();
            sizes.clear();
            if start < clean {
                // Intern the one clean representative first: bucket 0 holds
                // every clean member and any dirty members that still match.
                partition.signature(self.members[start], &mut nodes, &mut signatures);
                sizes.push(clean - start);
            }
            for &id in &self.members[clean..end] {
                let group = partition.signature(id, &mut nodes, &mut signatures).index();
                if group == sizes.len() {
                    sizes.push(0);
                }
                sizes[group] += 1;
                assignments.push((id, group));
            }
            #[cfg(test)]
            {
                self.evaluations += end - clean + usize::from(start < clean);
            }
            self.blocks[b.index()].clean = end;
            if sizes.len() <= 1 {
                continue;
            }
            // Counting-sort only dirty members into contiguous buckets. The
            // potentially huge clean prefix already belongs to bucket 0.
            starts.clear();
            let mut next = start;
            for &size in &sizes {
                starts.push(next);
                next += size;
            }
            cursor.clear();
            cursor.extend_from_slice(&starts);
            cursor[0] = clean;
            for &(id, group) in &assignments {
                self.swap(self.position[id.index()], cursor[group]);
                cursor[group] += 1;
            }
            let largest = sizes
                .iter()
                .enumerate()
                .max_by_key(|&(_, size)| size)
                .unwrap()
                .0;
            for (group, (&start, &size)) in starts.iter().zip(&sizes).enumerate() {
                let block = Block {
                    start,
                    clean: start + size,
                    end: start + size,
                };
                if group == largest {
                    self.blocks[b.index()] = block;
                } else {
                    let new_id = BlockId::from_usize(self.blocks.len());
                    for &id in &self.members[start..start + size] {
                        partition.blocks[id.index()] = new_id;
                        #[cfg(test)]
                        {
                            self.moves[id.index()] += 1;
                        }
                    }
                    self.blocks.push(block);
                }
            }
            // Finish all ID updates before invalidating predecessors: a parent
            // can lie in this same block (cycles), or in another new piece.
            // All smaller pieces lie before or after the retained largest
            // range. Save their IDs before dirty() can reorder those ranges.
            assignments.clear();
            let kept = &self.blocks[b.index()];
            for &id in self.members[start..kept.start]
                .iter()
                .chain(&self.members[kept.end..end])
            {
                assignments.push((id, 0));
            }
            for &(id, _) in &assignments {
                for edge in self.offset[id.index()]..self.offset[id.index() + 1] {
                    self.dirty(self.parents[edge], partition);
                }
            }
        }
    }
}

pub(crate) fn refine(partition: &mut Partition<'_>) -> usize {
    let mut worklist = Worklist::new(partition);
    worklist.finish(partition);
    worklist.steps
}

#[cfg(test)]
#[path = "../tests/support/hopcroft.rs"]
mod tests;
