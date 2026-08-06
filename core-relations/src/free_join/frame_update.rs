//! A data-structure for low-overhead buffering of updates in a free join
//! execution.
//!
//! Free Join is a recursive algorithm that that discovers a candidate binding for a particular
//! variable in a query and then recursively runs the rest of the join restricted for that binding
//! holding. Once the "sub-join" finishes, the outer recursive call backtracks, adds a separate
//! binding, and then repeats.
//!
//! The Free Join paper observed that this resulted in poor cache behavior because for every cell
//! iterated over in an outer stage, we had to do several other steps on successive inner stages.
//! Instead, we can accumulate a set of new bindings in a separate buffer and then iterate over
//! those bindings in recursive calls. When parallelism is enabled, this data-structure allows us
//! hand over an entire batch of recursive calls to a separate thread to process independently.

use crate::free_join::execute::AtomRows;
use crate::offsets::OffsetRange;

use crate::Value;

use super::{AtomId, Variable};

#[derive(Debug)]
pub(super) enum UpdateInstr<'rows, 'exec> {
    PushBinding(Variable, Value),
    RefineAtom(AtomId, AtomRows<'rows, 'exec>),
    /// Refine an atom to a dense offset range, avoiding an Arc<TrieNode> allocation.
    RefineAtomDense(AtomId, OffsetRange),
    /// Marks the end of the current frame. Time to make a recursive call.
    EndFrame,
}

/// Compact instruction stored in [`FrameUpdates`]. Variable-sized atom rows
/// live in a side buffer so the common binding and frame-marker instructions
/// do not inherit the size of [`AtomRows`]'s largest variant.
#[derive(Debug)]
enum BufferedUpdateInstr {
    PushBinding(Variable, Value),
    RefineAtom(AtomId),
    RefineAtomDense(AtomId, OffsetRange),
    EndFrame,
}

/// A flat buffer of updates that is used to prepare a sequence of recursive calls to free join.
#[derive(Default)]
pub(super) struct FrameUpdates<'rows, 'exec> {
    updates: Vec<BufferedUpdateInstr>,
    subsets: Vec<AtomRows<'rows, 'exec>>,
    frames: usize,
    last_start: usize,
    last_subset_start: usize,
}

impl<'rows, 'exec> FrameUpdates<'rows, 'exec> {
    pub(super) fn with_capacity(capacity: usize) -> FrameUpdates<'rows, 'exec> {
        FrameUpdates {
            // A two-prober frame commonly contains one binding, two atom
            // refinements, and the end marker.
            updates: Vec::with_capacity(capacity.saturating_mul(4)),
            subsets: Vec::with_capacity(capacity.saturating_mul(2)),
            frames: 0,
            last_start: 0,
            last_subset_start: 0,
        }
    }

    /// Bind `var` to `val` in the current frame.
    pub(super) fn push_binding(&mut self, var: Variable, val: Value) {
        self.updates
            .push(BufferedUpdateInstr::PushBinding(var, val));
    }

    /// Refine `atom` to consider only the given `subset` in the current frame.
    pub(super) fn refine_atom(&mut self, atom: AtomId, node: impl Into<AtomRows<'rows, 'exec>>) {
        self.subsets.push(node.into());
        self.updates.push(BufferedUpdateInstr::RefineAtom(atom));
    }

    /// Refine `atom` to consider only the given dense offset range, without
    /// allocating an Arc<TrieNode> eagerly.
    pub(super) fn refine_atom_dense(&mut self, atom: AtomId, range: OffsetRange) {
        self.updates
            .push(BufferedUpdateInstr::RefineAtomDense(atom, range));
    }

    /// Roll back the updates to the last frame start. Note that repeated calls
    /// to this method will still only roll back one frame (total).
    pub(super) fn rollback(&mut self) {
        self.updates.truncate(self.last_start);
        self.subsets.truncate(self.last_subset_start);
    }

    /// Finish the current frame and prepare for the next one.
    pub(super) fn finish_frame(&mut self) {
        self.updates.push(BufferedUpdateInstr::EndFrame);
        self.last_start = self.updates.len();
        self.last_subset_start = self.subsets.len();
        self.frames += 1;
    }

    /// Get the number of frames that have been finished.
    pub(super) fn frames(&self) -> usize {
        self.frames
    }

    pub(super) fn clear(&mut self) {
        self.updates.clear();
        self.subsets.clear();
        self.frames = 0;
        self.last_start = 0;
        self.last_subset_start = 0;
    }

    pub(super) fn drain(&mut self, mut f: impl FnMut(UpdateInstr<'rows, 'exec>)) {
        // Reset bookkeeping before invoking user code. Both drains still own
        // and drop their remaining elements if `f` unwinds, leaving the
        // buffer reusable after a caught panic.
        self.frames = 0;
        self.last_start = 0;
        self.last_subset_start = 0;
        {
            let mut subsets = self.subsets.drain(..);
            for update in self.updates.drain(..) {
                let update = match update {
                    BufferedUpdateInstr::PushBinding(var, val) => {
                        UpdateInstr::PushBinding(var, val)
                    }
                    BufferedUpdateInstr::RefineAtom(atom) => {
                        let rows = subsets
                            .next()
                            .expect("refine instruction is missing its atom rows");
                        UpdateInstr::RefineAtom(atom, rows)
                    }
                    BufferedUpdateInstr::RefineAtomDense(atom, range) => {
                        UpdateInstr::RefineAtomDense(atom, range)
                    }
                    BufferedUpdateInstr::EndFrame => UpdateInstr::EndFrame,
                };
                f(update);
            }
            debug_assert!(subsets.next().is_none());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use crate::{Value, numeric_id::NumericId, offsets::RowId};

    use super::{AtomId, AtomRows, BufferedUpdateInstr, FrameUpdates, UpdateInstr, Variable};

    fn range(start: usize, end: usize) -> crate::OffsetRange {
        crate::OffsetRange::new(RowId::from_usize(start), RowId::from_usize(end))
    }

    #[test]
    fn buffered_update_instruction_stays_compact() {
        assert!(size_of::<BufferedUpdateInstr>() <= 24);
        assert!(size_of::<BufferedUpdateInstr>() < size_of::<UpdateInstr<'static, 'static>>());
    }

    #[test]
    fn drain_preserves_instruction_order_and_side_buffer_pairing() {
        let atom0 = AtomId::from_usize(0);
        let atom1 = AtomId::from_usize(1);
        let var = Variable::from_usize(2);
        let mut updates = FrameUpdates::with_capacity(2);
        updates.push_binding(var, Value::from_usize(7));
        updates.refine_atom(atom0, AtomRows::Dense(range(1, 3)));
        updates.refine_atom_dense(atom1, range(4, 5));
        updates.finish_frame();
        updates.refine_atom(atom1, AtomRows::Dense(range(8, 13)));
        updates.finish_frame();

        let mut decoded = Vec::new();
        updates.drain(|update| match update {
            UpdateInstr::PushBinding(variable, value) => {
                decoded.push((0, variable.index(), value.index(), 0))
            }
            UpdateInstr::RefineAtom(atom, AtomRows::Dense(rows)) => {
                decoded.push((1, atom.index(), rows.start.index(), rows.end.index()))
            }
            UpdateInstr::RefineAtomDense(atom, rows) => {
                decoded.push((2, atom.index(), rows.start.index(), rows.end.index()))
            }
            UpdateInstr::EndFrame => decoded.push((3, 0, 0, 0)),
            UpdateInstr::RefineAtom(_, _) => panic!("expected dense test rows"),
        });

        assert_eq!(
            decoded,
            vec![
                (0, 2, 7, 0),
                (1, 0, 1, 3),
                (2, 1, 4, 5),
                (3, 0, 0, 0),
                (1, 1, 8, 13),
                (3, 0, 0, 0),
            ]
        );
        assert_eq!(updates.frames(), 0);

        // Side-buffer IDs restart after a drain.
        updates.refine_atom(atom0, AtomRows::Dense(range(21, 22)));
        updates.finish_frame();
        let mut seen = None;
        updates.drain(|update| {
            if let UpdateInstr::RefineAtom(_, AtomRows::Dense(rows)) = update {
                seen = Some(rows);
            }
        });
        assert_eq!(seen, Some(range(21, 22)));
    }

    #[test]
    fn rollback_discards_partial_side_payloads_and_rebases_ids() {
        let atom = AtomId::from_usize(0);
        let mut updates = FrameUpdates::with_capacity(2);
        updates.refine_atom(atom, AtomRows::Dense(range(1, 2)));
        updates.finish_frame();

        updates.refine_atom(atom, AtomRows::Dense(range(3, 4)));
        updates.refine_atom(atom, AtomRows::Dense(range(5, 6)));
        updates.rollback();
        updates.rollback();
        updates.refine_atom(atom, AtomRows::Dense(range(7, 8)));
        updates.finish_frame();

        assert_eq!(updates.frames(), 2);
        let mut starts = Vec::new();
        updates.drain(|update| {
            if let UpdateInstr::RefineAtom(_, AtomRows::Dense(rows)) = update {
                starts.push(rows.start.index());
            }
        });
        assert_eq!(starts, vec![1, 7]);
    }

    #[test]
    fn clear_resets_checkpoints_and_allows_reuse() {
        let atom = AtomId::from_usize(0);
        let mut updates = FrameUpdates::with_capacity(1);
        updates.refine_atom(atom, AtomRows::Dense(range(1, 2)));
        updates.finish_frame();
        updates.refine_atom(atom, AtomRows::Dense(range(3, 4)));
        updates.clear();
        assert_eq!(updates.frames(), 0);

        updates.refine_atom(atom, AtomRows::Dense(range(5, 6)));
        updates.rollback();
        updates.refine_atom(atom, AtomRows::Dense(range(7, 8)));
        updates.finish_frame();
        let mut starts = Vec::new();
        updates.drain(|update| {
            if let UpdateInstr::RefineAtom(_, AtomRows::Dense(rows)) = update {
                starts.push(rows.start.index());
            }
        });
        assert_eq!(starts, vec![7]);
    }
}
