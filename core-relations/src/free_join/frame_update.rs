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
#[path = "frame_update_tests.rs"]
mod tests;
