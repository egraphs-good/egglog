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

use std::sync::Arc;

use crate::free_join::execute::TrieNode;
use crate::numeric_id::define_id;
use crate::offsets::OffsetRange;

use crate::Value;

use super::{AtomId, Variable};

define_id!(pub SubsetId, u32, "An offset into a buffer of subsets");

#[derive(Debug)]
pub(super) enum UpdateInstr {
    PushBinding(Variable, Value),
    RefineAtom(AtomId, Arc<TrieNode>),
    /// Refine an atom to a dense offset range, avoiding an Arc<TrieNode> allocation.
    RefineAtomDense(AtomId, OffsetRange),
    /// Marks the end of the current frame. Time to make a recursive call.
    EndFrame,
}

/// A flat buffer of updates that is used to prepare a sequence of recursive calls to free join.
#[derive(Default)]
pub(super) struct FrameUpdates {
    updates: Vec<UpdateInstr>,
    frames: usize,
    last_start: usize,
}

impl FrameUpdates {
    pub(super) fn with_capacity(capacity: usize) -> FrameUpdates {
        FrameUpdates {
            updates: Vec::with_capacity(capacity * 2),
            frames: 0,
            last_start: 0,
        }
    }

    /// Bind `var` to `val` in the current frame.
    pub(super) fn push_binding(&mut self, var: Variable, val: Value) {
        self.updates.push(UpdateInstr::PushBinding(var, val));
    }

    /// Refine `atom` to consider only the given `subset` in the current frame.
    pub(super) fn refine_atom(&mut self, atom: AtomId, node: Arc<TrieNode>) {
        self.updates.push(UpdateInstr::RefineAtom(atom, node));
    }

    /// Refine `atom` to consider only the given dense offset range, without
    /// allocating an Arc<TrieNode> eagerly.
    pub(super) fn refine_atom_dense(&mut self, atom: AtomId, range: OffsetRange) {
        self.updates.push(UpdateInstr::RefineAtomDense(atom, range));
    }

    /// Roll back the updates to the last frame start. Note that repeated calls
    /// to this method will still only roll back one frame (total).
    pub(super) fn rollback(&mut self) {
        self.updates.truncate(self.last_start);
    }

    /// Finish the current frame and prepare for the next one.
    pub(super) fn finish_frame(&mut self) {
        self.updates.push(UpdateInstr::EndFrame);
        self.last_start = self.updates.len();
        self.frames += 1;
    }

    /// Get the number of frames that have been finished.
    pub(super) fn frames(&self) -> usize {
        self.frames
    }

    pub(super) fn clear(&mut self) {
        self.updates.clear();
    }

    pub(super) fn drain(&mut self, f: impl FnMut(UpdateInstr)) {
        let start = 0;
        self.updates.drain(start..).for_each(f);
        self.frames = 0;
        self.last_start = 0;
    }

    // for debugging
    #[allow(dead_code)]
    pub(super) fn updates(&self) -> &[UpdateInstr] {
        &self.updates
    }
}
