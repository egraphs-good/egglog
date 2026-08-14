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
