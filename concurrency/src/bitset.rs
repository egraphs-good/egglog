//! A simple dense bitset that leverages [`crate::ReadOptimizedLock`] for safe
//! resizing and provides wait-free reads and writes when no resizing is
//! required.

use std::{
    mem,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::ReadOptimizedLock;

pub struct BitSet {
    data: ReadOptimizedLock<Vec<AtomicU64>>,
}

impl BitSet {
    pub fn with_capacity(n: usize) -> Self {
        let n = n.next_multiple_of(64).next_power_of_two();
        let cells = n / 64;
        let mut data = Vec::with_capacity(cells);
        data.resize_with(cells, AtomicU64::default);
        BitSet {
            data: ReadOptimizedLock::new(data),
        }
    }

    /// Get the value of the bit at index `i`. If the bitset has not been
    /// initialized up to `i`, then this method returns `false`.
    pub fn get(&self, i: usize) -> bool {
        let cell = i / 64;
        let bit = i % 64;
        let reader = self.data.read();
        reader
            .get(cell)
            .map(|x| x.load(Ordering::Acquire) & (1 << bit) != 0)
            .unwrap_or(false)
    }

    /// Set the bit at index `i` to `val`. If the bitset has not been
    /// initialized up to `i`, the bitset will be resized. The resizing
    /// operation will block until all current readers have finished.
    pub fn set(&self, i: usize, val: bool) {
        let cell = i / 64;
        let bit = i % 64;
        let handle = self.data.read();
        if let Some(cell) = handle.get(cell) {
            if val {
                cell.fetch_or(1 << bit, Ordering::Release);
            } else {
                cell.fetch_and(!(1 << bit), Ordering::Release);
            }
            return;
        }
        mem::drop(handle);
        let mut writer = self.data.lock();
        if cell >= writer.len() {
            writer.resize_with((cell + 1).next_power_of_two(), AtomicU64::default);
        }
        mem::drop(writer);
        self.set(i, val);
    }
}
