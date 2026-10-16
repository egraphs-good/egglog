//! End-to-end rebuild coverage, independent of the cost model's decisions.
//!
//! Include as a child of `table::rebuild` so the fixture can compare the two
//! complete implementations without adding a production path-selection knob.

use std::collections::BTreeMap;

use egglog_concurrency::ThreadPool;

use crate::{
    ColumnId, RowId, TableId, Value, WrappedTable, numeric_id::NumericId, table_spec::Table,
    uf::DisplacedTable,
};

use super::SortedWritesTable;

// Three rebuildable key columns, one ordinary payload, and a timestamp.
type TestRow = [u32; 5];
type TestKey = [u32; 3];

fn key(row: &TestRow) -> TestKey {
    row[..3].try_into().unwrap()
}

fn input_row(id: u32, timestamp: u32) -> TestRow {
    let value = 1_000 + 3 * id;
    // Exercise both repeated values within a row and a shared, high-fanout ID.
    [
        value,
        if id.is_multiple_of(3) { value } else { 100 },
        value + 2,
        id,
        timestamp,
    ]
}

fn new_table() -> SortedWritesTable {
    SortedWritesTable::new(
        3,
        5,
        ColumnId::new(4),
        vec![ColumnId::new(0), ColumnId::new(1), ColumnId::new(2)],
        Box::new(|_, old, new, out| {
            let merged = [
                new[0],
                new[1],
                new[2],
                old[3].min(new[3]),
                old[4].max(new[4]),
            ];
            if old == merged {
                false
            } else {
                out.extend_from_slice(&merged);
                true
            }
        }),
    )
}

fn logical_rows(table: &SortedWritesTable) -> Vec<TestRow> {
    let mut rows = Vec::new();
    table.scan_generic(table.all().as_ref(), |_, row| {
        rows.push(std::array::from_fn(|column| row[column].rep()));
    });
    rows.sort_unstable();
    rows
}

struct Fixture {
    automatic: SortedWritesTable,
    full: SortedWritesTable,
    incremental: SortedWritesTable,
    uf: DisplacedTable,
    expected: BTreeMap<TestKey, TestRow>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            automatic: new_table(),
            full: new_table(),
            incremental: new_table(),
            uf: DisplacedTable::default(),
            expected: BTreeMap::new(),
        }
    }

    fn tables_mut(&mut self) -> [&mut SortedWritesTable; 3] {
        [&mut self.automatic, &mut self.full, &mut self.incremental]
    }

    fn insert_rows(&mut self, rows: impl IntoIterator<Item = TestRow>) {
        let rows: Vec<_> = rows.into_iter().collect();
        for row in &rows {
            self.expected
                .entry(key(row))
                .and_modify(|old| {
                    old[3] = old[3].min(row[3]);
                    old[4] = old[4].max(row[4]);
                })
                .or_insert(*row);
        }
        empty_execution_state!(state);
        for table in self.tables_mut() {
            {
                let mut buffer = table.new_buffer();
                for row in &rows {
                    buffer.stage_insert(&row.map(Value::new));
                }
            }
            table.merge(&mut state);
        }
        self.assert_rows("after insertion");
    }

    fn union(&mut self, edges: impl IntoIterator<Item = (u32, u32)>, timestamp: u32) {
        {
            let mut buffer = self.uf.new_buffer();
            for (left, right) in edges {
                buffer.stage_insert(&[Value::new(left), Value::new(right), Value::new(timestamp)]);
            }
        }
        empty_execution_state!(state);
        self.uf.merge(&mut state);
    }

    fn canonicalize_expected(&mut self, timestamp: u32) {
        let mut canonical = BTreeMap::<TestKey, TestRow>::new();
        for original in self.expected.values() {
            let mut row = *original;
            for value in &mut row[..3] {
                *value = self.uf.underlying_uf().find_naive(Value::new(*value)).rep();
            }
            if row[..3] != original[..3] {
                row[4] = timestamp;
            }
            canonical
                .entry(key(&row))
                .and_modify(|old| {
                    old[3] = old[3].min(row[3]);
                    old[4] = old[4].max(row[4]);
                })
                .or_insert(row);
        }
        self.expected = canonical;
    }

    fn rebuild(&mut self, timestamp: u32, label: &str) {
        self.canonicalize_expected(timestamp);
        // The same displaced-ID history and canonical map drive all modes.
        // The automatic mode retains its own recent-updates cursor across calls.
        let source = WrappedTable::new(self.uf.clone());
        let columns = vec![ColumnId::new(0), ColumnId::new(1), ColumnId::new(2)];
        let rebuilder = source.rebuilder(&columns).unwrap();
        let next_timestamp = Value::new(timestamp);
        empty_execution_state!(state);
        self.automatic
            .apply_rebuild(TableId::new(0), &source, next_timestamp, &mut state);
        self.full
            .rebuild_nonincremental(&*rebuilder, next_timestamp, &mut state);
        self.incremental.rebuild_incremental(
            &source,
            &*rebuilder,
            rebuilder.hint_col().unwrap(),
            source.all(),
            next_timestamp,
            &mut state,
        );
        for table in self.tables_mut() {
            table.merge(&mut state);
        }
        // A path may conservatively return `changed` for stale candidates.
        // Compare the committed rows, including timestamps, rather than flags.
        self.assert_rows(label);
    }

    fn assert_rows(&self, label: &str) {
        let expected: Vec<_> = self.expected.values().copied().collect();
        for (mode, table) in [
            ("automatic", &self.automatic),
            ("full", &self.full),
            ("incremental", &self.incremental),
        ] {
            assert_eq!(logical_rows(table), expected, "{label}: {mode}");
            // Validate point lookup too, not just the scan's logical contents.
            for row in &expected {
                let found = table.get_row(&key(row).map(Value::new)).unwrap();
                let actual: TestRow = std::array::from_fn(|column| found.vals[column].rep());
                assert_eq!(actual, *row, "{label}: {mode} lookup");
            }
        }
    }

    fn last_physical_value(&self) -> u32 {
        let last = self.automatic.data.next_row().index() - 1;
        self.automatic
            .data
            .get_row(RowId::from_usize(last))
            .unwrap()[2]
            .rep()
    }

    fn compact_warm_tables(&mut self) {
        // Build current-generation reverse indexes before making tombstones.
        for table in self.tables_mut() {
            table.refresh_rebuild_index();
        }
        let removed = *self.expected.values().find(|row| row[3] == 500).unwrap();
        self.expected.remove(&key(&removed)).unwrap();
        empty_execution_state!(state);
        for table in self.tables_mut() {
            table
                .new_buffer()
                .stage_remove(&key(&removed).map(Value::new));
            table.merge(&mut state);
        }
        let mut replacement = *self.expected.values().find(|row| row[3] == 600).unwrap();
        replacement[3] = 42;
        replacement[4] = 7;
        self.insert_rows([replacement]);
        for table in self.tables_mut() {
            assert!(table.has_stale_rows());
            let generation = table.version().major;
            table.rehash();
            assert_ne!(table.version().major, generation);
            assert!(!table.has_stale_rows());
        }
        self.assert_rows("after compaction with a previously warm index");
    }
}

#[test]
fn rebuild_paths_preserve_rows_across_index_and_table_lifecycles() {
    // These worker counts cover execution configurations. The compact fixture
    // deliberately does not claim to cross the full-scan parallel cutoff.
    for workers in [1, 4, 16] {
        ThreadPool::new(workers).install(|| {
            let mut fixture = Fixture::new();
            fixture.insert_rows((0..9_950).map(|id| input_row(id, 0)));
            fixture.union([(fixture.last_physical_value(), 0)], 1);
            fixture.rebuild(1, "below the small-table threshold");

            fixture.insert_rows((9_950..12_000).map(|id| input_row(id, 2)));
            // A change at the end of the physical range must not be missed by
            // the selector's bounded row sample, even with a cold index.
            fixture.union([(fixture.last_physical_value(), 9)], 3);
            fixture.rebuild(3, "after growth across the threshold with a cold index");
            fixture.rebuild(4, "no new displaced IDs");

            fixture.automatic.refresh_rebuild_index();
            fixture.union([(input_row(11_000, 0)[2], 10)], 5);
            fixture.rebuild(5, "sparse change with a warm index");

            // Two distinct keys become one; the payload merge is independent
            // of insertion order and timestamps must follow the changed row.
            fixture.union(
                [0, 3]
                    .into_iter()
                    .flat_map(|id| input_row(id, 0)[..3].to_vec())
                    .map(|value| (value, 1)),
                6,
            );
            fixture.rebuild(6, "canonicalization merges colliding keys");
            assert_eq!(fixture.expected.len(), 11_999);

            fixture.compact_warm_tables();
            fixture.union([(input_row(7_999, 0)[2], 11)], 8);
            fixture.rebuild(
                8,
                "sparse change after reverse-index generation invalidation",
            );

            fixture.union([(100, 99)], 9);
            fixture.rebuild(9, "one displaced value with high fanout");
            fixture.union(
                (0..12_000)
                    .map(|id| {
                        let row = input_row(id, 0);
                        (row[2], row[0])
                    })
                    // Include irrelevant global UF changes as well: the
                    // candidate finder must tolerate misses in its index.
                    .chain((0..6_000).map(|id| (80_000 + 2 * id, 80_001 + 2 * id))),
                10,
            );
            fixture.rebuild(10, "dense changes mixed with unrelated displaced IDs");
            assert_eq!(fixture.expected.len(), 11_998);
            fixture.rebuild(11, "idempotent rebuild after dense canonicalization");
        });
    }
}
