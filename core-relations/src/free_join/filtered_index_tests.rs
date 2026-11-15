use std::sync::{Arc, Barrier};

use egglog_concurrency::ThreadPool;
use egglog_reports::ReportLevel;

use super::*;
use crate::{
    PlanStrategy,
    query::CachedPlan,
    table::SortedWritesTable,
    table_shortcuts::{TimestampedTable, v},
};

fn eq(column: u32, value: usize) -> Constraint {
    Constraint::EqConst {
        col: ColumnId::new(column),
        val: v(value),
    }
}

struct Fixture {
    db: Database,
    facts: TableId,
    driver: TableId,
    output: TableId,
    atom: AtomId,
    plan: CachedPlan,
    arity: usize,
    keep_payload: bool,
}

impl Fixture {
    fn new(arity: usize, keep_payload: bool) -> Self {
        let mut db = Database::new();
        // (key, second join key, payload, subsumed, category, timestamp).
        let facts = db.add_table(
            SortedWritesTable::new(
                1,
                6,
                ColumnId::new(5),
                vec![],
                Box::new(|_, old, new, out| {
                    if old == new {
                        false
                    } else {
                        out.extend_from_slice(new);
                        true
                    }
                }),
            ),
            [],
            [],
        );
        let mut add_set = |arity| {
            db.add_table(
                TimestampedTable::new(arity, arity, Box::new(|_, _, _, _| false)),
                [],
                [],
            )
        };
        let driver = add_set(arity);
        let output = add_set(2);
        // SortedWritesTable requires monotone timestamps between merges;
        // writes within a merge may be reordered by their hash shards.
        for x in 0..128 {
            {
                let mut facts_buf = db.new_buffer(facts);
                let mut driver_buf = db.new_buffer(driver);
                facts_buf.stage_insert(&[
                    v(x),
                    v(1_000 + x),
                    v(10_000 + x),
                    v(usize::from(x % 3 == 0)),
                    v(x % 2),
                    v(x),
                ]);
                if x < 16 {
                    driver_buf.stage_insert(&[v(x), v(1_000 + x)][..arity]);
                }
            }
            db.merge_all();
        }

        let mut builder = db.new_rule_set();
        let mut query = builder.new_rule();
        query.set_plan_strategy(PlanStrategy::PureSize);
        query.set_no_decomp(true);
        let x = query.new_var();
        let y = query.new_var();
        let payload = query.new_var();
        let subsumed = query.new_var();
        let category = query.new_var();
        let timestamp = query.new_var();
        query
            .add_atom(driver, &[x.into(), y.into()][..arity], &[])
            .unwrap();
        let atom = query
            .add_atom(
                facts,
                &[
                    x.into(),
                    y.into(),
                    payload.into(),
                    subsumed.into(),
                    category.into(),
                    timestamp.into(),
                ],
                &[eq(3, 0), eq(4, 0)],
            )
            .unwrap();
        let mut rule = query.build();
        rule.insert(
            output,
            &[x.into(), if keep_payload { payload } else { x }.into()],
        )
        .unwrap();
        let rule = rule.build_with_description("filtered-catalog");
        let plan = builder.build().build_cached_plan(rule);
        Self {
            db,
            facts,
            driver,
            output,
            atom,
            plan,
            arity,
            keep_payload,
        }
    }

    fn run(&mut self, lower: usize, upper: usize) {
        self.db.clear_table(self.output);
        let table = self.db.get_table(self.facts);
        let mut expected = table
            .scan(table.all().as_ref())
            .iter()
            .filter(|(_, row)| {
                row[3] == v(0)
                    && row[4] == v(0)
                    && row[5] >= v(lower)
                    && row[5] < v(upper)
                    && self
                        .db
                        .get_table(self.driver)
                        .get_row(&row[..self.arity])
                        .is_some()
            })
            .map(|(_, row)| vec![row[0], row[if self.keep_payload { 2 } else { 0 }]])
            .collect::<Vec<_>>();
        expected.sort();
        let mut builder = self.db.new_rule_set();
        let added = builder.add_rule_from_cached_plan(
            &self.plan,
            &[
                (
                    self.atom,
                    Constraint::GeConst {
                        col: ColumnId::new(5),
                        val: v(lower),
                    },
                ),
                (
                    self.atom,
                    Constraint::LtConst {
                        col: ColumnId::new(5),
                        val: v(upper),
                    },
                ),
            ],
        );
        let rules = builder.build();
        let report = self.db.run_rule_set(&rules, ReportLevel::TimeOnly, None);
        if added.is_some() {
            assert_eq!(report.num_matches("filtered-catalog"), expected.len());
        } else {
            assert!(expected.is_empty());
        }
        let table = self.db.get_table(self.output);
        let mut actual = table
            .scan(table.all().as_ref())
            .iter()
            .map(|(_, row)| row.to_vec())
            .collect::<Vec<_>>();
        actual.sort();
        assert_eq!(actual, expected);
    }

    /// Inspect only indexes execution actually acquired. Calling a catalog
    /// getter here would conceal a regression that never selects this path.
    fn refresh_counts(&self) -> Vec<usize> {
        let info = self.db.get_table_info(self.facts);
        let mut counts = Vec::new();
        let expected = [eq(3, 0), eq(4, 0)];
        if self.arity == 1 {
            info.column_indexes.for_each(|(col, constraints), index| {
                if constraints.is_empty() {
                    return;
                }
                assert_eq!(*col, ColumnId::new(0));
                assert_eq!(constraints.as_slice(), &expected);
                counts.push(index.get().unwrap().refresh_count());
            });
        } else {
            info.indexes.for_each(|(cols, constraints), index| {
                if constraints.is_empty() {
                    return;
                }
                assert_eq!(cols.as_slice(), &[ColumnId::new(0), ColumnId::new(1)]);
                assert_eq!(constraints.as_slice(), &expected);
                counts.push(index.get().unwrap().refresh_count());
            });
        }
        counts
    }
}

#[test]
fn filtered_catalog_reuses_indexes_and_refreshes_after_mutation() {
    for workers in [1, 4] {
        ThreadPool::new(workers).install(|| {
            for (arity, keep_payload) in [(1, false), (1, true), (2, false), (2, true)] {
                let mut fixture = Fixture::new(arity, keep_payload);
                fixture.run(1, 120);
                assert_eq!(fixture.refresh_counts(), [1]);
                fixture.run(3, 121);
                assert_eq!(fixture.refresh_counts(), [1]);

                // A touched but unchanged table resets the lock without
                // scanning or rebuilding the existing index.
                drop(fixture.db.new_buffer(fixture.facts));
                fixture.db.merge_all();
                fixture.run(0, 128);
                assert_eq!(fixture.refresh_counts(), [1]);

                // Subsumption replaces a qualifying row with a nonqualifying
                // one. A payload update leaves a stale candidate, while an
                // append introduces a new qualifying key.
                {
                    let mut buf = fixture.db.new_buffer(fixture.facts);
                    buf.stage_insert(&[v(2), v(1_002), v(10_002), v(1), v(0), v(130)]);
                    buf.stage_insert(&[v(4), v(1_004), v(20_004), v(0), v(0), v(130)]);
                    buf.stage_insert(&[v(130), v(1_130), v(20_130), v(0), v(0), v(130)]);
                    fixture
                        .db
                        .new_buffer(fixture.driver)
                        .stage_insert(&[v(130), v(1_130)][..arity]);
                }
                fixture.db.merge_all();
                assert!(fixture.db.get_table(fixture.facts).has_stale_rows());
                fixture.run(0, 200);
                assert_eq!(fixture.refresh_counts(), [2]);

                // Deletion does not change the append offset. Stale catalog
                // candidates must still be checked without disabling reuse.
                fixture.db.new_buffer(fixture.facts).stage_remove(&[v(8)]);
                fixture.db.merge_all();
                fixture.run(0, 201);
                assert_eq!(fixture.refresh_counts(), [2]);

                let snapshot = fixture.db.clone();
                fixture.db.clear_table(fixture.facts);
                fixture.run(0, 202);
                // Restoring the snapshot must retain independent, valid
                // filtered indexes, including their predicates.
                let cleared = std::mem::replace(&mut fixture.db, snapshot);
                fixture.run(0, 202);
                assert_eq!(fixture.refresh_counts(), [2]);
                fixture.db = cleared;
                for x in 0..96 {
                    {
                        let mut buf = fixture.db.new_buffer(fixture.facts);
                        buf.stage_insert(&[
                            v(x),
                            v(1_000 + x),
                            v(30_000 + x),
                            v(x % 2),
                            v(0),
                            v(200 + x),
                        ]);
                    }
                    fixture.db.merge_all();
                }
                fixture.run(0, 400);
                assert_eq!(fixture.refresh_counts(), [3]);
            }
        });
    }
}

#[test]
fn filtered_catalog_keys_are_canonical_and_single_flight() {
    let fixture = Fixture::new(2, true);
    let info = fixture.db.get_table_info(fixture.facts);
    let barrier = Barrier::new(8);
    let handles = std::thread::scope(|scope| {
        let tasks = (0..8)
            .map(|worker| {
                let barrier = &barrier;
                scope.spawn(move || {
                    let mut constraints = [eq(3, 0), eq(4, 0)];
                    if worker % 2 == 0 {
                        constraints.reverse();
                    }
                    barrier.wait();
                    get_index_from_tableinfo(
                        info,
                        &[ColumnId::new(0), ColumnId::new(1)],
                        &constraints,
                    )
                })
            })
            .collect::<Vec<_>>();
        tasks
            .into_iter()
            .map(|task| task.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(
        handles
            .windows(2)
            .all(|pair| Arc::ptr_eq(&pair[0], &pair[1]))
    );
    assert_eq!(handles[0].get().unwrap().refresh_count(), 1);
    let other_predicate = get_index_from_tableinfo(
        info,
        &[ColumnId::new(0), ColumnId::new(1)],
        &[eq(3, 1), eq(4, 0)],
    );
    assert!(!Arc::ptr_eq(&handles[0], &other_predicate));
    assert!(
        handles[0]
            .get()
            .unwrap()
            .get_subset(&[v(2), v(1_002)])
            .is_some()
    );
    assert!(
        other_predicate
            .get()
            .unwrap()
            .get_subset(&[v(2), v(1_002)])
            .is_none()
    );
    let reversed_columns = get_index_from_tableinfo(
        info,
        &[ColumnId::new(1), ColumnId::new(0)],
        &[eq(3, 0), eq(4, 0)],
    );
    assert!(!Arc::ptr_eq(&handles[0], &reversed_columns));
    assert!(
        reversed_columns
            .get()
            .unwrap()
            .get_subset(&[v(1_002), v(2)])
            .is_some()
    );
}

#[test]
fn narrow_timestamp_deltas_do_not_create_filtered_catalogs() {
    for arity in [1, 2] {
        let mut fixture = Fixture::new(arity, true);
        fixture.run(0, 32);
        assert!(fixture.refresh_counts().is_empty());
    }
}

#[test]
fn filtered_catalog_respects_uncacheable_keys_and_predicates() {
    for arity in [1, 2] {
        for uncacheable in [0, 3] {
            let mut fixture = Fixture::new(arity, true);
            let mut builder = fixture.db.new_rule_set();
            builder
                .add_rule_from_cached_plan(&fixture.plan, &[])
                .unwrap();
            let rules = builder.build();
            // Keep the already-processed sparse header while forbidding
            // persistent indexes on a key or a predicate column.
            fixture.db.tables[fixture.facts]
                .spec
                .uncacheable_columns
                .insert(ColumnId::new(uncacheable), true);
            let report = fixture.db.run_rule_set(&rules, ReportLevel::TimeOnly, None);
            assert_eq!(report.num_matches("filtered-catalog"), 5);
            assert!(fixture.refresh_counts().is_empty());
        }
    }
}
