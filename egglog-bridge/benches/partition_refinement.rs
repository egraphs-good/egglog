use divan::{Bencher, counter::ItemsCount};
use egglog_bridge::partition_refinement::refinement::{PartitionRefinement, add_block_hash_table};
use egglog_core_relations::{Database, Value};
use egglog_numeric_id::NumericId;
use rand::{Rng, rng};
use std::sync::Arc;

type Row = [Value; 3];

const ROW_COUNT: usize = 20_000;
const COLLISION_PCTS: [usize; 4] = [1, 10, 50, 90];

fn main() {
    divan::main();
}

fn bucket_count(rows: usize, collision_pct: usize) -> usize {
    let unique_pct = 100usize.saturating_sub(collision_pct);
    let unique = rows.saturating_mul(unique_pct) / 100;
    unique.max(1)
}

fn generate_rows<const COLLISION_PCT: usize>(rows: usize) -> (Arc<Vec<Row>>, usize) {
    let hash_buckets = bucket_count(rows, COLLISION_PCT);
    let block_buckets = bucket_count(rows, COLLISION_PCT);
    let mut rng = rng();
    let mut out = Vec::with_capacity(rows);
    for key in 0..rows {
        let hash = Value::from_usize(rng.random_range(0..hash_buckets));
        let block = Value::from_usize(rng.random_range(0..block_buckets));
        out.push([Value::from_usize(key), hash, block]);
    }
    (Arc::new(out), block_buckets)
}

fn setup_db(rows: &[Row], block_buckets: usize) -> (Database, PartitionRefinement) {
    let mut db = Database::new();
    let block_counter = db.add_counter();
    let ts_counter = db.add_counter();
    let table = add_block_hash_table(&mut db);
    db.with_execution_state(|state| {
        state.inc_counter_by(block_counter, block_buckets.max(1));
        let ts = Value::from_usize(state.read_counter(ts_counter));
        for row in rows {
            state.stage_insert(table.table, &[row[0], row[1], row[2], ts]);
        }
    });
    db.merge_all();
    db.inc_counter(ts_counter);
    let refinement = PartitionRefinement::new(table, block_counter, ts_counter);
    (db, refinement)
}

#[divan::bench(consts = COLLISION_PCTS, sample_count = 10)]
fn split_blocks<const COLLISION_PCT: usize>(bench: Bencher) {
    let (rows, block_buckets) = generate_rows::<COLLISION_PCT>(ROW_COUNT);
    bench
        .with_inputs({
            let rows = Arc::clone(&rows);
            move || setup_db(rows.as_slice(), block_buckets)
        })
        .input_counter(|_| ItemsCount::new(ROW_COUNT))
        .bench_values(|(mut db, mut refinement)| {
            db.with_execution_state(|state| refinement.split_blocks(state));
            db.merge_all();
        });
}

#[divan::bench(consts = COLLISION_PCTS, sample_count = 10)]
fn merge_blocks<const COLLISION_PCT: usize>(bench: Bencher) {
    let (rows, block_buckets) = generate_rows::<COLLISION_PCT>(ROW_COUNT);
    bench
        .with_inputs({
            let rows = Arc::clone(&rows);
            move || setup_db(rows.as_slice(), block_buckets)
        })
        .input_counter(|_| ItemsCount::new(ROW_COUNT))
        .bench_values(|(mut db, mut refinement)| {
            db.with_execution_state(|state| refinement.merge_blocks(state));
            db.merge_all();
        });
}
