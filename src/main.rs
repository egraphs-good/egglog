use schemars::schema_for;

#[cfg(feature = "bin")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let schema = schema_for!(egglog::ast::Command);
    println!("{}", serde_json::to_string_pretty(&schema).unwrap());

    egglog::cli(egglog::EGraph::default())
}
