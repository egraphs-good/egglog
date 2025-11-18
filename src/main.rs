// use egglog::ast::Command;

#[cfg(feature = "bin")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    // let schema = schemars::schema_for!(Vec<egglog::ast::Command>);
    // println!("{}", serde_json::to_string_pretty(&schema).unwrap());

    egglog::cli(egglog::EGraph::default())
}
