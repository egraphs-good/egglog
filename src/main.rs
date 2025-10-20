#[cfg(feature = "bin")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    egglog::cli(egglog::EGraph::default());
}
