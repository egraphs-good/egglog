mod poach;
use poach::poach;

#[cfg(feature = "bin")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    poach();
}
