mod free_var_set;
mod lambda;
mod rise;

fn main() {
    env_logger::init();
    rise::run_rise_benchmark();
}
