fn main() {
    println!("cargo:rerun-if-changed=src/ast/parse.lalrpop");
    lalrpop::process_root().unwrap();
}
